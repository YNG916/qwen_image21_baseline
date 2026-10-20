"""Explicit, logged sbatch submission. Never called by full/train/infer automatically."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from qwen_mvwd.common import file_hash, write_json
from tools import workflow as wf


def submit(args):
    from tools.full_training import full_cache, load_plan
    if not 1 <= args.chain <= 8:
        raise ValueError("--chain must be 1..8; it is a fixed job count, not an infinite requeue")
    if not 1 <= args.through_epoch <= 10:
        raise ValueError("--through-epoch must be 1..10")
    if args.after is not None and not re.fullmatch(r"[0-9]+", args.after):
        raise ValueError("--after expects one numeric Slurm job ID")
    experiment = wf.resolve_experiment(args.experiment)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment.name):
        raise ValueError("Slurm experiment names must use letters, digits, dots, underscores or hyphens")
    config, plan = load_plan(experiment)
    if args.stage == "train":
        for split in ("train", "val"):
            full_cache(experiment, config, plan, split)
    script = wf.ROOT / "slurm/full_train_4h100.sbatch"
    base = ["sbatch", "--parsable", "--chdir=" + str(wf.ROOT), "--export=ALL",
            "--output=" + str(experiment / "logs/slurm-%j.log")]
    arguments = [str(script), experiment.name, args.stage, str(args.through_epoch)]
    if args.dry_run:
        for i in range(args.chain):
            dependency = args.after if i == 0 else f"JOB_ID_FROM_SUBMISSION_{i}"
            command = base + (["--dependency=afterany:" + dependency] if dependency else []) + arguments
            print(json.dumps({"cwd": str(wf.ROOT), "argv": command, "submitted": False}))
        return
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is unavailable; submit from the cluster login node")
    # No dataset/model files are changed. Freeze what was submitted separately from
    # the run's native/core snapshots. Job execution has its own source/receipt.
    name = "slurm_submission_" + uuid.uuid4().hex[:12]
    snapshot = experiment / "source/slurm" / (name + ".sbatch")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script, snapshot)
    arguments[0] = str(snapshot)
    receipt_file = experiment / "steps" / (name + ".workflow.json")
    receipt = {"schema": "qwen-mvwd-slurm-submission-v1", "action": "batch", "status": "running",
               "started_utc": wf.utc(), "experiment": experiment.name, "stage": args.stage,
               "through_epoch": args.through_epoch, "requested_jobs": args.chain,
               "script_sha256": file_hash(snapshot), "script_snapshot": str(snapshot.relative_to(experiment)),
               "jobs": [], "exit_code": None}
    write_json(receipt_file, receipt)
    log_file = experiment / "logs" / (name + ".workflow.log")
    dependency = args.after
    try:
        with log_file.open("x") as log:
            for _ in range(args.chain):
                command = base + (["--dependency=afterany:" + dependency] if dependency else []) + arguments
                # These variables can describe a previous interactive allocation and
                # must not leak into a newly scheduled batch job.
                env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"}
                env["QWEN_REPO_ROOT"] = str(wf.ROOT)
                result = subprocess.run(command, cwd=wf.ROOT, env=env, capture_output=True, text=True)
                event = {"argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
                log.write(json.dumps(event) + "\n"); log.flush()
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
                job_id = result.stdout.strip().split(";", 1)[0]
                if not re.fullmatch(r"[0-9]+", job_id):
                    raise ValueError("Unexpected sbatch response; inspect the submission log before retrying")
                receipt["jobs"].append({"job_id": job_id, "afterany": dependency, "argv": command})
                write_json(receipt_file, receipt)
                print(f"Submitted {job_id}" + (f" afterany:{dependency}" if dependency else ""), flush=True)
                dependency = job_id
        receipt.update(status="completed", exit_code=0)
    except BaseException as exc:
        receipt.update(status="failed", exit_code=getattr(exc, "returncode", 1), error=f"{type(exc).__name__}: {exc}")
        # Already submitted jobs remain queued and are explicitly listed in this receipt.
        raise
    finally:
        receipt["finished_utc"] = wf.utc()
        write_json(receipt_file, receipt)
        wf.refresh(experiment)
    print(f"Slurm output: {experiment / 'logs/slurm-JOB_ID.log'}")
