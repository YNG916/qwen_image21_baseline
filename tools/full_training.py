"""Full-data four-H100 recipe, orchestration and recovery; no model code changes."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import signal

from qwen_mvwd.common import digest, file_hash, frame_selection, implementation_hash, load_config, read_json, within, write_json
from qwen_mvwd.data import Level1Data
from qwen_mvwd.selection import PROTOCOL_KEYS
from qwen_mvwd.training_state import checkpoint_receipt
from tools import workflow as wf

SCHEMA = "qwen-mvwd-full-training-v1"
WORLD = 4
EPOCHS = 10


def jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def query_rows(data):
    return list(data.queries())


def frame_rows(data):
    return [{"episode_id": r["id"], "scene": r["scene"],
             "frames": frame_selection(r["frames"], data.config["frames_per_episode"], r["id"], data.config["frame_sample_seed"]),
             "robots": data.config["robots"]} for r in data.records]


def verify_reference(config, datasets, reference):
    """Compare actual train query identities, split snapshots and asset manifests."""
    directory = reference / "training/train_gpu4_b16"
    metadata_path = directory / "run_metadata.json"
    meta = read_json(metadata_path)
    control_path = directory / "training_control.json"
    control = read_json(control_path)
    if control["max_epochs"] != EPOCHS:
        raise ValueError("Reference training budget differs from ten epochs")
    reference_data = {r["split"]: r for r in meta["datasets"]}
    sources = {str(p): file_hash(p) for p in (metadata_path, control_path)}
    for split, data in datasets.items():
        old = reference_data[split]
        source = directory / (split + ".jsonl")
        sources[str(source)] = file_hash(source)
        actual = data.provenance()
        if (old["release_id"] != actual["release_id"] or old["split_sha256"] != actual["split_sha256"]
                or file_hash(source) != actual["split_sha256"] or old["episode_ids"] != actual["episode_ids"]
                or old["query_count"] != len(query_rows(data)) or old["target_size"] != config["target_hw"]
                or old["robots"] != config["robots"] or old["bev_size"] != config["bev_size"]
                or old["robot_assets"]["manifest_sha256"] != actual["robot_assets"]["manifest_sha256"]):
            raise ValueError(f"Reference {split} data/queries/geometry/assets do not match")
        if old["frame_sampling"] != {"policy": "stratified_hash", "frames_per_episode": 2, "seed": 42}:
            raise ValueError("Reference frame-selection protocol differs")
    training_plan = read_json(reference / "evaluations/training_plan.json")
    frame_plan = (reference / training_plan["sampling_plan_reference"]).resolve()
    expected = frame_rows(datasets["train"])
    with frame_plan.open() as stream:
        actual_frames = [json.loads(line) for line in stream if line.strip()]
    if actual_frames != expected:
        raise ValueError("Exact Top2Pano training episode/frame/robot query identities differ")
    sources[str(frame_plan)] = file_hash(frame_plan)
    return {"experiment": str(reference), "sources_sha256": sources,
            "exact_training_frame_plan_matched": True, "train_val_split_snapshots_matched": True,
            "validation_frame_policy_matched": True, "asset_manifest_matched": True,
            "reference_sample_visits_excluding_padding": len(query_rows(datasets["train"])) * EPOCHS,
            "differences": ["pretraining and adaptation", "RGB supervision versus Top2Pano RGB/depth objectives",
                            "text robot assets versus rendered robot assets", "batch size and update count",
                            "Qwen primary checkpoint is the fixed ten-epoch endpoint; flow-loss best is diagnostic only"]}


def setup(args):
    experiment = wf.EXPERIMENTS / wf.simple_name(args.experiment)
    if experiment.exists():
        raise FileExistsError("Experiment exists; use full --stage prepare/train/all to continue")
    config = read_json(wf.ROOT / "configs/examples/level1_camera_cues_4h100.json")
    workspace = wf.ROOT.parent.parent
    config["dataset_root"] = str(Path(args.dataset_root or workspace / "datasets/MultiViewWorldDataset_v1").resolve())
    datasets = {split: Level1Data(config, split) for split in ("train", "val")}
    counts = {split: len(query_rows(data)) for split, data in datasets.items()}
    if counts != {"train": 111582, "val": 8100}:
        raise ValueError(f"Expected full baseline support (111582/8100 queries), got {counts}")
    if args.chunk_episodes < 1:
        raise ValueError("chunk-episodes must be positive")
    reference = Path(args.reference or wf.ROOT.parent / "top2pano/artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4").resolve()
    comparison = verify_reference(config, datasets, reference)
    effective = WORLD * config["training"]["micro_batch_size"] * config["training"]["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(counts["train"] / effective)
    config["training"].update(max_steps=steps_per_epoch * EPOCHS, checkpoint_every=1000)
    with tempfile.TemporaryDirectory() as temporary:
        config_file = Path(temporary) / "config.json"
        write_json(config_file, config)
        load_config(config_file)
        wf.main(["init", args.experiment, "--config", str(config_file), "--purpose",
                 "Full Top2Pano-matched L1 data exposure: 111582 train queries x10 epochs; four H100, effective batch16; fixed endpoint baseline"])
    # Recreate dataset identities with the final immutable training configuration.
    datasets = {split: Level1Data(config, split) for split in ("train", "val")}
    plan = {"schema": SCHEMA, "created_utc": wf.utc(), "config_sha256": digest(config),
            "implementation_sha256": implementation_hash(), "world_size": WORLD, "epochs": EPOCHS,
            "effective_batch": effective, "steps_per_epoch": steps_per_epoch,
            "max_steps": steps_per_epoch * EPOCHS, "checkpoint_every": 1000,
            "validate_every": steps_per_epoch, "validation_seed": 137,
            "initialization": "pretrained backbone + fresh LoRA; no development checkpoint warm start",
            "checkpoint_selection": "primary: fixed epoch10 endpoint; best full-val fixed-flow proxy: diagnostic only",
            "reference": comparison, "splits": {}}
    for split, data in datasets.items():
        frames = experiment / "selections" / (split + "_full_frames.jsonl")
        jsonl(frames, frame_rows(data))
        parts = []
        for i, start in enumerate(range(0, len(data.records), args.chunk_episodes)):
            records = data.records[start:start + args.chunk_episodes]
            body = {"schema": "qwen-mvwd-selection-v1", "release_id": data.meta["release_id"],
                    "split": split, "split_sha256": data.split_hash,
                    "query_protocol": {k: config[k] for k in PROTOCOL_KEYS},
                    "policy": {"name": "full-split-contiguous-chunk-v1", "chunk_index": i, "episode_offset": start},
                    "episodes": [{"id": r["id"], "scene": r["scene"]} for r in records]}
            name = f"{split}_part{i:03d}"
            selection = experiment / "selections" / name / "selection.json"
            write_json(selection, {**body, "selection_sha256": digest(body)})
            parts.append({"name": name, "selection": str(selection.relative_to(experiment)),
                          "selection_sha256": file_hash(selection), "episodes": len(records), "queries": len(records) * 6})
        plan["splits"][split] = {"episodes": len(data.records), "scenes": len({r["scene"] for r in data.records}),
                "queries": counts[split], "dataset": data.provenance(), "cache_identity": data.cache_identity(),
                "frames": str(frames.relative_to(experiment)), "frames_sha256": file_hash(frames), "parts": parts}
    plan["exposure"] = {"unique_queries": counts["train"], "nonpadding_visits": counts["train"] * EPOCHS,
                        "padded_queries_per_epoch": steps_per_epoch * effective,
                        "padding_repeats_per_epoch": steps_per_epoch * effective - counts["train"],
                        "actual_visits_at_endpoint": steps_per_epoch * effective * EPOCHS}
    write_json(experiment / "plans/full_training.json", plan)
    write_json(experiment / "reports/FULL_SETUP.json", {"schema": SCHEMA, "plan_sha256": file_hash(experiment / "plans/full_training.json"),
        "reference_verified": comparison, "environment": environment(), "ground_truth_loaded": False, "gpu_work_started": False})
    for name, options in (("val_preview", ["--split", "val", "--episodes-per-scene", "10"]),
                          ("train_preview", ["--split", "train", "--scene-count", "10", "--episodes-per-scene", "1"])):
        wf.main(["select", args.experiment, name, *options, "--selection-seed", "42"])
    wf.refresh(experiment)
    print(f"Full-data recipe ready: {experiment}\n{steps_per_epoch} steps/epoch; {EPOCHS} epochs; {plan['max_steps']} updates")


def environment():
    import importlib.metadata
    versions = {}
    for key in ("torch", "diffusers", "transformers", "peft", "accelerate", "safetensors"):
        try:
            versions[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return {"host": socket.gethostname(), "python": sys.executable, "pid": os.getpid(), "versions": versions,
            "variables": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in
                ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE")}}


def load_plan(experiment):
    plan_path = experiment / "plans/full_training.json"
    if file_hash(plan_path) != read_json(experiment / "reports/FULL_SETUP.json")["plan_sha256"]:
        raise ValueError("Frozen full-training plan changed")
    plan = read_json(plan_path)
    config = load_config(experiment / "config.json")
    manifest = read_json(experiment / "manifest.json")
    if plan["schema"] != SCHEMA or plan["config_sha256"] != digest(config):
        raise ValueError("Full-training config/plan changed")
    if file_hash(experiment / "config.json") != manifest["config_sha256"]:
        raise ValueError("Frozen config changed")
    if plan["implementation_sha256"] != implementation_hash():
        raise ValueError("Frozen model/data implementation changed")
    for name, checksum in manifest["source_sha256"].items():
        if file_hash(wf.ROOT / name) != checksum:
            raise ValueError("Frozen source changed: " + name)
    for split, row in plan["splits"].items():
        if file_hash(experiment / row["frames"]) != row["frames_sha256"]:
            raise ValueError("Full frame plan changed")
        data = Level1Data(config, split)
        if data.provenance() != row["dataset"] or data.cache_identity() != row["cache_identity"]:
            raise ValueError("Pinned full dataset changed")
        for part in row["parts"]:
            if file_hash(experiment / part["selection"]) != part["selection_sha256"]:
                raise ValueError("Feature partition selection changed")
    return config, plan


def cache_manifest(experiment, config, plan, split):
    return {"schema": "qwen-mvwd-features-v1", "complete": True, "config_sha256": digest(config),
            "cache_identity": plan["splits"][split]["cache_identity"], "dataset": plan["splits"][split]["dataset"],
            "implementation_sha256": plan["implementation_sha256"], "samples": [],
            "encoding_processes": WORLD, "selection_limited": False}


def checked_part(path, config, data, expected):
    manifest = read_json(path / "manifest.json")
    if (not manifest.get("complete") or manifest.get("config_sha256") != digest(config)
            or manifest.get("implementation_sha256") != implementation_hash()
            or manifest.get("dataset") != data.provenance() or manifest.get("cache_identity") != data.cache_identity()
            or [r["query"] for r in manifest["samples"]] != expected):
        raise ValueError("Feature partition identity/support mismatch: " + str(path))
    return manifest


def completed_part(experiment, config, split, part):
    data = Level1Data(config, split, selection=experiment / part["selection"])
    expected = query_rows(data)
    # A terminated writer may have completed its native manifest before the outer runner exited.
    for path in sorted((experiment / "features").glob(part["name"] + "_attempt*"), reverse=True):
        file = path / "manifest.json"
        if file.exists():
            try:
                complete = read_json(file).get("complete")
            except json.JSONDecodeError:
                complete = False
            if complete:
                return path, checked_part(path, config, data, expected)
    return None


def next_name(experiment, prefix):
    for i in range(1, 10000):
        name = f"{prefix}_attempt{i:03d}"
        if not any((experiment / folder / name).exists() for folder in ("features", "training")) and not any(
                (experiment / "steps" / (name + suffix)).exists() for suffix in (".json", ".workflow.json")):
            return name
    raise ValueError("Too many attempts")


def execute(experiment, action, name, extra):
    args = argparse.Namespace(action=action, experiment=experiment.name, name=name,
                              dry_run=False, processes=WORLD, review_name=None)
    wf.run_workflow(args, extra)


def assemble_cache(experiment, config, plan, split):
    """Hard-link encoded tensors; do not duplicate TBs or change their bytes."""
    destination = experiment / "features" / (split + "_full")
    destination.mkdir(exist_ok=True)
    manifest = cache_manifest(experiment, config, plan, split)
    expected = query_rows(Level1Data(config, split))
    for part in plan["splits"][split]["parts"]:
        found = completed_part(experiment, config, split, part)
        if found is None:
            raise ValueError("Incomplete encoding partition: " + part["name"])
        source, receipt = found
        for row in receipt["samples"]:
            src = within(source, row["path"])
            relative = Path("parts") / part["name"] / row["path"]
            dest = destination / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not os.path.samefile(src, dest):
                    raise ValueError("Conflicting assembled feature: " + str(dest))
            else:
                os.link(src, dest)
            if src.stat().st_size == 0:
                raise ValueError("Empty feature tensor")
            # Native encoding recorded SHA256; hard links are the same inode. The frozen
            # training loader verifies that SHA256 again on every actual sample read.
            manifest["samples"].append({**row, "path": str(relative), "query_index": len(manifest["samples"])})
    if [r["query"] for r in manifest["samples"]] != expected:
        raise ValueError("Assembled cache differs from the complete query plan")
    file = destination / "manifest.json"
    if file.exists():
        if read_json(file) != manifest:
            raise ValueError("Completed full-cache manifest changed")
    else:
        temporary = destination / "manifest.pending.json"
        write_json(temporary, manifest)
        temporary.replace(file)
    return destination


def full_cache(experiment, config, plan, split):
    path = experiment / "features" / (split + "_full")
    file = path / "manifest.json"
    if not file.exists():
        raise ValueError("Run ./mvwd full EXP --stage prepare first")
    manifest = read_json(file)
    identity = cache_manifest(experiment, config, plan, split)
    if any(manifest.get(k) != v for k, v in identity.items() if k != "samples"):
        raise ValueError("Full cache identity mismatch")
    if [r["query"] for r in manifest["samples"]] != query_rows(Level1Data(config, split)):
        raise ValueError("Full cache does not contain the exact full query set")
    return path, manifest


def prepare(experiment, config, plan, event):
    # A conservative estimate is recorded; filesystem free space does not prove user quota.
    remaining = sum(part["queries"] for split, row in plan["splits"].items() for part in row["parts"]
                    if completed_part(experiment, config, split, part) is None)
    required = math.ceil(remaining * 23 * 2**20 * 1.15) + 60 * 2**30
    free = shutil.disk_usage(experiment).free
    event("storage", free_bytes=free, estimated_additional_bytes=required, user_quota_checked=False,
          note="23 MiB/query estimate +15% headroom +60 GiB checkpoint reserve; hard links do not duplicate tensors")
    if free < required:
        raise OSError("Insufficient filesystem free space for estimated remaining features/checkpoints")
    for split, row in plan["splits"].items():
        for part in row["parts"]:
            if completed_part(experiment, config, split, part) is not None:
                event("reuse_encoding_partition", split=split, part=part["name"])
                continue
            name = next_name(experiment, part["name"])
            event("encode_partition", split=split, part=part["name"], run=name)
            execute(experiment, "prepare", name, ["--split", split, "--selection", str(experiment / part["selection"])])
            if completed_part(experiment, config, split, part) is None:
                raise ValueError("Encoding exited without a complete partition")
        path = assemble_cache(experiment, config, plan, split)
        event("full_cache_ready", split=split, path=str(path), queries=row["queries"])


def resume_contract(config, plan, train, val):
    return {"config_sha256": digest(config), "cache_sha256": digest(train),
            "implementation_sha256": plan["implementation_sha256"], "sample_count": len(train["samples"]),
            "world_size": WORLD, "accumulation": config["training"]["gradient_accumulation_steps"],
            "micro_batch_size": config["training"]["micro_batch_size"],
            "padded_epoch_samples": plan["steps_per_epoch"] * plan["effective_batch"],
            "sampler": "exact-layout-buckets-shuffled-pad-full-effective-batch-v3",
            "validation": {"cache_sha256": digest(val), "every": plan["validate_every"],
                           "seed": plan["validation_seed"], "probe": "one-fixed-noise-and-timestep-per-query-v1"}}


def checkpoints(experiment, contract):
    found = []
    for file in sorted((experiment / "training").glob("epoch*_attempt*/*/metadata.json")):
        try:
            meta = read_json(file)
        except json.JSONDecodeError:
            # A killed checkpoint writer may leave partial JSON; there is no commit receipt.
            continue
        if meta.get("complete"):
            if meta.get("resume_contract") != contract:
                raise ValueError("Checkpoint belongs to a different full-training contract: " + str(file))
            found.append((meta["progress"]["step"], file.parent))
    # Prefer final over periodic saves at the same global step.
    return sorted(found, key=lambda x: (x[0], x[1].name == "final", str(x[1])), reverse=True)


def checkpoint_index(experiment, plan, contract, verify_latest=True):
    found = checkpoints(experiment, contract)
    latest = None
    if found:
        latest = found[0][1]
        if verify_latest:
            checkpoint_receipt(latest, contract)
    index = {"latest": str(latest.relative_to(experiment)) if latest else None,
             "latest_step": found[0][0] if found else 0, "target_step": plan["max_steps"],
             "primary_selection": plan["checkpoint_selection"], "epochs": {},
             "all_checkpoints": [{"step": step, "path": str(path.relative_to(experiment))} for step, path in found]}
    for step, path in reversed(found):
        if step % plan["steps_per_epoch"] == 0:
            index["epochs"][str(step // plan["steps_per_epoch"])] = str(path.relative_to(experiment))
    write_json(experiment / "reports/CHECKPOINTS.json", index)
    for name, relative in {"latest": index["latest"], **{f"epoch_{int(e):02d}": p for e, p in index["epochs"].items()},
                           "final": index["epochs"].get(str(plan["epochs"]))}.items():
        if relative is None:
            continue
        link = experiment / "training" / name
        if link.exists() and not link.is_symlink():
            raise ValueError("Checkpoint alias collides with a directory: " + str(link))
        temporary = link.with_name(name + ".pending")
        if temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(os.path.relpath(experiment / relative, link.parent), target_is_directory=True)
        temporary.replace(link)
    return latest


def log_rows(path):
    """A preempted writer can leave one partial trailing line; keep original bytes."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def summarize(experiment, plan):
    """Aggregate each attempt separately, including retry/replay compute."""
    attempts, validation = [], []
    for folder in sorted((experiment / "training").glob("epoch*_attempt*")):
        rows = log_rows(folder / "loss.jsonl")
        vals = log_rows(folder / "validation.jsonl")
        validation += [{"run": folder.name, **v} for v in vals]
        if rows:
            # This is backend loop elapsed time; final validation/save/initial setup are
            # excluded. Full wall times are preserved separately in native receipts.
            attempts.append({"run": folder.name, "logged_updates": len(rows), "last_step": rows[-1]["step"],
                             "loop_seconds": rows[-1]["elapsed_seconds"],
                             "samples_per_second": rows[-1]["samples_per_second"],
                             "rank0_peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows)})
    phases = []
    for file in sorted((experiment / "steps").glob("*.json")):
        if file.name.endswith(".workflow.json") or not (file.stem.startswith(("train_part", "val_part", "epoch"))):
            continue
        try:
            row = read_json(file)
        except json.JSONDecodeError:
            row = {"status": "incomplete_receipt"}
        elapsed = None
        if row.get("finished_utc") and row.get("started_utc"):
            elapsed = (dt.datetime.fromisoformat(row["finished_utc"]) - dt.datetime.fromisoformat(row["started_utc"])).total_seconds()
        phases.append({"run": file.stem, "status": row.get("status"), "exit_code": row.get("exit_code"),
                       "wall_seconds": elapsed, "allocated_gpu_hours": elapsed * WORLD / 3600 if elapsed is not None else None})
    index_path = experiment / "reports/CHECKPOINTS.json"
    index = read_json(index_path) if index_path.exists() else {"latest_step": 0}
    best = min(validation, key=lambda row: row["flow_loss"]) if validation else None
    result = {"schema": SCHEMA, "updated_utc": wf.utc(), "plan": "../plans/full_training.json", "exposure": plan["exposure"],
              "checkpointed_steps": index["latest_step"], "completed_budget": index["latest_step"] == plan["max_steps"],
              "attempts": attempts, "native_phases": phases, "validation": validation,
              "best_flow_proxy_observation": best,
              "best_proxy_is_not_primary_image_quality_checkpoint": True,
              "logged_sample_visits_including_replayed_attempts": sum(r["logged_updates"] for r in attempts) * plan["effective_batch"],
              "timing_note": "Phase wall time includes model load, full-val probes and checkpoint writes; loop metrics exclude final save/probe. Interrupted receipts may have unknown wall time. GPU-hours describe allocated four-GPU wall time, not FLOPs. Encoding is listed separately from training."}
    write_json(experiment / "reports/FULL_TRAINING.json", result)
    lines = [f"# Full-data training: {experiment.name}", "", f"Checkpointed progress: {index['latest_step']} / {plan['max_steps']} updates.", "",
             "Four H100; BF16; microbatch1/rank; accumulation4; effective batch16; gradient checkpointing disabled.",
             "Full train111582 / val8100 queries. Ten epochs = 1115840 actual visits, including20 padding repeats.", "",
             "Primary checkpoint: fixed ten-epoch endpoint. Full-val fixed-noise flow loss is diagnostic; it does not establish image/camera quality.", "",
             "[Checkpoint paths](CHECKPOINTS.json) · [Machine summary](FULL_TRAINING.json) · [Plan](../plans/full_training.json)", "",
             "| Attempt | Last step | Loop hours | Queries/s | Rank0 peak GiB |", "|---|---:|---:|---:|---:|"]
    for row in attempts:
        lines.append(f"| [{row['run']}](../training/{row['run']}/) | {row['last_step']} | {row['loop_seconds']/3600:.3f} | {row['samples_per_second']:.3f} | {row['rank0_peak_allocated_gib']:.2f} |")
    lines += ["", "Each phase has native/workflow logs and receipts. GPU utilization, memory, power and temperature are sampled every30 seconds in the full-driver GPU JSONL log. All periodic and epoch checkpoints are retained, including optimizer/RNG/sampler state. Rerun the same command to resume; only completed checkpoint updates count toward the declared budget.", ""]
    (experiment / "reports/FULL_TRAINING.md").write_text("\n".join(lines))


def train(experiment, config, plan, through_epoch, event):
    train_path, train_receipt = full_cache(experiment, config, plan, "train")
    val_path, val_receipt = full_cache(experiment, config, plan, "val")
    contract = resume_contract(config, plan, train_receipt, val_receipt)
    latest = checkpoint_index(experiment, plan, contract)
    for epoch in range(1, through_epoch + 1):
        target = epoch * plan["steps_per_epoch"]
        if latest and read_json(latest / "metadata.json")["progress"]["step"] >= target:
            continue
        name = next_name(experiment, f"epoch{epoch:02d}")
        extra = ["--cache", str(train_path), "--validation-cache", str(val_path),
                 "--validate-every", str(plan["validate_every"]), "--validation-seed", str(plan["validation_seed"]),
                 "--max-steps", str(target)]
        if latest:
            extra += ["--resume-from", str(latest)]
        event("train_epoch", epoch=epoch, target_step=target, run=name, resume_from=str(latest) if latest else None)
        try:
            execute(experiment, "train", name, extra)
        finally:
            latest = checkpoint_index(experiment, plan, contract)
            summarize(experiment, plan)
        if latest is None or read_json(latest / "metadata.json")["progress"]["step"] != target:
            raise ValueError("Training phase exited without its target checkpoint")
        event("epoch_complete", epoch=epoch, checkpoint=str(latest))


def gpu_monitor(path, stop):
    command = ["nvidia-smi", "--query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]
    with path.open("x") as stream:
        while not stop.is_set():
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                row = {"utc": wf.utc(), "argv": command, "exit_code": result.returncode,
                       "csv": result.stdout.splitlines(), "stderr": result.stderr}
            except Exception as exc:
                row = {"utc": wf.utc(), "error": str(exc)}
            stream.write(json.dumps(row) + "\n"); stream.flush()
            stop.wait(30)


def main_full(args):
    # Let orchestration record failure and terminate its entire child process group.
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise KeyboardInterrupt("Full driver received SIGTERM; resume from the last complete checkpoint")
    signal.signal(signal.SIGTERM, terminated)
    try:
        return run_full(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_full(args):
    if args.stage == "setup":
        if args.dry_run:
            raise ValueError("Setup reads dataset metadata and creates the recipe; use --stage plan for a read-only existing plan")
        setup(args)
        return
    experiment = wf.resolve_experiment(args.experiment)
    config, plan = load_plan(experiment)
    if not 1 <= args.through_epoch <= EPOCHS:
        raise ValueError("through-epoch must be between1 and10")
    if args.stage == "plan" or args.dry_run:
        print(json.dumps({"experiment": experiment.name, "stage": args.stage, "world_size": WORLD,
                          "splits": {k: {"episodes": v["episodes"], "queries": v["queries"], "encoding_parts": len(v["parts"])} for k, v in plan["splits"].items()},
                          "steps_per_epoch": plan["steps_per_epoch"], "target_step": args.through_epoch * plan["steps_per_epoch"],
                          "max_steps": plan["max_steps"], "exposure": plan["exposure"],
                          "primary_checkpoint": plan["checkpoint_selection"], "gpu_work_started": False}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Enter your existing four-H100 allocation before prepare/train/all")
    import torch
    if torch.cuda.device_count() != WORLD or not all("H100" in torch.cuda.get_device_name(i) for i in range(WORLD)):
        raise RuntimeError("This measured recipe requires exactly four visible H100 GPUs")
    lock_file = (experiment / "steps/full_driver.lock").open("a")
    with lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another full driver is running for this experiment") from None
        name = next_name(experiment, "full_" + args.stage)
        receipt_file = experiment / "steps" / (name + ".workflow.json")
        receipt = {"schema": SCHEMA, "action": "full", "stage": args.stage, "status": "running", "started_utc": wf.utc(),
                   "through_epoch": args.through_epoch, "environment": environment(), "automation": wf.snapshot_tools(experiment),
                   "plan_sha256": file_hash(experiment / "plans/full_training.json"), "argv": sys.argv}
        write_json(receipt_file, receipt)
        stop = threading.Event()
        monitor = threading.Thread(target=gpu_monitor, args=(experiment / "logs" / (name + ".gpu.jsonl"), stop), daemon=True)
        monitor.start()
        with (experiment / "logs" / (name + ".workflow.log")).open("x") as log:
            def event(kind, **fields):
                row = {"utc": wf.utc(), "event": kind, **fields}
                log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
            try:
                if args.stage in ("prepare", "all"):
                    prepare(experiment, config, plan, event)
                if args.stage in ("train", "all"):
                    train(experiment, config, plan, args.through_epoch, event)
                receipt.update(status="completed", exit_code=0)
            except BaseException as exc:
                receipt.update(status="failed", exit_code=getattr(exc, "returncode", 1), error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                stop.set(); monitor.join(timeout=16)
                receipt["finished_utc"] = wf.utc(); write_json(receipt_file, receipt)
                summarize(experiment, plan)
                wf.refresh(experiment)
