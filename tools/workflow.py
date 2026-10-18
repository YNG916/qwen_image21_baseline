"""One interface for MVWD experiments; native training/cache contracts stay unchanged."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_mvwd.common import digest, file_hash, read_json, write_json

ARTIFACTS = ROOT / "artifacts/mvwd"
EXPERIMENTS = ARTIFACTS / "experiments"
ACTIONS = ("init", "select", "prepare", "train", "infer", "eval", "review", "status")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def simple_name(value):
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Experiment and run names must be simple directory names")
    return value


def resolve_experiment(name):
    """Resolve a historical diagnostic alias to its owning training experiment."""
    visited = set()
    while name not in visited:
        visited.add(name)
        experiment = EXPERIMENTS / simple_name(name)
        redirect = experiment / "REDIRECT.json"
        if not redirect.exists():
            return experiment
        name = read_json(redirect)["experiment"]
    raise ValueError("Experiment redirect cycle")


def require_owned_path(experiment, value, category, flag):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    try:
        path.relative_to((experiment / category).resolve())
    except ValueError:
        raise ValueError(
            f"{flag} must belong to {experiment.name}/{category}/. "
            "Use the experiment that owns this training checkpoint or prediction; "
            "do not create another experiment for inference/evaluation."
        ) from None


def bind_training_outputs(args, experiment, extra):
    """Keep checkpoint inference, resumes and evaluations with their training."""
    extra = list(extra)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        path = Path(checkpoint)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("--checkpoint expects a path below EXP/training/, e.g. main2000/final")
        if any(x.split("=", 1)[0] in ("--lora", "--zero-shot") for x in extra):
            raise ValueError("Use one of --checkpoint, --lora or --zero-shot")
        extra += ["--lora", str(experiment / "training" / path)]
    inputs = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    inputs.add_argument("--lora", action="append", default=[])
    inputs.add_argument("--resume-from", action="append", default=[])
    inputs.add_argument("--predictions", nargs="+", action="append", default=[])
    known, _ = inputs.parse_known_args(extra)
    if args.action == "infer":
        for path in known.lora:
            require_owned_path(experiment, path, "training", "--lora")
    elif args.action == "train":
        for path in known.resume_from:
            require_owned_path(experiment, path, "training", "--resume-from")
    elif args.action == "eval":
        for group in known.predictions:
            for path in group:
                require_owned_path(experiment, path, "predictions", "--predictions")
    return extra


def read_steps(experiment):
    steps = {}
    for path in sorted((experiment / "steps").glob("*.json")):
        try:
            steps[path.stem] = read_json(path)
        except (OSError, json.JSONDecodeError):
            steps[path.stem] = {"status": "writing"}
    return steps


def refresh(experiment):
    """Rebuild a concise overview without touching raw scores, logs or images."""
    if not (experiment / "manifest.json").exists():
        return
    steps = read_steps(experiment)
    lines = [f"# {experiment.name}", "", read_json(experiment / "manifest.json").get("purpose", ""), "",
             "One experiment owns one training trajectory, including resumed checkpoints and all inference/evaluation runs.", "",
             "Config: [config.json](../config.json). Logs: [logs/](../logs/). Source: [source/](../source/).", "",
             "| Step | Status | Log |", "|---|---|---|"]
    for name, row in steps.items():
        if name.endswith(".workflow"):
            if name.removesuffix(".workflow") in steps:
                continue
            name = name.removesuffix(".workflow")
        workflow = steps.get(name + ".workflow", {})
        status = workflow.get("status", row.get("status", "unknown"))
        logfile = name + ".log" if (experiment / "logs" / (name + ".log")).exists() else name + ".workflow.log"
        lines.append(f"| {name} | {status} | [log](../logs/{logfile}) |")
    for folder in sorted((experiment / "predictions").glob("*")):
        if not folder.is_dir() or not (folder / "metadata.json").exists():
            continue
        meta = read_json(folder / "metadata.json")
        lines += ["", f"## Predictions: {folder.name}", "",
                  f"Split: {meta['dataset']['split']}; complete: {meta['complete']}; views: {meta.get('view_count', '?')}; seed: {meta['base_seed']}.", "",
                  f"[Images](../predictions/{folder.name}/)"]
        if (folder / "index.html").exists():
            lines.append(f"[BEV / GT / prediction gallery](../predictions/{folder.name}/index.html)")
    lines += ["", "## Metrics", "", "| Evaluation | PSNR ↑ | LPIPS ↓ | MV-GTWarp-DINO ↓ | Scope |", "|---|---:|---:|---:|---|"]
    for path in sorted((experiment / "evaluations").glob("*/metrics.json")):
        result = read_json(path)
        def cell(key):
            value = result.get("metrics", {}).get(key, {}).get("mean")
            return f"{value:.6f}" if isinstance(value, (float, int)) else "—"
        scope = "formal" if result.get("benchmark_ready") else "preview / diagnostic"
        metric_lines = [f"# {path.parent.name}", "", f"Scope: {scope}. Raw scores: [metrics.json](metrics.json).", "",
                        "| Metric | Mean | Episode count | 95% CI |", "|---|---:|---:|---|"]
        for key, value in result.get("metrics", {}).items():
            mean = value.get("mean")
            mean_text = f"{mean:.6f}" if isinstance(mean, (float, int)) else "undefined"
            ci = value.get("ci95")
            ci_text = f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci and all(isinstance(x, (float, int)) for x in ci) else "—"
            metric_lines.append(f"| {key} | {mean_text} | {value.get('episode_count', '—')} | {ci_text} |")
        (path.parent / "METRICS.md").write_text("\n".join(metric_lines) + "\n")
        lines.append(f"| [{path.parent.name}](../evaluations/{path.parent.name}/metrics.json) | {cell('psnr')} | {cell('lpips')} | {cell('mv_gtwarp_dino')} | {scope} |")
    if (experiment / "plans/full_training.json").exists():
        lines += ["", "## Full-data training", "", "[Frozen plan](../plans/full_training.json)"]
        for name in ("FULL_TRAINING.md", "CHECKPOINTS.json"):
            if (experiment / "reports" / name).exists():
                lines.append(f"[{name}]({name})")
    imports = sorted((experiment / "provenance").glob("*/REDIRECT.json"))
    if imports:
        lines += ["", "## Imported diagnostics", ""]
        for path in imports:
            label = path.parent.name
            lines.append(f"- {label}: outputs are included above; [original reports](../provenance/{label}/reports/REVIEW.md), "
                         f"[checksummed consolidation receipt](../provenance/{label}/consolidation.json). Original paths remain aliases.")
    lines += ["", "GT/BEV visualizations are exported after completed generation. Training-set diagnostics do not establish held-out generalization.", ""]
    (experiment / "reports").mkdir(exist_ok=True)
    (experiment / "reports/OVERVIEW.md").write_text("\n".join(lines))
    refresh_index()


def refresh_index():
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    lines = ["# MVWD experiments", "", "Generated by `./mvwd status`. One experiment per independent training; resumes, inference and evaluation stay within it. Historical diagnostic aliases are excluded.", "",
             "| Experiment | Completed steps | Failed steps | Overview |", "|---|---:|---:|---|"]
    for folder in sorted(EXPERIMENTS.iterdir()):
        if not folder.is_dir() or (folder / "REDIRECT.json").exists():
            continue
        steps = read_steps(folder)
        logical = {k: steps.get(k + ".workflow", v) for k, v in steps.items() if not k.endswith(".workflow")}
        logical.update({k.removesuffix(".workflow"): v for k, v in steps.items()
                        if k.endswith(".workflow") and k.removesuffix(".workflow") not in steps})
        completed = sum(r.get("status") == "completed" for r in logical.values())
        failed = sum(r.get("status") == "failed" for r in logical.values())
        target = f"experiments/{folder.name}/reports/OVERVIEW.md" if (folder / "reports/OVERVIEW.md").exists() else f"experiments/{folder.name}/"
        lines.append(f"| {folder.name} | {completed} | {failed} | [open]({target}) |")
    (ARTIFACTS / "EXPERIMENTS.md").write_text("\n".join(lines) + "\n")


def snapshot_tools(experiment):
    files = [ROOT / "mvwd", ROOT / "review_predictions.py", ROOT / "tools/workflow.py", ROOT / "tools/visualizations.py", ROOT / "tools/full_training.py"]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in files}
    identity = digest(hashes)
    directory = experiment / "source/automation" / identity
    for path in files:
        destination = directory / path.relative_to(ROOT)
        if destination.exists():
            if file_hash(destination) != hashes[str(path.relative_to(ROOT))]:
                raise ValueError("Automation snapshot was modified")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return {"sha256": identity, "files": hashes, "path": str(directory.relative_to(experiment))}


def stream_command(command, log):
    print("Running:", json.dumps(command, ensure_ascii=False), flush=True)
    log.write(json.dumps({"argv": command, "started_utc": utc()}) + "\n"); log.flush()
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        for line in child.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        raise


def run_workflow(args, extra):
    experiment = resolve_experiment(args.experiment)
    if not (experiment / "manifest.json").exists():
        raise ValueError("Experiment does not exist; use ./mvwd init first")
    simple_name(args.name)
    native_kind = {"eval": "evaluate"}.get(args.action, args.action)
    name = (getattr(args, "eval_name", None) or args.name + "_eval") if args.action == "eval" else args.name
    if args.action == "review":
        name = args.review_name or args.name + "_review"
    simple_name(name)
    if any(x.split("=", 1)[0] in ("--config", "--output") for x in extra):
        raise ValueError("Use the frozen experiment config; --config/--output belong to the runner")
    if args.action == "eval":
        if not any(x.split("=", 1)[0] == "--predictions" for x in extra):
            extra += ["--predictions", str(experiment / "predictions" / args.name)]
        for flag, value in (("--metrics-repo", ROOT.parent / "top2pano"),
                            ("--weights-dir", ROOT.parent / "top2pano/models/metrics")):
            if not any(x.split("=", 1)[0] == flag for x in extra):
                extra += [flag, str(value)]
        if not any(x.split("=", 1)[0] == "--metrics" for x in extra):
            extra += ["--metrics", "psnr", "lpips", "mv_gtwarp_dino"]
            if "--preview" not in extra:
                extra += ["fid"]
    extra = bind_training_outputs(args, experiment, extra)
    processes = getattr(args, "processes", 1)
    commands = []
    if args.action != "review":
        commands.append([sys.executable, str(ROOT / "run.py"), "run", experiment.name,
                         "--name", name, "--kind", native_kind, "--processes", str(processes), "--", *extra])
    if args.action in ("infer", "review"):
        commands.append([os.environ.get("QWEN_PYTHON", sys.executable), str(ROOT / "review_predictions.py"),
                         "--predictions", str(experiment / "predictions" / args.name)])
    if args.dry_run:
        for command in commands:
            print(json.dumps({"cwd": str(ROOT), "argv": command}, ensure_ascii=False))
        return
    device = "cuda"
    for i, value in enumerate(extra):
        if value.startswith("--device="):
            device = value.split("=", 1)[1]
        elif value == "--device" and i + 1 < len(extra):
            device = extra[i + 1]
    cpu_eval = device.split(":", 1)[0] == "cpu"
    if (args.action in ("train", "prepare", "infer") or (args.action == "eval" and not cpu_eval)) and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Enter an existing GPU allocation before this command; ./mvwd never submits jobs")
    receipt_path = experiment / "steps" / (name + ".workflow.json")
    if receipt_path.exists():
        raise FileExistsError(f"Run already exists: {name}. Use a new run name; eval/review also accept --name.")
    if args.action != "review" and (experiment / "steps" / (name + ".json")).exists():
        raise FileExistsError(f"Native step already exists: {name}; use a new run name")
    with receipt_path.open("x") as stream:
        stream.write("{}\n")
    receipt = {"schema": "qwen-mvwd-workflow-v1", "action": args.action, "name": name,
        "status": "running", "exit_code": None, "started_utc": utc(), "argv": commands,
        "cwd": str(ROOT), "python": sys.executable, "config_sha256": file_hash(experiment / "config.json"),
        "environment": {k: v for k, v in os.environ.items()
            if k.startswith("SLURM_") or k in ("CUDA_VISIBLE_DEVICES", "CONDA_PREFIX", "HF_HUB_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    write_json(receipt_path, receipt)
    try:
        receipt["automation"] = snapshot_tools(experiment)
        write_json(receipt_path, receipt)
        with (experiment / "logs" / (name + ".workflow.log")).open("x") as log:
            for command in commands:
                stream_command(command, log)
        receipt.update(status="completed", exit_code=0)
    except BaseException as exc:
        receipt.update(status="failed", exit_code=getattr(exc, "returncode", 1), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["finished_utc"] = utc(); write_json(receipt_path, receipt)
        try:
            refresh(experiment)
        except Exception as exc:
            print(f"Overview refresh failed: {exc}", file=sys.stderr)
    print(f"Saved: {experiment / 'reports/OVERVIEW.md'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./mvwd", description="MVWD: one entry for training, inference, evaluation and artifact review. All relative paths use the repository root.")
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init", help="Start a new independent training experiment (not needed for infer/eval)")
    init.add_argument("experiment")
    source = init.add_mutually_exclusive_group(required=True)
    source.add_argument("--config"); source.add_argument("--from", dest="from_experiment", help="Reuse configuration for a new independent training; existing checkpoints stay in their original experiment")
    init.add_argument("--purpose", default="MVWD experiment")
    native_help = {
        "select": "Options: --split train|val --episodes-per-scene N [--scene-count N] [--selection-seed 42]",
        "prepare": "Options: --split train|val [--selection FILE] [--max-samples N]\nExample: ./mvwd prepare EXP train_features --processes 4 --split train --selection FILE",
        "train": "Options: --cache DIR [--validation-cache DIR --validate-every N]\n         [--resume-from CHECKPOINT_DIR] [--max-steps TOTAL]\nExample: ./mvwd train EXP fit6000 --processes 4 --cache DIR --resume-from DIR --max-steps 6000",
        "infer": "Options: --split train|val|test [--selection FILE] [--max-episodes N]\n         --checkpoint TRAIN_RUN/final | --lora CHECKPOINT_DIR | --zero-shot\n         [--seed 42] [--cpu-offload]\nGT, BEV, multi-camera comparisons and index.html are exported automatically after generation.",
        "eval": "By default NAME refers to predictions/NAME; evaluation is saved as NAME_eval (override with --name).\nOptions: [--predictions DIR [DIR ...]] [--preview] [--device cuda|cpu]\n         [--metrics psnr lpips fid mv_gtwarp_dino]\n         [--metrics-repo DIR] [--weights-dir DIR]\nUse --preview for train/val subsets. Formal evaluation requires the full test protocol.",
        "review": "NAME refers to an existing inference run. This command is CPU-only; it never regenerates predictions.",
    }
    for action in ACTIONS[1:-1]:
        p = sub.add_parser(action, formatter_class=argparse.RawDescriptionHelpFormatter,
                           epilog=native_help[action], help={"infer": "Generate, then automatically export GT/BEV comparisons", "eval": "Evaluate an inference run (or explicit --predictions)", "review": "CPU-only GT/BEV backfill for an existing inference run"}.get(action, action))
        p.add_argument("experiment"); p.add_argument("name", help="Run name; for eval/review, the inference run name")
        p.add_argument("--dry-run", action="store_true", help="Print commands without running, allocating or writing anything")
        if action in ("train", "prepare"):
            p.add_argument("--processes", type=int, default=1)
        if action == "infer":
            p.add_argument("--checkpoint", help="Checkpoint relative to this experiment's training/, e.g. main2000/final")
        if action == "eval":
            p.add_argument("--name", dest="eval_name", help="Unique evaluation name; allows multiple evaluations of the same inference")
        if action == "review":
            p.add_argument("--name", dest="review_name", help="Unique review step name when retrying")
    full = sub.add_parser("full", help="Full-data four-H100 setup, encoding, training and automatic recovery")
    full.add_argument("experiment")
    full.add_argument("--stage", choices=("setup", "plan", "prepare", "train", "all"), default="plan")
    full.add_argument("--dataset-root", help="Setup only; defaults to the workspace MVWD dataset")
    full.add_argument("--reference", help="Setup only; completed Top2Pano experiment for exact data-protocol checks")
    full.add_argument("--chunk-episodes", type=int, default=1024, help="Setup only; bounded, restartable encoding partitions")
    full.add_argument("--through-epoch", type=int, default=10, help="Cumulative training target,1..10; default10")
    full.add_argument("--dry-run", action="store_true")
    status = sub.add_parser("status", help="Show results and refresh the artifact index")
    status.add_argument("experiment", nargs="?")
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.action == "full":
        if extra:
            parser.error("Unrecognized full arguments: " + " ".join(extra))
        from tools.full_training import main_full
        main_full(args)
    elif args.action == "init":
        if extra:
            parser.error("Unrecognized init arguments: " + " ".join(extra))
        simple_name(args.experiment)
        config = args.config or str(resolve_experiment(args.from_experiment) / "config.json")
        subprocess.run([sys.executable, str(ROOT / "run.py"), "create", args.experiment,
                        "--config", config, "--purpose", args.purpose], cwd=ROOT, check=True)
        experiment = EXPERIMENTS / args.experiment
        snapshot_tools(experiment); refresh(experiment)
    elif args.action == "status":
        if extra:
            parser.error("Unrecognized status arguments: " + " ".join(extra))
        if args.experiment:
            experiment = resolve_experiment(args.experiment)
            if not (experiment / "manifest.json").exists():
                raise ValueError("Unknown experiment")
            refresh(experiment)
            print((experiment / "reports/OVERVIEW.md").read_text())
        else:
            refresh_index(); print((ARTIFACTS / "EXPERIMENTS.md").read_text())
    else:
        if args.action == "review" and extra:
            parser.error("Unrecognized review arguments: " + " ".join(extra))
        run_workflow(args, extra)


if __name__ == "__main__":
    main()
