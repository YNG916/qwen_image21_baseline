#!/usr/bin/env bash
#SBATCH --account=hk-project-test-p0029577
#SBATCH --partition=gpu-h100
#SBATCH --job-name=qwen-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=artifacts/mvwd/experiments/20261018_l1_full10ep_4h100_v1/logs/slurm-%j.log

# Prefer: ./mvwd batch EXP [--after JOB_ID] [--chain 2].
# Direct sbatch: submit from the repo root; override --output for another EXP.
# Arguments: EXP [train|all|prepare] [cumulative_target_epoch].
set -euo pipefail
: "${SLURM_JOB_ID:?This script must run as a Slurm batch job}"
repo_root="${QWEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd -- "$repo_root"
repo_root="$PWD"
experiment="${1:-20261018_l1_full10ep_4h100_v1}"
stage="${2:-train}"
through_epoch="${3:-10}"
if [[ ! "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid experiment name: $experiment" >&2; exit 2
fi
case "$stage" in train|all|prepare) ;; *) echo 'Stage must be train, all or prepare' >&2; exit 2 ;; esac
if [[ ! "$through_epoch" =~ ^([1-9]|10)$ ]]; then
  echo 'Cumulative target epoch must be 1..10' >&2; exit 2
fi
if [[ -f "$HOME/.config/multiview/storage.sh" ]]; then
  source "$HOME/.config/multiview/storage.sh"
fi
workspace_root="${MULTIVIEW_STORAGE_ROOT:-$(cd "$repo_root/../.." && pwd)}"
task_python="${QWEN_PYTHON:-$workspace_root/envs/qwen-image21/bin/python}"
experiment_root="$repo_root/artifacts/mvwd/experiments/$experiment"
[[ -x "$task_python" && -x ./mvwd && -f "$experiment_root/plans/full_training.json" ]] || {
  echo 'Missing Python, mvwd entry, or initialized full-training experiment' >&2; exit 2;
}
export PYTHONUNBUFFERED=1
# Do not overwrite Slurm's GPU visibility. One srun task starts four Accelerate workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export SLURM_EXPORT_ENV=ALL
restart="${SLURM_RESTART_COUNT:-0}"
receipt="$experiment_root/steps/slurm_job_${SLURM_JOB_ID}_restart_${restart}.json"
snapshot="$experiment_root/source/slurm/${SLURM_JOB_ID}_restart_${restart}.sbatch"
mkdir -p -- "$(dirname -- "$snapshot")"
if [[ -e "$snapshot" ]]; then
  cmp -s -- "${BASH_SOURCE[0]}" "$snapshot" || { echo 'Conflicting Slurm source snapshot' >&2; exit 2; }
else
  cp -- "${BASH_SOURCE[0]}" "$snapshot"
fi
record_state() {
  "$task_python" - "$receipt" "$1" "${2:-}" "$snapshot" "$repo_root" "$experiment" "$stage" "$through_epoch" <<'PY'
import datetime, hashlib, json, os, pathlib, sys
path, status, code, snapshot, root, experiment, stage, through = sys.argv[1:]
file = pathlib.Path(path)
row = json.loads(file.read_text()) if file.exists() else {
    "schema": "qwen-mvwd-slurm-job-v1", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cwd": root, "experiment": experiment, "stage": stage, "through_epoch": int(through),
    "command": ["./mvwd", "full", experiment, "--stage", stage, "--through-epoch", through],
    "script_snapshot": snapshot, "script_sha256": hashlib.sha256(pathlib.Path(snapshot).read_bytes()).hexdigest(),
    "environment": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") or k in
                    ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE")}}
row.update(status=status, exit_code=int(code) if code else None)
if status != "running":
    row["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
tmp = file.with_suffix(".pending")
tmp.write_text(json.dumps(row, indent=2) + "\n")
tmp.replace(file)
PY
}
step_pid=''
final_status=failed
on_exit() {
  code=$?
  trap - EXIT TERM INT
  record_state "$final_status" "$code" || true
  exit "$code"
}
on_signal() {
  trap - TERM INT
  final_status=interrupted
  echo "Received stop signal; ending the driver. Recovery uses the last complete checkpoint." >&2
  if [[ -n "$step_pid" ]]; then
    # srun forwards SIGTERM to the task; the full driver closes its worker group.
    kill -TERM "$step_pid" 2>/dev/null || true
    wait "$step_pid" || true
  fi
  exit 143
}
trap on_exit EXIT
trap on_signal TERM INT
record_state running
printf 'Job %s; experiment %s; stage %s; cumulative epoch %s\n' "$SLURM_JOB_ID" "$experiment" "$stage" "$through_epoch"
srun --unbuffered --ntasks=1 --kill-on-bad-exit=1 \
  ./mvwd full "$experiment" --stage "$stage" --through-epoch "$through_epoch" &
step_pid=$!
if wait "$step_pid"; then
  final_status=completed
else
  code=$?
  exit "$code"
fi
