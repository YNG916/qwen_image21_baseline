#!/usr/bin/env bash
# Run inside an existing, user-owned four-GPU allocation. Never submits a job.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through srun in an existing allocation}"
source "$HOME/.config/multiview/storage.sh"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21"
qwen_python="$MULTIVIEW_STORAGE_ROOT/envs/qwen-image21/bin/python"
metric_python="$MULTIVIEW_STORAGE_ROOT/envs/mvwd-metrics/bin/python"
experiment=20261018_l1_4h100_main_v1
exp_root="artifacts/mvwd/experiments/$experiment"
# The fixed experiment and its selections must have been created before running this driver.
"$qwen_python" run.py run "$experiment" --name encode_train4800 --kind prepare --processes 4 -- \
  --split train --selection "$exp_root/selections/train800/selection.json"
"$qwen_python" run.py run "$experiment" --name ddp_smoke2 --kind train --processes 4 -- \
  --cache "$exp_root/features/encode_train4800" --max-steps 2
"$qwen_python" run.py run "$experiment" --name main1000 --kind train --processes 4 -- \
  --cache "$exp_root/features/encode_train4800" --validation-cache "$exp_root/features/encode_val180" \
  --validate-every 250 --max-steps 1000
"$qwen_python" run.py run "$experiment" --name main2000 --kind train --processes 4 -- \
  --cache "$exp_root/features/encode_train4800" --validation-cache "$exp_root/features/encode_val180" \
  --validate-every 250 --resume-from "$exp_root/training/main1000/final" --max-steps 2000

# Three matched validation sets and a fixed-noise camera diagnostic run independently.
CUDA_VISIBLE_DEVICES=0 "$qwen_python" run.py run "$experiment" --name val_zero180 --kind infer -- \
  --split val --selection "$exp_root/selections/val30/selection.json" --zero-shot --seed 42 > "$exp_root/logs/driver_val_zero180.log" 2>&1 &
pid_zero=$!
CUDA_VISIBLE_DEVICES=1 "$qwen_python" run.py run "$experiment" --name val1000_180 --kind infer -- \
  --split val --selection "$exp_root/selections/val30/selection.json" --lora "$exp_root/training/main1000/final" --seed 42 > "$exp_root/logs/driver_val1000_180.log" 2>&1 &
pid_mid=$!
CUDA_VISIBLE_DEVICES=2 "$qwen_python" run.py run "$experiment" --name val2000_180 --kind infer -- \
  --split val --selection "$exp_root/selections/val30/selection.json" --lora "$exp_root/training/main2000/final" --seed 42 > "$exp_root/logs/driver_val2000_180.log" 2>&1 &
pid_final=$!
CUDA_VISIBLE_DEVICES=3 "$qwen_python" run.py run "$experiment" --name camera_val2000 --kind camera_diagnose -- \
  --split val --lora "$exp_root/training/main2000/final" --seed 42 > "$exp_root/logs/driver_camera_val2000.log" 2>&1 &
pid_camera=$!
wait "$pid_zero"; wait "$pid_mid"; wait "$pid_final"; wait "$pid_camera"

metric_pids=()
for pair in '0 val_zero180' '1 val1000_180' '2 val2000_180'; do
  read -r gpu name <<< "$pair"
  CUDA_VISIBLE_DEVICES="$gpu" "$metric_python" run.py run "$experiment" --name "${name}_metrics" --kind evaluate -- \
    --predictions "$exp_root/predictions/$name" --metrics-repo ../top2pano --weights-dir ../top2pano/models/metrics \
    --metrics psnr lpips mv_gtwarp_dino --device cuda --preview > "$exp_root/logs/driver_${name}_metrics.log" 2>&1 &
  metric_pids+=("$!")
done
for pid in "${metric_pids[@]}"; do wait "$pid"; done
"$qwen_python" run.py run "$experiment" --name camera_val2000_review --kind camera_review -- \
  --diagnostics "$exp_root/diagnostics/camera_val2000"
for name in val_zero180 val1000_180 val2000_180; do
  "$qwen_python" review_predictions.py --predictions "$exp_root/predictions/$name"
done
"$qwen_python" -c 'from pathlib import Path; import datetime; Path("artifacts/mvwd/experiments/20261018_l1_4h100_main_v1/reports/DRIVER_COMPLETE.txt").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+"\n")'
