# Slurm launchers

`full_train_4h100.sbatch` runs the existing full-data workflow on one four-H100 node, with automatic checkpoint recovery. All results remain under the original training experiment.

Prefer `./mvwd batch EXP` from the repository root. It freezes the submitted script, selects the correct experiment log path and records job IDs. Use `--after JOB_ID` or `--chain 2` for sequential jobs; use `--dry-run` to inspect without submitting.

The script defaults to `20261018_l1_full10ep_4h100_v1`, stage `train`, target epoch10. Direct `sbatch slurm/full_train_4h100.sbatch` from the repo root therefore resumes this experiment. Other experiments need explicit arguments and `--output`.

See [full training](../docs/FULL_TRAINING.md) for resources, commands, logs, interruption behavior and dependencies. No launcher submits its own successor or changes the training configuration.
