# Working on this repository

- The human interface is `./mvwd`. Add commands there rather than adding standalone train/infer/eval shell scripts.
- The user runs experiments themselves. Do not launch model training, inference, GPU evaluation, GPU profiling, or submit/allocate Slurm resources unless the user explicitly asks. Implementing automation and CPU tests does not authorize GPU experiments.
- Save every experiment under `artifacts/mvwd/experiments/NAME/`, with config/source snapshots, unique step names, logs, checkpoints, predictions, metrics and reports. Never overwrite or relocate historical predictions, checkpoints or raw scores to reorganize the repository.
- `infer` must finish GT-free generation before exporting GT/BEV comparisons. Record post-generation GT access separately; never change `ground_truth_loaded=false` in generation metadata to describe later visualization.
- Place comparison images beside the corresponding predictions. Preserve original prediction pixels, dataset files and robot assets. Use checksums and reject conflicting existing exports.
- `run.py` and `qwen_mvwd/` define frozen training/cache/resume contracts. Avoid changing them for workflow-only improvements; historical experiments have strict source/config checks.
- New workflow utilities live in `tools/`. One-off historical utilities live in `tools/legacy/`; the README must show the current simple workflow.
