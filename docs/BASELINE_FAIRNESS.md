# Level 1 comparison protocol and current training budgets

The 20261018 four-H100 Qwen run is a development experiment. It is not a training-budget-matched competitor to the completed full-data Top2Pano run, and must not be presented as a final fair ranking against that model.

| Quantity | Completed Top2Pano run | Current Qwen development budget |
|---|---:|---:|
| Training scene IDs | 40 | 40 |
| Unique training episodes | 18,597 | 800 |
| Fixed frames per episode | 2 | 2 |
| Robot views per frame | 3 | 3 |
| Unique training queries | 111,582 | 4,800 |
| Epochs over those queries | 10 | 6.667 |
| Nominal sample visits excluding distributed padding | 1,115,820 | 32,000 |
| Effective batch | 64 | 16 |
| Optimizer updates | 17,440 actual | 2,000 completed |
| Validation queries | 8,100 | 180 |

Top2Pano sources (local, not included in this Git repository):
`../top2pano/artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4/training/train_gpu4_b16/training_control.json`,
`evaluations/train_gpu4_b16_convergence.json`, and `reports/REVIEW.md` under that experiment.
The recorded training finished ten epochs in about 36h17m on four H100s. This elapsed time includes its training-loop overhead and validation; it is not a model-only FLOP measurement or total project cost.

Top2Pano's DDP sampler adds two repeated queries per epoch, and its last per-device batch is partial. Therefore `updates × nominal effective_batch` is not its exact sample-visit count. Report distributed padding and partial batches explicitly. Qwen pads to full effective batches; its current 4,800-query epoch divides exactly by effective batch16.

For the main controlled comparison:

1. Use the same release, official scene split, full 18,597 train episodes, hashed stratified two-frame selection with seed42, all three observers, and full-FoV RGB256×448. Match the actual query identities, not just scene counts.
2. Predeclare a common data-exposure budget, such as ten epochs over that identical query set, with a consistent validation/checkpoint-selection protocol. Report actual visits, repeats and optimizer updates. Optimizer steps, epoch counts over different subsets, or GPU counts alone are insufficient matching criteria.
3. Select practical batch sizes, learning rates and adaptation methods separately for each architecture. LoRA and Top2Pano's trainable components need not have identical parameter counts or learning rates. Document pretrained checkpoints and trainable parameters.
4. Evaluate the same test queries, inference seed list, resolution, aggregation and metrics. No method uses target RGB/depth as inference input; GT evaluation is separate. Keep test out of tuning.
5. Disclose supervision and asset encodings: Qwen currently uses RGB supervision with text assets; Top2Pano's public implementation includes RGB/depth objectives and renders the same GT-aligned robot assets. Distinguish loss availability from effective gradient paths. These are method differences; matching sample visits does not erase them or unequal pretraining.
6. Also report compute: GPU model/count, elapsed training/validation time and GPU-hours. Report feature/SAM preprocessing and inference cost separately so cached pipelines do not hide their preparation expense. A fixed-compute-budget result is useful as a separately labeled comparison, using the same allowed training pool; it is not a fixed-data-exposure result.

With effective batch16 and Qwen's sampler, full-data ten-epoch training would require `ceil(111582/16) × 10 = 69,740` optimizer updates. This is a budget calculation, not an experiment already launched or an ETA. A 48-hour allocation does not itself establish feasibility; measure actual DDP speed and cache storage first. The current development pipeline is retained unchanged so generated-view quality can inform whether a full-data run is warranted.

## Prepared full-data recipe

`20261018_l1_full10ep_4h100_v1` has a CPU-verified plan through `./mvwd full`; no full-data GPU training result is implied by setup. It matches all actual train episode/frame/robot identities, train/val split snapshots and GT-aligned robot asset manifest to the completed Top2Pano run. It uses111582/8100 full train/val queries, effective batch16,6974 updates/epoch,69740 updates over10 epochs, and1115840 actual sample visits including20 padding repeats. It initializes fresh LoRA from the pinned pretrained Qwen backbone; the development subset checkpoint is not a warm start.

The Qwen primary result is predeclared as the fixed ten-epoch endpoint. Full-val fixed-noise flow loss is recorded for diagnostics; it is not image quality or geometric accuracy. Top2Pano's existing run selected a checkpoint by val loss, so compare fixed-budget endpoints when available and separately label any best-validation variants rather than claiming identical selection rules. All data/compute exposure, padding, preparation, retries and architectural/supervision differences still need disclosure. See [full training](FULL_TRAINING.md).
