# 日常操作

从repo根目录使用 `./mvwd`；入口自动选择Qwen或metrics Python。需要GPU的操作只在已有Slurm allocation里执行。
可以先用 `--dry-run`查看即将执行的命令。工具不会申请节点、提交Slurm作业或在后台自动开始下一轮训练。

## 按训练创建实验

```bash
./mvwd init MY_EXPERIMENT --config configs/level1_camera_cues_4h100.local.json \
  --purpose 'Describe the data support and training budget'
```

一个 experiment 对应一次独立训练及其续训轨迹。所有 checkpoint 的推理、不同 seed / split / 子集的检查及评估都留在该 experiment。检查已有 checkpoint 时直接使用原 experiment，不执行 init。Zero-shot 参考可以放在同一 experiment。

本机配置不随Git发布；新机器从`configs/examples/`复制并设置绝对dataset_root。`--from OLD_EXPERIMENT`只复用配置，用于开始另一轮独立训练；不会复制旧权重或训练状态。创建后配置不可修改，步骤名不可复用。

## 选数据、编码和训练

```bash
exp=MY_EXPERIMENT
root="artifacts/mvwd/experiments/$exp"

./mvwd select "$exp" train80 --split train --episodes-per-scene 2 --selection-seed 42
./mvwd select "$exp" val3 --split val --episodes-per-scene 1 --selection-seed 42

./mvwd prepare "$exp" train_features --processes 4 --split train \
  --selection "$root/selections/train80/selection.json"
./mvwd prepare "$exp" val_features --processes 4 --split val \
  --selection "$root/selections/val3/selection.json"

./mvwd train "$exp" fit1000 --processes 4 \
  --cache "$root/features/train_features" \
  --validation-cache "$root/features/val_features" --validate-every 250 \
  --max-steps 1000
```

上述为小子集示例，不是全量训练。正式全量可省略prepare的`--selection`；缓存存储量应先检查。有效batch是进程数×每卡micro_batch_size×gradient_accumulation_steps；四卡示例配置为16。

完整恢复使用新运行名，数据、配置、核心源码、world size和固定验证协议保持一致：

```bash
./mvwd train "$exp" fit2000 --processes 4 \
  --cache "$root/features/train_features" \
  --validation-cache "$root/features/val_features" --validate-every 250 \
  --resume-from "$root/training/fit1000/final" --max-steps 2000
```

`--max-steps`是累计目标，不是额外步数。不要为这些workflow脚本改动重建特征缓存：此次未修改`qwen_mvwd/`或`run.py`。更换模型/数据实现则仍需遵守原有严格hash校验。

## 推理、对照和评估

```bash
./mvwd infer "$exp" val2000 --split val \
  --selection "$root/selections/val3/selection.json" \
  --checkpoint fit2000/final --seed 42

./mvwd eval "$exp" val2000 --preview
./mvwd status "$exp"
```

`--checkpoint`相对于当前 experiment 的 `training/`。完整路径写法 `--lora "$root/training/fit2000/final"` 仍可用。入口会检查 infer 的权重、train 的恢复权重、eval 的预测归属，避免把同一次训练的结果写到其他 experiment。

infer先完成全部GT-free生成并关闭生成阶段，再自动导出GT/BEV对照。`index.html`位于`predictions/val2000/`，可以本地浏览器打开。对应原始分数和易读表位于`evaluations/val2000_eval/metrics.json`与`METRICS.md`。
同一份预测需要另做评估时，使用 `./mvwd eval "$exp" val2000 --name val2000_psnr --preview --metrics psnr`；结果写入同一 experiment 的 `evaluations/val2000_psnr/`，不会覆盖已有评估。

新生成的对照包括逐相机四列图（global BEV / 实际condition BEV / GT / prediction），以及episode/frame级多机器人三行图（condition BEV / GT / prediction）。生成元数据中的`ground_truth_loaded=false`不因后处理读取GT而改变；后处理独立记录在`visualization_receipt.json`。

默认preview计算PSNR、LPIPS、MV-GTWarp-DINO；完整test不加`--preview`时默认包含FID，需要三个seed的完整预测支持，不能把单seed/子集称为正式结果：

```bash
# 先按正式协议分别完成三个seed的全test推理，再执行：
./mvwd eval "$exp" test_k3 --predictions \
  "$root/predictions/test42" "$root/predictions/test43" "$root/predictions/test44"
```

`--metrics-repo`与`--weights-dir`默认指向同workspace的`../top2pano`与其`models/metrics`；其他机器可以显式覆盖。指标数学不在此repo重新实现。

## 给既有结果补导对照

CPU即可，不需要模型权重加载或GPU allocation：

```bash
./mvwd review 20261018_l1_4h100_main_v1 train2000_60 --name train60_review_check
./mvwd status 20261018_l1_4h100_main_v1
```

原预测与旧的两列GT对照不覆盖；已有图必须像素一致才能复用。完整导出可校验重用；如果之前的review工作流失败，使用新review名称：

```bash
./mvwd review EXP INFER_NAME --name REVIEW_RETRY_NAME
```

导出文件若被修改，校验会报错并给出路径，不会替换原文件。修复冲突应先确认来源。

## 查看结果和日志

- `./mvwd status`：生成并显示`artifacts/mvwd/EXPERIMENTS.md`。
- `./mvwd status EXP`：生成并显示`reports/OVERVIEW.md`，汇总每步状态、画廊、指标和日志链接。
- 原始执行信息：`steps/NAME.json`和`logs/NAME.log`。
- 整个用户命令（含推理后导出）的状态：`steps/NAME.workflow.json`和`logs/NAME.workflow.log`。
- `infer`原始步骤成功而GT导出失败时，原预测仍保留；工作流会标记failed并返回非零退出码。

无需把结果分散复制到新的根目录。`config.json`、原始指标和预测保持不变；overview、METRICS.md与实验索引是可以重新生成的汇总文件。

## 已归并的历史检查

`20261018_l1_trainfit_check_v1` 没有独立训练，已归入 `20261018_l1_4h100_main_v1`：

- `selections/train10/`、`predictions/train2000_60/`、`evaluations/train2000_60_metrics/` 位于训练 experiment 下。
- 对应 steps 与 logs 也在同一训练 experiment 下，原始 JSON、日志、图片和指标保留。
- 原 config/source/reports 与逐文件 SHA256 归并记录位于 `provenance/20261018_l1_trainfit_check_v1/`。
- 旧目录保留兼容链接；`./mvwd` 使用旧名时会解析到训练 experiment，索引不再把它列为独立实验。

原始执行回执仍记录当时实际使用的旧路径；这些路径通过兼容链接继续可读。只有派生的 overview、指标表和索引会重新生成。
