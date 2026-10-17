# 从当前实现继续

本文记录原开发集群上的已有环境与实验，不是新克隆仓库的安装步骤。新环境请先按 [README.md](README.md) 初始化。`artifacts/` 未随 Git 发布；文中的既有结果路径仅适用于保留了本地产物的工作区。

代码位置：`$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21`。
已建环境：`$MULTIVIEW_STORAGE_ROOT/envs/qwen-image21`。
固定权重已在共享 HF cache；GPU 节点可设置 `HF_HUB_OFFLINE=1`，加载器会解析固定本地快照。

## 先看已完成的记录

- Independent：`artifacts/mvwd/experiments/20261017_l1_local_snapshot_smoke/`
- Joint：`artifacts/mvwd/experiments/20261017_l1_joint_local_snapshot_smoke/`
- 训练记录：各实验 `training/*/loss.jsonl`、`steps/`、`logs/`。
- 真实生成：`predictions/val_one_episode/`。每张图旁的 `.comparison.png` 左 GT、右预测。
- Joint 四指标检查：`evaluations/val_four_metrics_preview/metrics.json`。
- 最终结论与限制：各实验 `reports/REVIEW.md`。

已有名称不能重复执行；失败步骤也保留。不要重新执行 README 中与现有实验重名的 create 命令。

## 进入现有 GPU allocation

在作业仍运行时：

```bash
srun --jobid=YOUR_JOB_ID --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --pty bash -l
source ~/.config/multiview/storage.sh
conda activate "$MULTIVIEW_STORAGE_ROOT/envs/qwen-image21"
cd "$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21"
export HF_HUB_OFFLINE=1
```

先用 `nvidia-smi` 检查分配到的卡；CUDA 中的 device 0 不一定是节点物理 GPU 0。

## 接下来优先做什么

1. 查看训练子集的预测与 GT 对照，确认透视输出、机器人身份和朝向是否开始学习。
2. 固定生成 seed 比较同帧三个实际相机，再增加相机位置/方向干预的独立诊断；像素变化不等于控制正确。
3. 扩大到按训练场景平衡选取的小子集，继续用 val 选模；不要用 test 调参。
4. 验证后再准备正式全集训练，不直接把 100 或 1000 步当作充分训练。
5. Level 1 稳定后增加 Level 2：现有三类 event 的语言化、目标定位、before/after 监督和状态执行器。

例如，继续在同一个 12 样本缓存上做新的 300 步诊断（从预训练权重重新开始，**不是恢复 100 步 optimizer**）：

```bash
python run.py run 20261017_l1_local_snapshot_smoke \
  --name overfit_train12_300steps --kind train -- \
  --cache artifacts/mvwd/experiments/20261017_l1_local_snapshot_smoke/features/encode_train12 \
  --max-steps 300
```

这个命令尚未作为正式训练配置选定。若修改配置或训练代码，创建新实验并按校验要求重建缓存。

## 正式训练前的工程工作

当前冻结编码缓存实测：Independent 12 samples 约 232 MiB，Joint 4 groups 约 100 MiB。
按当前格式粗略外推，全集 Independent 缓存约 2 TiB；需要先规划配额、分片与条件缓存复用。
NPZ 随机帧读取仍可能解压完整视频数组。

训练目前没有 optimizer 恢复、自动 val/early stopping 或学习率搜索；正式长训前应补充这些功能。
多卡入口通过 `run ... --processes N --kind train` 调用 Accelerate，实际多卡验证状态以 REVIEW.md 为准。
不要根据两个 optimizer steps 或六张验证图宣称几何一致性、相机控制或泛化已成立。

本阶段机器人条件为资产元数据、身份颜色、状态与 BEV 标记；GLB 投影、额外 camera adapter 和几何一致性 loss 是后续独立版本。
