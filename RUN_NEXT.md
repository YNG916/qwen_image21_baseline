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

固定噪声诊断已新增，旧实验的源码 hash 与当前实现不同。请使用新实验；旧缓存不能跳过 hash 校验直接训练。
当前 300 步小子集实验为 `20261017_l1_overfit300_v1`，其缓存已在新源码下重建，且与旧 12 样本逐 tensor 核验完全一致。
训练从预训练权重重新开始，不是恢复旧 100 步 optimizer。现已完成300步、相机诊断、六图推理和PSNR检查；结论见该实验 `reports/REVIEW.md`。

## 正式训练前的工程工作

当前冻结编码缓存实测：Independent 12 samples 约 232 MiB，Joint 4 groups 约 100 MiB。
按当前格式粗略外推，全集 Independent 缓存约 2 TiB；需要先规划配额、分片与条件缓存复用。
NPZ 随机帧读取仍可能解压完整视频数组。

训练目前没有 optimizer 恢复、自动 val/early stopping 或学习率搜索；正式长训前应补充这些功能。
多卡入口通过 `run ... --processes N --kind train` 调用 Accelerate，实际多卡验证状态以 REVIEW.md 为准。
不要根据两个 optimizer steps 或六张验证图宣称几何一致性、相机控制或泛化已成立。

本阶段机器人条件为资产元数据、身份颜色、状态与 BEV 标记；GLB 投影、额外 camera adapter 和几何一致性 loss 是后续独立版本。

## 新增：先运行 camera_diagnose

现已提供固定噪声的相机诊断入口，操作见 README 的“固定噪声的相机响应诊断”。
保留现有资产文本条件，不新增状态/资产消融。
使用新实验，依次比较 zero-shot、旧两步 LoRA 和旧 100 步 LoRA；配置使用原本机 Independent JSON。
两个旧 LoRA 路径分别为：

- `artifacts/mvwd/experiments/20261017_l1_local_snapshot_smoke/training/train_two_steps/final`
- `artifacts/mvwd/experiments/20261017_l1_local_snapshot_smoke/training/overfit_train12_100steps/final`

先确认 repeat_0 与 recorded_0 像素一致，再检查三个真实相机的 GT 对照，最后人工评价没有 GT 的虚拟相机干预。
这是训练小子集诊断，不能作为 val/test 泛化结果；通过后用相同入口在 val 上检查。

## 本机新增实验

- `20261017_l1_camera_response_v1`：zero-shot、两步、100步的固定实际噪声相机诊断；24图与GT后处理已完成。
- `20261017_l1_overfit300_v1`：相同12样本的300步拟合和后续相机复查。

先读前者 `reports/REVIEW.md` 与 `reports/recorded_models_comparison.png`，再读后者的最终报告。
旧100步模型在当前诊断帧对不同目标相机输出近乎相同墙面；不能仅依据PSNR提高扩大到正式全集训练。

### 300步复查结论

原协议同六个query/seed的训练PSNR：2步7.2014、100步10.6257、300步11.3648 dB。
固定实际噪声的三相机诊断中，300步仍对目标相机变化不敏感；该训练帧的旋转/平移/焦距干预没有正确透视响应。
不要把拟合分数提高视为相机控制通过。下一轮先以独立配置测试目标相机BEV位置/视锥标记和简洁朝向描述，机器人资产文本继续保留。
新条件只能从K/T和已知状态计算，不使用GT depth、场景mesh或visibility。使用新实验并重建编码缓存。
