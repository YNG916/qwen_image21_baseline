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

## 当前优先步骤

当前工程实现为 `a6bd52e`，最新实验 `20261017_l1_balanced_resume_v2`。
已固定40训练场景/80 episodes/480视图和3验证场景/18视图；缓存实测10.25/0.37 GiB。场景ID不重叠，房间类型、朝向和FOV覆盖仍有限。
先查看该实验 `reports/REVIEW.md`、`reports/comparisons*`、`reports/data_support_camera_comparison.png`，以实际完成的步骤为准。

1. 用同一批val图和固定噪声诊断判断相机控制、机器人身份及布局；flow loss、PSNR下降/上升均不能替代图像检查。
2. 先在固定小子集检验相机条件是否能充分拟合，并检查原始BEV的局部细节/朝向编码；从已知相机导出的局部BEV可作为新实验。不要只因flow loss下降就直接开长训。下方命令保留为同配置/源码/缓存的恢复示例，固定18-query flow探针。
3. 另用 `select --split val --episodes-per-scene 3` 建立更大的val图像支持，再用相同checkpoint和seed推理；无需更改训练的val-cache恢复协议。不要用test调参。
4. 检查RGB BEV的遮挡/分辨率限制。若充分训练后仍不能跟随实际相机，再单独评估camera/ray adapter；保留现有文本资产条件。
5. Level1的基本视角和布局稳定后，再规划正式训练/test以及Level2的指令、状态更新和before/after监督。

本阶段300步终点完成后，下一次在有效GPU shell内可使用：

```bash
source ~/.config/multiview/storage.sh
cd "$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21"
qwen_python="$MULTIVIEW_STORAGE_ROOT/envs/qwen-image21/bin/python"
exp_root=artifacts/mvwd/experiments/20261017_l1_balanced_resume_v2
export HF_HUB_OFFLINE=1
"$qwen_python" run.py run 20261017_l1_balanced_resume_v2 \
  --name balanced1000 --kind train -- \
  --cache "$exp_root/features/encode_train480" \
  --validation-cache "$exp_root/features/encode_val18" --validate-every 40 \
  --resume-from "$exp_root/training/balanced300/final" --max-steps 1000
```

这是恢复用法示例，不表示300步已被验证集选为最佳模型；是否继续应先查看生成图和最终报告。1000是总更新数，不是额外1000步；该配置仍只使用480-query小子集，不是全集训练。新步骤名不能与已有步骤重复。改变代码或恢复协议需要新实验，不能跳过hash校验。

## 以下为历史实验记录

固定噪声诊断已新增，旧实验的源码 hash 与当前实现不同。请使用新实验；旧缓存不能跳过 hash 校验直接训练。
当前 300 步小子集实验为 `20261017_l1_overfit300_v1`，其缓存已在新源码下重建，且与旧 12 样本逐 tensor 核验完全一致。
训练从预训练权重重新开始，不是恢复旧 100 步 optimizer。现已完成300步、相机诊断、六图推理和PSNR检查；结论见该实验 `reports/REVIEW.md`。

## 正式训练前的工程工作

当前冻结编码缓存实测：Independent 12 samples 约 232 MiB，Joint 4 groups 约 100 MiB。
按当前格式粗略外推，全集 Independent 缓存约 2 TiB；需要先规划配额、分片与条件缓存复用。
NPZ 随机帧读取仍可能解压完整视频数组。

新版已补充 optimizer/随机状态/采样进度恢复及固定 val flow-loss 探针，使用方法见 README。自动 early stopping 和学习率搜索仍未加入；真实图像验证需单独运行 infer/evaluate。
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
不要把拟合分数提高视为相机控制通过。该独立配置对照已完成，见下节，机器人资产文本继续保留。
新条件只能从K/T和已知状态计算，不使用GT depth、场景mesh或visibility。使用新实验并重建编码缓存。

## 新增可选相机条件版本

已新增 `camera_conditioning: bev_frustum_text_v1`，模板为 `configs/examples/level1_camera_cues.json`。
复制到被Git忽略的 `configs/level1_camera_cues.local.json` 并修改dataset_root，创建新实验。
第二张条件BEV加入目标相机CAM标记、光轴和2米光学射线扇形，文本加入位置/方向/FOV摘要；机器人资产文本、状态、clean BEV和训练方法保留。
诊断干预会同步重算标记与文本；具体定义和边界见README。

本机实验：`20261017_l1_camera_cues_v1`，同12个样本、300更新；通过 `reports/cache_control.json` 核对目标和clean BEV latent完全一致。
原矩阵文本模式的真实八case输入逐字节兼容证明在 `20261017_l1_matrix_compat_v2/reports/compatibility.json`。
后者同时保存旧矩阵文本300步LoRA在相同val episode上的对照；两者的完整质量结论以各 `reports/REVIEW.md` 为准。
使用新配置重建缓存并训练新LoRA；原LoRA继续使用原配置与匹配资产，不要混用条件版本。

### 相机提示v1的实际结果与下一阶段

已完成24项测试、300步/1200次样本访问、train/val各6张普通预测、新旧两模式匹配val，以及固定实际噪声的相机诊断。
原版/新版PSNR：train 11.3648/11.5906，val 7.9376/8.8781 dB。两版的目标、raw BEV、机器人状态、相机与seed匹配，证据见 `paired_conditioning_comparison.json`。
相机遵从仍未通过：训练同帧的三视角接近同一墙面，val没有正确恢复货架走廊，有错误物体和标记残留。
训练只覆盖Beechwood_0_garden的2个episodes，val只覆盖grocery_store_half_stocked的1个episode；不得将这些数值作为正式test主表。

下一阶段先扩充训练场景和位姿覆盖，建立场景平衡的小型固定子集；补充optimizer/随机状态恢复和周期性固定val后再扩大训练。
先判断充分数据/训练是否能建立真实视角关联，再单独评估camera/ray adapter或几何监督。当前正式test和Level2均未启动。


## 场景平衡与恢复工程阶段

新实验：`20261017_l1_balanced_resume_v2`。v1 的 select 子命令分发失败保留日志，v2 已修复并加入 CLI 回归测试。

- `selections/train80/selection.json`：40个训练场景，每场景2 episodes，共80 episodes/480固定视图。
- `selections/val3/selection.json`：3个验证场景，每场景1 episode，共3 episodes/18固定视图。
- `selections/train_smoke/selection.json`：2个训练场景/2 episodes/12视图，用于真实模型的恢复验证。
- `diagnostics/train80_plan`、`val3_plan`：只读元数据的查询数量核对，不读取GT。
- `reports/cpu_tests.log`：CPU检查；训练循环测试核对连续/恢复一致及验证不改变训练结果。

该实验使用可选相机提示v1，不改动文本机器人资产条件。最终 GPU 核验与缓存/训练预算以本实验 `reports/REVIEW.md` 为准。480视图训练与18视图验证是小规模阶段，仍不代表正式test。


### 恢复核验结果

32项CPU测试通过；实际训练循环的连续/恢复与插入验证两项等价性均通过。
H100单卡4步与2→4步、双卡2步与1→2步均完成；双卡12样本补齐为16次访问，正确恢复到epoch1/cursor0。
GPU只读审计 `reports/restore_audit.json` 确认LoRA、AdamW以及随机状态在加载后与断点完全一致。
独立启动的BF16训练未达到逐位一致：单卡/双卡最终adapter最大绝对差约3.32e-4/1.99e-4；差异已在中断前出现。对应JSON保留 `passed_exact: false`，不能把它报告为GPU精确重现通过。各rank采样进度和随机状态一致，恢复起点固定val与保存前一致。
缓存/预算在 `reports/stage_plan.json`；相机条件覆盖单独记录，正式test与Level2均未开始。


### 本阶段最终结果（2026-10-18）

已完成480-query缓存、120步→完整恢复至300步、两个终点各18张固定val图、同一GPU/同一实现的三指标，以及8case固定噪声相机复查；23个runner步骤均exit0。
缓存train/val实测10.2527/0.3743 GiB；最终checkpoint为 `training/balanced300/final`，step300/epoch2/cursor240/1200次样本访问。120、250、300步均有完整且有限的256张量LoRA/optimizer状态，120步参考保留。

| 更新数 | PSNR ↑ | LPIPS ↓ | MV-GTWarp-DINO ↑ |
| --- | ---: | ---: | ---: |
| 120 | 9.347860 | 0.768392 | 0.696741 |
| 300 | 9.017678 | 0.765130 | 0.722660 |

这是3 episodes、单seed、18视图和6个有效GT重叠视图对的preview，未算FID，未运行正式test。固定val flow从0.364313降到0.323090，但图像几何仍未通过。
三个场景首帧的9张普通预测仍有旋转/俯视、布局和机器人形体错误。另一个匹配旧实验的val帧中，8cases输入/几何/实际seed逐hash相同；300步新模型仍输出近似同一错误房间，真实三相机PSNR约4.80/4.93/4.57，repeat像素一致，虚拟相机不打GT分数。

完整证据：该实验 `reports/REVIEW.md`、`final_summary.json`、`comparisons_val120_val300/` 和 `data_support_camera_comparison.png`。
机器人资产文本和隐藏top_body的共享定义保留；不能把生成形体正确性视为已解决。下一阶段优先诊断相机输入表示和BEV可见信息；1000步命令只是恢复示例，当前300步不是被验证选定的最佳模型。
