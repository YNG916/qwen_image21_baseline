# Qwen-Image-2.1 × MVWD Level 1

独立 baseline，固定读取正式 MVWD release / split；不修改 Top2Pano、数据集或机器人资产。
当前实现 Level 1。Level 2 的语言标注、编辑训练与状态执行器尚未实现。
本地既有实验的后续操作见 [RUN_NEXT.md](RUN_NEXT.md)。实验产物、权重、数据和本机配置不随 Git 仓库发布。

## 实际条件与能力边界

- 两张参考图：clean environment BEV + 同一 BEV 的米制网格、全部机器人位置/朝向/身份标注。
- 文本包含共享局部米制坐标中的实际 RGB 相机 K/T、全部机器人计划 base pose、测量相机高度、资产部件名称与身份颜色。
- 共享 `MVWD_RobotAssets_v1_gt_aligned`，验证 template fingerprint 和读取的资产元数据 hash。
- 本版本使用资产元数据，**不渲染 GLB，也不输入机器人参考渲染图**；不能宣称实现完整 mesh conditioning。
- Independent 每次输出一个指定相机；Joint 每次输出固定 0/1/2 横向画布，裁出三张完整视野图。
- BEV 保持比例，透视图不 crop；目标 K 按完整视野 resize 更新。世界坐标 Z-up、CV 相机右/下/前。
- 推理不使用目标 RGB/depth/semantic、after 状态、world BEV、场景 mesh 或 GT visibility。仅从 generation_metrics 中提取 trajectory.floor_index 定位楼层，其余结果字段不参与条件。
- 相机通过标注图与数值文本输入。本版本没有额外 ray encoder、共享显式 3D 状态或几何 loss；不保证严格三维一致性。

## 与其他 baseline 对比的输入协议

本版本保持文本机器人资产条件，不强制增加 mesh 渲染或资产外观消融。
允许所有方法访问同一份共享机器人资产、实际相机标定、同帧机器人状态；各方法如何编码这些信息属于方法设计，需在结果表中明确披露。
当前 Qwen 使用资产元数据和 RGB 监督；Top2Pano 带机器人版本使用纹理 mesh 投影与 RGB/depth 训练路径，不能描述为完全相同的条件表达或监督。
统一 train/val/test、目标分辨率、采样帧和评估指标，报告预训练权重、训练样本范围、预算及 checkpoint 选择规则。
Level 1 的测试目标视图、场景 mesh、GT depth/semantic/visibility 不得作为生成条件；GT 仅用于训练监督或生成后的评估。

默认共享包为 `MVWD_RobotAssets_v1_gt_aligned`，其中 `chassis_link/visual/top_body` 已按共享可见性修正移除；prompt 部件列表从该包读取。
这是固定资产版本的修正，不在推理时查询 GT 可见性，也不等于文本条件能硬性保证生成图不幻觉出该外壳。
其他正常的传感器头、顶盖等部件保留；更换资产包必须记录 manifest 和修正来源。

## 文件

`qwen_mvwd/data.py`：无模拟器、SAM、Torch 依赖的数据与相机适配。

`qwen_mvwd/backend.py`：多参考图冻结编码、条件 token/latent 校验、target-only flow loss、LoRA 训练。

`qwen_mvwd/tasks.py`：数据检查、缓存、推理与现有指标 API 的文件适配。

`qwen_mvwd/selection.py`：场景平衡固定清单；`training_state.py`：确定性采样、随机状态和断点完整性检查。

`run.py`：不可变配置、源码快照、唯一命令/日志/退出码记录。

## 环境

使用独立 Python 3.11 环境。不要向 top2pano 或 mvwd-metrics 环境升级 Diffusers/Transformers。

```bash
git clone https://github.com/YNG916/qwen_image21_baseline.git
cd qwen_image21_baseline
conda create -n qwen-image21 python=3.11 pip -y
conda activate qwen-image21
python -m pip install -r requirements.lock.txt

cp configs/examples/level1_independent.json configs/level1_independent.json
cp configs/examples/level1_joint.json configs/level1_joint.json
```

把两个本地配置中的 `dataset_root` 改为实际数据集绝对路径；需要时设置 `robot_assets_root`。
本地配置已被 Git 忽略，模板在 `configs/examples/`。JSON 配置不展开环境变量。
原始数据、完整机器人资产包以及评估使用的 Top2Pano `mvwd_metrics` 需要自行提供。
`--metrics-repo` 指向包含该模块的 Top2Pano checkout，`--weights-dir` 指向对应指标权重目录。

模型 revision 和 Diffusers commit 已固定，`requirements.lock.txt` 记录本次 Linux/CUDA 12.8 环境。
需要适配其他硬件时可使用 `requirements.txt`，并记录实际安装的依赖版本。
可自行设置 `HF_HOME` 保存模型缓存；已有集群环境可以继续使用 `~/.config/multiview/storage.sh`。
原版 checkpoint 用于训练；不使用 Turbo 做普通 SFT。

## 先检查条件

```bash
python run.py create 20261017_l1_independent_smoke \
  --config configs/level1_independent.json --purpose 'L1 independent 最小链路核验'
python run.py run 20261017_l1_independent_smoke \
  --name cpu_inputs --kind inspect -- \
  --split train --max-episodes 2 --preview-samples 3 --with-targets
```

查看 `diagnostics/cpu_inputs/sample_*/condition_*.png`、`geometry.json` 和 `target.png`。
不传 `--with-targets` 时，检查入口也不会读取目标图。

## 已分配 GPU 内验证

在已有 Slurm allocation 的 GPU shell 中执行；代码不申请或提交新作业。

```bash
python run.py run 20261017_l1_independent_smoke \
  --name encode_train12 --kind prepare -- \
  --split train --max-episodes 2 --max-samples 12

python run.py run 20261017_l1_independent_smoke \
  --name train_two_steps --kind train -- \
  --cache artifacts/mvwd/experiments/20261017_l1_independent_smoke/features/encode_train12 \
  --max-steps 2

python run.py run 20261017_l1_independent_smoke \
  --name val_one_episode --kind infer -- \
  --lora artifacts/mvwd/experiments/20261017_l1_independent_smoke/training/train_two_steps/final \
  --split val --max-episodes 1 --seed 42
```

两步训练只验证数值、梯度、保存与加载，不能说明收敛或图像质量。
完整模型首次加载会下载固定权重；模型缓存遵循 `HF_HOME`，已有集群环境由 storage.sh 设置。
显存不足时推理支持 `--cpu-offload`；编码与训练的显存应实际测量。

Joint 用 `configs/level1_joint.json` 创建**另一个实验**，不要修改已创建实验的配置。
Independent 和 Joint 的训练缓存及 LoRA 不可混用。

## 训练协议

第一版：attention LoRA rank=alpha=16、AdamW、LR=5e-5、BF16 基础权重、FP32 adapter 参数、gradient checkpointing。
每卡 batch=1，可变长度条件不拼成不同样本的 batch；通过梯度累积和 DDP 增大有效 batch。
同一参考图的视觉编码器和 VAE 使用一致的 resize，编码后检查视觉 slots 与 latent slots 对齐。
目标 latent 使用确定性 posterior mode；条件保持 clean，仅对目标添加噪声，使用模型 scheduler 的训练时间表计算 flow loss。
没有新增 depth/geometry loss，GT depth 只在评估中使用。

编码缓存会记录每个 tensor 文件 hash、配置/实现 hash 和完整样本选择。先测 12 个样本的磁盘量，再规划全量缓存；不能把全数据集条件编码放到内存。
编码缓存暂不支持中断续写；失败步骤保留，使用新步骤名重试。新版训练支持下述完整断点恢复；旧 LoRA-only checkpoint 仍只用于推理。
正式长训前先验证全部目标 attention 参数有有限梯度、检查条件响应，并在 val 上选模。
当前 `max_steps=1000` 是起点配置，不是已经选定的充分训练预算。

## 评估

使用现有 mvwd-metrics 环境，复用其 PSNR / LPIPS / FID / MV-GTWarp-DINO 数学实现。
Qwen 自己验证 manifest 和 Independent/Joint seed 策略，不伪造 Top2Pano 的逐相机 seed。

```bash
conda activate mvwd-metrics
python run.py run 20261017_l1_independent_smoke \
  --name val_psnr_preview --kind evaluate -- \
  --predictions artifacts/mvwd/experiments/20261017_l1_independent_smoke/predictions/val_one_episode \
  --metrics-repo ../top2pano --weights-dir ../top2pano/models/metrics \
  --preview --metrics psnr --device cpu
```

正式测试需要完整 test 3338 episodes、固定两帧、三个机器人、seeds=42/43/44、256×448。
三个 seed 的预测分别保存后，一次传给 `--predictions`；去掉 `--preview` 并使用全部四项指标。
GT 只由评估入口读取。小样本 FID 只作诊断；MV-GTWarp-DINO 必须与图像质量指标共同解释。
每个实验的 config、源码、steps、logs、features、training、predictions、evaluations、reports 集中保存。

## 验证与后续

```bash
MVWD_METRICS_REPO=../top2pano python -m unittest discover -s tests -v
```

测试覆盖坐标变换、K/ray 一致性、三视角排列、GT 条件隔离、split/asset 校验、seed、flow 梯度及 PSNR 评估适配。
数学 contract 测试使用小型替身，不能替代真实 Qwen GPU 检查。真实运行状态以各实验 `reports/REVIEW.md` 为准。

下一阶段先做训练小子集的 BEV/相机/机器人条件响应，再开展正式 L1 训练。Level 2 在 L1 路径稳定后添加事件语言化和 before→after 编辑监督。

## 固定噪声的相机响应诊断

保留现有文本机器人资产条件。诊断仅支持 Independent，使用 train/val，默认一个 episode 的第一组固定采样帧。
每组八个 case：三个真实相机、相机 0 完全重复、相机 0 绕世界 Z 轴旋转 ±30°、沿相机右方向平移 0.5 m、焦距乘 1.25。
同组全部 case 和不同 checkpoint 共用相机 0 query 派生的实际噪声 seed；参考 BEV、所有机器人 base 状态和资产文本保持一致。
虚拟查询相机变化不移动机器人，不验证空间碰撞，也没有对应 GT。

```bash
python run.py create camera_response_v1 \
  --config configs/level1_independent.json --purpose '固定噪声相机响应诊断'
python run.py run camera_response_v1 --name inputs --kind camera_diagnose -- \
  --split train --max-groups 1 --dry-run
python run.py run camera_response_v1 --name zero_shot --kind camera_diagnose -- \
  --split train --max-groups 1 --zero-shot --seed 42
# 对两个 LoRA 分别运行，使用不同步骤名和对应 checkpoint 路径：
python run.py run camera_response_v1 --name lora100 --kind camera_diagnose -- \
  --split train --max-groups 1 --lora /path/to/lora/final --seed 42
python run.py run camera_response_v1 --name lora100_review --kind camera_review -- \
  --diagnostics artifacts/mvwd/experiments/camera_response_v1/diagnostics/lora100
```

`camera_diagnostics.py` 的计划/生成路径不读取目标图；`camera_review` 在生成完成后才读取 GT。
输出包括逐 case 输入、prompt、完整相机、实际 seed、文件 hash，以及审阅用目标相机射线图。默认 matrix_text 模式的审阅射线图不输入模型；下述可选版本明确把相机标记加入第二张条件图。
review 生成 GT/预测对照、真实相机 PSNR、与三个真实 GT 的 MSE、重复生成一致性和相对锚点的像素变化。
虚拟相机不计算 GT 分数；像素变化或最近 GT 判断都不能替代几何遵从评价。
这些产物使用独立 schema，不能作为正式 benchmark predictions 传入 evaluate。

默认 matrix_text 保持旧输入协议，训练损失不变；旧 LoRA 配置、release 和资产校验仍须通过。
由于实验 runner 和全包实现 hash 已改变，运行需创建新实验；训练缓存继续执行严格 hash 校验，不复用旧缓存进行新训练。

## 可选相机条件：bev_frustum_text_v1

这是独立的 Independent 条件版本，默认配置仍为原矩阵文本。配置模板为 `configs/examples/level1_camera_cues.json`：

```bash
cp configs/examples/level1_camera_cues.json configs/level1_camera_cues.local.json
# 修改本地 dataset_root，然后使用这个配置创建新实验。
python run.py create camera_cues_v1 \
  --config configs/level1_camera_cues.local.json --purpose '目标相机标记与简洁位姿文本对照'
python run.py run camera_cues_v1 --name inputs --kind camera_diagnose -- \
  --split train --max-groups 1 --dry-run
```

`camera_conditioning: bev_frustum_text_v1` 在第二张机器人标注BEV上加入洋红色CAM标记、光轴箭头和固定2米范围的光学边界射线扇形。
相机位置来自实际RGB camera-to-world，方向来自CV相机+Z，边界射线由K反投影再投到BEV。它不是GT可见性、可通行区域或遮挡估计。
文本额外给出局部米制相机中心、光轴单位向量、从+X向+Y的水平朝向、pitch及由完整视野边界像素中心计算的FOV；完整K/T仍保留。
机器人资产文本、身份颜色、全部机器人base状态、clean BEV和训练损失不变；不读取目标RGB/depth/semantic、场景mesh或GT visibility。
虚拟相机诊断会从原机器人标注BEV重画相机提示，并重算文本，避免旧标记/摘要残留。默认矩阵文本模式仍保持两张条件图不变。

该版本需要自己的编码缓存和LoRA，不能用旧LoRA直接推理；与旧版比较时匹配query、目标图、训练预算和生成seed，而不是要求条件图/文本hash相同。
本轮控制实验使用相同12个训练样本、rank16、学习率5e-5、梯度累积4、300个更新，从预训练权重重新开始。小子集结果不代表泛化或正式benchmark。

本机该版本已通过24项测试，并完成同12样本/300步及一个held-out val episode的真实GPU对照。
原版/提示版六图PSNR为train 11.36/11.59、val 7.94/8.88 dB；固定噪声对照仍有相机不敏感、布局错误和标记残留。
这些是单场景训练的小样本诊断，不能判相机遵从通过或作为正式benchmark；本版本保留为可复现的可选实验，不替换默认条件。


## 场景平衡子集、固定验证与训练恢复

`selection.py` 从已校验的官方 split 元数据生成清单，不读取 GT。按 SHA256(seed, scene/episode) 排序，每个场景取相同数量 episode，再按轮次交错场景。清单固定 release、split hash、episode/scene、机器人及帧采样协议，可同时用于 `inspect`、`prepare`、`infer` 的 `--selection`；不能再叠加 `--scene` 或 `--max-episodes`。省略清单时保留旧入口行为，`--max-episodes` 仍是 split 前缀。

以下为 Independent 小规模阶段：train 每场景2个 episode（40场景/80 episodes/480视图），val 每场景1个（3场景/3 episodes/18视图）；各 episode 仍为固定2帧×3机器人。不要用 test 设计子集或选择 checkpoint。

```bash
python run.py create balanced_l1_v1 --config configs/level1_camera_cues.local.json \
  --purpose '场景平衡训练与固定val'
exp_root=artifacts/mvwd/experiments/balanced_l1_v1
python run.py run balanced_l1_v1 --name train80 --kind select -- \
  --split train --episodes-per-scene 2 --selection-seed 42
python run.py run balanced_l1_v1 --name val3 --kind select -- \
  --split val --episodes-per-scene 1 --selection-seed 42
python run.py run balanced_l1_v1 --name train_features --kind prepare -- \
  --split train --selection "$exp_root/selections/train80/selection.json"
python run.py run balanced_l1_v1 --name val_features --kind prepare -- \
  --split val --selection "$exp_root/selections/val3/selection.json"
python run.py run balanced_l1_v1 --name train250 --kind train -- \
  --cache "$exp_root/features/train_features" \
  --validation-cache "$exp_root/features/val_features" --validate-every 50 --max-steps 250
python run.py run balanced_l1_v1 --name continue500 --kind train -- \
  --cache "$exp_root/features/train_features" \
  --validation-cache "$exp_root/features/val_features" --validate-every 50 --max-steps 500 \
  --resume-from "$exp_root/training/train250/final"
python run.py run balanced_l1_v1 --name val500 --kind infer -- \
  --split val --selection "$exp_root/selections/val3/selection.json" \
  --lora "$exp_root/training/continue500/final" --seed 42
```

`prepare` 允许 train/val，禁止 test；val 的 GT 编码只用于验证损失，推理仍从原始条件独立构造输入，不读取此缓存或目标图。
`validation.jsonl` 在开始/恢复时、指定间隔及最终更新处记录全部固定 val queries 的 flow MSE。每个 query 固定噪声和 timestep，并隔离/恢复训练随机状态。它是去噪代理指标；checkpoint 质量还要运行固定 seed 的 val 生成、现有 `evaluate --preview` 和相机诊断，不能根据该损失宣称几何准确。

新版 checkpoint 包含可直接推理的 LoRA、每 rank 的 AdamW/随机状态、下一个采样位置及完整性 hash；不保存冻结的完整 Qwen 权重。`--max-steps` 是包含已完成步骤的总目标。仅在完整 optimizer 边界保存。恢复要求相同代码、配置、缓存、GPU进程数和验证协议；旧 LoRA-only checkpoint 不支持 optimizer 恢复。换配置/数据/进程数属于新实验。
CPU小模型已验证连续训练/恢复逐位相同，并确认验证不改变训练结果；真实H100上已逐项审计LoRA、AdamW与随机状态恢复。默认BF16 GPU的独立训练存在数值差异，固定seed和保存RNG不保证跨运行逐位相同。

训练按 seed 和 epoch 固定打乱；每 epoch 将排列前缀重复补齐至完整有效 batch，日志的样本访问量包含补齐项。loss 现为全部累积 microbatches 和 ranks 的均值，与历史只记录最后一个 microbatch 的日志数值不宜直接比较。尚未加入自动 early stopping 或学习率调度。

本机工程验证记录见 `RUN_NEXT.md`；该阶段仍保留机器人资产文本和隐藏 top_body 的共享资产定义。


当前场景平衡工程阶段已实际完成40训练场景/80 episodes/480视图缓存、120→300步完整恢复及固定3场景/18视图验证，使用可选相机提示v1；train/val缓存约10.25/0.37 GiB。32项CPU测试、单/双卡恢复和GPU状态加载审计已完成，GPU独立运行不保证逐位一致。
同18视图、单seed的120/300步PSNR为9.35/9.02，LPIPS为0.768/0.765，MV-GTWarp-DINO为0.697/0.723（仅6个有效视图对）。真实相机仍近似同一错误画面，布局/机器人形体未达标；这些是工程诊断，不能作为正式test结果。下一步与本机产物位置见 [RUN_NEXT.md](RUN_NEXT.md)。
