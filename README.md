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

## 文件

`qwen_mvwd/data.py`：无模拟器、SAM、Torch 依赖的数据与相机适配。

`qwen_mvwd/backend.py`：多参考图冻结编码、条件 token/latent 校验、target-only flow loss、LoRA 训练。

`qwen_mvwd/tasks.py`：数据检查、缓存、推理与现有指标 API 的文件适配。

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
本版缓存和训练不支持中断恢复。失败步骤保留，使用新步骤名重试；已保存 LoRA 可直接推理。
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
