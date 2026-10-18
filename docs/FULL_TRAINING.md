# 全量四卡 H100 训练

所有操作仍使用 `./mvwd`。本机已完成 **CPU 配置与协议核验** 的 experiment 为：

```bash
exp=20261018_l1_full10ep_4h100_v1
```

没有开始编码或训练。与开发实验独立，从固定 Qwen 预训练模型初始化新的 LoRA。后续训练、续训、infer、eval 全部使用该 experiment。

## 运行

在自己申请的四卡 H100 shell 内执行；不需要手动激活环境：

```bash
source ~/.config/multiview/storage.sh
cd "$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21"
exp=20261018_l1_full10ep_4h100_v1

# CPU 读取并显示冻结计划，不运行模型。
./mvwd full "$exp" --stage plan

# 编码全部 train/val，然后训练至累计10 epochs。
./mvwd full "$exp" --stage all --through-epoch 10
```

工具不申请 allocation，也不提交 Slurm。需要恰好四张可见 H100。普通 shell 前台执行；关闭终端会影响任务，请在已有的持久会话中运行。

当前速度的训练主体线性估计约70小时，另有编码、全验证及保存开销；这不是全量实测 ETA。48小时 allocation 预计需要跨 allocation 恢复。进入新的同规格 allocation 后，**重复同一条 `--stage all` 命令**即可：已完成编码块会复用，训练自动读取最近完整 checkpoint；无需填写恢复路径。

如果希望先查看一个完整 epoch，可先用 `--through-epoch 1`，检查后再改成10；配置、缓存和恢复协议不变。只编码或只训练分别使用 `--stage prepare` / `--stage train`。这些阶段不会自动进行图像生成或测试集评估。

## 固定协议和效率配置

| 项目 | 设置 |
|---|---|
| 训练数据 | 40场景，18,597 episodes，111,582 queries |
| 验证数据 | 1,350 episodes，8,100 queries，全量固定 flow-loss probe |
| 帧与输出 | 每 episode 两帧，hash分层采样 seed42，三机器人，RGB256×448 |
| 训练预算 | 10 epochs；每 epoch6,974步；累计69,740步 |
| 样本访问 | 1,115,820次非padding访问；加20次padding，共1,115,840次 |
| 硬件与精度 | 单节点四 H100，BF16，四进程 DDP |
| Batch | 每卡microbatch1 ×四卡×梯度累积4 =16 |
| 梯度检查点 | 关闭；沿用实测配置，避免较低吞吐 |
| LoRA | attention q/k/v/out，rank/alpha16，lr5e-5 |
| 恢复保存 | 每1,000步 + 每个epoch末，保存完整状态 |
| 验证 | 每epoch末全部8,100个固定噪声/时间步查询；启动/恢复时也执行一次 |
| 主要checkpoint | 第10个epoch固定预算终点；最低flow proxy仅诊断，不替代图像/几何质量 |

使用现有测量中性价比较好的配置，不宣称经过新的全量硬件调优。每卡batch2曾显著增加显存，吞吐提升有限；仅扩大梯度累积也不会按比例缩短训练。

与已完成Top2Pano run逐项核验 train/val split文件、实际训练 episode/frame/robot 清单、输出协议和资产manifest。两者的预训练、监督与资产编码仍有差异；Top2Pano原先按val loss选择模型，而本实验主要报告固定预算终点，需在最终比较中披露，另报告两种选择规则的结果时明确标注。详见[公平比较协议](BASELINE_FAIRNESS.md)。

## 缓存和中断恢复

编码被分为19个train块与2个val块，每块最多1,024 episodes。每块使用四卡；原始日志和回执保存在同一experiment。完整块会复用；中断块重新编码到新的attempt目录，保留失败记录。完整缓存合并使用**硬链接**，不会复制数TB张量；训练时仍逐样本核验SHA256。

按已有缓存估算，train+val特征约2.5TiB；实现按23MiB/query保守估计、加15%余量和60GiB checkpoint储备检查文件系统可用空间。**文件系统free不代表个人quota已核实**，运行前应确认自己的workspace容量。失败块残留可能额外占用一个块的空间；不会自动删除。

SIGTERM/Ctrl-C会结束整个训练子进程组并记录失败；不能触发即时checkpoint。恢复点是最近一次已经完整写入且SHA256通过的状态，最多重放约1,000步；强制终止时也可能需要重做当前编码块。SIGKILL或节点故障无法保证最终回执被写完，已有原始日志仍保留，重跑时使用新attempt名。

保存的状态包括LoRA、每rank optimizer、Python/NumPy/PyTorch/CUDA RNG、训练noise generator及采样进度。恢复必须保持四卡、配置、数据、缓存、核心源码、验证频率及seed一致。不要在训练期间编辑`run.py`或`qwen_mvwd/`；此次实现未修改这些核心文件。

## 日志、checkpoint与结果

```text
experiments/EXP/
├── plans/full_training.json          # 冻结预算、数据身份、参考核验与编码分块
├── selections/                      # 完整帧清单、编码块、固定可视化子集
├── features/
│   ├── train_part000_attempt001/     # 原始编码块与checksum
│   ├── train_full/                   # 硬链接合并，完整训练缓存
│   └── val_full/                     # 完整验证缓存
├── training/
│   ├── epoch01_attempt001/
│   │   ├── loss.jsonl                # 每步loss、累计样本量、吞吐、rank0显存高水位
│   │   ├── validation.jsonl          # 固定全val flow-loss probe
│   │   ├── step_001000/              # 完整恢复状态
│   │   └── final/                    # 第1个epoch完整状态
│   ├── epoch_01 -> …/final           # 已完成epoch的便捷链接
│   ├── latest -> …                   # 最近完整状态；中断后重跑时刷新
│   └── final -> …                    # 完成第10个epoch后建立
├── logs/
│   ├── full_all_attempt001.workflow.log  # driver阶段、复用、恢复与完成事件
│   ├── full_all_attempt001.gpu.jsonl     # 每30秒GPU利用率/显存/功率/温度
│   └── epoch01_attempt001*.log           # 原始训练及workflow日志
├── steps/                           # argv、退出码、环境、Slurm、工具版本
└── reports/
    ├── OVERVIEW.md
    ├── FULL_SETUP.json               # CPU协议检查、版本和计划hash
    ├── FULL_TRAINING.md / .json       # 每attempt吞吐/耗时/显存/访问量及全val记录
    └── CHECKPOINTS.json              # 全部checkpoint、epoch路径及最近进度
```

所有周期与epoch checkpoint保留，不自动清理。约79份完整状态，按开发实验每份约578MiB估算，约45GiB。epoch重启时会重复一次固定验证；该开销进入原始阶段wall time，验证不改变训练随机流。训练与编码wall time/GPU-hours分开列出；失败attempt和重放样本也会计入日志汇总。被强制终止而没有结束时间的阶段，其wall time标记未知，不伪造完整耗时。

```bash
./mvwd status "$exp"
tail -f "artifacts/mvwd/experiments/$exp/training/epoch01_attempt001/loss.jsonl"
```

## 查看生成结果

本机已建立 `val_preview`（180视图）和 `train_preview`（60视图）固定选择。epoch1完成后可以显式运行：

```bash
root="artifacts/mvwd/experiments/$exp"
./mvwd infer "$exp" val_ep01_seed42 --split val \
  --selection "$root/selections/val_preview/selection.json" \
  --checkpoint epoch_01 --seed 42
./mvwd eval "$exp" val_ep01_seed42 --preview
```

推理后自动导出GT、BEV、多机器人对照及`index.html`，均紧邻预测。最终模型改用`--checkpoint final`及新的运行名。小子集结果是开发诊断；正式全test三seed推理/评估沿用[日常操作指南](WORKFLOW.md)。

## 新建另一轮全量训练

只有独立重训时新建experiment：

```bash
./mvwd full NEW_EXPERIMENT --stage setup \
  --dataset-root "$MULTIVIEW_STORAGE_ROOT/datasets/MultiViewWorldDataset_v1" \
  --reference "$MULTIVIEW_STORAGE_ROOT/baselines/top2pano/artifacts/mvwd/experiments/20261013_fullscene_2frames_gpu4"
./mvwd full NEW_EXPERIMENT --stage all --through-epoch 10
```

setup只读取metadata并生成冻结配置、源码快照和计划。新机器需要可验证的Top2Pano参考记录；路径可覆盖。全量setup默认不修改现有条件设计或引入新的监督输入。
