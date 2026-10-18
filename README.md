# Qwen-Image-2.1 × MVWD

BEV 条件的机器人相机视图生成 baseline。目前实现 Level 1；Level 2 尚未实现。
**日常只使用一个入口：`./mvwd`。** 不需要手动切换推理/评估环境，产物和日志自动归档。

## 开始使用

原集群已有环境可直接运行：

```bash
source ~/.config/multiview/storage.sh
cd "$MULTIVIEW_STORAGE_ROOT/baselines/qwen_image21"
./mvwd --help
./mvwd status
```

如果当前 shell 尚未设置存储路径，先 `source ~/.config/multiview/storage.sh`。入口自身也会读取这个文件。
新机器的安装步骤、固定依赖和模型配置见 [实现与安装说明](docs/IMPLEMENTATION.md)。环境位置可覆盖：

```bash
export QWEN_PYTHON=/path/to/qwen-image21/bin/python
export MVWD_METRICS_PYTHON=/path/to/mvwd-metrics/bin/python
```

默认复用本地模型缓存（`HF_HUB_OFFLINE=1`）；首次下载需显式设置 `HF_HUB_OFFLINE=0`。
所有相对路径以 repo 根目录为基准。GPU 命令需要先进入你自己申请的 Slurm allocation；入口不会申请资源或提交任务。

## 常用操作

| 命令 | 功能 |
|---|---|
| `./mvwd init EXP --config FILE` | 创建实验，冻结配置和源码 |
| `./mvwd init EXP --from OLD_EXP` | 复制已有实验配置，创建新实验 |
| `./mvwd select EXP NAME …` | 建立固定训练/验证子集 |
| `./mvwd prepare EXP NAME …` | 编码训练/验证特征，支持 `--processes 4` |
| `./mvwd train EXP NAME …` | 训练或完整恢复，支持 `--processes 4` |
| `./mvwd infer EXP NAME …` | 推理，然后自动导出 GT/BEV/多相机对照和浏览页 |
| `./mvwd eval EXP INFER_NAME --preview` | 评估已有预测，结果保存为 `INFER_NAME_eval` |
| `./mvwd review EXP INFER_NAME` | 仅用 CPU 为旧预测补导或校验对照，不重新推理 |
| `./mvwd status [EXP]` | 查看结果与日志，更新实验索引和易读指标报告 |

每个命令可用 `--help` 查看参数。`prepare/train/infer/eval/review` 支持 `--dry-run`：只打印命令，不执行或创建产物。
同一实验内每次运行使用不同 NAME；失败记录也保留。重新 review 可用 `--name NEW_REVIEW_NAME`。

例如，使用现有2000步checkpoint进行训练集检查：

```bash
exp=MY_TRAINFIT_CHECK
prior=artifacts/mvwd/experiments/20261018_l1_4h100_main_v1

./mvwd init "$exp" --from 20261018_l1_4h100_main_v1 --purpose 'Training-set fitting check'
./mvwd select "$exp" train10 --split train --scene-count 10 --episodes-per-scene 1 --selection-seed 42
./mvwd infer "$exp" train60 --split train \
  --selection "artifacts/mvwd/experiments/$exp/selections/train10/selection.json" \
  --lora "$prior/training/main2000/final" --seed 42
./mvwd eval "$exp" train60 --preview
./mvwd status "$exp"
```

此例是开发诊断：固定选择策略产生的60个查询属于该checkpoint训练过的4800-query子集。它不表示正式test结果。
训练和恢复的完整示例见 [操作指南](docs/WORKFLOW.md)。

## 产物在哪里

```text
artifacts/mvwd/
├── EXPERIMENTS.md                      # 自动实验索引
└── experiments/EXP/
    ├── config.json / manifest.json     # 冻结配置和实验信息
    ├── source/                        # 模型代码与自动化工具版本快照
    ├── selections/                    # 固定子集
    ├── features/                      # 训练/验证特征
    ├── training/NAME/                 # LoRA、optimizer/RNG状态、loss
    ├── predictions/NAME/              # 原预测、GT/BEV对照、index.html
    ├── evaluations/NAME_eval/         # metrics.json、METRICS.md、逐项指标
    ├── steps/                         # argv、状态、退出码、环境、版本
    ├── logs/                          # 原始日志及完整工作流日志
    └── reports/OVERVIEW.md             # 自动结果总览与链接
```

推理目录中的 `index.html` 可在浏览器打开；按episode/frame排列，每列一个机器人，三行依次为条件BEV、GT、预测。预测旁同时保存：

- `frame_XXX.png`：原预测，保持不变。
- `frame_XXX.gt.png`、`frame_XXX.camera_bev.png`：GT和实际输入的相机/机器人标注BEV。
- `frame_XXX.comparison.png`：GT / prediction。
- `frame_XXX.bev_comparison.png`：global BEV / condition BEV / GT / prediction。
- `frame_XXX.camera.json`：相机和条件元数据。
- episode级 `env_bev.png`、`bev_calibration.json`、`frame_XXX.multiview_comparison.png`。

`visualization_receipt.json`、`visualizations.jsonl`记录独立的GT后处理阶段与文件校验和。**GT只在完整生成结束后用于对照和评估，不进入模型推理条件。**
历史实验目录不搬迁、不覆盖；旧的两列GT对照仍保留。数据、权重和本机产物不随Git发布。

## 当前结果与协议

最新训练集检查：`20261018_l1_trainfit_check_v1`，60视图，PSNR **12.3514**、LPIPS **0.6617**、MV-GTWarp-DINO **0.7240**。
同一checkpoint的val180：9.2728 / 0.7259 / 0.7267。两组分布不同，不能将分数差值直接解释为纯泛化差距；人工抽查中训练样本也存在房间、朝向和机器人错误。

模型输入为全局RGB BEV、机器人/相机标注及数值/资产文本。使用GT对齐且隐藏top_body的共享机器人资产；不增加mesh渲染、GT深度输入或局部BEV条件。相机和跨视图几何尚未达标。
公平训练预算见 [BASELINE_FAIRNESS.md](docs/BASELINE_FAIRNESS.md)；历史记录见 [HISTORY.md](docs/HISTORY.md)。

`qwen_mvwd/` 是训练/数据核心；`tools/` 是统一工作流与可视化实现；`tools/legacy/` 仅保留历史开发工具。
`run.py`、`review_predictions.py` 保留为兼容入口，正常使用无需直接调用。
