# Transformer 期中作业实现（Encoder-Decoder架构）
本仓库是「大模型基础与应用」期中作业的开源代码，严格遵循《Description_of_the_Assignment.pdf》要求，实现了完整的Encoder-Decoder Transformer，并在IWSLT2017英德翻译数据集上验证性能。


## 一、作业目标匹配
| 作业要求 | 本实现对应功能 |
|----------|----------------|
| 1.10 实现核心组件 | 完成multi-head self-attention、position-wise FFN、残差+LayerNorm、相对位置编码 |
| 1.11 搭建Transformer | 实现Encoder+Decoder架构（非仅Encoder），支持Seq2Seq任务 |
| 1.12 代码开源 | 提供完整Git仓库、运行脚本、依赖清单 |
| 1.14 仓库结构 | 包含src/、scripts/、results/、requirements.txt、README.md |
| 1.16 可重现性 | 固定随机种子42，提供exact命令行，硬件要求明确 |
| 1.20 挑战任务 | 实现相对位置编码、AdamW优化器、梯度裁剪、学习率调度 |


## 二、仓库结构
transformer-midterm/
├── src/                  # 源代码目录（核心实现）
│   ├── config.py         # 超参数配置（含固定随机种子）
│   ├── model.py          # Transformer 全架构实现（Encoder+Decoder）
│   ├── train.py          # 训练/评估逻辑（含早停、可视化）
│   └── utils.py          # 工具函数（掩码、BLEU计算、数据预处理）
├── scripts/              # 运行脚本目录
│   └── run.sh            # 一键训练/评估脚本（作业要求的 exact 命令）
├── results/              # 实验结果目录（作业要求存放曲线与表格）
│   ├── train_val_ppl.png # 训练/验证 PPL 曲线（自动生成）
│   ├── ablation_table.csv # 消融实验结果表（手动补充实验数据）
│   └── translation_samples.txt # 翻译样本（自动生成）
├── requirements.txt      # 依赖库清单（含版本号，确保环境一致）
└── README.md             # 完整运行说明（含硬件要求、命令行示例）


## 三、环境准备
### 1. 硬件要求
- **推荐配置**：NVIDIA GPU（RTX 3060 6GB及以上），训练20轮约1.5小时；
- **最低配置**：CPU（i7-12700H），训练20轮约6小时；
- 内存要求：≥16GB（避免数据加载时内存溢出）。

### 2. 软件安装
#### 方法1：一键安装（推荐）
直接运行脚本安装所有依赖：
```bash
pip install -r requirements.txt
=======
# annotated-transformer-master
大模型中期作业：从零构建Transformer并验证性能
