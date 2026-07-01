# CIFAR-10 ResNet-18 课后任务项目

本项目用于完成课后任务：

1. 从零训练 ResNet-18 变体，在 CIFAR-10 验证集上尽量超过 85% Accuracy。
2. 加载 ImageNet 预训练 ResNet-18，在 CIFAR-10 上微调，并与从零训练进行对比。
3. 自动输出训练日志、Loss/Accuracy 曲线、测试集准确率报告。

## 1. 环境安装

```bash
conda create -n cifar10_resnet python=3.10 -y
conda activate cifar10_resnet
pip install -r requirements.txt
```

如有 GPU，请优先安装与你 CUDA 匹配的 PyTorch 版本。

## 2. 项目结构

```text
cifar10_resnet18_homework_project/
├── models/resnet_cifar.py      # 从零训练用 CIFAR 版 ResNet-18
├── utils/data.py               # CIFAR-10 数据加载与增强
├── utils/engine.py             # 训练/验证通用函数
├── utils/plot.py               # 曲线绘制
├── train_scratch.py            # 从零训练脚本
├── train_finetune.py           # ImageNet 预训练微调脚本
├── test.py                     # 测试集评估脚本
├── plot_curves.py              # 单独绘图脚本
└── outputs/                    # 运行后自动生成
```

## 3. 从零训练 ResNet-18 变体

推荐命令：

```bash
python train_scratch.py \
  --data-dir ./data \
  --out-dir ./outputs/scratch \
  --epochs 120 \
  --batch-size 128 \
  --lr 0.1
```

运行后生成：

```text
outputs/scratch/
├── best_model.pth
├── history.csv
├── train.log
├── loss_curve.png
└── accuracy_curve.png
```

## 4. 微调 ImageNet 预训练 ResNet-18

推荐命令：

```bash
python train_finetune.py \
  --data-dir ./data \
  --out-dir ./outputs/finetune \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.01
```

运行后生成：

```text
outputs/finetune/
├── best_model.pth
├── history.csv
├── train.log
├── loss_curve.png
└── accuracy_curve.png
```

## 5. 测试集最终准确率

从零训练模型：

```bash
python test.py \
  --mode scratch \
  --ckpt ./outputs/scratch/best_model.pth \
  --data-dir ./data
```

微调模型：

```bash
python test.py \
  --mode finetune \
  --ckpt ./outputs/finetune/best_model.pth \
  --data-dir ./data
```

运行后会在对应输出目录生成：

```text
test_report.txt
```

