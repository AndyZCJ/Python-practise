# CV-ResNet

这是一个使用 **PyTorch** 进行图像分类练习的小项目，当前重点是：
- 组织数据加载流程
- 练习简化版 ResNet 结构
- 训练并保存分类结果与可视化曲线

从代码来看，这个项目当前使用的是 **CIFAR-10** 数据集。

## 当前目录结构

```text
CV-ResNet/
├── dataset.py
├── main.py
├── model.py
├── ResNet.py
├── trainer.py
├── data/          # 本地数据目录，已在仓库根 .gitignore 中忽略
└── outputs/       # 训练输出目录，已在仓库根 .gitignore 中忽略
```

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 项目入口，负责解析参数、构建模型、启动训练和保存结果 |
| `dataset.py` | 构建 CIFAR-10 的 `DataLoader` |
| `ResNet.py` | 放置简化版 ResNet 结构 |
| `model.py` | 其他模型定义或试验性模型入口 |
| `trainer.py` | 训练与验证逻辑 |

## 当前产物

训练完成后，代码会把结果保存到 `outputs/`，例如：
- `best.pt`
- `train_history.json`
- `train_loss.png`
- `val_acc.png`

## 运行方式

建议先进入项目目录，再运行：

```powershell
python .\main.py --epochs 5 --lr 1e-3 --batch_size 64
```

## 适合学习的点

这个项目适合练习：
- 图像数据增强与标准化
- 分类模型训练流程
- 如何保存最好权重与训练历史
- 如何把单个实验脚本拆成 `dataset / model / trainer / main` 的结构

