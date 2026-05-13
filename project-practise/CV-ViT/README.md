# CV-ViT

这是一个使用 **PyTorch** 练习 Vision Transformer 思路的小项目。

从当前实现来看，这个项目与 `CV-ResNet` 的整体训练框架相似，但把核心模型替换成了一个简化版 `MiniViT`。

## 当前目录结构

```text
CV-ViT/
├── dataset.py
├── main.py
├── trainer.py
├── vit.py
├── data/          # 本地数据目录，已在仓库根 .gitignore 中忽略
└── outputs/       # 训练输出目录，已在仓库根 .gitignore 中忽略
```

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 项目入口，负责训练参数、模型构建与结果保存 |
| `dataset.py` | 构建 CIFAR-10 数据加载器 |
| `vit.py` | 简化版 ViT 模型实现 |
| `trainer.py` | 训练与评估逻辑 |

## 当前特点

- 使用 `CIFAR-10` 作为训练数据
- 会在训练过程中记录 loss 和 accuracy
- 会把最好权重与训练曲线保存到 `outputs/`

## 运行方式

建议在当前目录运行：

```powershell
python .\main.py --epochs 5 --lr 1e-3 --batch_size 64
```

## 适合学习的点

这个项目适合用来对比：
- 卷积模型和 Transformer 模型在组织方式上的差异
- 数据管线与训练器如何复用
- 如何把注意力模型接到一个最小可运行的分类训练流程中

