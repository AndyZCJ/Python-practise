# project-practise

这里收集的是一组**相对更完整、更接近“小项目”形态**的练习内容。

和根目录的 notebook 或 `DRL/` 下的单文件脚本相比，这里的内容通常更强调：
- 目录拆分
- 模块职责
- 独立入口脚本
- 输出文件组织
- 从“能跑”逐步过渡到“更像项目”

## 当前包含的项目

| 路径 | 简介 |
| --- | --- |
| `CV-ResNet/` | 使用 PyTorch 训练简化版 ResNet 做图像分类 |
| `CV-ViT/` | 使用 PyTorch 训练简化版 Vision Transformer |
| `CV-SwinTransformer/` | 练习窗口注意力与 Swin Block 的局部实现 |
| `CV-DiffusionModel/` | 扩散模型方向的预留目录 |
| `DDQN-CartPole/` | 模块化的 Double DQN 强化学习小项目 |
| `PPO-CartPole/` | 面向学习的 PPO 骨架项目，故意保留 TODO |
| `log_analyzer/` | 一个带有解析、统计和测试的日志分析小工具 |

此外，这个目录下还有两个独立练习脚本：

| 文件 | 简介 |
| --- | --- |
| `MovingAverageMonitor.py` | 使用滑动平均做简单告警判断 |
| `OnlineDecisionSystem.py` | 使用状态机方式实现触发、冷却与重置逻辑 |

## 建议阅读顺序

如果你想按“从简单到更项目化”的顺序看，可以参考：
1. `MovingAverageMonitor.py`
2. `OnlineDecisionSystem.py`
3. `log_analyzer/`
4. `DDQN-CartPole/`
5. `CV-ResNet/`
6. `CV-ViT/`
7. `CV-SwinTransformer/`
8. `PPO-CartPole/`

## 说明

每个子目录下都尽量补了自己的 `README.md`，建议从对应目录的 README 开始阅读，这样能更快知道：
- 这个项目的目标是什么
- 当前做到了哪一步
- 从哪个文件切入最合适
- 如何运行

