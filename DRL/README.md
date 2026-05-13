# DRL

这个目录存放的是一组**偏脚本式的深度强化学习练习**，主要围绕 `CartPole` 环境展开。

这里的代码特点是：
- 更接近“边学边写”的实验脚本
- 大多把模型、训练循环和可视化放在同一个文件里
- 适合快速对照算法流程，而不是作为可复用工程模板

## 当前包含的内容

| 文件 | 简介 |
| --- | --- |
| `DQN_CartPole.py` | 使用经验回放和目标网络实现基础 DQN |
| `DDQN_CartPole.py` | 在 DQN 基础上练习 Double DQN |
| `ActorCritic.py` | 单步 Actor-Critic 练习实现 |
| `A2C.py` | Advantage Actor-Critic 的基础实验版本 |
| `A3C.py` | 使用多进程的异步 Actor-Critic 练习 |

## 代码风格说明

这些脚本并不是统一工程风格，主要目的是帮助理解：
- 状态、动作、奖励如何组织
- 不同强化学习算法的更新方式有什么差别
- 目标网络、优势函数、并行训练分别在什么地方出现

## 依赖提示

从当前文件可见，这一组脚本主要依赖：
- `torch`
- `gym`
- `numpy`
- `matplotlib`
- `tqdm`

另外，`DQN_CartPole.py` 和 `DDQN_CartPole.py` 会使用仓库根目录下的 `utils.py` 中的 `ReplayBuffer`。

## 适合的阅读顺序

建议按下面顺序阅读：
1. `DQN_CartPole.py`
2. `DDQN_CartPole.py`
3. `ActorCritic.py`
4. `A2C.py`
5. `A3C.py`

这样能从单线程、单文件脚本逐步过渡到更复杂的策略梯度和多进程实现。

## 说明

如果你后续想把这一组脚本整理成更工程化的项目结构，可以参考 `project-practise/DDQN-CartPole/` 的拆分方式。

