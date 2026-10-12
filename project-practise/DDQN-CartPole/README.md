# DDQN-CartPole

这是一个把 **Double DQN** 按模块拆开的强化学习练习项目，目标环境是 `CartPole-v1`。

和 `DRL/DDQN_CartPole.py` 相比，这里的组织方式更接近一个小型项目，而不是单文件脚本。

## 当前目录结构

```text
DDQN-CartPole/
├── buffer.py
├── DDQN.py
├── model.py
├── trainer.py
└── main.py
```

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 程序入口，负责读取超参数并启动训练 |
| `trainer.py` | 组织环境交互、经验回放采样与训练循环 |
| `DDQN.py` | Double DQN 智能体逻辑 |
| `buffer.py` | 经验回放缓冲区 |
| `model.py` | Q 网络结构定义 |

## 当前特点

从代码可见，这个项目已经具备了常见的强化学习项目拆分方式：
- agent
- buffer
- model
- trainer
- main

这比 `DRL/` 里的单文件脚本更适合继续扩展。

## 运行方式

建议在当前目录下运行：

```powershell
python .\main.py --epoch 5 --num_episode 500 --batch_size 64
```

## 适合学习的点

这个项目适合练习：
- DQN 和 DDQN 的差异
- 为什么动作选择和目标 Q 值计算要分开
- 如何把强化学习实验从单脚本整理成模块化结构
- 如何管理经验回放缓冲区与训练器

