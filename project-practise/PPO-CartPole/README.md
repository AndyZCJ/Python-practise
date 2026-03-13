# PPO-CartPole

这是一个**学习型项目骨架**，用于帮助你自己实现基于 **PyTorch** 的 **PPO** 算法，并在 **CartPole** 环境上完成训练。

## 设计原则

- 不直接实现算法核心逻辑
- 不补全模型结构
- 不提供训练循环细节
- 所有关键位置使用 `TODO` 标记
- 所有注释、说明与文档字符串都使用中文

## 目录结构

```text
PPO-CartPole/
│
├── configs/
│   └── config.py
├── env/
│   └── cartpole_env.py
├── models/
│   ├── policy_network.py
│   └── value_network.py
├── algorithms/
│   └── ppo.py
├── buffers/
│   └── rollout_buffer.py
├── trainer/
│   └── trainer.py
├── utils/
│   ├── logger.py
│   └── seed.py
├── train.py
└── README.md
```

## 文件说明

### `configs/config.py`
用于放置 PPO 训练的超参数配置骨架。你可以在这里补充学习率、折扣因子、裁剪系数、批大小等配置。

### `env/cartpole_env.py`
用于封装 CartPole 环境的创建、重置、步进与关闭逻辑。你可以在这里统一环境接口。

### `models/policy_network.py`
用于定义策略网络骨架。你需要自己设计输入层、隐藏层与输出层，并输出离散动作的 logits。

### `models/value_network.py`
用于定义价值网络骨架。你需要自己实现状态价值估计网络，并输出标量价值。

### `algorithms/ppo.py`
用于放置 PPO 智能体接口。你需要自己实现动作采样、动作评估、损失计算与参数更新等核心内容。

### `buffers/rollout_buffer.py`
用于管理采样得到的轨迹数据。你需要自己决定如何存储状态、动作、奖励、终止标记、旧策略概率与价值估计。

### `trainer/trainer.py`
用于组织训练流程。你需要自己设计 rollout 采样、优势计算、参数更新、评估和保存模型的完整流程。

### `utils/logger.py`
用于记录训练指标、保存历史数据和打印摘要。你可以在这里扩展控制台日志、文件日志和可视化输出。

### `utils/seed.py`
用于统一设置随机种子，帮助你做实验复现。

### `train.py`
项目入口文件。你可以从这里串联配置、随机种子、训练器构建与训练启动。

## 推荐补全顺序

1. 先补全 `env/cartpole_env.py`
2. 再实现 `models/policy_network.py` 与 `models/value_network.py`
3. 然后实现 `buffers/rollout_buffer.py`
4. 接着补全 `algorithms/ppo.py`
5. 最后完成 `trainer/trainer.py` 与 `train.py`

## 你可以重点思考的问题

- 策略网络应该输出 logits 还是概率
- 价值网络输出的张量形状应该如何统一
- rollout 数据应该按时间步组织还是按批次组织
- GAE 的计算过程如何实现
- PPO 裁剪目标如何写得清晰且稳定
- 训练过程中应该记录哪些关键指标

## 语法检查

你可以在项目根目录下运行下面的命令检查语法：

```powershell
python -m compileall .\project-practise\PPO-CartPole
```

这个骨架是故意不完整的，目的是让你亲自实现每个关键模块，从而真正理解 PPO 的实现过程。

