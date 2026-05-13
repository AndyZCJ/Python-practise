# PPO-CartPole

这是一个用于学习 **PPO + CartPole + PyTorch** 的**骨架型项目**。

它的定位不是“直接给出完整答案”，而是：
- 提前把目录拆好
- 把模块边界划清楚
- 在关键位置保留 `TODO`
- 让你自己把 PPO 的核心逻辑补出来

## 项目定位

这个项目适合用来练习：
- 如何组织一个最小 PPO 项目结构
- 策略网络、价值网络与缓冲区分别负责什么
- rollout、优势估计、损失计算和更新步骤如何串起来
- 怎样从一个可读的骨架出发，逐步补成完整实现

## 当前特点

- 不直接实现算法核心逻辑
- 不补全模型结构细节
- 不提供完整训练循环
- 关键函数保留中文 `TODO`
- 文档字符串、注释和说明都使用中文

也就是说，这个项目是**故意不完整**的，目标是让你亲手实现 PPO，而不是直接复制结果。

## 目录结构

```text
PPO-CartPole/
├── algorithms/
│   └── ppo.py
├── buffers/
│   └── rollout_buffer.py
├── configs/
│   └── config.py
├── env/
│   └── cartpole_env.py
├── models/
│   ├── policy_network.py
│   └── value_network.py
├── trainer/
│   └── trainer.py
├── utils/
│   ├── logger.py
│   └── seed.py
├── train.py
└── README.md
```

## 从哪里开始看

建议按下面顺序补全：

1. `configs/config.py`
2. `env/cartpole_env.py`
3. `models/policy_network.py`
4. `models/value_network.py`
5. `buffers/rollout_buffer.py`
6. `algorithms/ppo.py`
7. `trainer/trainer.py`
8. `train.py`

这样更容易先把数据流和模块边界建立清楚，再回头实现 PPO 更新细节。

## 各模块作用

| 文件 | 作用 |
| --- | --- |
| `configs/config.py` | 放置训练超参数和默认配置 |
| `env/cartpole_env.py` | 封装环境创建、重置、步进与关闭 |
| `models/policy_network.py` | 定义策略网络骨架 |
| `models/value_network.py` | 定义价值网络骨架 |
| `buffers/rollout_buffer.py` | 存储轨迹数据与训练所需字段 |
| `algorithms/ppo.py` | 实现动作选择、评估与 PPO 更新接口 |
| `trainer/trainer.py` | 组织采样、优势计算、更新与日志记录 |
| `utils/logger.py` | 记录训练指标 |
| `utils/seed.py` | 固定随机种子 |
| `train.py` | 项目入口，用于串联配置与训练器 |

## 补全时值得重点思考的问题

- 策略网络输出应该是 logits 还是概率
- 价值网络输出 shape 如何保持稳定
- 缓冲区该按“字段分桶”还是“transition 列表”来存
- GAE 的计算顺序和终止状态处理怎么写
- PPO clip loss、value loss、entropy bonus 如何组合
- 一次 rollout 后如何切 mini-batch 做多轮更新

## 使用建议

如果你只是想检查骨架是否能被 Python 正常解析，可以在仓库根目录运行：

```powershell
python -m compileall .\project-practise\PPO-CartPole
```

如果你准备真正补代码，建议每补完一个模块就单独做一次最小测试，而不是把所有逻辑一次性写完。

## 一句话总结

这个目录不是现成答案，而是一个为“自己动手实现 PPO”准备的学习型项目框架。

