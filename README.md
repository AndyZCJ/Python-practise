# Python Practise
这是一个以**学习、练习、做实验**为核心目标的 Python 项目仓库。
仓库里包含：
- 强化学习算法练习
- 计算机视觉模型练习
- 日志分析等小型工程练习
- Notebook 形式的随手实验与笔记
- 一些用于理解状态机、缓存、经验回放等概念的小脚本
> 这个仓库不是面向生产环境的完整产品，而是一个持续迭代的练习集合。
> 很多代码都保留了实验性质、学习痕迹和可继续扩展的空间。
## 仓库结构
```text
Python Practise/
├── DRL/                     # 一组经典深度强化学习脚本练习
├── math/                    # 数学相关笔记与 notebook
├── project-practise/        # 相对更完整的小项目与工程化练习
├── playground.ipynb         # 零散实验 notebook
├── project-practise.ipynb   # 项目方向 notebook
├── python-practise.ipynb    # Python 语法与练习 notebook
├── pytorch-practise.ipynb   # PyTorch 相关 notebook
└── utils.py                 # 一些脚本共用的小工具
```
## 小项目速览
| 路径 | 类型 | 简介 |
| --- | --- | --- |
| `DRL/` | 强化学习脚本集合 | 包含 DQN、DDQN、Actor-Critic、A2C、A3C 等 CartPole 练习实现 |
| `math/` | 数学笔记 | 当前主要是线性代数 notebook |
| `project-practise/CV-ResNet/` | 视觉分类练习 | 使用 PyTorch 训练简化版 ResNet 进行 CIFAR-10 分类 |
| `project-practise/CV-ViT/` | 视觉分类练习 | 使用 PyTorch 实现简化版 ViT 并进行分类实验 |
| `project-practise/CV-SwinTransformer/` | 模块级练习 | 主要练习窗口划分、窗口注意力和 Swin Block |
| `project-practise/CV-DiffusionModel/` | 预留方向 | 为扩散模型练习预留的目录，目前仍是占位状态 |
| `project-practise/DDQN-CartPole/` | 强化学习小项目 | 将 DDQN 按模块拆分为 agent、buffer、trainer 与入口脚本 |
| `project-practise/PPO-CartPole/` | 学习型骨架项目 | 故意保留 TODO 的 PPO 学习框架，用于自己补全算法 |
| `project-practise/log_analyzer/` | 工程化小练习 | 解析日志、聚合统计并输出摘要结果 |
| `project-practise/MovingAverageMonitor.py` | 小练习脚本 | 用滑动平均实现简单监控与告警判断 |
| `project-practise/OnlineDecisionSystem.py` | 小练习脚本 | 用状态机方式模拟在线触发、冷却与重置逻辑 |
## 这个仓库适合怎么使用
### 1. 当作练习集合来阅读
可以按主题阅读：
- 想看强化学习：先看 `DRL/`，再看 `project-practise/DDQN-CartPole/`，最后看 `project-practise/PPO-CartPole/`
- 想看计算机视觉：看 `project-practise/CV-ResNet/`、`project-practise/CV-ViT/`、`project-practise/CV-SwinTransformer/`
- 想看更偏工程化的 Python 项目：看 `project-practise/log_analyzer/`
### 2. 当作“自己继续补全”的起点
仓库里有些项目是完整度较高的实验脚本，有些则是故意保留学习空间的骨架项目。
尤其是 `project-practise/PPO-CartPole/`，它的目标就是让你自己补核心逻辑，而不是直接给出答案。
### 3. 当作 Notebook + 脚本混合工作区
根目录下的 notebook 更适合做：
- 语法测试
- 张量实验
- 小段原型代码验证
- 公式和想法草稿
## 依赖说明
从当前代码可以看到，仓库里主要使用到这些库：
- `torch`
- `torchvision`
- `gym` / `gymnasium`
- `matplotlib`
- `numpy`
- `pytest`
- `tqdm`
不同子项目的依赖并不完全一致，建议进入对应目录后再按需安装。
## 运行方式建议
由于这是练习仓库，不同项目的运行入口不同。常见方式包括：
```powershell
python .\project-practise\CV-ResNet\main.py
python .\project-practise\CV-ViT\main.py
python .\project-practise\DDQN-CartPole\main.py
python .\project-practise\log_analyzer\main.py .\project-practise\log_analyzer\log.txt
python .\project-practise\PPO-CartPole\train.py
```
如果某个项目使用相对路径读取数据或输出结果，更稳妥的方式是先进入该项目目录，再执行对应入口脚本。
## 数据与输出文件说明
仓库根目录的 `.gitignore` 已经对以下内容做了排除：
- Python 缓存
- 测试缓存
- 虚拟环境
- 训练输出目录
- 模型权重
- 常见数据目录与压缩数据包
这样可以避免把实验过程中产生的大文件、缓存文件和本地环境文件提交到仓库。
## 文档约定
为了让仓库更容易浏览：
- 根目录 `README.md` 负责总览
- 每个小项目目录尽量提供自己的 `README.md`
- 与学习目的强相关的项目，会明确写出“当前完成度”和“建议阅读顺序”
如果后续新增子项目，建议同步补一个简短 `README.md`，说明：
1. 这个项目是干什么的
2. 当前做到了哪一步
3. 从哪个文件开始看
4. 如何运行
## 当前定位
这个仓库的关键词不是“封装完善”，而是：
- 练习
- 理解
- 试错
- 迭代
- 自己动手实现
如果你是在学习算法、PyTorch、强化学习或小型项目组织方式，这个仓库就是为这种目的服务的。
