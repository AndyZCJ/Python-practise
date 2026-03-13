from typing import Any, Dict, Optional

from algorithms.ppo import PPO
from buffers.rollout_buffer import RolloutBuffer
from configs.config import PPOConfig
from env.cartpole_env import CartPoleEnv
from utils.logger import TrainingLogger


class PPOTrainer:
    """
    TODO: 组织 PPO 在 CartPole 上的训练流程。

    你需要完成:
        1. 创建环境、智能体、缓冲区与日志器
        2. 设计采样阶段、更新阶段与评估阶段的衔接方式
        3. 规划模型保存、指标记录与可视化输出
    """

    def __init__(
        self,
        config: PPOConfig,
        env: Optional[CartPoleEnv] = None,
        agent: Optional[PPO] = None,
        buffer: Optional[RolloutBuffer] = None,
        logger: Optional[TrainingLogger] = None,
    ):
        super().__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger

        # TODO: 在这里补充组件初始化与依赖注入逻辑

    def collect_rollouts(self) -> Dict[str, Any]:
        """
        TODO: 从环境中采样一批轨迹。

        你需要完成:
            1. 控制单轮采样步数或回合数
            2. 将状态、动作、奖励、对数概率和价值写入缓冲区
            3. 处理 episode 结束与环境重置
        """

        # TODO: 实现 rollout 采样逻辑
        pass

    def update_agent(self) -> Dict[str, float]:
        """
        TODO: 调用 PPO 智能体完成一次参数更新。

        返回:
            metrics: 训练指标字典
        """

        # TODO: 从缓冲区取出批次并调用智能体更新接口
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        TODO: 对当前策略进行评估。

        返回:
            metrics: 评估指标字典，例如平均回报、成功率等
        """

        # TODO: 实现独立评估流程
        pass

    def train(self) -> None:
        """
        TODO: 串联 PPO 训练主流程。

        你需要完成:
            1. 循环执行采样、优势计算、参数更新与日志记录
            2. 根据评估结果保存最佳模型
            3. 控制训练终止条件
        """

        # TODO: 实现训练主循环
        pass



def build_trainer(
    config: PPOConfig,
    env: Optional[CartPoleEnv] = None,
    agent: Optional[PPO] = None,
    buffer: Optional[RolloutBuffer] = None,
    logger: Optional[TrainingLogger] = None,
) -> PPOTrainer:
    """
    TODO: 构建训练器实例。
    """

    return PPOTrainer(
        config=config,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )

