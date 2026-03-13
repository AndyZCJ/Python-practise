from typing import Any, Dict, Optional
import torch.distributions as D
import torch

from configs.config import PPOConfig
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork


class PPO:
    """
    TODO: 实现 PPO 智能体的核心接口。

    你需要完成:
        1. 组织策略网络与价值网络
        2. 定义 PPO 更新所需的优化器与损失项
        3. 实现动作采样、优势估计与参数更新流程
    """

    def __init__(
        self,
        config: PPOConfig,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.config = config
        self.policy_network = policy_network
        self.value_network = value_network
        self.device = device

        # TODO: 定义优化器、学习率调度器与其他训练状态

    def select_action(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        根据当前策略为状态采样动作。

        参数:
            state: 形状可以是 (state_dim,) 或 (B, state_dim)

        返回:
            result: 建议至少包含以下键
                action: 形状为 (B,) 或标量动作
                log_prob: 形状为 (B,) 的动作对数概率
                value: 形状为 (B, 1) 的状态价值估计
        """
        value = self.value_network(state)
        dist = D.Categorical(self.policy_network(state))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return {"action": action, "log_prob":log_prob, "value":value}

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        TODO: 评估一批状态和动作。

        参数:
            states: 形状为 (B, state_dim)
            actions: 形状为 (B,)

        返回:
            result: 建议至少包含以下键
                log_prob: 形状为 (B,)
                entropy: 形状为 (B,)
                value: 形状为 (B, 1)
        """

        # TODO: 实现 PPO 更新阶段所需的动作评估逻辑
        pass

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        TODO: 根据一批 rollout 数据执行 PPO 更新。

        参数:
            batch: 建议包含 states、actions、old_log_probs、returns、advantages 等张量

        返回:
            metrics: 训练指标字典，例如策略损失、价值损失、熵奖励等
        """

        # TODO: 实现 PPO 裁剪目标、价值函数损失与多轮小批量更新
        pass

    def save(self, save_path: str) -> None:
        """
        TODO: 保存策略网络、价值网络及训练状态。
        """

        # TODO: 根据你的实验需求实现模型保存逻辑
        pass

    def load(self, load_path: str) -> None:
        """
        TODO: 加载策略网络、价值网络及训练状态。
        """

        # TODO: 根据你的实验需求实现模型加载逻辑
        pass



def build_ppo_agent(
    config: PPOConfig,
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    device: Optional[torch.device] = None,
) -> PPO:
    """
    TODO: 构建 PPO 智能体实例。

    你需要完成:
        1. 统一智能体初始化入口
        2. 如有需要，在这里补充断言、日志或设备迁移逻辑
    """

    return PPO(
        config=config,
        policy_network=policy_network,
        value_network=value_network,
        device=device,
    )

