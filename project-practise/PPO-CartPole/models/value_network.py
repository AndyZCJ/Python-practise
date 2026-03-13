import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    TODO: 实现 CartPole 的价值网络。

    输入:
        state: (B, state_dim)

    输出:
        state_value: (B, 1)

    你需要完成:
        1. 设计状态价值函数网络结构
        2. 输出单个状态价值估计
        3. 根据需要补充权重初始化策略
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        参数:
            state: 形状为 (B, state_dim) 的状态张量

        返回:
            state_value: 形状为 (B, 1) 的状态价值张量
        """
        return self.net(state)



def build_value_network(state_dim: int, hidden_dim: int) -> ValueNetwork:
    """
    TODO: 构建价值网络实例。

    你需要完成:
        1. 如有需要，在这里统一网络初始化流程
        2. 按实验需求补充模型检查或参数打印逻辑
    """

    return ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim)

