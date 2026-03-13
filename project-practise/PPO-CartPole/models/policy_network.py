import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    输入:
        state: (B, state_dim)

    输出:
        action_logits: (B, action_dim)

    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        参数:
            state: 形状为 (B, state_dim) 的状态张量

        返回:
            action_logits: 形状为 (B, action_dim) 的动作分数张量
        """
        output = torch.softmax(self.net(state), dim=-1)
        return output



def build_policy_network(state_dim: int, hidden_dim: int, action_dim: int) -> PolicyNetwork:
    """
    TODO: 构建策略网络实例。

    你需要完成:
        1. 如有需要，在这里统一网络初始化流程
        2. 按实验需求补充模型检查或参数打印逻辑
    """

    return PolicyNetwork(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

