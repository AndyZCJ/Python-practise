from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class PPOConfig:
    """
    TODO: 定义 PPO 训练所需的配置项。

    你需要完成:
        1. 补充环境参数、模型参数、训练参数与日志参数
        2. 为每个超参数设置合理默认值
        3. 明确哪些参数需要从命令行或配置文件覆盖
    """

    env_name: str = "CartPole-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1
    state_dim: Optional[int] = None
    action_dim: Optional[int] = None
    buffer_size: int = 10000
    hidden_dim: int = 128
    rollout_steps: int = 128 #cartpole常见取值128, 256, 512, 1024
    total_epochs: int = 10
    learning_rate: float = 1e-3
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    value_coef: float = 0.5 #价值损失项的权重系数
    entropy_coef: float = 0.01 #熵正则项的权重
    batch_size: int = 32
    update_epochs: int = 10
    log_interval: int = 10
    save_dir: str = "outputs"



def get_default_config() -> PPOConfig:
    """
    TODO: 返回一个默认配置对象。

    你需要完成:
        1. 根据设备情况选择 CPU 或 CUDA
        2. 按照实验需求修改默认超参数
        3. 如有需要，对接命令行参数或外部配置文件
    """

    # TODO: 如果你希望支持命令行解析，可以在这里补充转换逻辑
    return PPOConfig()

