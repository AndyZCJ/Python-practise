import random
from typing import Optional
import os
import torch
import numpy as np


class SeedManager:
    """
    TODO: 统一管理实验随机种子。

    你需要完成:
        1. 设置 Python、NumPy、PyTorch 等库的随机种子
        2. 处理 CUDA 与确定性设置
        3. 如有需要，对环境随机种子进行同步
    """

    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

        # TODO: 在这里记录与你的实验复现相关的配置

    def apply(self, deterministic: Optional[bool] = None) -> None:
        """
        TODO: 应用随机种子设置。

        参数:
            deterministic: 是否启用更严格的确定性行为
        """
        # TODO: 实现随机种子设置逻辑
        random.seed(self.seed)
        os.environ['PYTHONHASHED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False




def set_global_seed(seed: int, deterministic: Optional[bool] = None) -> SeedManager:
    """
    TODO: 创建并应用随机种子管理器。

    你需要完成:
        1. 在这里统一管理外部调用入口
        2. 根据需要立即调用 apply 方法
    """

    manager = SeedManager(seed=seed)

    # TODO: 如有需要，在这里调用 manager.apply(deterministic=deterministic)
    return manager

