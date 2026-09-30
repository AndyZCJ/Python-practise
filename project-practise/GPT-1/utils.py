import torch
import os
import random
import numpy as np
def setup_seed(seed: int = 42, deterministic: bool = True) -> None:

    # Python 哈希与随机数
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 禁止 cuDNN 自动寻找最快算法
        torch.backends.cudnn.benchmark = False

        # 使用确定性的 cuDNN 算法
        torch.backends.cudnn.deterministic = True

        # 对非确定性操作直接报错
        torch.use_deterministic_algorithms(True)
    else:
        # 输入尺寸固定时通常可以提高性能
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False