from typing import Dict, List, Optional


class TrainingLogger:
    """
    TODO: 实现训练日志记录器。

    你需要完成:
        1. 记录训练过程中的标量指标
        2. 管理控制台输出、文件输出与可视化数据
        3. 按需保存训练历史，便于后续分析
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        self.history: Dict[str, List[float]] = {}

        # TODO: 初始化日志文件、历史记录容器与输出目录

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        TODO: 记录一次训练或评估指标。

        参数:
            metrics: 指标名字到标量值的映射
            step: 当前训练步数或迭代编号
        """

        # TODO: 实现指标记录逻辑
        pass

    def save_history(self, file_name: Optional[str] = None) -> None:
        """
        TODO: 保存历史指标到文件。
        """

        # TODO: 将历史记录序列化到磁盘
        pass

    def print_summary(self, metrics: Dict[str, float]) -> None:
        """
        TODO: 以清晰格式打印关键指标摘要。
        """

        # TODO: 实现控制台摘要输出
        pass



def build_logger(save_dir: str) -> TrainingLogger:
    """
    TODO: 构建训练日志记录器。
    """

    return TrainingLogger(save_dir=save_dir)

