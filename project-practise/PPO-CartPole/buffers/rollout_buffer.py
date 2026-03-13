from typing import Any, Dict, List


class RolloutBuffer:
    """
    TODO: 实现 PPO 训练使用的 rollout 缓冲区。

    你需要完成:
        1. 设计状态、动作、奖励、终止标记等数据的存储结构
        2. 记录策略输出的对数概率与价值估计
        3. 计算回报与优势，并整理成可训练批次
    """

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.storage: Dict[str, List[Any]] = {"states":[],
            "actions":[],
            "rewards":[],
            "next_states":[],
            "dones":[],
            "log_probs":[],
            "values":[]}

        # TODO: 初始化各类轨迹字段，例如 states、actions、rewards、dones 等

    def add(self, transition: Dict[str, Any]) -> None:
        """
        TODO: 向缓冲区写入一条时间步数据。

        参数:
            transition: 建议包含 state、action、reward、done、log_prob、value 等字段
        """
        filed_mapping = {
            "states":"state",
            "actions":"action",
            "rewards":"reward",
            "next_states":"next_state",
            "dones":"done",
            "log_probs":"log_prob",
            "values":"value"
        }
        for buffer_key, trans_key in filed_mapping.items():
            self.storage[buffer_key].append(transition[trans_key])

    def compute_returns_and_advantages(self, last_value: Any) -> None:
        """
        TODO: 根据轨迹数据计算回报与优势。

        参数:
            last_value: 最后一个状态的价值估计，可用于 bootstrap
        """

        # TODO: 实现 GAE 或你计划使用的优势估计方法
        pass

    def get_batch(self) -> Dict[str, Any]:
        """
        TODO: 将缓冲区内容整理为训练批次。

        返回:
            batch: 建议包含 states、actions、old_log_probs、returns、advantages 等字段
        """

        # TODO: 实现批量整理与张量转换逻辑
        pass

    def clear(self) -> None:
        self.storage.clear()

    def __len__(self) -> int:
        return len(self.storage["states"])



def build_rollout_buffer(capacity: int) -> RolloutBuffer:
    """
    TODO: 构建 rollout 缓冲区实例。
    """

    return RolloutBuffer(capacity=capacity)

