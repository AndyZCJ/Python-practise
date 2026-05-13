from typing import Any, Dict, List
from configs.config import PPOConfig
import torch
import random
class RolloutBuffer:
    """
    TODO: 实现 PPO 训练使用的 rollout 缓冲区。

    你需要完成:
        1. 设计状态、动作、奖励、终止标记等数据的存储结构
        2. 记录策略输出的对数概率与价值估计
        3. 计算回报与优势，并整理成可训练批次
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.storage: Dict[str, List[Any]] = self._create_empty_storage()

    def _create_empty_storage(self) -> Dict[str, List[Any]]:
        """
        说明:
            这里集中定义所有字段，便于 clear 和初始化时复用。
        """
        return {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "log_probs": [],
            "values": [],
            "returns": [],
            "advantages": [],
        }

    def add(self, transition: Dict[str, Any]) -> None:
        """
        参数:
            transition: 包含 state、action、reward、next_state、done、log_prob、value
        """
        field_mapping = {
            "states": "state",
            "actions": "action",
            "rewards": "reward",
            "next_states": "next_state",
            "dones": "done",
            "log_probs": "log_prob",
            "values": "value"
        }

        if self.__len__() < self.config.buffer_size:
            for buffer_key, trans_key in field_mapping.items():
                self.storage[buffer_key].append(transition[trans_key])
        else:
            for key in field_mapping:
                if self.storage[key]:
                    del self.storage[key][0] #删掉最老的元素

            for buffer_key, trans_key in field_mapping.items():
                self.storage[buffer_key].append(transition[trans_key])

        # TODO: 新增轨迹后，旧的 returns / advantages 需要重新计算
        self.storage["returns"] = []
        self.storage["advantages"] = []

    def compute_returns_and_advantages(self, last_value: Any) -> None:
        """
        参数:
            last_value: 最后一个状态的价值估计
        """

        assert len(self.storage["rewards"]) == len(self.storage["dones"]) == len(self.storage["values"])
        assert len(self.storage["states"]) > 0


        # TODO: 实现 GAE 或你计划使用的优势估计方法

        values = self.storage["values"]
        rewards = self.storage["rewards"]
        dones = self.storage["dones"]
        values = values + [last_value]


        T = self.__len__()
        advantages = [0.0] * T
        returns = [0.0] * T
        advantage = 0

        for t in reversed(range(T)):
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantage = delta + self.config.gae_lambda * self.config.gamma * (1 - dones[t]) * advantage
            advantages[t] = advantage

        for t in range(T):
            returns[t] = advantages[t] + values[t]

        self.storage["advantages"] = advantages
        self.storage["returns"] = returns

    def get_batch(self) -> Dict[str, Any]:
        """
        TODO: 将缓冲区内容整理为训练批次。

        返回:
            batch: 建议包含 states、actions、old_log_probs、returns、advantages 等字段
        """

        # TODO: 实现批量整理与张量转换逻辑
        assert len(self)>0, "buffer为空"
        assert len(self.storage["returns"]) == len(self),"请先计算returns再获取batch"
        assert len(self.storage["advantages"]) == len(self),"请先计算advantages再获取batch"
        assert all(self.__len__()==len(self.storage[buf]) for buf, _ in self.storage.items())

        indices = random.sample(range(self.__len__()), self.config.batch_size)
        sampled = {key: [self.storage[key][i] for i in indices] for key in self.storage}
        states = torch.stack([
            state.detach().to(dtype=torch.float32).reshape(-1)
            if isinstance(state, torch.Tensor)
            else torch.tensor(state, dtype=torch.float32).reshape(-1)
            for state in sampled["states"]
        ])
        actions = torch.tensor(sampled["actions"], dtype=torch.long)
        next_states = torch.tensor(sampled["next_states"], dtype=torch.float32)
        old_log_probs = torch.tensor(sampled["log_probs"], dtype=torch.float32)
        returns = torch.tensor(sampled["returns"], dtype=torch.float32)
        advantages = torch.tensor(sampled["advantages"], dtype=torch.float32)
        dones = torch.tensor(sampled["dones"], dtype=torch.long)

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "dones": dones,
            "old_log_probs": old_log_probs,
            "returns": returns,
            "advantages": advantages,
        }

    def clear(self) -> None:
        """
        清空缓冲区中的所有已采样数据。
        """
        self.storage = self._create_empty_storage()

    def __len__(self) -> int:
        return len(self.storage["states"])



def build_rollout_buffer(config: PPOConfig) -> RolloutBuffer:
    """
    TODO: 构建 rollout 缓冲区实例。

    参数:
        config: PPO 训练配置对象
    """

    return RolloutBuffer(config=config)

