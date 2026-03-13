class OnlineDecisionSystem:
    """
    这是一个极简的在线决策系统骨架。

    现实含义（简化）：
    ----------------
    系统连续接收一个数值流（例如价格、信号强度、评分等）。

    系统维护内部状态：
    - position: 当前是否处于“已触发”状态（0 或 1）
    - cooldown: 冷却计数器（防止连续触发）
    - last_trigger_value: 上一次触发时对应的输入值

    系统规则：
    --------
    1. 当系统未触发(position == 0)，且当前值 x > threshold：
       → 触发一次（position = 1）
       → 记录 last_trigger_value
       → 启动 cooldown

    2. 当处于冷却期 cooldown > 0：
       → 每一步 cooldown 减 1
       → 冷却期内禁止再次触发

    3. 当 position == 1，且当前值比 last_trigger_value 低 delta：
       → 系统 reset（position 归零，清空 last_trigger_value）

    step(x) 返回：
        "TRIGGER"
        "RESET"
        "HOLD"
    """

    def __init__(self, threshold=5.0, delta=2.0, cooldown_steps=2):
        self.threshold = threshold
        self.delta = delta
        self.cooldown_steps = cooldown_steps

        self.position = 0
        self.cooldown = 0
        self.last_trigger_value = None

    def step(self, x):
        """
        输入：
            x: float

        输出：
            action: str
        """

        # ===== TODO 1 =====
        # 冷却期更新逻辑：
        # 如果 cooldown > 0，则 cooldown -= 1
        if self.cooldown > 0:
            self.cooldown -= 1


        # ===== TODO 2 =====
        # reset 条件：
        # 如果 position == 1 且 last_trigger_value 不为 None
        # 且 x < last_trigger_value - delta
        #
        # reset 含义：
        # - position 归零
        # - cooldown 清零
        # - last_trigger_value 置为 None
        #
        # reset 后直接返回 "RESET"
        if self.position ==1 and self.last_trigger_value is not  None and x < self.last_trigger_value-self.delta:
            self.position = 0
            self.cooldown = 0
            self.last_trigger_value = None
            return "RESET"

        # ===== 触发逻辑（不要修改）=====
        if self.position == 0 and self.cooldown == 0 and x > self.threshold:
            self.position = 1
            self.cooldown = self.cooldown_steps
            self.last_trigger_value = x
            return "TRIGGER"

        return "HOLD"


if __name__ == "__main__":
    system = OnlineDecisionSystem(threshold=5, delta=2, cooldown_steps=2)

    stream = [1, 3, 6, 7, 4, 3, 6, 2, 8]

    for v in stream:
        action = system.step(v)
        print(f"x={v}, pos={system.position}, cooldown={system.cooldown}, last={system.last_trigger_value}, action={action}")
