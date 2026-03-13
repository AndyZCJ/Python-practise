class MovingAverageMonitor:
    """
    一个简单的滑动平均监控器。

    系统行为：
    - 连续接收数值输入
    - 维护最近 window_size 个数的平均值
    - 如果当前值比滑动平均高出 threshold，则触发告警

    这是一个“状态驱动”的系统。
    """

    def __init__(self, window_size=3, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold

        self.buffer = []
        self.current_avg = None

    def step(self, x):
        """
        输入：
            x: float

        输出：
            alert: bool
        """

        # ===== TODO 1 =====
        # 更新 buffer：
        # - 把 x 加入 buffer
        # - 如果 buffer 长度超过 window_size，移除最早的元素
        self.buffer.append(x)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # ===== TODO 2 =====
        # 更新 current_avg
        # 注意：buffer 可能为空
        if len(self.buffer)>0:
            self.current_avg = sum(self.buffer) / len(self.buffer)


        # ===== 决策逻辑（不要修改）=====
        if self.current_avg is None:
            return False

        if x > self.current_avg + self.threshold:
            return True

        return False


if __name__ == "__main__":
    monitor = MovingAverageMonitor(window_size=3, threshold=1.5)

    data = [1, 2, 3, 10, 4, 5]

    for d in data:
        alert = monitor.step(d)
        print(f"x={d}, avg={monitor.current_avg}, alert={alert}")
