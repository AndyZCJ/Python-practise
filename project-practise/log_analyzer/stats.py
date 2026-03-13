from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from datetime import datetime


class StatsAggregator:

    def __init__(self):
        self.level_counter: Counter[str] = Counter()
        self.error_counter: Counter[str] = Counter()
        self.hour_counter: Dict[int, int] = defaultdict(int)

    def update(self, record: dict) -> None:

        level = record["level"]

        self.level_counter[level] += 1

        if level == "ERROR":
            self.error_counter[record["message"]] += 1

        hour = record["timestamp"].hour
        self.hour_counter[hour] += 1

    def summary(self, top_k: int = 3) -> dict:

        return {
            "INFO": self.level_counter["INFO"],
            "WARNING": self.level_counter["WARNING"],
            "ERROR": self.level_counter["ERROR"],
            "top_errors": self.error_counter.most_common(top_k),
            "per_hour": dict(self.hour_counter),
        }

