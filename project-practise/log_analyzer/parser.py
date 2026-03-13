import re
from datetime import datetime
from typing import Optional, Dict

pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)"
)


def parse_line(line: str) -> Optional[Dict]:

    m = pattern.match(line.strip())
    if not m:
        return None

    data = m.groupdict()

    try:
        data["timestamp"] = datetime.strptime(
            data["timestamp"], "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return None

    if data["level"] not in ("INFO", "WARNING", "ERROR"):
        return None

    return data

