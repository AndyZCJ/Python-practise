from pathlib import Path
import glob
import os

def collect_log_files(path):
    """
    如果 path 是文件：
        返回 [path]

    如果 path 是目录：
        返回该目录下所有 .log 文件（不递归）

    如果都不是：
        raise ValueError
    """
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = list(p.glob("*.log"))
        if not files:
            ValueError("No .log files found in directory")
            return files
    else:
        raise ValueError
