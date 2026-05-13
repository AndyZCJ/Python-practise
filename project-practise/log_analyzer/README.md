# log_analyzer

这是一个偏工程化的 Python 小项目，用来对日志文件做简单分析。

它会：
- 解析日志行
- 统计不同日志级别数量
- 统计高频错误消息
- 按小时汇总日志数量
- 把摘要写到 `summary.csv`

## 当前目录结构

```text
log_analyzer/
├── io_utils.py
├── parser.py
├── stats.py
├── main.py
├── log.txt
├── summary.csv
└── tests/
```

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `main.py` | 程序入口，负责读取路径、聚合结果并输出摘要 |
| `parser.py` | 解析单行日志，提取时间、级别和消息 |
| `stats.py` | 聚合统计逻辑 |
| `io_utils.py` | 收集日志文件路径 |
| `tests/test_parser.py` | 针对解析函数的最小测试 |
| `log.txt` | 示例日志输入 |
| `summary.csv` | 一次运行后的示例输出 |

## 运行方式

建议在当前目录执行：

```powershell
python .\main.py .\log.txt --top-k 3
```

如果要跑测试，可以使用：

```powershell
pytest .\tests\test_parser.py -q
```

## 适合学习的点

这个项目适合练习：
- 用正则表达式解析结构化文本
- 用 `Counter` 和 `defaultdict` 做统计
- 把脚本拆分为解析、统计、I/O 和入口模块
- 给小型工具补最基本的测试

## 说明

这里的 `log.txt` 和 `summary.csv` 是项目示例文件，因此没有被根目录 `.gitignore` 一刀切忽略。

