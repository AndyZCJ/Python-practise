import argparse
import csv
import logging

from parser import parse_line
from stats import StatsAggregator
from io_utils import collect_log_files


# ---------------- Logging Setup ----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- Main Pipeline ----------------

def main():

    parser = argparse.ArgumentParser(description="Simple log analyzer")

    parser.add_argument(
        "path",
        help="Log file or directory containing .log files"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top K most frequent error messages"
    )

    args = parser.parse_args()

    logger.info("Starting log analyzer")

    aggregator = StatsAggregator()

    try:
        files = collect_log_files(args.path)
    except ValueError as e:
        logger.error(str(e))
        return

    logger.info("Found %d log files", len(files))

    for file in files:
        logger.info("Processing %s", file)

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                record = parse_line(line)
                if record:
                    aggregator.update(record)

    result = aggregator.summary(args.top_k)

    logger.info("Summary generated")

    # -------- Console output --------

    print("\n===== Log Summary =====")
    print(f"INFO: {result['INFO']}")
    print(f"WARNING: {result['WARNING']}")
    print(f"ERROR: {result['ERROR']}")

    print("\nTop Errors:")
    for msg, cnt in result["top_errors"]:
        print(f"{cnt}x {msg}")

    print("\nLogs per hour:")
    for hour, cnt in sorted(result["per_hour"].items()):
        print(f"{hour}:00 -> {cnt}")

    # -------- CSV Output --------

    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "count"])

        for hour, cnt in sorted(result["per_hour"].items()):
            writer.writerow([hour, cnt])

    logger.info("summary.csv generated")


if __name__ == "__main__":
    main()

