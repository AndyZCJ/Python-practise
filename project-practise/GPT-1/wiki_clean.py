from pathlib import Path
import json
import re
from opencc import OpenCC
import unicodedata
converter = OpenCC("t2s")
INPUT_FILE = Path("data/extracted/AA/wiki_00")
OUTPUT_FILE = Path("data/processed/wiki_clean_00.jsonl")
MIN_TEXT_LENGTH = 100
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)#统一兼容字符和全角字符

    text = converter.convert(text)#繁体转简体

    text = text.replace("\r\n", "\n").replace("\r", "\n") #统一windows、旧Mac换行符

    cleaned_lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)# 将制表符及连续空格压缩成一个普通空格

        line = line.strip()# 去掉每行首尾空白

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def has_template_marker(text: str) -> bool:
    return "{{" in text or "}}" in text

def has_table_marker(text: str) -> bool:
    return "{|" in text or "|}" in text

def clean_file(input_file: Path, output_file: Path) -> dict:
    stats = {
        "input_count": 0,
        "kept_count": 0,
        "short_text_count": 0,
        "template_marker_count": 0,
        "table_marker_count": 0,
        "json_fail_count": 0,
        "empty_text_count": 0,
        "total_clean_character_count": 0
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_file.open("r", encoding="utf-8") as source,
        output_file.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            line = line.strip()

            if not line:
                continue

            stats["input_count"] += 1

            try:
                article = json.loads(line)
            except json.JSONDecodeError as error:
                stats["json_fail_count"] += 1
                print(
                    f"JSON 解析失败：第 {line_number} 行，"
                    f"错误信息：{error}"
                )
                continue

            raw_text = article.get("text")

            if not isinstance(raw_text, str) or not raw_text.strip():
                stats["empty_text_count"] += 1
                continue

            if has_template_marker(raw_text):
                stats["template_marker_count"] += 1
                continue

            if has_table_marker(raw_text):
                stats["table_marker_count"] += 1
                continue

            cleaned_text = normalize_text(raw_text)

            if len(cleaned_text) < MIN_TEXT_LENGTH:
                stats["short_text_count"] += 1
                continue

            raw_title = article.get("title", "")
            cleaned_title = normalize_text(raw_title) if isinstance(
                raw_title, str
            ) else ""

            cleaned_article = {
                "id": article.get("id"),
                "title": cleaned_title,
                "text": cleaned_text,
                "url": article.get("url")
            }

            destination.write(
                json.dumps(cleaned_article, ensure_ascii=False) + "\n"
            )

            stats["kept_count"] += 1
            stats["total_clean_character_count"] += len(cleaned_text)
    return stats

if __name__ == "__main__":
    result = clean_file(INPUT_FILE, OUTPUT_FILE)

    print("清洗完成：")
    for name, value in result.items():
        print(f"{name}{value}")