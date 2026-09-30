from pathlib import Path
import json
import re
from opencc import OpenCC
import unicodedata


def contains_symbol(obj, symbol):
    if isinstance(obj, str):
        return symbol in obj
    return False



def file_reading(folder):
    stats = {
        'file_count': 0,
        'json_count': 0,
        'total_character_count': 0,
        'average_character_count': 0,
        'template_marker_count':0, 
        'table_marker_count': 0,
        'longest_text_len': 0,
        'shortest_text_len':0,
        'empty_text_count': 0,
        'not_empty_text_count': 0,
        'short_text_count': 0,
        'success_count': 0,
        'fail_count': 0
    }

    failed_lines = []
    shortest = float('+inf')
    longest = 0
    for file in Path(folder).rglob("wiki_*"):
        if not file.is_file():
            continue

        stats['file_count'] += 1

        with open(file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                stats['json_count'] += 1
                try:
                    j = json.loads(line)
                    text = j.get("text")
                    stats['success_count'] += 1
                    if not text:
                        stats['empty_text_count'] += 1
                        continue
                    stats['not_empty_text_count'] += 1
                    if len(text) < 100:
                        stats['short_text_count'] += 1
                        
                    if contains_symbol(text, "{{") or contains_symbol(text, "}}"):
                        stats["template_marker_count"] += 1
                    if contains_symbol(text, "{|") or contains_symbol(text, "|}"):
                        stats["table_marker_count"] += 1
                    
                    stats['total_character_count'] += len(text)
                    if len(text) > longest:
                        longest = len(text)
                    if len(text) < shortest:
                        shortest = len(text)
                except json.JSONDecodeError as e:
                    stats['fail_count'] += 1
                    failed_lines.append({
                        'file': str(file),
                        'line': line_number,
                        'error': str(e)
                    })
    if stats['not_empty_text_count'] != 0:
        stats['average_character_count'] = stats['total_character_count']/stats['not_empty_text_count']
    else:
        stats['average_character_count'] = 0
    stats['longest_text_len'] = longest
    stats['shortest_text_len'] = shortest
    return stats, failed_lines


if __name__ == "__main__":
    file = "data/extracted/AA"
    stats, failed_lines = file_reading(file)
    print(stats, failed_lines)