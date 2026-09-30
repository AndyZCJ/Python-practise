from pathlib import Path
import json
import re
from opencc import OpenCC
import unicodedata

INPUT_FILE = Path("data/processed/wiki_clean_00.jsonl")

def file_check(input_file: Path) -> dict:
    stats = {
        "num_check": False,
        "type_check": False,
        "template_marker_check": False,
        "table_marker_check": False,
        "character_check": False
    }