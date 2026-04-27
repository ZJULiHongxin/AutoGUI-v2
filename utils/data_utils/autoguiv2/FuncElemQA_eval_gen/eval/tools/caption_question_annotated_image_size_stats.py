#!/usr/bin/env python3
"""
Aggregate `annotated_image_size` values from captioning_mode question result files.
"""

import glob
import json
import os
from collections import Counter
from typing import Iterable, List, Tuple

CAPTIONING_RESULT_DIRS: List[str] = [
    "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode",
]

RESULT_PATTERN = "**/*_result.json"


def iter_result_files(base_dirs: Iterable[str]) -> Iterable[str]:
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        pattern = os.path.join(base_dir, RESULT_PATTERN)
        for path in glob.iglob(pattern, recursive=True):
            if os.path.isfile(path):
                yield path


def load_questions(path: str) -> List[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    questions = data.get("questions")
    if not isinstance(questions, list):
        result_section = data.get("result")
        if isinstance(result_section, dict):
            questions = result_section.get("questions")
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, dict)]


def normalize_image_size(value) -> str:
    """
    Convert annotated_image_size values to a stable string representation for counting.
    """
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if isinstance(item, (int, float)):
                items.append(str(item))
            else:
                items.append(json.dumps(item, sort_keys=True))
        return "[" + ", ".join(items) + "]"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def compute_image_size_counts() -> Tuple[Counter, int]:
    counter: Counter = Counter()

    for path in iter_result_files(CAPTIONING_RESULT_DIRS):
        for question in load_questions(path):
            if "annotated_image_size" in question:
                counter[normalize_image_size(question["annotated_image_size"])] += 1
    total = sum(counter.values())
    return counter, total


def main() -> None:
    counter, total = compute_image_size_counts()
    print("Total questions with annotated_image_size:", total)
    print("\nAll annotated_image_size counts:")
    for value, count in counter.most_common():
        percentage = (count / total * 100) if total else 0.0
        print(f"  {value}: {count} ({percentage:.2f}%)")

    top_three = counter.most_common(3)
    print("\nTop 3 annotated_image_size values:")
    for rank, (value, count) in enumerate(top_three, start=1):
        percentage = (count / total * 100) if total else 0.0
        print(f"  {rank}. {value}: {count} ({percentage:.2f}%)")

    top_three_total = sum(count for _, count in top_three)
    overall_percentage = (top_three_total / total * 100) if total else 0.0
    print(f"\nTop 3 combined percentage: {overall_percentage:.2f}%")


if __name__ == "__main__":
    main()

