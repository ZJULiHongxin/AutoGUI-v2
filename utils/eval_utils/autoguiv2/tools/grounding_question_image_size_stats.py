#!/usr/bin/env python3
"""
Aggregate `image_size` values from grounding_mode question result files.
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Iterable, List, Tuple

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
    Convert image_size values to a stable string representation for counting.
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


def compute_image_size_counts(directories: Iterable[str]) -> Tuple[Counter, int]:
    counter: Counter = Counter()

    for path in iter_result_files(directories):
        for question in load_questions(path):
            if "image_size" in question:
                counter[normalize_image_size(question["image_size"])] += 1
    total = sum(counter.values())
    return counter, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        help="Grounding-mode result directories to scan recursively (e.g. ./tasks/region/grounding_mode).",
    )
    args = parser.parse_args()

    counter, total = compute_image_size_counts(args.directories)
    print("Total questions with image_size:", total)
    print("\nAll image_size counts:")
    for value, count in counter.most_common():
        percentage = (count / total * 100) if total else 0.0
        print(f"  {value}: {count} ({percentage:.2f}%)")

    top_three = counter.most_common(3)
    print("\nTop 3 image_size values:")
    for rank, (value, count) in enumerate(top_three, start=1):
        percentage = (count / total * 100) if total else 0.0
        print(f"  {rank}. {value}: {count} ({percentage:.2f}%)")

    top_three_total = sum(count for _, count in top_three)
    overall_percentage = (top_three_total / total * 100) if total else 0.0
    print(f"\nTop 3 combined percentage: {overall_percentage:.2f}%")


if __name__ == "__main__":
    main()

