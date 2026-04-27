#!/usr/bin/env python3
"""
Compute word statistics for question fields in grounding_mode result files.
"""

import glob
import json
import os
import re
from typing import Iterable, List, Tuple

GROUNDING_RESULT_DIRS: List[str] = [
    "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/grounding_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/grounding_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/grounding_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/grounding_mode",
]

RESULT_PATTERN = "**/*_result.json"
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")


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


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def compute_stats() -> Tuple[int, int]:
    total_questions = 0
    total_words = 0

    for path in iter_result_files(GROUNDING_RESULT_DIRS):
        for question_data in load_questions(path):
            question_text = question_data.get("question")
            if isinstance(question_text, str) and question_text.strip():
                total_questions += 1
                total_words += count_words(question_text)
    return total_questions, total_words


def main() -> None:
    total_questions, total_words = compute_stats()
    average = (total_words / total_questions) if total_questions else 0.0
    print(f"Total question count: {total_questions}")
    print(f"Total word count (question field only): {total_words}")
    print(f"Average words per question: {average:.4f}")


if __name__ == "__main__":
    main()

