#!/usr/bin/env python3
import glob
import json
import os
import re
from typing import Iterable, List

TARGET_DIRS: List[str] = [
    "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode",
]

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def iter_result_files() -> Iterable[str]:
    for directory in TARGET_DIRS:
        pattern = os.path.join(directory, "**", "*_result.json")
        for path in glob.glob(pattern, recursive=True):
            yield path


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(WORD_PATTERN.findall(text))


def extract_option_contexts(question: dict) -> List[str]:
    contexts: List[str] = []

    options = question.get("options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            context = option.get("option_context")
            if isinstance(context, str):
                contexts.append(context)
            elif isinstance(context, list):
                contexts.extend(str(item) for item in context if item is not None)

    alt_context = question.get("option_context")
    if isinstance(alt_context, str):
        contexts.append(alt_context)
    elif isinstance(alt_context, list):
        contexts.extend(str(item) for item in alt_context if item is not None)

    return contexts


def main() -> None:
    total_words = 0
    total_questions = 0

    for file_path in iter_result_files():
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {file_path}: {exc}")
            continue

        questions = data.get("questions")
        if not isinstance(questions, list):
            result_section = data.get("result")
            if isinstance(result_section, dict):
                questions = result_section.get("questions")
        if not isinstance(questions, list):
            continue

        for question in questions:
            if not isinstance(question, dict):
                continue

            word_count = count_words(question.get("question", ""))
            for context in extract_option_contexts(question):
                word_count += count_words(context)

            total_words += word_count
            total_questions += 1

    average = total_words / total_questions if total_questions else 0.0

    print(f"Total question count: {total_questions}")
    print(f"Total word count (question + option_contexts): {total_words}")
    print(f"Average words per question (including option_contexts): {average:.4f}")


if __name__ == "__main__":
    main()

