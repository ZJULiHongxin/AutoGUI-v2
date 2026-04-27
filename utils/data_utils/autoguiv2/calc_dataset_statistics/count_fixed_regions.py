#!/usr/bin/env python3
"""
Quick utility to count the number of corrected bounding boxes stored in the cache.

The correction discovery logic is shared with `reannotate_func_after_fixing.py`.
This script simply reports how many unique node corrections are pending (or were
produced) after human adjustment, optionally grouped by namespace/model/version.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from utils.data_utils.autoguiv2.reannotate_func_after_fixing import (
    CorrectionTask,
    discover_all_corrections,
    discover_corrections,
)


def _normalize_image_path(root_path: str, cache_dir: str) -> str:
    abs_path = os.path.abspath(root_path)
    cache_dir_abs = os.path.abspath(cache_dir)
    try:
        rel_path = os.path.relpath(abs_path, cache_dir_abs)
    except ValueError:
        # Different drives or other OS-specific issues; fall back to absolute path.
        return abs_path
    if rel_path.startswith(".."):
        return abs_path
    return rel_path


def _parse_groupings(group_by: Iterable[str]) -> Tuple[str, ...]:
    valid_keys = {"namespace", "model", "version", "image"}
    parsed: List[str] = []
    for key in group_by:
        key_normalized = key.lower()
        if key_normalized not in valid_keys:
            raise ValueError(
                f"Invalid group key '{key}'. Expected one of: "
                f"{', '.join(sorted(valid_keys))}"
            )
        parsed.append(key_normalized)
    return tuple(parsed)


def _tasks_for_args(args: argparse.Namespace) -> List[CorrectionTask]:
    # If the user keeps all wildcard selectors, we can rely on the bulk helper
    # to avoid redundant directory scans.
    if args.namespace == "*" and args.target_model == "*" and args.version == "*":
        return discover_all_corrections(args.cache_dir)
    return discover_corrections(
        args.cache_dir,
        args.namespace,
        args.target_model,
        args.version,
    )


def _group_tasks(
    tasks: Iterable[CorrectionTask],
    group_keys: Tuple[str, ...],
    cache_dir: str,
) -> Dict[str, int]:
    if not group_keys:
        try:
            total = len(tasks)  # type: ignore[arg-type]
        except TypeError:
            total = sum(1 for _ in tasks)
        return {"TOTAL": total}

    counter: Dict[str, int] = defaultdict(int)
    for task in tasks:
        key_parts = []
        for key in group_keys:
            if key == "namespace":
                key_parts.append(task.namespace)
            elif key == "model":
                key_parts.append(task.model_name)
            elif key == "version":
                key_parts.append(task.version)
            elif key == "image":
                key_parts.append(_normalize_image_path(task.root_image_path, cache_dir))
        counter[" / ".join(key_parts)] += 1
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _group_unique_images(
    tasks: Iterable[CorrectionTask],
    group_keys: Tuple[str, ...],
    cache_dir: str,
) -> Dict[str, int]:
    if not group_keys:
        unique = {
            _normalize_image_path(task.root_image_path, cache_dir)
            for task in tasks
        }
        return {"TOTAL_IMAGES": len(unique)}

    image_sets: Dict[str, set] = defaultdict(set)
    for task in tasks:
        key_parts = []
        for key in group_keys:
            if key == "namespace":
                key_parts.append(task.namespace)
            elif key == "model":
                key_parts.append(task.model_name)
            elif key == "version":
                key_parts.append(task.version)
            elif key == "image":
                key_parts.append(_normalize_image_path(task.root_image_path, cache_dir))
        group_key = " / ".join(key_parts)
        image_sets[group_key].add(
            _normalize_image_path(task.root_image_path, cache_dir)
        )

    counts = {key: len(images) for key, images in image_sets.items()}
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _save_counts(payload: Dict[str, int], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count the number of corrected bounding boxes inside the cache."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/cache",
        help="Root directory containing cached annotation runs.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="*",
        help="Namespace (benchmark) selector. Use '*' to include all.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="gemini-2.5-pro-thinking",
        help="Original annotation model selector. Use '*' to include all.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="*",
        help="Annotation version selector. Use '*' to include all.",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="*",
        default=("namespace", "model", "version"),
        help=(
            "Optional grouping keys. Choose any combination of "
            "'namespace', 'model', 'version', 'image'. "
            "Defaults to namespace/model/version."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the counts as JSON.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit printed groups to the top-N entries.",
    )
    args, _ = parser.parse_known_args()

    group_keys = _parse_groupings(args.group_by)

    cache_dir_abs = os.path.abspath(args.cache_dir)

    print("Scanning cache for corrected bounding boxes...")
    tasks: List[CorrectionTask] = list(_tasks_for_args(args))
    total = len(tasks)
    unique_images_total = len(
        {
            _normalize_image_path(task.root_image_path, cache_dir_abs)
            for task in tasks
        }
    )
    print(
        f"Discovered {total} unique corrected regions across "
        f"{unique_images_total} unique images."
    )

    grouped_counts = _group_tasks(tasks, group_keys, cache_dir_abs)
    grouped_image_counts = _group_unique_images(tasks, group_keys, cache_dir_abs)
    if args.top is not None and args.top > 0:
        items = list(grouped_counts.items())[: args.top]
        image_items = list(grouped_image_counts.items())[: args.top]
    else:
        items = list(grouped_counts.items())
        image_items = list(grouped_image_counts.items())

    if group_keys:
        label = " / ".join(group_keys)
        print(f"\nCounts grouped by {label}:")
        for key, count in items:
            print(f"  {key}: {count}")
        print(f"\nUnique images grouped by {label}:")
        for key, count in image_items:
            print(f"  {key}: {count}")

    if args.output_json:
        payload = {
            "total_corrections": total,
            "unique_image_total": unique_images_total,
            "grouped_counts": grouped_counts,
            "grouped_unique_images": grouped_image_counts,
            "group_keys": group_keys,
            "cache_dir": cache_dir_abs,
            "namespace": args.namespace,
            "target_model": args.target_model,
            "version": args.version,
        }
        output_path = _save_counts(payload, args.output_json)
        print(f"\nCounts saved to {output_path}")


if __name__ == "__main__":
    main()
