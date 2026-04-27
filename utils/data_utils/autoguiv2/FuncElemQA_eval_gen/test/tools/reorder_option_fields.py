#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reorder option fields in JSON files under a directory.

Target order for option dictionaries:
1) label
2) region_id
3) bbox
4) metrics
5) remaining keys in their original order

Rules:
- Operates on any list value under keys named "options".
- Additionally, any dict that contains a "label" key is treated as an option dict.
- Works recursively through the JSON structure.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

PREFERRED_ORDER = ["label", "region_id", "bbox", "metrics"]


def reorder_option_dict(option: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with keys reordered to preferred order then remaining in original order."""
    # JSON in Python 3.7+ preserves dict insertion order on load, so we can rely on it
    reordered: Dict[str, Any] = {}
    for key in PREFERRED_ORDER:
        if key in option:
            reordered[key] = option[key]
    for key, value in option.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def is_option_dict(d: Dict[str, Any]) -> bool:
    """Heuristic: treat any dict containing a 'label' as an option dict."""
    return isinstance(d, dict) and "label" in d


def process_node(node: Any) -> Any:
    """Recursively process a JSON node, reordering option dicts."""
    if isinstance(node, dict):
        processed: Dict[str, Any] = {}
        for key, value in node.items():
            # Recurse first
            new_value = process_node(value)
            # If this is an options list, reorder each dict item
            if key == "options" and isinstance(new_value, list):
                new_list: List[Any] = []
                for item in new_value:
                    if isinstance(item, dict) and is_option_dict(item):
                        new_list.append(reorder_option_dict(item))
                    else:
                        new_list.append(process_node(item))
                processed[key] = new_list
            else:
                processed[key] = new_value
        # If the dict itself looks like an option, reorder it as well
        if is_option_dict(processed):
            return reorder_option_dict(processed)
        return processed
    elif isinstance(node, list):
        return [process_node(item) for item in node]
    else:
        return node


def process_file(path: Path, write: bool = True, indent: int = 2, ensure_ascii: bool = False) -> bool:
    """Process a single JSON file; return True if file changed and was written."""
    try:
        with path.open("r", encoding="utf-8") as f:
            original_text = f.read()
        data = json.loads(original_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"Skip (read error): {path} -> {e}")
        return False

    processed = process_node(data)
    try:
        new_text = json.dumps(processed, ensure_ascii=ensure_ascii, indent=indent)
        new_text += "\n"  # ensure trailing newline for POSIX friendliness
    except (TypeError, ValueError) as e:
        print(f"Skip (dump error): {path} -> {e}")
        return False

    if new_text == original_text:
        return False

    if write:
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(new_text)
            return True
        except OSError as e:
            print(f"Write failed: {path} -> {e}")
            return False
    return False


def iter_json_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".json"):
                yield Path(dirpath) / name


def main():
    parser = argparse.ArgumentParser(description="Reorder option fields in JSON files under a directory.")
    parser.add_argument("target_dir", help="Directory containing JSON files to process.")
    parser.add_argument("--dry-run", action="store_true", help="Only report files that would change.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent width (default: 2).")
    parser.add_argument("--ascii", action="store_true", help="Use ensure_ascii=True when writing JSON.")
    args = parser.parse_args()

    root = Path(args.target_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Not a directory: {root}")
        return

    changed = 0
    total = 0
    for jf in iter_json_files(root):
        total += 1
        if args.dry_run:
            # simulate processing and check if would change
            try:
                with jf.open("r", encoding="utf-8") as f:
                    original_text = f.read()
                data = json.loads(original_text)
                processed = process_node(data)
                new_text = json.dumps(processed, ensure_ascii=args.ascii, indent=args.indent) + "\n"
                if new_text != original_text:
                    print(f"Would change: {jf}")
                    changed += 1
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Skip (error): {jf} -> {e}")
        else:
            if process_file(jf, write=True, indent=args.indent, ensure_ascii=args.ascii):
                print(f"Changed: {jf}")
                changed += 1

    mode = "would change" if args.dry_run else "changed"
    print(f"Done: {mode} {changed} file(s) out of {total}.")


if __name__ == "__main__":
    main()


