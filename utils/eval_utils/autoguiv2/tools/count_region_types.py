#!/usr/bin/env python3
import argparse
import os
import json
from typing import Any, Iterable, List, Tuple, Dict
from collections import Counter
import sys
from pathlib import Path


DIRECTORIES: List[str] = []


# ===== 6-class taxonomy mapping helpers =====
MAIN_PARENT_CATEGORIES: List[str] = [
    "Primary Interface Containers",
    "Global Navigation & Structure",
    "Content & Data Display",
    "Interaction & Input",
    "Contextual & Temporary Regions",
]
OTHERS_BUCKET = "Others"


def build_leaf_to_parent_mapping() -> Dict[str, str]:
    """
    Build mapping: leaf type -> one of the 5 main parent categories.
    Anything not in this mapping will be assigned to 'Others'.
    """
    try:
        # Ensure repository root is importable when this file is run directly.
        for parent in Path(__file__).resolve().parents:
            taxonomy_path = parent / "utils" / "data_utils" / "autoguiv2" / "classify_region_types" / "classify_functional_regions.py"
            if taxonomy_path.exists() and str(parent) not in sys.path:
                sys.path.append(str(parent))
                break
        from utils.data_utils.autoguiv2.classify_region_types.classify_functional_regions import TAXONOMY  # type: ignore
    except Exception:
        return {}

    try:
        # Evaluate TAXONOMY string without extra leaf types
        taxonomy_dict: Dict[str, Dict[str, str]] = eval(TAXONOMY.replace("{extra_leaf_types}", ""))
    except Exception:
        return {}

    leaf_to_parent: Dict[str, str] = {}
    for parent, leaves in taxonomy_dict.items():
        if parent not in MAIN_PARENT_CATEGORIES:
            continue
        if not isinstance(leaves, dict):
            continue
        for leaf in leaves.keys():
            leaf_to_parent[str(leaf)] = parent
    return leaf_to_parent


LEAF_TO_PARENT = build_leaf_to_parent_mapping()


def resolve_parent_category(leaf_type: str) -> str:
    """
    Resolve 6-bucket parent for a given leaf type string.
    - Exact match to known leaf -> its main parent
    - 'Other: ...' / 'Other; ...' / 'Unclassified' -> Others
    - Anything else not matched -> Others
    """
    s = (leaf_type or "").strip()
    if not s:
        return OTHERS_BUCKET
    lower = s.lower()
    if lower.startswith("other:") or lower.startswith("other;") or lower == "other":
        return OTHERS_BUCKET
    if lower == "unclassified":
        return OTHERS_BUCKET
    return LEAF_TO_PARENT.get(s, OTHERS_BUCKET)


def normalize_key(name: str) -> str:
    """
    归一化键名：去除非字母数字字符，转小写，用于识别 region type 变体。
    例如：'region_type' / 'region type' / 'RegionType' -> 'regiontype'
    """
    return "".join(ch.lower() for ch in name if ch.isalnum())


REGION_TYPE_KEY_NORM = "regiontype"


def iter_json_paths(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".json"):
                yield os.path.join(dirpath, fname)


def extract_region_types(obj: Any) -> Iterable[Any]:
    """
    递归提取所有键名等价于 region type 的值。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if normalize_key(str(k)) == REGION_TYPE_KEY_NORM:
                yield v
            # 继续向下递归
            yield from extract_region_types(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_region_types(item)
    # 其他基础类型忽略


def load_json_robust(path: str) -> Iterable[Any]:
    """
    尝试加载 JSON。
    - 优先使用标准 JSON（json.load）
    - 若失败，尝试按 JSONL/NDJSON 逐行解析
    返回一个可迭代对象，每项为一个解析得到的 JSON 文档（对象或数组等）
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [data]
    except json.JSONDecodeError:
        docs: List[Any] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 非 JSONL 格式，放弃逐行解析
                        return []
        except OSError:
            return []
        return docs
    except OSError:
        return []


def count_region_types_in_dir(directory: str) -> Tuple[Counter, int, int]:
    """
    返回：
      - Counter：region type -> 数量
      - files_total：扫描到的 JSON 文件数
      - files_parsed：成功解析的 JSON 文件数（标准 JSON 或 JSONL 至少成功一条）
    """
    counter: Counter = Counter()
    files_total = 0
    files_parsed = 0
    for path in iter_json_paths(directory):
        files_total += 1
        docs = load_json_robust(path)
        if not docs:
            continue
        files_parsed += 1
        for doc in docs:
            for value in extract_region_types(doc):
                # 将值统一为字符串便于统计
                key = str(value)
                counter[key] += 1
    return counter, files_total, files_parsed


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n=== {title} ===")
    total = sum(counter.values())
    print(f"总计: {total}")
    for key, cnt in counter.most_common():
        parent = resolve_parent_category(key)
        print(f"[{parent}] {key}\t{cnt}")


def summarize_by_parent(counter: Counter) -> Counter:
    """
    Aggregate counts into the 6 buckets (5 mains + Others).
    """
    agg: Counter = Counter()
    for leaf, cnt in counter.items():
        parent = resolve_parent_category(leaf)
        agg[parent] += cnt
    # Ensure all buckets present
    for parent in MAIN_PARENT_CATEGORIES + [OTHERS_BUCKET]:
        _ = agg[parent]
    return agg


def print_parent_summary(title: str, parent_counter: Counter) -> None:
    print(f"\n=== {title} ===")
    total = sum(parent_counter.values())
    print(f"总计: {total}")
    for parent in MAIN_PARENT_CATEGORIES + [OTHERS_BUCKET]:
        print(f"{parent}\t{parent_counter[parent]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count region_type values in generated FuncRegion JSON files."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        default=DIRECTORIES,
        help="Directories to scan recursively. Pass captioning_mode/grounding_mode directories explicitly.",
    )
    args = parser.parse_args()

    all_counter: Counter = Counter()
    dir_stats: List[Tuple[str, Counter, int, int]] = []

    if not args.directories:
        parser.error("Please provide at least one directory to scan.")

    for d in args.directories:
        if not os.path.isdir(d):
            print(f"[警告] 目录不存在，跳过：{d}")
            continue
        counter, files_total, files_parsed = count_region_types_in_dir(d)
        dir_stats.append((d, counter, files_total, files_parsed))
        all_counter.update(counter)

    # 输出每个目录统计
    for d, counter, files_total, files_parsed in dir_stats:
        print(f"\n目录: {d}")
        print(f"JSON 文件数: {files_total}，成功解析: {files_parsed}")
        print_counter("按 region type 统计", counter)
        dir_parent_summary = summarize_by_parent(counter)
        print_parent_summary("按 6 大类统计（含 Others）", dir_parent_summary)

    # 合并汇总
    print_counter("\n所有目录汇总（Grand Total）", all_counter)
    grand_parent_summary = summarize_by_parent(all_counter)
    print_parent_summary("\n所有目录 6 大类汇总（Grand Total, 含 Others）", grand_parent_summary)


if __name__ == "__main__":
    main()

