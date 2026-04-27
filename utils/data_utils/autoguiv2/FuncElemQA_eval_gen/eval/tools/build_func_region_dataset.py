#!/usr/bin/env python3
"""
Build a dataset that pairs target-region functionalities with distractor functionalities.

For each captioning-mode result JSON, we gather:
  * annotated image path
  * the target region's type and functionality
  * (N-1) functionalities sampled from non-target reannotated nodes, where N is the number of region_ids

The output JSON contains a list of entries with randomized option orderings and the index
of the correct (target) functionality. The number of options equals the number of region_ids.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

RESULT_DIRS: Sequence[Path] = (
    Path("/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode"),
    Path("/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode"),
    Path("/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode"),
    Path("/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode"),
)

CACHE_ROOT = Path("/mnt/vdb1/hongxin_li/AutoGUIv2/cache")


def iter_result_files(directories: Sequence[Path]) -> Iterable[Path]:
    """Yield every *_result.json path under the provided directories."""
    for directory in directories:
        if not directory.is_dir():
            logging.warning("Result directory does not exist: %s", directory)
            continue
        for path in sorted(directory.glob("*_result.json")):
            if path.is_file():
                yield path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_dataset_name(question: Dict, fallback: Optional[str]) -> Optional[str]:
    dataset = question.get("dataset_name")
    if dataset:
        return dataset
    if fallback:
        return fallback
    annotated_path = question.get("annotated_image_path")
    if annotated_path:
        parts = Path(annotated_path).parts
        # The dataset name should immediately follow "AutoGUIv2"
        if "AutoGUIv2" in parts:
            idx = parts.index("AutoGUIv2")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def parse_image_id(question: Dict, result_image_name: Optional[str]) -> Optional[str]:
    image_name = question.get("image_name") or result_image_name
    if not image_name:
        annotated_path = question.get("annotated_image_path")
        if annotated_path:
            return Path(annotated_path).stem.split("_")[-1]
        return None
    return Path(image_name).stem


def parse_region_id_from_filename(path: Path) -> str:
    return path.name.split("_", 1)[0]


def find_reannotated_file(nodes_dir: Path, region_id: str) -> Optional[Path]:
    """
    Return the first *_reannotated*.json whose filename starts with the region_id.
    """
    candidates = sorted(nodes_dir.glob(f"{region_id}_*reannotated*.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(nodes_dir.glob(f"{region_id}*reannotated*.json"))
    return candidates[0] if candidates else None


def find_nodes_dir(
    dataset: str,
    model: Optional[str],
    image_id: str,
    cache_root: Path,
) -> Optional[Path]:
    """
    Locate the nodes directory containing reannotated files for a given image.

    Search strategy (in order of preference):
      1. Look under cache/<dataset>/<model>/<version>/<image_id>/nodes.
      2. Look for subdirectories whose names contain the image_id (e.g. version/<image_id>-suffix/nodes).
      3. Fallback: search descendants containing the image_id and ending with /nodes.

    Directories that already contain *_reannotated*.json are preferred.
    """

    def iter_model_roots(root: Path) -> Iterable[Path]:
        if model and (root / model).is_dir():
            yield root / model
        else:
            for candidate in sorted(root.iterdir()):
                if candidate.is_dir():
                    yield candidate

    def iter_version_dirs(model_root: Path) -> Iterable[Path]:
        def version_sort_key(path: Path) -> Tuple[int, str]:
            name = path.name.lower()
            bak_penalty = 1 if "bak" in name else 0
            return (bak_penalty, name)

        for version in sorted((p for p in model_root.iterdir() if p.is_dir()), key=version_sort_key):
            yield version

    def candidate_score(path: Path) -> Tuple[int, Path]:
        has_reannotated = 1 if any(path.glob("*_reannotated*.json")) else 0
        return (has_reannotated, path)

    dataset_root = cache_root / dataset
    if not dataset_root.exists():
        return None

    best_candidate: Optional[Path] = None
    best_score = (-1, Path())

    for model_root in iter_model_roots(dataset_root):
        for version_dir in iter_version_dirs(model_root):
            candidates: List[Path] = []

            direct_candidate = version_dir / image_id / "nodes"
            if direct_candidate.is_dir():
                candidates.append(direct_candidate)

            for subdir in version_dir.iterdir():
                if not subdir.is_dir():
                    continue
                if image_id in subdir.name:
                    candidate = subdir / "nodes"
                    if candidate.is_dir():
                        candidates.append(candidate)

            for candidate in candidates:
                score = candidate_score(candidate)
                if score[0] == 1:
                    return candidate
                if score > best_score:
                    best_candidate = candidate
                    best_score = score

    # Final fallback: recursive search (limited depth) to avoid expensive globbing.
    search_queue = list(iter_model_roots(dataset_root))
    visited: set[Path] = set()
    while search_queue:
        current = search_queue.pop()
        if current in visited or not current.exists():
            continue
        visited.add(current)
        if current.name == "nodes" and image_id in str(current):
            score = candidate_score(current)
            if score[0] == 1:
                return current
            if score > best_score:
                best_candidate = current
                best_score = score
        if current.is_dir():
            for child in current.iterdir():
                if child.is_dir():
                    search_queue.append(child)

    return best_candidate


def load_corrected_bbox(path: Path) -> Optional[Tuple[int, int, int, int]]:
    data = load_json(path)
    bbox = data.get("corrected_bbox")
    if (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(coord, (int, float)) for coord in bbox)
    ):
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            return (int(x1), int(y1), int(x2), int(y2))
    return None


def bboxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def load_revised_functionality(path: Path) -> Optional[str]:
    data = load_json(path)
    new_func = data.get("new_functionality") or {}
    revised = new_func.get("revised functionality")
    if isinstance(revised, str) and revised.strip():
        return revised.strip()
    return None


def gather_distractor_paths(
    nodes_dir: Path,
    excluded_region_ids: Iterable[str],
) -> List[Path]:
    excluded_set = {region_id for region_id in excluded_region_ids if region_id}
    candidates = []
    for path in nodes_dir.glob("*_reannotated*.json"):
        if not path.is_file():
            continue
        region_id = parse_region_id_from_filename(path)
        if region_id not in excluded_set:
            candidates.append(path)
    return candidates


def build_entry(
    question: Dict,
    target_option: Dict,
    distractor_funcs: List[Tuple[str, str]],
    rng: random.Random,
) -> Dict:
    options = [(target_option["functionality"], target_option["region_id"])]
    options.extend(distractor_funcs)
    rng.shuffle(options)
    correct_index = next(
        idx for idx, (_, region_id) in enumerate(options) if region_id == target_option["region_id"]
    )

    return {
        "annotated_image_path": question.get("annotated_image_path"),
        "target_region_id": target_option["region_id"],
        "target_region_type": target_option.get("region_type"),
        "options": [func for func, _ in options],
        "option_region_ids": [region_id for _, region_id in options],
        "correct_index": correct_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct functionality dataset with distractor options.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for sampling distractor functionalities.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    rng = random.Random(args.seed)
    entries: List[Dict] = []
    stats_skipped_no_nodes = 0
    stats_skipped_no_target = 0
    stats_skipped_no_distractors = 0
    stats_skipped_no_target_bbox = 0
    stats_skipped_overlap = 0

    nodes_cache: Dict[Tuple[str, str, str], Optional[Path]] = {}
    bbox_cache: Dict[Path, Optional[Tuple[int, int, int, int]]] = {}

    for result_path in iter_result_files(RESULT_DIRS):
        try:
            result_json = load_json(result_path)
        except json.JSONDecodeError as exc:
            logging.warning("Failed to parse %s: %s", result_path, exc)
            continue

        metadata = result_json.get("metadata") or {}
        model_name = metadata.get("model")
        result_block = result_json.get("result") or {}
        questions = result_block.get("questions") or []
        result_image_name = result_block.get("image_name")

        for question in questions:
            dataset_name = extract_dataset_name(question, fallback=None)
            image_id = parse_image_id(question, result_image_name)
            target_region_id = question.get("target_region_id")

            if not dataset_name or not image_id or not target_region_id:
                logging.debug(
                    "Skipping question due to missing dataset/image/target: file=%s",
                    result_path,
                )
                stats_skipped_no_target += 1
                continue

            target_option = None
            for option in question.get("options", []):
                if option.get("region_id") == target_region_id:
                    if option.get("functionality") and option.get("region_type"):
                        target_option = option
                    break

            if not target_option:
                logging.debug(
                    "Target option missing functionality or not found (region_id=%s) in %s",
                    target_region_id,
                    result_path,
                )
                stats_skipped_no_target += 1
                continue

            cache_key = (dataset_name, model_name or "", image_id)
            if cache_key not in nodes_cache:
                nodes_cache[cache_key] = find_nodes_dir(dataset_name, model_name, image_id, CACHE_ROOT)

            nodes_dir = nodes_cache[cache_key]
            if not nodes_dir:
                logging.debug(
                    "Nodes directory not found for dataset=%s model=%s image_id=%s (file=%s)",
                    dataset_name,
                    model_name,
                    image_id,
                    result_path,
                )
                stats_skipped_no_nodes += 1
                continue

            target_reannotated_path = find_reannotated_file(nodes_dir, target_region_id)
            if not target_reannotated_path:
                logging.debug(
                    "Target reannotated file not found (region_id=%s) in %s",
                    target_region_id,
                    nodes_dir,
                )
                stats_skipped_no_target_bbox += 1
                continue

            if target_reannotated_path not in bbox_cache:
                bbox_cache[target_reannotated_path] = load_corrected_bbox(target_reannotated_path)
            target_bbox = bbox_cache[target_reannotated_path]
            if not target_bbox:
                logging.debug(
                    "Target bounding box missing or invalid: %s",
                    target_reannotated_path,
                )
                stats_skipped_no_target_bbox += 1
                continue

            excluded_region_ids: Set[str] = {target_region_id}
            for option in question.get("options", []):
                region_id = option.get("region_id")
                if isinstance(region_id, str):
                    excluded_region_ids.add(region_id)
            
            # 获取 region_ids 以确定需要的选项总数
            region_ids_list = question.get("region_ids", [])
            for region_id in region_ids_list:
                if isinstance(region_id, str):
                    excluded_region_ids.add(region_id)
            
            # 计算需要的干扰项数量（总选项数 - 1个目标选项）
            required_distractor_count = len(region_ids_list) - 1
            if required_distractor_count < 1:
                logging.debug(
                    "region_ids count too small (count=%d) in %s",
                    len(region_ids_list),
                    result_path,
                )
                stats_skipped_no_distractors += 1
                continue

            distractor_paths = gather_distractor_paths(nodes_dir, excluded_region_ids)
            if not distractor_paths:
                logging.debug(
                    "No eligible distractor files in %s (target=%s)",
                    nodes_dir,
                    target_region_id,
                )
                stats_skipped_no_distractors += 1
                continue

            eligible_distractors: List[Path] = []
            for path in distractor_paths:
                if path not in bbox_cache:
                    bbox_cache[path] = load_corrected_bbox(path)
                bbox = bbox_cache[path]
                if bbox and not bboxes_overlap(target_bbox, bbox):
                    eligible_distractors.append(path)

            # 检查是否有足够的非重叠干扰项
            if len(eligible_distractors) < required_distractor_count:
                logging.debug(
                    "Not enough non-overlapping distractors in %s (target=%s, required=%d, available=%d)",
                    nodes_dir,
                    target_region_id,
                    required_distractor_count,
                    len(eligible_distractors),
                )
                stats_skipped_overlap += 1
                continue

            # 精确采样所需数量的干扰项
            selected_paths = rng.sample(eligible_distractors, required_distractor_count)

            distractor_funcs: List[Tuple[str, str]] = []
            for path in selected_paths:
                revised = load_revised_functionality(path)
                if revised:
                    distractor_funcs.append((revised, parse_region_id_from_filename(path)))

            # 确保获取到足够数量的有效功能描述
            if len(distractor_funcs) < required_distractor_count:
                logging.debug(
                    "Selected distractors missing revised functionality in %s (required=%d, got=%d)",
                    nodes_dir,
                    required_distractor_count,
                    len(distractor_funcs),
                )
                stats_skipped_no_distractors += 1
                continue

            entry = build_entry(question, target_option, distractor_funcs, rng)
            entries.append(entry)

    output_payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_entries": len(entries),
        "skipped": {
            "no_target_info": stats_skipped_no_target,
            "no_nodes_dir": stats_skipped_no_nodes,
            "no_distractors": stats_skipped_no_distractors,
            "no_target_bbox": stats_skipped_no_target_bbox,
            "no_non_overlapping_distractors": stats_skipped_overlap,
        },
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logging.info(
        "Dataset written to %s (entries=%d, skipped=%s)",
        args.output,
        len(entries),
        output_payload["skipped"],
    )


if __name__ == "__main__":
    main()

