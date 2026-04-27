#!/usr/bin/env python3
"""
Calculate statistics for the FuncElemCap dataset.

This script summarizes the dataset from multiple perspectives, including:
    - Average word counts for functionality annotations, description annotations, and captioning questions
    - Top-5 image resolutions and their proportions
    - Distribution of interaction types (Plotly bar chart)
    - Proportions of density classes (Plotly pie chart)
    - Proportions of similarity group sizes (Plotly pie chart)

By default it expects the cached HuggingFace dataset saved by
`31_convert_elemcap_to_hf_dataset.py` at
`/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemCap`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

from tqdm import tqdm
try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError as err:  # pragma: no cover - handled at runtime
    raise ImportError(
        "datasets library is required. Install with: pip install datasets"
    ) from err

try:
    import plotly.express as px
except ImportError as err:  # pragma: no cover - handled at runtime
    raise ImportError(
        "plotly is required. Install with: pip install plotly"
    ) from err

logger = logging.getLogger(__name__)

ELEGANT_QUALITATIVE_PALETTE = [
    "#264653",
    "#287271",
    "#2a9d8f",
    "#8ab17d",
    "#babb74",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#d67ab1",
    "#6d597a",
]

CACHE_SUBDIR = "cache"
CACHE_FILENAME = "elemfunccap_image_attributes.json"


def _attribute_cache_path(output_dir: str) -> str:
    cache_dir = _ensure_output_dir(os.path.join(output_dir, CACHE_SUBDIR))
    return os.path.join(cache_dir, CACHE_FILENAME)


def _load_cached_attributes(cache_path: str) -> Dict[str, object] | None:
    if not os.path.exists(cache_path):
        logger.debug("Image attribute cache not found at %s.", cache_path)
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except json.JSONDecodeError:
        logger.warning("Cache at %s is invalid JSON. It will be ignored.", cache_path)
        return None

    if not isinstance(payload, dict):
        logger.warning("Cache at %s does not contain a dictionary payload.", cache_path)
        return None

    logger.info("Loaded cached image attributes from %s.", cache_path)
    return payload


def _save_cached_attributes(cache_path: str, payload: Dict[str, object]) -> None:
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2, ensure_ascii=False)
    logger.info("Saved image attribute cache to %s.", cache_path)


def _parse_target_element(target_element_str: str) -> Dict[str, str]:
    """Parse target_element JSON string to extract functionality and description."""
    if not target_element_str:
        return {"functionality": "", "description": ""}
    
    try:
        if isinstance(target_element_str, dict):
            target_elem = target_element_str
        else:
            target_elem = json.loads(target_element_str)
        
        return {
            "functionality": target_elem.get("functionality", "") or "",
            "description": target_elem.get("description", "") or "",
        }
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.debug("Failed to parse target_element: %s", target_element_str)
        return {"functionality": "", "description": ""}


def _collect_dataset_attributes(dataset: Dataset) -> Dict[str, object]:
    total_entries = len(dataset)
    if total_entries == 0:
        raise ValueError("Dataset is empty. Cannot compute statistics.")

    logger.info(
        "Scanning %s dataset entries to build the image attribute cache.",
        f"{total_entries:,}",
    )

    functionality_word_sum = 0
    description_word_sum = 0
    question_word_sum = 0
    resolutions: Counter = Counter()
    interaction_type_counter: Counter = Counter()
    density_class_counter: Counter = Counter()
    similarity_group_size_counter: Counter = Counter()
    dataset_name_counter: Counter = Counter()

    for record in tqdm(
        dataset,
        desc="Computing statistics",
        total=total_entries,
    ):
        # Parse target_element to get functionality and description
        target_element_str = record.get("target_element", "")
        target_elem_data = _parse_target_element(target_element_str)
        
        functionality_word_sum += _word_count(target_elem_data.get("functionality", ""))
        description_word_sum += _word_count(target_elem_data.get("description", ""))
        question_word_sum += _word_count(record.get("statepred_candidates_string", "") + " What will happen to the given GUI if I click on the target element highlighted with a red rectangle?")

        image_size = record.get("image_size")
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            width, height = image_size
            try:
                resolution = f"{int(width)}x{int(height)}"
            except (TypeError, ValueError):
                resolution = None
            if resolution:
                resolutions[resolution] += 1

        interaction_type = record.get("interaction_type", "unknown") or "unknown"
        density_class = record.get("density_class", "unknown") or "unknown"
        group_size = record.get("num_elements_in_group", -1)
        dataset_name = record.get("dataset_name", "unknown") or "unknown"

        try:
            group_size_int = int(group_size)
        except (TypeError, ValueError):
            group_size_int = -1

        interaction_type_counter[interaction_type] += 1
        density_class_counter[density_class] += 1
        similarity_group_size_counter[str(group_size_int)] += 1
        dataset_name_counter[dataset_name] += 1

    logger.info(
        "Finished scanning dataset: %s resolutions, %s interaction types detected.",
        len(resolutions),
        len(interaction_type_counter),
    )

    attribute_snapshot: Dict[str, object] = {
        "total_entries": total_entries,
        "functionality_word_total": functionality_word_sum,
        "description_word_total": description_word_sum,
        "question_word_total": question_word_sum,
        "resolutions": dict(resolutions),
        "interaction_type_counts": dict(interaction_type_counter),
        "density_class_counts": dict(density_class_counter),
        "similarity_group_size_counts": dict(similarity_group_size_counter),
        "dataset_name_counts": dict(dataset_name_counter),
    }

    return attribute_snapshot


def _word_count(text: str) -> int:
    """Return the number of word-like tokens in the input string."""
    if not text:
        return 0
    tokens = re.findall(r"\b[\w'-]+\b", str(text))
    return len(tokens)


def _load_dataset(dataset_path: str | None, dataset_id: str | None, split: str) -> Dataset:
    """Load the requested dataset split from disk or HuggingFace hub."""
    if dataset_path and os.path.exists(dataset_path):
        data_obj = load_from_disk(dataset_path)
    elif dataset_id:
        data_obj = load_dataset(dataset_id, split=None)
    else:
        raise ValueError(
            "Please provide either a valid --dataset-path or --dataset-id for loading."
        )

    if isinstance(data_obj, Dataset):
        dataset = data_obj
    elif isinstance(data_obj, DatasetDict):
        if split not in data_obj:
            raise ValueError(
                f"Requested split '{split}' not found. Available splits: {list(data_obj.keys())}"
            )
        dataset = data_obj[split]
    else:
        raise TypeError(f"Unsupported dataset type: {type(data_obj)}")

    return dataset


def _ensure_output_dir(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def _top_k(counter: Counter, k: int = 5) -> List[Tuple[str, int]]:
    return counter.most_common(k)


def _counter_to_proportions(counter: Counter, total: int) -> Dict[str, float]:
    if total == 0:
        return {key: 0.0 for key in counter}
    return {key: value / total for key, value in counter.items()}


def _plot_bar(counter: Counter, title: str, x_label: str, y_label: str, output: str) -> str:
    labels = list(counter.keys())
    counts = [counter[label] for label in labels]
    fig = px.bar(
        x=labels,
        y=counts,
        labels={"x": x_label, "y": y_label},
        title=title,
        text=[f"{count}" for count in counts],
        color=labels,
        color_discrete_sequence=ELEGANT_QUALITATIVE_PALETTE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=60, r=40, t=80, b=120),
        showlegend=False,
    )
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _plot_pie(counter: Counter, title: str, output: str) -> str:
    labels = list(counter.keys())
    values = [counter[label] for label in labels]
    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        hole=0.35,
        color=labels,
        color_discrete_sequence=ELEGANT_QUALITATIVE_PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=40, r=40, t=80, b=40))
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def compute_statistics(
    dataset: Dataset,
    output_dir: str,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Dict[str, object]:
    cache_path = _attribute_cache_path(output_dir)
    attribute_snapshot: Dict[str, object] | None = None

    if use_cache and not refresh_cache:
        attribute_snapshot = _load_cached_attributes(cache_path)

    if attribute_snapshot is None:
        if refresh_cache:
            logger.info("Refreshing image attribute cache as requested.")
        else:
            logger.info("No usable cache found. Recomputing dataset attributes.")
        attribute_snapshot = _collect_dataset_attributes(dataset)
        if use_cache or refresh_cache:
            _save_cached_attributes(cache_path, attribute_snapshot)
    else:
        logger.info("Using cached image attributes to generate statistics and plots.")

    total_entries = attribute_snapshot.get("total_entries", 0)
    if total_entries == 0:
        raise ValueError("Cached dataset statistics are empty. Recompute with --refresh-cache.")

    functionality_total = attribute_snapshot.get("functionality_word_total", 0)
    description_total = attribute_snapshot.get("description_word_total", 0)
    question_total = attribute_snapshot.get("question_word_total", 0)

    avg_functionality_words = functionality_total / total_entries if total_entries else 0.0
    avg_description_words = description_total / total_entries if total_entries else 0.0
    avg_question_words = question_total / total_entries if total_entries else 0.0

    resolutions_counter = Counter(attribute_snapshot.get("resolutions", {}))
    interaction_type_counter = Counter(attribute_snapshot.get("interaction_type_counts", {}))
    density_class_counter = Counter(attribute_snapshot.get("density_class_counts", {}))
    similarity_group_size_counter = Counter(attribute_snapshot.get("similarity_group_size_counts", {}))
    dataset_name_counter = Counter(attribute_snapshot.get("dataset_name_counts", {}))

    top_resolutions = _top_k(resolutions_counter, k=5)

    logger.info("Rendering Plotly charts.")

    plots_dir = _ensure_output_dir(os.path.join(output_dir, "plots"))
    interaction_plot_path = os.path.join(plots_dir, "interaction_type_distribution.html")
    density_plot_path = os.path.join(plots_dir, "density_class_proportions.html")
    similarity_plot_path = os.path.join(plots_dir, "similarity_group_sizes.html")

    _plot_bar(
        interaction_type_counter,
        title="Interaction Type Distribution",
        x_label="Interaction Type",
        y_label="Number of Questions",
        output=interaction_plot_path,
    )
    _plot_pie(
        density_class_counter,
        title="Density Class Proportions",
        output=density_plot_path,
    )
    _plot_pie(
        similarity_group_size_counter,
        title="Similarity Group Size Proportions",
        output=similarity_plot_path,
    )

    logger.info("Charts saved to %s.", plots_dir)

    summary = {
        "total_entries": total_entries,
        "average_word_counts": {
            "functionality": round(avg_functionality_words, 2),
            "description": round(avg_description_words, 2),
            "question": round(avg_question_words, 2),
        },
        "top_resolutions": [
            {
                "resolution": res,
                "count": count,
                "proportion": round(count / total_entries, 4),
            }
            for res, count in top_resolutions
        ],
        "interaction_type_distribution": OrderedDict(
            sorted(interaction_type_counter.items(), key=lambda item: item[1], reverse=True)
        ),
        "density_class_distribution": OrderedDict(
            sorted(density_class_counter.items(), key=lambda item: item[1], reverse=True)
        ),
        "similarity_group_size_distribution": OrderedDict(
            sorted(similarity_group_size_counter.items(), key=lambda item: int(item[0]))
        ),
        "dataset_name_distribution": OrderedDict(
            sorted(dataset_name_counter.items(), key=lambda item: item[1], reverse=True)
        ),
        "plot_files": {
            "interaction_type_distribution_html": interaction_plot_path,
            "density_class_proportions_html": density_plot_path,
            "similarity_group_sizes_html": similarity_plot_path,
        },
    }

    return summary


def save_summary(summary: Dict[str, object], output_dir: str) -> str:
    summary_path = os.path.join(output_dir, "func_elemcap_statistics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary JSON written to %s.", summary_path)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive statistics for the FuncElemCap dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemCap",
        help="Path to the local HuggingFace dataset directory (produced by save_to_disk).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Optional HuggingFace dataset ID (used if --dataset-path is not provided or missing).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to analyze (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "elemfunccap_stats_output"),
        help="Directory to store statistics summary and plots.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute and overwrite the cached image attributes.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading and saving the image attribute cache.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logger.info(
        "Starting FuncElemCap statistics computation for split '%s'.",
        args.split,
    )

    output_dir = _ensure_output_dir(args.output_dir)
    use_cache = not args.no_cache

    dataset = _load_dataset(args.dataset_path, args.dataset_id, args.split)
    logger.info("Dataset loaded from %s.", args.dataset_path or args.dataset_id or "dataset ID")

    summary = compute_statistics(
        dataset,
        output_dir,
        use_cache=use_cache,
        refresh_cache=args.refresh_cache,
    )
    summary_path = save_summary(summary, output_dir)

    print("\n=== FuncElemCap Dataset Statistics ===")
    print(f"Total entries: {summary['total_entries']}")
    avg = summary["average_word_counts"]
    print(
        f"Average word counts - Functionality: {avg['functionality']}, "
        f"Description: {avg['description']}, "
        f"Question: {avg['question']}"
    )
    print("\nTop-5 image resolutions:")
    for item in summary["top_resolutions"]:
        proportion_pct = round(item["proportion"] * 100, 2)
        print(f"  {item['resolution']}: {item['count']} entries ({proportion_pct}%)")
    print("\nInteraction type distribution (see Plotly chart for details):")
    for interaction_type, count in summary["interaction_type_distribution"].items():
        print(f"  {interaction_type}: {count}")
    print("\nDensity class proportions chart saved to:")
    print(f"  {summary['plot_files']['density_class_proportions_html']}")
    print("Similarity group size proportions chart saved to:")
    print(f"  {summary['plot_files']['similarity_group_sizes_html']}")
    print("Interaction type distribution chart saved to:")
    print(f"  {summary['plot_files']['interaction_type_distribution_html']}")
    print(f"\nSummary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()

