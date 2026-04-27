import argparse
import json
import sys
from typing import Any, Dict


def summarize_value(value: Any):
    if isinstance(value, dict):
        return {
            "__type__": "dict",
            "__len__": len(value),
            "__keys__sample": list(value.keys())[:20],
        }
    if isinstance(value, list):
        return {
            "__type__": "list",
            "__len__": len(value),
            "__first_item_type__": type(value[0]).__name__ if value else None,
        }
    if isinstance(value, (bytes, bytearray)):
        return {"__type__": "bytes", "__len__": len(value)}
    return {"__type__": type(value).__name__}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a single sample from a HuggingFace dataset to understand structure."
    )
    parser.add_argument(
        "--hf-dataset-id",
        type=str,
        default="HongxinLi/AutoGUIv2-FuncRegionGnd",
        help="HuggingFace dataset ID.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="test",
        help="Dataset split to load (e.g., test/validation/train).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the sample to inspect.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for HuggingFace datasets.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, writes the inspected summary to this JSON file.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        print("Failed to import 'datasets'. Please install it in your environment.", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    ds_kwargs: Dict[str, Any] = {}
    if args.hf_cache_dir:
        ds_kwargs["cache_dir"] = args.hf_cache_dir

    ds = load_dataset(args.hf_dataset_id, split=args.hf_split, **ds_kwargs)
    if args.index < 0 or args.index >= len(ds):
        print(f"Index {args.index} is out of range for split size {len(ds)}.", file=sys.stderr)
        sys.exit(2)

    item = ds[int(args.index)]

    # High-level keys and summarized types
    keys = sorted(item.keys())
    focus = {k: summarize_value(item.get(k)) for k in keys}

    # Dive into commonly relevant fields for options/density/area/similar counts
    extra: Dict[str, Any] = {}
    candidate_fields = [
        "options",
        "option_density_classes",
        "option_density_class",
        "density_classes",
        "option_num_similar_elements",
        "option_area_classes",
        "area_class",
        "density_class",
        "density",
        "density_bucket",
        "num_similar_elements",
        "num_similar",
        "similar_count",
        "target_elem_id",
        "group_index",
        "answer_index",
        "label_index",
        "option_index",
    ]
    for k in candidate_fields:
        if k in item:
            v = item[k]
            if isinstance(v, (bytes, bytearray)):
                extra[k] = {"__type__": "bytes", "__len__": len(v)}
            elif isinstance(v, list):
                # For lists, show up to first 3 elements (summarized)
                extra[k] = {
                    "__type__": "list",
                    "__len__": len(v),
                    "head": v[:3],
                }
            elif isinstance(v, dict):
                # For dicts, show keys and a few example entries
                example_items = []
                for i, (dk, dv) in enumerate(v.items()):
                    if i >= 3:
                        break
                    example_items.append([dk, dv])
                extra[k] = {
                    "__type__": "dict",
                    "__len__": len(v),
                    "__keys__sample": list(v.keys())[:20],
                    "__examples__": example_items,
                }
            else:
                extra[k] = v

    # If options is present and is a list of dicts, peek into first few
    if isinstance(item.get("options"), list):
        opt_list = item["options"]
        opt_details = []
        for opt in opt_list[:3]:
            if isinstance(opt, dict):
                opt_details.append(
                    {
                        "__keys__": list(opt.keys()),
                        "density_class": opt.get("density_class"),
                        "density": opt.get("density"),
                        "density_bucket": opt.get("density_bucket"),
                        "area_class": opt.get("area_class"),
                        "num_similar_elements": opt.get("num_similar_elements"),
                        "num_similar": opt.get("num_similar"),
                        "similar_count": opt.get("similar_count"),
                        "meta": summarize_value(opt.get("meta")),
                    }
                )
            else:
                opt_details.append({"__type__": type(opt).__name__})
        extra["options.__peek__"] = opt_details

    summary = {
        "keys": keys,
        "focus": focus,
        "extra": extra,
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()


