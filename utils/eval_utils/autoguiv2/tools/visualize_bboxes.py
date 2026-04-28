#!/usr/bin/env python3
"""
Visualize ground-truth and predicted bounding boxes from an AutoGUIv2 eval JSON.

Usage:
    python visualize_bboxes.py \
        --results-json /absolute/path/to/2025-11-12_10-30-31.json \
        --output-dir /absolute/path/for/annotated/images
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, cast

from PIL import Image, ImageDraw


def _ensure_ratio_bbox(
    bbox: Sequence[float],
    scale_hint: int = 1000,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert a bbox into 0-1 ratios, accepting inputs in 0-1, 0-scale_hint, or pixel space.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        scale_hint: value used when coordinates are in 0-scale_hint range.
        width: image width for interpreting pixel coordinates.
        height: image height for interpreting pixel coordinates.

    Returns:
        A tuple of four floats in [0, 1] or None if input is invalid.
    """
    if not isinstance(bbox, Iterable):
        return None

    coords: List[float] = []
    for val in bbox:
        try:
            coords.append(float(val))
        except (TypeError, ValueError):
            return None
    coords_tuple = tuple(coords)

    if len(coords_tuple) != 4:
        return None

    typed_bbox = cast(Tuple[float, float, float, float], coords_tuple)

    max_val = max(typed_bbox)
    if max_val <= 1.0:
        return typed_bbox  # already 0-1 ratios

    # Assume 0-scale_hint normalization (default 0-1000)
    if 0 < scale_hint and max_val <= scale_hint:
        return tuple(c / scale_hint for c in typed_bbox)

    # Fall back to interpreting as absolute pixel coordinates
    if width is None or height is None:
        raise ValueError(
            "Pixel bbox provided but image dimensions are unavailable for normalization."
        )

    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive for pixel bboxes.")

    x_min = max(0.0, min(float(width), typed_bbox[0])) / width
    y_min = max(0.0, min(float(height), typed_bbox[1])) / height
    x_max = max(0.0, min(float(width), typed_bbox[2])) / width
    y_max = max(0.0, min(float(height), typed_bbox[3])) / height
    return (x_min, y_min, x_max, y_max)


def _ratio_to_pixel_bbox(
    ratio_bbox: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """
    Convert a bbox in ratios to pixel coordinates, clamping to image bounds.
    """
    x_min, y_min, x_max, y_max = ratio_bbox
    x_min = max(0, min(width, x_min * width))
    y_min = max(0, min(height, y_min * height))
    x_max = max(0, min(width, x_max * width))
    y_max = max(0, min(height, y_max * height))
    return (
        int(round(x_min)),
        int(round(y_min)),
        int(round(x_max)),
        int(round(y_max)),
    )


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[int, int, int, int],
    color: str,
    stroke: int = 3,
) -> None:
    draw.rectangle(bbox, outline=color, width=stroke)


def _parse_bbox_arg(bbox_arg: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse a CLI bbox string formatted as x_min,y_min,x_max,y_max into floats.
    """
    if bbox_arg is None:
        return None

    cleaned = bbox_arg.replace(",", " ").split()
    if len(cleaned) != 4:
        raise ValueError(
            f"Invalid bbox '{bbox_arg}'. Expected four numbers "
            "in the format x_min,y_min,x_max,y_max."
        )

    try:
        coords = tuple(float(val) for val in cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in bbox '{bbox_arg}'.") from exc

    return coords


def visualize_entry(
    image_path: Path,
    gt_bbox: Optional[Sequence[float]],
    pred_bbox: Optional[Sequence[float]],
    output_path: Path,
    scale_hint: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    if gt_bbox is not None:
        gt_ratio = _ensure_ratio_bbox(
            gt_bbox, scale_hint=scale_hint, width=width, height=height
        )
        if gt_ratio is not None:
            gt_pixels = _ratio_to_pixel_bbox(gt_ratio, width, height)
            _draw_bbox(draw, gt_pixels, color="green")

    if pred_bbox is not None:
        pred_ratio = _ensure_ratio_bbox(
            pred_bbox, scale_hint=scale_hint, width=width, height=height
        )
        if pred_ratio is not None:
            pred_pixels = _ratio_to_pixel_bbox(pred_ratio, width, height)
            _draw_bbox(draw, pred_pixels, color="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay ground-truth and predicted bounding boxes on screenshots."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        help="Absolute path to the evaluation JSON file (e.g. .../2025-11-12_10-30-31.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store annotated images. Defaults to sibling 'visualizations/<timestamp>'.",
    )
    parser.add_argument(
        "--scale-hint",
        type=int,
        default=1000,
        help="Normalization scale used when bbox coordinates exceed 1 (default: 1000).",
    )
    parser.add_argument(
        "--center-acc",
        choices=["true", "false"],
        default=None,
        help="仅可视化 center_acc 为指定布尔值的样本；未指定时展示全部。",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="绝对路径到单张图片；若未提供 results-json，可用此选项与 bbox 参数手动绘制。",
    )
    parser.add_argument(
        "--gt-bbox",
        type=str,
        help="绿色框坐标，格式 x_min,y_min,x_max,y_max；可使用 0-1 比例、0-scale_hint 范围或像素值。",
    )
    parser.add_argument(
        "--pred-bbox",
        type=str,
        help="红色框坐标，格式 x_min,y_min,x_max,y_max；可使用 0-1 比例、0-scale_hint 范围或像素值。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="单图模式下的输出文件路径；未指定时默认写到原图同目录下的 *_annotated.png。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.results_json is None:
        if args.image is None:
            raise ValueError("Please provide either --results-json or --image.")

        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        gt_bbox = _parse_bbox_arg(args.gt_bbox)
        pred_bbox = _parse_bbox_arg(args.pred_bbox)

        if args.output_path is not None:
            output_path = args.output_path.expanduser().resolve()
        else:
            output_path = image_path.with_name(f"{image_path.stem}_annotated.png")

        visualize_entry(
            image_path=image_path,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            output_path=output_path,
            scale_hint=args.scale_hint,
        )

        print(f"Annotated image saved to: {output_path}")
        return

    results_json = args.results_json.expanduser().resolve()
    if not results_json.is_file():
        raise FileNotFoundError(f"JSON file not found: {results_json}")

    with results_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No 'results' found in {results_json}")

    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
    else:
        output_root = results_json.parent / "visualizations" / results_json.stem

    for entry in results:
        if args.center_acc is not None:
            desired_center_acc = args.center_acc == "true"
            entry_center_acc = entry.get("center_acc")
            if entry_center_acc is None or bool(entry_center_acc) != desired_center_acc:
                continue

        image_path = entry.get("image_path")
        if not image_path:
            continue

        image_path = Path(image_path)
        if not image_path.is_file():
            # Try relative to JSON file
            candidate = results_json.parent / image_path.name
            if candidate.is_file():
                image_path = candidate
            else:
                print(f"[WARN] Image not found for entry {entry.get('entry_id')}: {image_path}")
                continue

        entry_id = entry.get("entry_id") or image_path.stem
        output_path = output_root / f"{entry_id}.png"

        gt_bbox = entry.get("gt_bbox")
        pred_bbox = entry.get("pred_bbox")

        try:
            visualize_entry(
                image_path=image_path,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                output_path=output_path,
                scale_hint=args.scale_hint,
            )
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Failed to process {entry_id}: {exc}")

    print(f"Annotated images saved to: {output_root}")


if __name__ == "__main__":
    main()

