#!/usr/bin/env python3
"""
Visualize similarity groups produced by `embed_all_elems.py`.

For every OmniParser JSON and its corresponding embedding file, this script
renders the matched screenshot with coloured bounding boxes for each group and
exports element crops for quick inspection.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def find_image_by_stem(images_root: Path, stem: str) -> Path:
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    for ext in exts:
        matches = list(images_root.rglob(stem + ext))
        if matches:
            return matches[0]
    return None


def clamp_bbox_to_image(
    bbox: Sequence[float], width: int, height: int
) -> Tuple[int, int, int, int]:
    """
    Convert a normalised bbox (x1, y1, x2, y2) into image pixel coordinates.
    """
    x1 = max(0, min(width - 1, int(bbox[0] * width)))
    y1 = max(0, min(height - 1, int(bbox[1] * height)))
    x2 = max(0, min(width, int(np.ceil(bbox[2] * width))))
    y2 = max(0, min(height, int(np.ceil(bbox[3] * height))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def setup_font(scale: float = 1.0) -> ImageFont.ImageFont:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(12 * scale))
    except IOError:
        font = ImageFont.load_default()
    return font


def draw_group_overlay(
    base_image: Image.Image,
    elements: List[dict],
    group: Iterable[int],
    group_color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(base_image)
    W, H = base_image.size
    for elem_idx in group:
        elem = elements[elem_idx]
        bbox = elem.get("bbox") or elem.get("bbox_global")
        if not bbox:
            continue
        x1, y1, x2, y2 = clamp_bbox_to_image(bbox, W, H)
        draw.rectangle([(x1, y1), (x2, y2)], outline=group_color, width=3)
        label = elem.get("content") or elem.get("type") or str(elem_idx)
        if len(label) > 40:
            label = label[:37] + "..."
        text_bg = (group_color[0], group_color[1], group_color[2], 160)
        text_pos = (x1, max(0, y1 - 14))
        text_size = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(text_size, fill=text_bg)
        draw.text(text_pos, label, fill=(0, 0, 0), font=font)


def save_group_crops(
    base_image: Image.Image,
    elements: List[dict],
    group: Iterable[int],
    group_dir: Path,
    group_color: Tuple[int, int, int],
) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    W, H = base_image.size
    for elem_idx in group:
        elem = elements[elem_idx]
        bbox = elem.get("bbox") or elem.get("bbox_global")
        if not bbox:
            continue
        x1, y1, x2, y2 = clamp_bbox_to_image(bbox, W, H)
        crop = base_image.crop((x1, y1, x2, y2))
        crop_path = group_dir / f"elem_{elem_idx:04d}.png"
        crop.save(crop_path)
        # Convert NumPy types to Python native types for JSON serialization
        bbox_serializable = [float(x) for x in bbox] if bbox else None
        meta = {
            "idx": int(elem_idx),
            "content": elem.get("content"),
            "type": elem.get("type"),
            "bbox": bbox_serializable,
            "color": group_color,
        }
        with open(group_dir / f"elem_{elem_idx:04d}.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def save_overlay_image(out_path: Path, image: Image.Image) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    image.save('test.png')


def generate_palette(n: int) -> List[Tuple[int, int, int]]:
    base_colors = [
        (244, 67, 54),
        (33, 150, 243),
        (76, 175, 80),
        (255, 193, 7),
        (156, 39, 176),
        (0, 188, 212),
        (255, 87, 34),
        (63, 81, 181),
        (205, 220, 57),
        (121, 85, 72),
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    palette = []
    for i in range(n):
        palette.append(base_colors[i % len(base_colors)])
    return palette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize similarity groups discovered by embed_all_elems.py"
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/",
        help="Root directory containing images (searched recursively)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help=(
            "Dataset base directory. If None and images-root contains 'images', "
            "uses its parent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store visualisations. Defaults to <base-dir>/omniparser_groups",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit")
    parser.add_argument(
        "--max-groups-per-image",
        type=int,
        default=None,
        help="Limit number of groups visualised per image",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip groups whose overlay.png already exists",
    )
    return parser.parse_known_args()


def main():
    args, _ = parse_args()

    images_root = Path(args.images_root).resolve()
    if args.base_dir is None:
        parts = list(images_root.parts)
        if "images" in parts:
            idx = parts.index("images")
            base_dir = Path(*parts[:idx])
        else:
            base_dir = images_root
    else:
        base_dir = Path(args.base_dir).resolve()

    omniparser_dir = base_dir / "omniparser"
    embedding_dir = base_dir / "omniparser_embeddings"
    if args.output_dir:
        out_root = Path(args.output_dir).resolve()
    else:
        out_root = base_dir / "omniparser_groups"
    out_root.mkdir(parents=True, exist_ok=True)

    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding directory missing: {embedding_dir}")

    embedding_files = sorted(embedding_dir.glob("**/*.npz"))
    if args.max_images is not None:
        embedding_files = embedding_files[: args.max_images]

    font = setup_font()

    for emb_path in tqdm(embedding_files, desc="Visualizing groups"):
        # Special handling
        if any(x in str(emb_path) for x in ['os_android', 'os_ios']): continue
        

        
        stem = str(emb_path).split("omniparser_embeddings/")[-1].rsplit(".", 1)[0]
        json_path = omniparser_dir / f"{stem}.json"
        if not json_path.exists():
            print(f"[WARN] Missing JSON for {stem}")
            continue

        npz = np.load(emb_path, allow_pickle=True)
        groups = npz.get("similar_groups", None)
        if groups is None or len(groups) == 0:
            continue
        groups = groups.tolist()

        try:
            with open(json_path, "r") as f:
                elements = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to read {json_path}: {exc}")
            continue

        image_path = find_image_by_stem(images_root, stem)
        if image_path is None or not image_path.exists():
            print(f"[WARN] Missing image for {stem}")
            continue

        base_image = Image.open(image_path).convert("RGB")

        group_limit = (
            min(len(groups), args.max_groups_per_image)
            if args.max_groups_per_image
            else len(groups)
        )
        palette = generate_palette(group_limit)

        for group_idx, group in enumerate(groups[:group_limit]):
            if not group:
                continue

            if any([((elements[i]['bbox'][2] - elements[i]['bbox'][0]) * (elements[i]['bbox'][3] - elements[i]['bbox'][1])) < 0.005 for i in group]): continue

            group_dir = out_root / stem / f"group_{group_idx:03d}"
            overlay_path = group_dir / "overlay.png"
            if args.skip_existing and overlay_path.exists():
                continue

            image_copy = base_image.copy()
            color = palette[group_idx % len(palette)]

            draw_group_overlay(image_copy, elements, group, color, font)
            save_group_crops(base_image, elements, group, group_dir, color)
            save_overlay_image(overlay_path, image_copy)

            metadata = {
                "stem": stem,
                "group_index": group_idx,
                "color": color,
                "element_indices": list(map(int, group)),
            }
            with open(group_dir / "group_meta.json", "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
