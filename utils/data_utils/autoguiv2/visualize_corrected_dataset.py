#!/usr/bin/env python3
"""
Visualize samples from the corrected HuggingFace dataset.

This script loads the corrected dataset and displays samples with bounding boxes
overlaid on the images, along with metadata.

Usage:
    python visualize_corrected_dataset.py --dataset-dir /path/to/corrected_datasets
    python visualize_corrected_dataset.py --dataset-dir /path/to/corrected_datasets --output-dir /path/to/output
    python visualize_corrected_dataset.py --dataset-dir /path/to/corrected_datasets --sample-limit 10
"""

import argparse
import os
import random
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from datasets import load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: 'datasets' package is required. Install with: pip install datasets")

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:
    class _Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = ""
    class _Style:
        RESET_ALL = ""
    Fore = _Fore()
    Style = _Style()


def debug_print(message: str, level: str = "info") -> None:
    level_to_color = {
        'info': Fore.CYAN,
        'step': Fore.BLUE,
        'success': Fore.GREEN,
        'warn': Fore.YELLOW,
        'error': Fore.RED,
        'title': Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def draw_bbox_on_image(
    image: Image.Image,
    bbox: list,
    label: str = "",
    color: str = "red",
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding box on image with optional label.

    Args:
        image: PIL Image
        bbox: [x_min, y_min, x_max, y_max] in absolute coordinates or 0-1000 scale
        label: Optional label text to draw
        color: Box color
        line_width: Line width for the box

    Returns:
        Image with bbox drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    W, H = img.size

    # Convert bbox to absolute coordinates if normalized (0-1000 scale)
    if all(0 <= coord <= 1000 for coord in bbox):
        x_min = bbox[0] * W / 1000
        y_min = bbox[1] * H / 1000
        x_max = bbox[2] * W / 1000
        y_max = bbox[3] * H / 1000
    else:
        x_min, y_min, x_max, y_max = bbox

    # Draw rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

    # Draw label if provided
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Get text bbox for background
        text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)

        # Draw background rectangle for text
        draw.rectangle(text_bbox, fill=color)

        # Draw text
        draw.text((x_min, y_min - 20), label, fill="white", font=font)

    return img


def visualize_sample(
    sample: dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Image.Image:
    """Visualize a single sample with bbox overlay.

    Args:
        sample: Dataset sample dictionary
        output_path: Optional path to save the visualization
        show: Whether to display the image

    Returns:
        Visualized image
    """
    # Get image
    image = sample.get('image')
    if image is None:
        raise ValueError("Sample has no image")

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Get bbox
    bbox = sample.get('bbox', [0, 0, 0, 0])

    # Create label
    action_type = sample.get('action_type', 'unknown')
    dataset_name = sample.get('dataset_name', 'unknown')
    label = f"{action_type} | {dataset_name}"

    # Draw bbox
    img_with_bbox = draw_bbox_on_image(image, bbox, label=label, color="red", line_width=3)

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_with_bbox.save(output_path)

    # Show if requested
    if show:
        img_with_bbox.show()

    return img_with_bbox


def create_info_panel(sample: dict, width: int = 600, height: int = 400) -> Image.Image:
    """Create an info panel with sample metadata.

    Args:
        sample: Dataset sample dictionary
        width: Panel width
        height: Panel height

    Returns:
        Info panel image
    """
    panel = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_bold = font

    y_offset = 10
    line_height = 20

    # Helper to draw wrapped text
    def draw_field(label: str, value: str, max_chars: int = 70):
        nonlocal y_offset
        draw.text((10, y_offset), f"{label}:", fill="blue", font=font_bold)
        y_offset += line_height

        # Wrap text
        words = str(value).split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines[:3]:  # Limit to 3 lines
            draw.text((20, y_offset), line, fill="black", font=font)
            y_offset += line_height
        if len(lines) > 3:
            draw.text((20, y_offset), "...", fill="gray", font=font)
            y_offset += line_height
        y_offset += 5

    # Draw fields
    draw_field("Dataset", sample.get('dataset_name', 'N/A'))
    draw_field("Image", sample.get('image_name', 'N/A'))
    draw_field("Action Type", sample.get('action_type', 'N/A'))
    draw_field("Question", sample.get('question', 'N/A'))
    draw_field("Action Intent", sample.get('action_intent', 'N/A'))
    draw_field("Description", sample.get('description', 'N/A'))
    draw_field("BBox", str(sample.get('bbox', [])))
    draw_field("Density", f"{sample.get('density_class', 'N/A')} ({sample.get('num_similar_elements', 'N/A')} similar)")

    return panel


def visualize_sample_with_info(
    sample: dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Image.Image:
    """Visualize a sample with bbox and info panel side by side.

    Args:
        sample: Dataset sample dictionary
        output_path: Optional path to save the visualization
        show: Whether to display the image

    Returns:
        Combined visualization image
    """
    # Get image with bbox
    img_with_bbox = visualize_sample(sample)

    # Resize image to reasonable size if too large
    max_height = 800
    if img_with_bbox.height > max_height:
        ratio = max_height / img_with_bbox.height
        new_width = int(img_with_bbox.width * ratio)
        img_with_bbox = img_with_bbox.resize((new_width, max_height), Image.Resampling.LANCZOS)

    # Create info panel
    info_panel = create_info_panel(sample, width=500, height=img_with_bbox.height)

    # Combine side by side
    combined_width = img_with_bbox.width + info_panel.width
    combined_height = max(img_with_bbox.height, info_panel.height)
    combined = Image.new('RGB', (combined_width, combined_height), color='white')
    combined.paste(img_with_bbox, (0, 0))
    combined.paste(info_panel, (img_with_bbox.width, 0))

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.save(output_path)

    # Show if requested
    if show:
        combined.show()

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Visualize samples from the corrected HuggingFace dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/volume/pt-coder/users/gji/projects/highres_autogui/utils/data_utils/autoguiv2/corrected_datasets",
        help="Path to the saved dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (if not provided, displays interactively)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of samples to visualize",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to visualize (default: test)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before visualization",
    )
    parser.add_argument(
        "--filter-dataset",
        type=str,
        default=None,
        help="Filter by dataset name (e.g., 'screenspot_pro')",
    )
    parser.add_argument(
        "--filter-action",
        type=str,
        default=None,
        help="Filter by action type (e.g., 'clicking')",
    )
    parser.add_argument(
        "--with-info",
        action="store_true",
        default=True,
        help="Include info panel alongside the image (default: True)",
    )
    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Do not include info panel",
    )
    args = parser.parse_args()

    if not HAS_DATASETS:
        debug_print("Error: 'datasets' package is required", level="error")
        return 1

    debug_print("=" * 60, level="title")
    debug_print("Visualize Corrected Dataset", level="title")
    debug_print("=" * 60, level="title")

    # Load dataset
    debug_print(f"\n📂 Loading dataset from: {args.dataset_dir}", level="step")
    try:
        dataset = load_from_disk(args.dataset_dir)
        split_data = dataset[args.split]
        debug_print(f"✅ Loaded {len(split_data)} samples from '{args.split}' split", level="success")
    except Exception as e:
        debug_print(f"❌ Failed to load dataset: {e}", level="error")
        return 1

    # Get indices
    indices = list(range(len(split_data)))

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(indices)

    # Apply filters
    filtered_indices = []
    for idx in indices:
        sample = split_data[idx]

        if args.filter_dataset and sample.get('dataset_name') != args.filter_dataset:
            continue
        if args.filter_action and sample.get('action_type') != args.filter_action:
            continue

        filtered_indices.append(idx)

    debug_print(f"📊 {len(filtered_indices)} samples after filtering", level="info")

    # Apply sample limit
    if args.sample_limit:
        filtered_indices = filtered_indices[:args.sample_limit]
        debug_print(f"📊 Limited to {len(filtered_indices)} samples", level="info")

    # Determine visualization mode
    include_info = args.with_info and not args.no_info

    # Visualize samples
    if args.output_dir:
        debug_print(f"\n💾 Saving visualizations to: {args.output_dir}", level="step")
        os.makedirs(args.output_dir, exist_ok=True)

        for i, idx in enumerate(tqdm(filtered_indices, desc="Visualizing")):
            sample = split_data[idx]

            # Create output filename
            dataset_name = sample.get('dataset_name', 'unknown')
            image_name = sample.get('image_name', f'sample_{idx}')
            safe_image_name = image_name.replace('/', '_').replace('\\', '_')
            output_filename = f"{i:04d}_{dataset_name}_{safe_image_name}"
            if not output_filename.endswith(('.png', '.jpg', '.jpeg')):
                output_filename += '.png'
            output_path = os.path.join(args.output_dir, output_filename)

            try:
                if include_info:
                    visualize_sample_with_info(sample, output_path=output_path)
                else:
                    visualize_sample(sample, output_path=output_path)
            except Exception as e:
                debug_print(f"⚠️  Failed to visualize sample {idx}: {e}", level="warn")

        debug_print(f"✅ Saved {len(filtered_indices)} visualizations to {args.output_dir}", level="success")
    else:
        # Interactive mode
        debug_print("\n🖼️  Interactive visualization mode", level="step")
        debug_print("   Press Enter to show next sample, 'q' to quit", level="info")

        for i, idx in enumerate(filtered_indices):
            sample = split_data[idx]

            print(f"\n[{i+1}/{len(filtered_indices)}] Sample {idx}")
            print(f"  Dataset: {sample.get('dataset_name', 'N/A')}")
            print(f"  Image: {sample.get('image_name', 'N/A')}")
            print(f"  Action: {sample.get('action_type', 'N/A')}")
            print(f"  Question: {sample.get('question', 'N/A')[:100]}...")

            try:
                if include_info:
                    visualize_sample_with_info(sample, show=True)
                else:
                    visualize_sample(sample, show=True)
            except Exception as e:
                debug_print(f"⚠️  Failed to visualize: {e}", level="warn")

            user_input = input("Press Enter for next, 'q' to quit: ")
            if user_input.lower() == 'q':
                break

    debug_print("\n✅ Done!", level="success")
    return 0


if __name__ == "__main__":
    exit(main())
