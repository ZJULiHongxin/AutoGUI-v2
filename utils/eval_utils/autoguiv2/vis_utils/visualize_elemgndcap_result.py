"""
Visualize Element Grounding and Captioning evaluation results.

This script loads evaluation outputs for grounding tasks (funcgnd, descgnd, intentgnd)
and element captioning / functional reasoning tasks, displaying annotated screenshots with
rich terminal summaries. For captioning evaluations, it also lists multiple-choice options,
highlights predictions vs. ground truth, and overlays the referenced element on the image.
The output directory is auto-generated if not provided.
"""

import os
import json
import argparse
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  rich library not available. Install with: pip install rich")

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

CAPTION_TASK_TYPES = {
    'func',
    'func-w-bbox',
    'func-w-ques',
    'desc',
    'desc-w-bbox',
    'desc-w-ques',
}


def is_caption_task(metadata: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
    """Determine if the evaluation file corresponds to caption (multi-choice) tasks."""
    task_type = str((metadata or {}).get('task_type', '')).lower()
    if task_type in CAPTION_TASK_TYPES:
        return True
    return any('correct' in (r or {}) for r in results)


def extract_cache_hash_and_root(image_path: str) -> Tuple[Optional[str], Optional[Path]]:
    """Extract the cached hash directory and cache root from an image path."""
    if not image_path:
        return None, None
    path = Path(image_path).resolve()
    parts = path.parts
    if 'images' not in parts:
        return None, None
    images_idx = parts.index('images')
    if images_idx + 1 >= len(parts):
        return None, None
    cache_hash = parts[images_idx + 1]
    cache_root = Path(*parts[:images_idx])
    return cache_hash, cache_root


def resolve_caption_cache_file(
    results: List[Dict[str, Any]],
    task_type: str,
    cache_override: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the dataset cache JSON file that stores caption metadata."""
    candidate_files: List[Path] = []
    cache_dir_override: Optional[Path] = None

    if cache_override:
        cache_path = Path(cache_override).expanduser()
        if cache_path.is_file():
            candidate_files.append(cache_path)
        elif cache_path.is_dir():
            # Defer adding until we know the hash
            cache_dir_override = cache_path
        else:
            cache_dir_override = None

    first_with_image = next((r for r in results if r and r.get('image_path')), None)
    cache_hash = None
    cache_root = None
    if first_with_image:
        cache_hash, cache_root = extract_cache_hash_and_root(first_with_image.get('image_path', ''))

    if cache_hash and cache_root:
        candidate_roots = [cache_root]
        if cache_dir_override:
            candidate_roots.insert(0, cache_dir_override)
        normalized_task_types = list({
            task_type,
            task_type.replace('_', '-'),
            task_type.replace('-w-', '-'),
            task_type.replace('-wo-', '-w-'),
            task_type.split('-')[0] if '-' in task_type else task_type,
        })
        for root in candidate_roots:
            for t in normalized_task_types:
                candidate_files.append(root / f"{cache_hash}_{t}.json")
    elif cache_dir_override:
        # If we only have override directory, try to find any json there
        candidate_files.extend(sorted(cache_dir_override.glob("*.json")))

    for candidate in candidate_files:
        if candidate.exists():
            return candidate
    return None


def load_caption_entries_map(
    results: List[Dict[str, Any]],
    task_type: str,
    cache_override: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load caption dataset entries, returning a map from entry_id to metadata."""
    cache_file = resolve_caption_cache_file(results, task_type, cache_override)
    if not cache_file:
        print("⚠️  Could not locate caption dataset cache file. Choices will be parsed from prompts.")
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        entries = cache_data.get('entries', [])
        if not isinstance(entries, list):
            print(f"⚠️  Unexpected entries format in cache file: {cache_file}")
            return {}
        return {entry.get('entry_id'): entry for entry in entries if entry.get('entry_id')}
    except Exception as exc:
        print(f"⚠️  Failed to load caption cache ({cache_file}): {exc}")
        return {}


def normalize_text(text: str, width: int = 88) -> str:
    """Nicely wrap long text for plain console output."""
    if not text:
        return ""
    return fill(text, width=width)


def load_results(result_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file
    
    Args:
        result_file: Path to result JSON file
    
    Returns:
        Dictionary containing metadata, metrics, and results
    """
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def normalize_bbox(bbox: List[float], scale: float = 1.0) -> List[float]:
    """Normalize bbox to 0-1 range
    
    Args:
        bbox: Bounding box [x_min, y_min, x_max, y_max]
        scale: Scale factor (if bbox is in 0-1000 range, use scale=1000)
    
    Returns:
        Normalized bbox [x_min, y_min, x_max, y_max] in 0-1 range
    """
    if scale != 1.0:
        return [x / scale for x in bbox]
    return bbox


def draw_bbox_on_image(image: np.ndarray, bbox: List[float], 
                       color: tuple, label: str = "", 
                       thickness: int = 3) -> np.ndarray:
    """Draw bounding box on image
    
    Args:
        image: Image as numpy array (RGB)
        bbox: Bounding box [x_min, y_min, x_max, y_max] normalized 0-1
        color: Color tuple (R, G, B)
        label: Optional label text
        thickness: Line thickness
    
    Returns:
        Annotated image
    """
    H, W = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x1 = int(bbox[0] * W)
    y1 = int(bbox[1] * H)
    x2 = int(bbox[2] * W)
    y2 = int(bbox[3] * H)
    
    # Clamp to image bounds
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Position label above the box if possible
        if y1 - label_height - 10 > 0:
            label_y = y1 - 5
        else:
            label_y = y2 + label_height + 10
        
        # Draw label background
        cv2.rectangle(image, 
                     (x1, label_y - label_height - 5),
                     (x1 + label_width + 10, label_y + 5),
                     color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return image


def annotate_caption_prediction(image: np.ndarray, summary_lines: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """Render prediction summary text block on the image."""
    if image is None or not summary_lines:
        return image
    y = 30
    for text, color in summary_lines:
        cv2.putText(
            image,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            lineType=cv2.LINE_AA
        )
        y += 32
    return image


def parse_choices_from_prompt(prompt: str) -> Dict[str, str]:
    """Parse multiple-choice options from the evaluation prompt string."""
    if not prompt:
        return {}
    lines = prompt.splitlines()
    choices: Dict[str, str] = {}
    in_options = False
    for raw_line in lines:
        line = raw_line.strip()
        if not in_options:
            if line.lower().startswith("options"):
                in_options = True
            continue
        if not line:
            continue
        if line.lower().startswith("now provide"):
            break
        if len(line) >= 3 and line[1] in [')', '.', '-'] and line[0].isalpha():
            letter = line[0].upper()
            text = line[2:].strip(" )-")
            if text:
                choices[letter] = text
    return choices


def visualize_result(result: Dict[str, Any], output_dir: Optional[str] = None,
                    show_image: bool = True, console: Optional[Console] = None) -> str:
    """Visualize a single result entry
    
    Args:
        result: Result dictionary
        output_dir: Optional directory to save visualized images
        show_image: Whether to display image (requires GUI)
        console: Rich console for terminal output
    
    Returns:
        Path to saved image (if output_dir provided)
    """
    # Extract data
    entry_id = result.get('entry_id', 'unknown')
    image_path = result.get('image_path', '')
    question = result.get('question', '')
    gt_bbox = result.get('gt_bbox', [])
    pred_bbox = result.get('pred_bbox', [])
    iou = result.get('iou', 0.0)
    center_acc = result.get('center_acc', False)
    inference_done = result.get('inference_done', False)
    error = result.get('error')
    
    # Metadata
    image_name = result.get('image_name', '')
    dataset_name = result.get('dataset_name', 'unknown')
    action_type = result.get('action_type', 'unknown')
    density_class = result.get('density_class', 'unknown')
    num_similar_elements = result.get('num_similar_elements', -1)
    processing_time = result.get('processing_time', 0.0)
    raw_response = result.get('response', '')
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return ""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Failed to load image: {image_path}")
        return ""
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    
    # Normalize bboxes (gt_bbox is already 0-1, pred_bbox might be 0-1000)
    if gt_bbox and len(gt_bbox) == 4:
        gt_bbox_norm = normalize_bbox(gt_bbox, scale=1.0)
        # Draw GT bbox in green
        image_copy = draw_bbox_on_image(
            image_copy, gt_bbox_norm, 
            color=(0, 255, 0),  # Green
            label="GT",
            thickness=3
        )
    
    if pred_bbox and len(pred_bbox) == 4:
        # Check if pred_bbox is in 0-1000 range or 0-1 range
        if max(pred_bbox) > 1.0:
            pred_bbox_norm = normalize_bbox(pred_bbox, scale=1000.0)
        else:
            pred_bbox_norm = normalize_bbox(pred_bbox, scale=1.0)
        
        # Draw predicted bbox in red
        image_copy = draw_bbox_on_image(
            image_copy, pred_bbox_norm,
            color=(255, 0, 0),  # Red
            label="Pred",
            thickness=3
        )
    
    # Display terminal output
    if console and RICH_AVAILABLE:
        # Question panel
        question_panel = Panel(
            question,
            title="[bold yellow]Question[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        )
        
        # Info table - use Text objects for proper markup
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column(style="cyan", width=20)
        info_table.add_column(style="white")
        
        info_table.add_row("Entry ID:", str(entry_id))
        info_table.add_row("Image:", str(image_name))
        info_table.add_row("Dataset:", str(dataset_name))
        info_table.add_row("Action Type:", str(action_type))
        info_table.add_row("Density Class:", str(density_class))
        info_table.add_row("Similar Elements:", str(num_similar_elements if num_similar_elements >= 0 else 'N/A'))
        info_table.add_row("Processing Time:", f"{processing_time:.2f}s")
        
        # Metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column(style="cyan", width=20)
        metrics_table.add_column(style="white")
        
        status_icon = "✅" if center_acc else "❌"
        status_label = Text(f"{status_icon} Center Accuracy:", style="bold")
        status_value = Text(str(center_acc), style="green" if center_acc else "red")
        metrics_table.add_row(status_label, status_value)
        
        metrics_table.add_row("IoU:", f"{iou:.3f}")
        metrics_table.add_row("Inference Done:", str(inference_done))
        
        if error:
            error_label = Text("Error:", style="bold red")
            error_value = Text(str(error), style="red")
            metrics_table.add_row(error_label, error_value)
        
        # Bbox coordinates
        bbox_table = Table(title="[bold]Bounding Boxes[/bold]", box=box.ROUNDED, show_header=True)
        bbox_table.add_column("Type", style="cyan", justify="center")
        bbox_table.add_column("Coordinates", style="white", justify="left")
        
        if gt_bbox and len(gt_bbox) == 4:
            bbox_table.add_row("Ground Truth", f"[{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
        
        if pred_bbox and len(pred_bbox) == 4:
            bbox_table.add_row("Predicted", f"[{pred_bbox[0]:.3f}, {pred_bbox[1]:.3f}, {pred_bbox[2]:.3f}, {pred_bbox[3]:.3f}]")
        
        # Print everything
        console.print("\n" + "═" * 80)
        console.print(question_panel)
        console.print("\n[bold]Information[/bold]")
        console.print(info_table)
        console.print("\n[bold]Metrics[/bold]")
        console.print(metrics_table)
        console.print("\n")
        console.print(bbox_table)

        if raw_response:
            raw_response_text = Text(raw_response, style="white")
            raw_response_panel = Panel(
                raw_response_text,
                title="[bold cyan]Model Raw Response[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            console.print("\n")
            console.print(raw_response_panel)

        console.print("═" * 80 + "\n")
    else:
        # Fallback to simple printing
        print("\n" + "=" * 80)
        print(f"Question: {question}")
        print(f"Entry ID: {entry_id}")
        print(f"Image: {image_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Action Type: {action_type}")
        print(f"Density Class: {density_class}")
        print(f"Similar Elements: {num_similar_elements if num_similar_elements >= 0 else 'N/A'}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Center Accuracy: {center_acc} {'✅' if center_acc else '❌'}")
        print(f"IoU: {iou:.3f}")
        print(f"Inference Done: {inference_done}")
        if error:
            print(f"Error: {error}")
        if gt_bbox and len(gt_bbox) == 4:
            print(f"GT Bbox: [{gt_bbox[0]:.3f}, {gt_bbox[1]:.3f}, {gt_bbox[2]:.3f}, {gt_bbox[3]:.3f}]")
        if pred_bbox and len(pred_bbox) == 4:
            print(f"Pred Bbox: [{pred_bbox[0]:.3f}, {pred_bbox[1]:.3f}, {pred_bbox[2]:.3f}, {pred_bbox[3]:.3f}]")
        if raw_response:
            print("Raw Response:")
            print(raw_response)
        print("=" * 80 + "\n")
    
    # Save image if output_dir provided
    output_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Convert back to BGR for saving
        image_bgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        safe_entry_id = entry_id.replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"vis_{safe_entry_id}.png")
        cv2.imwrite(output_path, image_bgr)
        cv2.imwrite('test.png', image_bgr)
    
    # Show image if requested (requires GUI)
    if show_image:
        try:
            # Try using matplotlib for display
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(image_copy)
            plt.axis('off')
            plt.title(f"{image_name}\nIoU: {iou:.3f} | Center Acc: {center_acc}")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            # Fallback: just save the image
            if not output_dir:
                print(f"⚠️  Could not display image: {e}")
    
    return output_path


def visualize_caption_result(
    result: Dict[str, Any],
    caption_entry: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    show_image: bool = True,
    console: Optional[Console] = None,
) -> str:
    """Visualize a caption (multi-choice) evaluation entry."""
    entry_id = result.get('entry_id', 'unknown')
    image_path = result.get('image_path') or (caption_entry or {}).get('image_path', '')
    question = result.get('question') or (caption_entry or {}).get('question', '')
    pred = result.get('pred')
    correct = result.get('correct')
    is_correct = result.get('is_correct', False)
    response = result.get('response', '')
    error = result.get('error')
    processing_time = result.get('processing_time', 0.0)
    action_type = (
        result.get('action_type')
        or (caption_entry or {}).get('action_type')
        or 'unknown'
    )
    dataset_name = result.get('dataset_name') or (caption_entry or {}).get('dataset_name', 'unknown')

    if not image_path or not os.path.exists(image_path):
        print(f"⚠️  Image not found: {image_path}")
        return ""

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Failed to load image: {image_path}")
        return ""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = image.copy()

    # Prepare choices
    choices = {}
    if caption_entry and isinstance(caption_entry.get('choices'), dict):
        choices = caption_entry['choices']
    else:
        choices = parse_choices_from_prompt(result.get('prompt', ''))
    choices_metadata = {}
    if caption_entry and isinstance(caption_entry.get('choices_metadata'), dict):
        choices_metadata = caption_entry['choices_metadata']

    target_bbox = []
    if caption_entry:
        target_bbox = caption_entry.get('target_bbox') or caption_entry.get('target_element', {}).get('bbox', [])
    if target_bbox and len(target_bbox) == 4:
        scale = 1000.0 if max(target_bbox) > 1 else 1.0
        target_norm = normalize_bbox(target_bbox, scale=scale)
        image_copy = draw_bbox_on_image(
            image_copy,
            target_norm,
            color=(0, 255, 0),
            label="Target",
            thickness=3
        )

    pred_type = result.get('predicted_candidate_type')
    if not pred_type and pred and choices_metadata:
        pred_type = choices_metadata.get(pred)

    summary_lines: List[Tuple[str, Tuple[int, int, int]]] = []
    pred_color = (0, 200, 0) if is_correct else (230, 70, 70)
    summary_lines.append((f"Pred: {pred or 'N/A'}", pred_color))
    summary_lines.append((f"Correct: {correct or 'N/A'}", (80, 200, 120)))
    status_text = "Status: ✅ Correct" if is_correct else "Status: ❌ Incorrect"
    summary_lines.append((status_text, pred_color))
    if pred_type:
        summary_lines.append((f"Predicted type: {pred_type}", (150, 150, 255)))
    image_copy = annotate_caption_prediction(image_copy, summary_lines)

    # Console / terminal output
    if console and RICH_AVAILABLE:
        question_panel = Panel(
            question,
            title="[bold yellow]Question[/bold yellow]",
            border_style="yellow",
            padding=(1, 2)
        )

        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column(style="cyan", width=22)
        info_table.add_column(style="white")
        info_table.add_row("Entry ID:", str(entry_id))
        info_table.add_row("Dataset:", str(dataset_name))
        info_table.add_row("Action Type:", str(action_type))
        info_table.add_row("Processing Time:", f"{processing_time:.2f}s")
        info_table.add_row("Prediction:", f"{pred or 'N/A'}")
        info_table.add_row("Correct:", f"{correct or 'N/A'}")
        status_text_rich = Text("✅ Correct" if is_correct else "❌ Incorrect", style="green" if is_correct else "bold red")
        info_table.add_row("Result:", status_text_rich)
        if pred_type:
            info_table.add_row("Predicted Type:", str(pred_type))
        if error:
            info_table.add_row("Error:", Text(str(error), style="bold red"))

        choices_table = Table(title="[bold]Choices[/bold]", box=box.ROUNDED, show_header=True, padding=(0, 1))
        choices_table.add_column("Letter", style="cyan", justify="center")
        choices_table.add_column("Description", style="white", justify="left")
        choices_table.add_column("Type", style="magenta", justify="center")
        choices_table.add_column("Flags", style="green", justify="center")

        for letter in sorted(choices.keys()):
            text_value = choices[letter]
            type_value = choices_metadata.get(letter, "unknown")
            flags = []
            flag_style = []
            if letter == correct:
                flags.append("GT")
                flag_style.append("green")
            if letter == pred:
                flags.append("Pred✅" if is_correct and letter == correct else "Pred")
                flag_style.append("bold red" if letter != correct else "green")
            if not flags:
                flags_text = "-"
            else:
                flags_text = " | ".join(flags)
            choices_table.add_row(
                letter,
                text_value,
                type_value,
                Text(flags_text, style=" ".join(flag_style) if flag_style else "white")
            )

        console.print("\n" + "═" * 80)
        console.print(question_panel)
        console.print("\n[bold]Information[/bold]")
        console.print(info_table)
        console.print("\n")
        console.print(choices_table)

        if response:
            response_panel = Panel(
                response,
                title="[bold cyan]Model Raw Response[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            console.print("\n")
            console.print(response_panel)

        console.print("═" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print(f"Question: {normalize_text(question)}")
        print(f"Entry ID: {entry_id}")
        print(f"Dataset: {dataset_name}")
        print(f"Action Type: {action_type}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Prediction: {pred or 'N/A'}")
        print(f"Correct: {correct or 'N/A'}")
        print(f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
        if pred_type:
            print(f"Predicted Type: {pred_type}")
        if error:
            print(f"Error: {error}")
        if choices:
            print("\nChoices:")
            for letter in sorted(choices.keys()):
                text_value = choices[letter]
                type_value = choices_metadata.get(letter, "unknown")
                tags = []
                if letter == correct:
                    tags.append("GT")
                if letter == pred:
                    tags.append("Pred✅" if letter == correct else "Pred❌")
                tag_suffix = f" [{' | '.join(tags)}]" if tags else ""
                print(f"  {letter}) {normalize_text(text_value)} [{type_value}]{tag_suffix}")
        if response:
            print("\nRaw Response:")
            print(response)
        print("=" * 80 + "\n")

    output_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        image_bgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        safe_entry_id = entry_id.replace('/', '_').replace('\\', '_')
        suffix = "_correct" if is_correct else "_incorrect"
        if pred:
            suffix += f"_pred-{pred}"
        output_path = os.path.join(output_dir, f"vis_{safe_entry_id}{suffix}.png")
        cv2.imwrite(output_path, image_bgr)
        cv2.imwrite('test.png', image_bgr)

    if show_image:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(image_copy)
            plt.axis('off')
            plt.title(f"{dataset_name} | Pred: {pred or 'N/A'} | Correct: {correct or 'N/A'} {'✅' if is_correct else '❌'}")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as exc:
            if not output_dir:
                print(f"⚠️  Could not display image: {exc}")

    return output_path


def main(args):
    """Main visualization function"""
    console = Console() if RICH_AVAILABLE else None
    
    # Load results
    data = load_results(args.result_file)
    
    results = data.get('results', [])
    metadata = data.get('metadata', {})
    metrics = data.get('metrics', {})
    
    # Determine task type from metadata or file path
    task_type = metadata.get('task_type') or None
    if not task_type:
        # Try to infer from file path
        result_file_path = args.result_file
        if 'funcgnd' in result_file_path:
            task_type = 'funcgnd'
        elif 'descgnd' in result_file_path:
            task_type = 'descgnd'
        elif 'intentgnd' in result_file_path:
            task_type = 'intentgnd'
        else:
            task_type = 'funcgnd'  # default

    caption_mode = is_caption_task(metadata, results)
    caption_entries_map: Dict[str, Dict[str, Any]] = {}
    if caption_mode:
        caption_entries_map = load_caption_entries_map(results, task_type or '', args.dataset_cache)
    
    # Task type display names
    task_display_names = {
        'funcgnd': 'Functionality Grounding',
        'descgnd': 'Description Grounding',
        'intentgnd': 'Intent Grounding',
        'func-w-bbox': 'Element Captioning (Func + Target Highlight)',
        'func-w-ques': 'Element Captioning (Func + Question Highlight)',
        'func': 'Element Captioning (Functionality)',
        'desc': 'Element Captioning (Description)',
    }
    default_display = 'Element Captioning' if caption_mode else 'Element Grounding'
    task_display = task_display_names.get(task_type, default_display)
    
    if console:
        console.print(f"[bold magenta]🔍 {task_display} Result Visualization[/bold magenta]")
        console.print("═" * 80 + "\n")
    else:
        print(f"🔍 {task_display} Result Visualization")
        print("=" * 80 + "\n")
    
    if not results:
        print("❌ No results found in file")
        return
    
    # Auto-generate output_dir if not provided
    if not args.output_dir:
        result_file_path = os.path.abspath(args.result_file)
        result_file_dir = os.path.dirname(result_file_path)
        result_file_basename = os.path.splitext(os.path.basename(result_file_path))[0]
        args.output_dir = os.path.join(result_file_dir, f"{result_file_basename}_vis")
    
    # Print summary
    summary_rows: List[Tuple[str, str]] = []
    if metrics:
        if 'total' in metrics:
            summary_rows.append(("Total Entries", str(metrics.get('total', 0))))
        if 'successful' in metrics:
            success_rate = metrics.get('success_rate')
            if success_rate is not None:
                summary_rows.append(("Successful", f"{metrics.get('successful', 0)} ({success_rate*100:.1f}%)"))
            else:
                summary_rows.append(("Successful", str(metrics.get('successful', 0))))
        if caption_mode:
            if 'accuracy' in metrics:
                summary_rows.append(("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.2f}%"))
        else:
            if 'avg_iou' in metrics:
                summary_rows.append(("Average IoU", f"{metrics.get('avg_iou', 0.0):.3f}"))
            if 'center_acc' in metrics:
                summary_rows.append(("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%"))
            if 'accuracy' in metrics and 'center_acc' not in metrics:
                summary_rows.append(("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.2f}%"))

    if console:
        if summary_rows:
            summary_table = Table(title="[bold]Evaluation Summary[/bold]", box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green", justify="right")
            for label, value in summary_rows:
                summary_table.add_row(label, value)
            console.print(summary_table)
            console.print("\n")
    else:
        for label, value in summary_rows:
            print(f"{label}: {value}")
        if summary_rows:
            print()

    # Filter results if needed
    if caption_mode:
        if args.filter_correct:
            results = [r for r in results if r.get('is_correct', False)]
        if args.filter_incorrect:
            results = [r for r in results if not r.get('is_correct', False)]
        if args.filter_action_type:
            allowed = {a.lower() for a in args.filter_action_type}
            results = [r for r in results if str(r.get('action_type', 'unknown')).lower() in allowed]
    else:
        if args.filter_successful:
            results = [r for r in results if r.get('center_acc', False)]
        if args.filter_failed:
            results = [r for r in results if not r.get('center_acc', False)]
        if args.filter_high_iou is not None:
            results = [r for r in results if r.get('iou', 0.0) >= args.filter_high_iou]
        if args.filter_low_iou is not None:
            results = [r for r in results if r.get('iou', 0.0) <= args.filter_low_iou]
    
    if args.entry_id:
        results = [r for r in results if args.entry_id in r.get('entry_id', '')]
    
    if args.limit:
        results = results[:args.limit]
    
    print(f"Visualizing {len(results)} result(s)...\n")
    
    # Print output directory info if saving
    if args.output_dir:
        abs_output_dir = os.path.abspath(args.output_dir)
        if console:
            console.print(f"[bold cyan]📁 Saving visualized images to:[/bold cyan] [bold]{abs_output_dir}[/bold]")
        else:
            print(f"📁 Saving visualized images to: {abs_output_dir}")
        print()
    
    # Visualize each result
    saved_images = []
    for i, result in enumerate(results):
        if console:
            console.print(f"[bold blue]Result {i+1}/{len(results)}[/bold blue]")
        else:
            print(f"\n{'='*80}")
            print(f"Result {i+1}/{len(results)}")
            print(f"{'='*80}")
        
        if caption_mode:
            caption_entry = caption_entries_map.get(result.get('entry_id'))
            output_path = visualize_caption_result(
                result,
                caption_entry=caption_entry,
                output_dir=args.output_dir,
                show_image=args.show_image and not args.output_dir,
                console=console
            )
        else:
            output_path = visualize_result(
                result,
                output_dir=args.output_dir,
                show_image=args.show_image and not args.output_dir,  # Only show if not saving
                console=console
            )
        
        if output_path:
            saved_images.append(output_path)
            # Print save confirmation for each image
            if console:
                console.print(f"[green]💾 Saved:[/green] {os.path.abspath(output_path)}")
            else:
                print(f"💾 Saved: {output_path}")
        
        # Pause between visualizations if requested or if showing images
        should_pause = args.pause or (args.show_image and not args.output_dir)
        if should_pause and i < len(results) - 1:
            try:
                print("Press Enter to continue to next result...")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
    
    # Print summary of saved images
    if saved_images:
        abs_output_dir = os.path.abspath(args.output_dir) if args.output_dir else ""
        if console:
            saved_files_list = "\n".join([f"  • {os.path.basename(img)}" for img in saved_images[:10]])
            if len(saved_images) > 10:
                saved_files_list += f"\n  ... and {len(saved_images) - 10} more files"
            
            summary_content = (
                f"[bold green]✅ Successfully saved {len(saved_images)} visualized image(s)[/bold green]\n\n"
                f"[bold cyan]Output Directory:[/bold cyan] {abs_output_dir}\n\n"
                f"[bold]Saved Files:[/bold]\n{saved_files_list}"
            )
            
            summary_panel = Panel(
                summary_content,
                title="[bold green]Save Summary[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
            console.print("\n")
            console.print(summary_panel)
        else:
            print(f"\n{'='*80}")
            print(f"✅ Successfully saved {len(saved_images)} visualized image(s)")
            print(f"📁 Output Directory: {abs_output_dir}")
            print(f"\nSaved Files:")
            for img in saved_images[:20]:
                print(f"  • {img}")
            if len(saved_images) > 20:
                print(f"  ... and {len(saved_images) - 20} more files")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize Element Grounding (funcgnd/descgnd/intentgnd) and Element Captioning evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all results (output dir auto-generated):
  python visualize_elemgnd_result.py --result-file results.json
  
  # Visualize and save to custom directory:
  python visualize_elemgnd_result.py --result-file results.json --output-dir ./vis_output
  
  # Visualize with pause after each result:
  python visualize_elemgnd_result.py --result-file results.json --pause
  
  # Visualize and save with pause:
  python visualize_elemgnd_result.py --result-file results.json --output-dir ./vis_output --pause
  
  # Visualize only successful predictions:
  python visualize_elemgnd_result.py --result-file results.json --filter-successful
  
  # Visualize only high IoU results:
  python visualize_elemgnd_result.py --result-file results.json --filter-high-iou 0.5
  
  # Visualize specific entry:
  python visualize_elemgnd_result.py --result-file results.json --entry-id "screenshot_001"

  # Review captioning results and only show incorrect cases:
  python visualize_elemgnd_result.py --result-file elemcap_results.json --filter-incorrect

  # Focus on hover/click caption entries:
  python visualize_elemgnd_result.py --result-file elemcap_results.json --filter-action-type hover click
        """
    )
    
    parser.add_argument("--result-file", type=str, default=["utils/eval_utils/autoguiv2/eval_results/funcgnd/gemini-2.5-pro-thinking/2025-11-08_12-43-44.json",
    "utils/eval_utils/autoguiv2/eval_results/AutoGUI-v2-ElemCap/gemini-2.5-pro-thinking/2025-11-10_13-20-29.json",][-1],
                       help="Path to evaluation result JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualized images (auto-generated from result file path if not provided)")
    parser.add_argument("--show-image", action="store_true", default=False,
                       help="Display images interactively (requires GUI)")
    parser.add_argument("--pause", action="store_true", default=False,
                       help="Pause after each result to allow inspection (press Enter to continue)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of results to visualize")
    parser.add_argument("--dataset-cache", type=str, default=None,
                       help="Optional path to caption dataset cache JSON (or directory containing caches) for enhanced metadata")
    
    # Filtering options
    parser.add_argument("--filter-successful", action="store_true",
                       help="Only visualize successful predictions")
    parser.add_argument("--filter-failed", action="store_true",
                       help="Only visualize failed predictions")
    parser.add_argument("--filter-high-iou", type=float, default=None,
                       help="Only visualize results with IoU >= threshold")
    parser.add_argument("--filter-low-iou", type=float, default=None,
                       help="Only visualize results with IoU <= threshold")
    parser.add_argument("--entry-id", type=str, default=None,
                       help="Filter by entry ID (substring match)")
    parser.add_argument("--filter-correct", action="store_true",
                       help="(Caption tasks) Only visualize correct predictions")
    parser.add_argument("--filter-incorrect", action="store_true",
                       help="(Caption tasks) Only visualize incorrect predictions")
    parser.add_argument("--filter-action-type", type=str, nargs='+', default=None,
                       help="(Caption tasks) Only visualize entries whose action type is in the provided list (case-insensitive)")
    
    args, _ = parser.parse_known_args()

    main(args)

