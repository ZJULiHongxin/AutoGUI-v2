"""
Visualize ground truth and predicted bounding boxes for Element Grounding evaluation results

This script loads evaluation results from funcgnd, descgnd, or intentgnd tasks and displays 
images with overlaid bounding boxes, along with elegant terminal output showing question 
and metadata. The output directory is auto-generated if not provided.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
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
    
    # Task type display names
    task_display_names = {
        'funcgnd': 'Functionality Grounding',
        'descgnd': 'Description Grounding',
        'intentgnd': 'Intent Grounding'
    }
    task_display = task_display_names.get(task_type, 'Element Grounding')
    
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
    if console:
        summary_table = Table(title="[bold]Evaluation Summary[/bold]", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")
        
        summary_table.add_row("Total Entries", str(metrics.get('total', 0)))
        summary_table.add_row("Successful", f"{metrics.get('successful', 0)} ({metrics.get('success_rate', 0.0)*100:.1f}%)")
        summary_table.add_row("Average IoU", f"{metrics.get('avg_iou', 0.0):.3f}")
        summary_table.add_row("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%")
        
        console.print(summary_table)
        console.print("\n")
    else:
        print(f"Total Entries: {metrics.get('total', 0)}")
        print(f"Successful: {metrics.get('successful', 0)} ({metrics.get('success_rate', 0.0)*100:.1f}%)")
        print(f"Average IoU: {metrics.get('avg_iou', 0.0):.3f}")
        print(f"Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%\n")
    
    # Filter results if needed
    if args.filter_successful:
        results = [r for r in results if r.get('center_acc', False)]
    
    if args.filter_failed:
        results = [r for r in results if not r.get('center_acc', False)]
    
    if args.filter_high_iou:
        results = [r for r in results if r.get('iou', 0.0) >= args.filter_high_iou]
    
    if args.filter_low_iou:
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
        description="Visualize Element Grounding evaluation results (funcgnd, descgnd, intentgnd)",
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
        """
    )
    
    parser.add_argument("--result-file", type=str, default="utils/eval_utils/autoguiv2/eval_results/funcgnd/gemini-2.5-pro-thinking/2025-11-08_12-43-44.json",
                       help="Path to evaluation result JSON file")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualized images (auto-generated from result file path if not provided)")
    parser.add_argument("--show-image", action="store_true", default=False,
                       help="Display images interactively (requires GUI)")
    parser.add_argument("--pause", action="store_true", default=False,
                       help="Pause after each result to allow inspection (press Enter to continue)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of results to visualize")
    
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
    
    args, _ = parser.parse_known_args()

    main(args)

