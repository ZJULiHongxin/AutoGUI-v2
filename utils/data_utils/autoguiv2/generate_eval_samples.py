import os
import json
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # Fallback if colorama is unavailable
    class _Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = ""
    class _Style:
        RESET_ALL = ""
    Fore = _Fore()
    Style = _Style()


def debug_print(message: str, level: str = "info") -> None:
    """Colorized debug print using colorama.

    Levels: info (cyan), step (blue), success (green), warn (yellow), error (red), title (magenta)
    """
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


def on_off(value: bool) -> str:
    return f"{Fore.GREEN}ON{Style.RESET_ALL}" if value else f"{Fore.YELLOW}OFF{Style.RESET_ALL}"


def load_cv_image(path: str) -> Optional[np.ndarray]:
    """Load image from local path into BGR numpy array."""
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_regions_from_raw_response(raw_response: str) -> Optional[List[Dict[str, Any]]]:
    """Best-effort extract a list of region dicts from an LLM raw response string.

    Returns a list like [root, child1, child2, ...] or None if parsing fails.
    """
    if not isinstance(raw_response, str) or len(raw_response) == 0:
        return None

    # Remove think tag content if present
    response = raw_response.split('</think>')[1] if '</think>' in raw_response else raw_response

    # Find last closing brace to avoid trailing commentary
    right_bracket_idx = response.rfind('}')
    if right_bracket_idx < 0:
        return None

    # If content after last '}' is too long, likely not a clean JSON
    if len(response) - right_bracket_idx > 10:
        return None

    try:
        raw_json_content = response[response.find('['):right_bracket_idx + 1] + ']'
        data = json.loads(raw_json_content)
        # Basic schema sanity check
        ok = True
        for item in data:
            if not all(k in item for k in ['bbox', 'dividable', 'type', 'description', 'functionality']):
                ok = False
                break
        return data if ok else None
    except Exception:
        return None


def match_child_region_from_parent(parent_raw_response: str,
                                   child_bbox_parent_norm: List[float],
                                   tolerance: float = 8.0) -> Optional[Dict[str, Any]]:
    """Match the child's region dict from the parent's raw response by bbox.

    - parent_raw_response contains regions with bbox in [0..1000] integer scale
    - child_bbox_parent_norm is [x1, y1, x2, y2] in 0..1 scale relative to the parent
    - tolerance is absolute pixel tolerance in 0..1000 space per coordinate
    """
    regions = extract_regions_from_raw_response(parent_raw_response)
    if not regions or len(regions) <= 1:
        return None

    target = [p * 1000.0 for p in child_bbox_parent_norm]
    # Round target for more stable comparisons
    target = [float(int(round(v))) for v in target]

    best_match = None
    best_err = 1e9
    for item in regions[1:]:
        bbox = item.get('bbox', None)
        if not bbox or len(bbox) != 4:
            continue
        # Compute L1 error in 0..1000 space
        err = sum(abs(float(b) - float(t)) for b, t in zip(bbox, target))
        if err < best_err:
            best_err = err
            best_match = item

    # Accept only if each coordinate is within tolerance on average
    if best_match is not None:
        bbox = best_match.get('bbox', [0, 0, 0, 0])
        diffs = [abs(float(b) - float(t)) for b, t in zip(bbox, target)]
        if all(d <= tolerance for d in diffs):
            return best_match
    return None


def draw_bbox(image: np.ndarray,
              xyxy: List[int],
              label: str = "",
              color: Tuple[int, int, int] = (0, 0, 255),
              thickness: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = map(int, xyxy)
    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), color, -1)
        cv2.putText(img, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def build_parent_index(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Return mapping child_id -> parent_id by scanning 'children' lists."""
    parent_of = {}
    for pid, pdata in nodes.items():
        for cid in pdata.get('children', []) or []:
            parent_of[str(cid)] = pid
    return parent_of


def generate_samples_for_image(sample: Dict[str, Any], debug_draw: bool, debug_dir: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Generate query-to-bbox and bbox-to-caption samples for a single annotated image.

    Returns (q2b_samples, b2c_samples, num_nodes_used)
    """
    image_path = sample.get('image_path')

    nodes: Dict[str, Dict[str, Any]] = sample.get('result', {})
    if not image_path or not nodes:
        return [], [], 0

    image_name = os.path.basename(image_path)
    # Root dims can be taken from any node (use root if present)
    root_node = nodes.get('0-0') or next(iter(nodes.values()))
    W, H = (root_node.get('root_size(wxh)') or [None, None])

    parent_of = build_parent_index(nodes)

    # Load image once if drawing
    img_cv = None
    if debug_draw and debug_dir:
        img_cv = load_cv_image(image_path)

    q2b: List[Dict[str, Any]] = []
    b2c: List[Dict[str, Any]] = []
    num_nodes_used = 0

    # Iterate nodes, skip root if it spans the entire image
    for node_id, node in nodes.items():
        
        bbox: List[int] = node.get('bbox_global') or []
        bbox_norm: List[float] = node.get('bbox_global_norm') or []
        if len(bbox) != 4 or len(bbox_norm) != 4:
            continue

        # Skip full-image root region
        if node_id == '0-0' or (bbox[0] == 0 and bbox[1] == 0 and (bbox[2] - bbox[0]) >= W and (bbox[3] - bbox[1]) >= H):
            continue

        desc = (node.get('description') or {})
        func = (node.get('functionality') or {})
        desc_wo = desc.get('wo_context') or desc.get('with_context')
        func_wo = func.get('wo_context') or func.get('with_context')

        # Resolve region type from parent raw_response when possible
        region_anno_file = node['node_image_path'].replace('.png', '_region-type.json')
        with open(region_anno_file) as f:
            region_anno = json.load(f)
        region_type = region_anno.get('type')

        parent_id = parent_of.get(node_id)
        if parent_id is not None:
            parent = nodes.get(parent_id, {})
            if (selected_idx := parent.get('selected_idx')) is not None:
                raw_response = parent.get('all_responses')[selected_idx]
            else:
                raw_response = parent.get('raw_response')

            child_rel = node.get('bbox_parent_norm') or []


        base_fields = {
            "image_path": image_path,
            "image_name": image_name,
            "image_size": [int(W), int(H)] if W is not None and H is not None else None,
            "node_id": node_id,
            "level": int(node_id.split('-')[0]) if '-' in node_id else None,
            "bbox": [int(x) for x in bbox],
            "bbox_norm": [float(x) for x in bbox_norm],
            "bbox_parent": node.get('bbox_parent'),
            "bbox_parent_norm": node.get('bbox_parent_norm'),
            "children": node.get('children') or [],
            "region_type": region_type,
            "dividable": dividable,
            "description": desc,
            "functionality": func,
        }

        # Query-to-BBox: one sample per available query type
        if isinstance(func_wo, str) and len(func_wo.strip()) > 0:
            q2b.append({
                **base_fields,
                "task": "query_to_bbox",
                "query_type": "functionality",
                "query": func_wo.strip(),
            })

        if isinstance(desc_wo, str) and len(desc_wo.strip()) > 0:
            q2b.append({
                **base_fields,
                "task": "query_to_bbox",
                "query_type": "description",
                "query": desc_wo.strip(),
            })

        # BBox-to-Caption: produce two variants if text exists
        if isinstance(func_wo, str) and len(func_wo.strip()) > 0:
            b2c.append({
                **base_fields,
                "task": "bbox_to_caption",
                "caption_type": "functionality",
                "caption": func_wo.strip(),
            })

        if isinstance(desc_wo, str) and len(desc_wo.strip()) > 0:
            b2c.append({
                **base_fields,
                "task": "bbox_to_caption",
                "caption_type": "description",
                "caption": desc_wo.strip(),
            })

        # Debug preview drawing per node (single file annotated with node_id)
        if debug_draw and debug_dir and img_cv is not None:
            preview = draw_bbox(img_cv, bbox, label=f"{node_id}")
            out_dir = os.path.join(debug_dir, os.path.splitext(image_name)[0])
            ensure_dir(out_dir)
            out_path = os.path.join(out_dir, f"preview_{node_id}.jpg")
            try:
                cv2.imwrite(out_path, preview)
            except Exception:
                pass

        num_nodes_used += 1

    return q2b, b2c, num_nodes_used


def main():
    parser = argparse.ArgumentParser(description="Convert annotated functional regions into evaluation samples")
    parser.add_argument("--anno-file", default='/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G.json', help="Path to annotation JSON produced by annotate_functional_regions.py")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save generated JSON files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--debug-draw", action="store_true", help="Draw bbox previews to images")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory to save debug preview images (defaults under output-dir)")
    parser.add_argument("--limit-images", type=int, default=0, help="Optionally limit how many images to process (0=all)")
    args, _ = parser.parse_known_args()

    # Banner
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("🧪 Generate Evaluation Samples - Run Configuration", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("", level="info")
    debug_print("📁 INPUT/OUTPUT", level="step")
    debug_print(f"   Annotation File: {Fore.CYAN}{args.anno_file}{Style.RESET_ALL}", level="info")

    output_dir = args.output_dir
    if output_dir is None or len(str(output_dir).strip()) == 0:
        # Place under the annotation file's directory by default
        output_dir = os.path.join(os.path.dirname(args.anno_file), "eval_samples")
    ensure_dir(output_dir)
    debug_print(f"   Output Dir: {Fore.CYAN}{output_dir}{Style.RESET_ALL}", level="info")

    if args.debug_dir is None or len(str(args.debug_dir).strip()) == 0:
        args.debug_dir = os.path.join(output_dir, "debug_images")
    if args.debug_draw:
        ensure_dir(args.debug_dir)
    debug_print(f"   Debug Draw: {on_off(args.debug_draw)}", level="info")
    debug_print(f"   Debug Dir: {Fore.CYAN}{args.debug_dir}{Style.RESET_ALL}", level="info")
    debug_print("", level="info")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Load annotations
    if not os.path.exists(args.anno_file):
        debug_print(f"Annotation file not found: {args.anno_file}", level="error")
        return

    with open(args.anno_file, 'r', encoding='utf-8') as f:
        anno_payload = json.load(f)

    results: Dict[str, Any] = anno_payload.get('results') or {}
    if not isinstance(results, dict) or len(results) == 0:
        debug_print("No results found in annotation file.", level="error")
        return

    # Sort for determinism
    image_entries = list(results.values())
    debug_print(f"Found {len(image_entries)} annotated image entries", level="info")

    if args.limit_images and args.limit_images > 0:
        image_entries = image_entries[: args.limit_images]
        debug_print(f"Limiting to first {len(image_entries)} images", level="warn")

    total_nodes_used = 0
    total_q2b = 0
    total_b2c = 0
    all_q2b: List[Dict[str, Any]] = []
    all_b2c: List[Dict[str, Any]] = []

    debug_print("", level="info")
    debug_print("🚀 Generating samples...", level="step")
    with tqdm(total=len(image_entries), desc="Images") as pbar:
        for entry in image_entries:
            q2b, b2c, used = generate_samples_for_image(entry, args.debug_draw, args.debug_dir)
            all_q2b.extend(q2b)
            all_b2c.extend(b2c)
            total_nodes_used += used
            total_q2b += len(q2b)
            total_b2c += len(b2c)
            pbar.update(1)

    # Write outputs
    q2b_out = os.path.join(output_dir, "query_to_bbox.json")
    b2c_out = os.path.join(output_dir, "bbox_to_caption.json")
    with open(q2b_out, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "source_annotation": os.path.abspath(args.anno_file),
                "num_images": len(image_entries),
                "num_regions_used": total_nodes_used,
                "num_samples": len(all_q2b),
                "task": "query_to_bbox",
            },
            "samples": all_q2b
        }, f, indent=2, ensure_ascii=False)

    with open(b2c_out, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "source_annotation": os.path.abspath(args.anno_file),
                "num_images": len(image_entries),
                "num_regions_used": total_nodes_used,
                "num_samples": len(all_b2c),
                "task": "bbox_to_caption",
            },
            "samples": all_b2c
        }, f, indent=2, ensure_ascii=False)

    debug_print("", level="info")
    debug_print("✅ Generation complete!", level="success")
    debug_print(f"   Images processed: {Fore.CYAN}{len(image_entries)}{Style.RESET_ALL}", level="info")
    debug_print(f"   Regions used: {Fore.CYAN}{total_nodes_used}{Style.RESET_ALL}", level="info")
    debug_print(f"   Query→BBox samples: {Fore.GREEN}{total_q2b}{Style.RESET_ALL}", level="info")
    debug_print(f"   BBox→Caption samples: {Fore.GREEN}{total_b2c}{Style.RESET_ALL}", level="info")
    debug_print(f"   Saved: {Fore.BLUE}{q2b_out}{Style.RESET_ALL}", level="info")
    debug_print(f"   Saved: {Fore.BLUE}{b2c_out}{Style.RESET_ALL}", level="info")


if __name__ == "__main__":
    main()

