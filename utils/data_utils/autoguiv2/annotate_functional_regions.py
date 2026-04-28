import os, json, cv2
import re
import traceback
import base64
import time
import argparse
import multiprocessing
import numpy as np

import random
random.seed(999)

from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, Manager
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO
from datetime import datetime
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

# Import utilities
import sys
sys.path.append('/'.join(__file__.split('/')[:-3]))
from utils.data_utils.task_prompt_lib import *
from utils.data_utils.misc import resize_image
from utils.openai_utils.openai import OpenAIModel
from utils.data_utils.autoguiv2.data_loaders import ScreenSpotPro, OSWORLDG, MMBenchGUI, AgentNet, AndroidControl, GUIOdyssey, AMEX, MagicUI

MAX_SIZE = 1920
# Track whether we're running under a multiprocessing pool (parallel mode)
parallel_mode = False

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

def load_image(img_file: str) -> Image.Image:
    """Load image from file path, supporting both local and S3 paths"""
    image = Image.open(img_file).convert("RGB")
    return image

def image_to_base64(image_or_path):
    """Convert an image to a base64 data URL.

    Accepts:
      - str: filesystem path to the image
      - np.ndarray: OpenCV image (BGR or grayscale)
      - PIL.Image.Image: PIL image instance
    """
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.svg': 'image/svg+xml',
    }

    # Case 1: path string
    if isinstance(image_or_path, str):
        ext = Path(image_or_path).suffix.lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    # Case 2: OpenCV image (numpy array)
    if isinstance(image_or_path, np.ndarray):
        # Encode as PNG by default
        success, buf = cv2.imencode('.png', image_or_path)
        if not success:
            raise ValueError("Failed to encode numpy image to PNG")
        binary_data = buf.tobytes()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    # Case 3: PIL Image
    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        # Prefer original format if present, otherwise PNG
        fmt = image_or_path.format if image_or_path.format else 'PNG'
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else 'image/png'
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str), numpy array, or PIL Image")

def validate_bounding_boxes(bboxes: List[List[float]], image_width: int, image_height: int) -> List[List[float]]:
    """Validate and normalize bounding boxes"""
    validated_bboxes = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        # Ensure x1 < x2 and y1 < y2
        if x1 >= x2 or y1 >= y2:
            continue

        validated_bboxes.append([x1, y1, x2, y2])

    return validated_bboxes

def draw_functional_regions(image: np.ndarray, regions: List[Dict], output_path: str = None):
    """Draw bounding boxes and labels for functional regions on the image"""
    image_copy = image.copy()

    # Elegant color palette for distinguishing regions
    colors = [
        (255, 100, 100),   # Red
        (100, 255, 100),   # Green
        (100, 100, 255),   # Blue
        (255, 255, 100),   # Yellow
        (255, 100, 255),   # Magenta
        (100, 255, 255),   # Cyan
        (255, 150, 100),   # Orange
        (150, 100, 255),   # Purple
        (100, 150, 255),   # Light Blue
        (255, 100, 150),   # Pink
        (150, 255, 100),   # Light Green
        (100, 255, 150),   # Mint
        (150, 150, 255),   # Lavender
        (255, 150, 150),   # Light Coral
        (150, 255, 150),   # Light Lime
    ]

    for i, region in enumerate(regions):
        bbox = region['bbox']
        functionality = region['functionality']

        # Select color (cycle through palette if more regions than colors)
        color = colors[i % len(colors)]

        # Convert normalized coordinates to pixel coordinates
        H, W = image.shape[:2]
        x1 = round(bbox[0] / 1000 * W)
        y1 = round(bbox[1] / 1000 * H)
        x2 = round(bbox[2] / 1000 * W)
        y2 = round(bbox[3] / 1000 * H)

        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{i}: {functionality[:20]}..."
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Position label above the box if possible
        if y1 - label_height - 5 > 0:
            label_y = y1 - 5
        else:
            label_y = y2 + label_height + 5

        cv2.rectangle(image_copy, (x1, label_y - label_height - 2),
                     (x1 + label_width, label_y + 2), color, -1)
        cv2.putText(image_copy, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if output_path:
        cv2.imwrite(output_path, image_copy)

    return image_copy

def extract_functional_regions_from_response(raw_response: str) -> List[Dict]:
    """Extract functional regions from LLM response"""
    regions = []

    # Try to extract JSON format first
    response = raw_response.split('</think>')[1] if '</think>' in raw_response else raw_response

    right_bracket_idx = response.rfind('}')
    
    # The JSON object is incomplete.
    if len(response) - right_bracket_idx > 10:
        return None

    raw_json_content = response[response.find('['):right_bracket_idx+1] + ']'

    valid = True
    try:
        data = json.loads(raw_json_content)
        for item in data:
            if not all(k in item for k in ['bbox',
                                           'dividable',
                                           'type',
                                           'description',
                                        #    'description_zh',
                                           'functionality',
                                        #    'functionality_zh'
                                           ]):
                valid = False
                break
            
            if not all(0 <= p <= 1000 for p in item['bbox']):
                valid = False
                break
            
            regions.append(item)

    except json.JSONDecodeError:
        valid = False

    return regions if valid else None

def save_checkpoint(results: Dict, output_file: str, metadata: Dict = None):
    """Save results to checkpoint file"""
    if metadata is None:
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    output = {
        "metadata": metadata,
        "results": dict(results),
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def load_checkpoint(output_file: str) -> Tuple[Dict, Dict]:
    """Load results from checkpoint file"""
    if not os.path.exists(output_file):
        return {}, {}

    try:
        with open(output_file, 'r') as f:
            checkpoint = json.load(f)

        results = checkpoint.get("results", {})
        new_results = {}

        for k, v in results.items():
            if 'error' not in v:
                new_results[k] = v

        print(f"Loaded existing results from {output_file} with {len(results)} processed images, skipped {len(results) - len(new_results)} errors")
        return new_results, checkpoint.get('metadata', {})
    except Exception as e:
        print(f"Error loading results from {output_file}: {e}")
        return {}, {}

class FunctionalRegionAnnotator:
    """Annotator for identifying functional regions in GUI images"""

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o", max_retries: int = 3, max_refine: int = 3, max_level: int = -1, cache_dir: str = None, cache_namespace: str = None, checking_model: str = None):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=16000
        )
        # Separate model for completeness/boundedness checking
        self.checking_model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=(checking_model or model),
            temperature=0.1,
            max_tokens=8192
        )
        self.max_retries = max_retries
        # Maximum times to refine a low-quality region proposal
        self.max_refine = max_refine
        # Maximum depth of the region tree; -1 means unlimited
        self.max_level = max_level
        self.cache_dir = cache_dir
        self.cache_namespace = cache_namespace

    # Class-level counters to track total LLM queries made during processing of a single image
    # These are reset at the start of each image annotation in annotate_image
    annotation_query_count = 0
    completeness_query_count = 0

    # def _safe_image_id(self, image_path: str) -> str:
    #     # Build a filesystem-safe id from image path
    #     # Keep basename and a short hash of the full path to avoid collisions
    #     import hashlib
    #     base = os.path.splitext(os.path.basename(image_path))[0]
    #     h = hashlib.sha1(image_path.encode('utf-8')).hexdigest()[:8]
    #     return f"{base}-{h}"

    def _get_image_cache_dir(self, image_path: str) -> str:
        if not self.cache_dir:
            return None
        #image_id = self._safe_image_id(image_path)
        image_id = os.path.splitext(image_path.rsplit('images/', 1)[-1])[0]
        # Optional namespace allows grouping by benchmark/model/version
        parts = [self.cache_dir]
        if self.cache_namespace:
            parts.append(self.cache_namespace)
        parts.append(image_id)
        image_cache_dir = os.path.join(*parts)
        os.makedirs(os.path.join(image_cache_dir, 'nodes'), exist_ok=True)
        return image_cache_dir

    def _write_node_cache(self, image_cache_dir: str, node_id: str, node_img_path: str, node_meta: Dict, result_tree: Dict, stack_snapshot: List[List[Any]]):
        if not image_cache_dir:
            return
        try:
            # Copy/symlink image into cache if it's outside; otherwise ensure it exists in cache
            # We always write/update a JSON alongside
            node_dir = os.path.join(image_cache_dir, 'nodes')
            os.makedirs(node_dir, exist_ok=True)
            # Save/ensure image at a stable cache path
            ext = os.path.splitext(node_img_path)[1] or '.png'
            cache_node_img_path = os.path.join(node_dir, f"{node_id}{ext}")
            if os.path.abspath(node_img_path) != os.path.abspath(cache_node_img_path):
                try:
                    # Copy file (avoid linking across FS)
                    import shutil
                    shutil.copy2(node_img_path, cache_node_img_path)
                except Exception:
                    pass

            # Write node meta
            meta_out = os.path.join(node_dir, f"{node_id}.json")
            with open(meta_out, 'w', encoding='utf-8') as f:
                json.dump(node_meta, f, indent=2, ensure_ascii=False)

            # Write/refresh tree and stack snapshots for Web UI
            tree_out = os.path.join(image_cache_dir, 'tree.json')
            with open(tree_out, 'w', encoding='utf-8') as f:
                json.dump(result_tree, f, indent=2, ensure_ascii=False)

            stack_out = os.path.join(image_cache_dir, 'stack.json')
            try:
                with open(stack_out, 'w', encoding='utf-8') as f:
                    json.dump([[s[0], s[1]] for s in stack_snapshot], f, indent=2, ensure_ascii=False)
            except Exception:
                # Stack may contain non-serializable items; best-effort
                pass
        except Exception:
            # Best-effort caching; never break main flow
            pass


    def check_region_completeness(self, region_image_path: str, whole_image_path: str | np.ndarray = None, debug: bool = False) -> Dict:
        """Check the completeness and boundedness of a region"""
        # Check box completeness and boundedness
        if whole_image_path is not None:
            # downsampled_whole_img, _ = resize_image(whole_image_path, 1600)
            content = [
                    {'type': 'text', 'text': "The original GUI screenshot:\n"},
                    {'type': 'image_url', 'image_url': {'url': image_to_base64(image_or_path=whole_image_path)}},
                    {'type': 'text', 'text': "The functional region cropped from the original GUI screenshot:\n"},
                    {'type': 'image_url', 'image_url': {'url': image_to_base64(image_or_path=region_image_path)}},
                    {'type': 'text', 'text': CHECK_REGION_COMPLETENESS_PROMPT.format(context_info=' The original GUI screenshot, with the cropped region marked with a red rectangle, is also provided for reference.')}
                ]
        else:
            content = [
                    {'type': 'image_url', 'image_url': {'url': image_to_base64(region_image_path)}},
                    {'type': 'text', 'text': CHECK_REGION_COMPLETENESS_PROMPT.format(context_info='')}
                ]

        check_region_completeness_messages = [{
            'role': 'user',
            'content': content
        }]

        judge_repeat = 0
        # Defaults to ensure variables are always defined even if all attempts fail
        check_region_completeness_response = ""
        completeness_score = 0

        while judge_repeat < 5:
            judge_repeat += 1

            if debug:
                debug_print(f"🔍 Checking region completeness (Model: {self.checking_model.model}, attempt {judge_repeat}/5)", level="step")
                debug_print(f"🖼️  Image: {os.path.basename(region_image_path)}", level="info")

            
            try:
                attempt_start_time = time.time()
                # Count each LLM query attempt for completeness judging
                FunctionalRegionAnnotator.completeness_query_count += 1
                success, check_region_completeness_response, _ = self.checking_model.get_model_response_with_prepared_messages(
                    check_region_completeness_messages, temperature=0.6 if judge_repeat > 1 else 0.0, max_new_tokens=8192
                )
                query_time = time.time() - attempt_start_time
                if debug:
                    debug_print(f"⏱️ LLM completeness query time: {query_time:.3f}s", level="info")
            except:
                if debug:
                    # Derive time from the best available context
                    try:
                        query_time  # type: ignore # may not exist yet
                    except NameError:
                        # If attempt_start_time was not defined due to a very early failure, fall back to 0.0
                        query_time = 0.0
                    debug_print(f"❌ API call failed (attempt {judge_repeat}/5, {query_time:.2f}s), retrying...", level="warn")
                continue

            if not success:
                if debug:
                    print(f"❌ API call failed (attempt {judge_repeat}/5, {query_time:.2f}s), retrying...")
                continue

            match = re.search(r'\d+', check_region_completeness_response.split('Score: ')[-1].split('\n')[0])
            if match:
                completeness_score = int(match.group()); 
                if debug:
                    debug_print(f"✅ Region analysis complete (attempt {judge_repeat}, {query_time:.2f}s)", level="success")
                    debug_print(f"📊 Completeness score: {completeness_score}/3", level="info")
                break
            else:
                continue

        # Safely parse boundedness
        if 'Boundedness:' in check_region_completeness_response:
            try:
                boundedness = 'yes' in check_region_completeness_response.split('Boundedness: ')[1].split('\n')[0].lower()
            except Exception:
                boundedness = False
        else:
            boundedness = False
        return completeness_score, boundedness, check_region_completeness_response

    def annotate_image(self, image_path: str, debug: bool = False, debug_draw: bool = False, debug_output_dir: str = None) -> Dict:
        """Annotate functional regions in a single image"""

        # Reset class-level counters at the start of processing this image
        FunctionalRegionAnnotator.annotation_query_count = 0
        FunctionalRegionAnnotator.completeness_query_count = 0

        raw_img = load_image(image_path)
        raw_img_wo_ext = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1]
        W, H = raw_img.size
        
        if H > W: # A smaller size for mobile GUIs
            MAX_SIZE = 1280
        else:
            MAX_SIZE = 1920

        stack = [['0-0', (0, 0, 1, 1), image_path, [W, H], {}, False]]
        level_cnt = {0: 0}
        result = {}
        context = None

        # Prepare cache for this image
        image_cache_dir = self._get_image_cache_dir(image_path)
        if image_cache_dir:
            try:
                # Save root image once for quick reference
                root_img_out = os.path.join(image_cache_dir, 'root.png')
                if not os.path.exists(root_img_out):
                    raw_img.save(root_img_out)
            except Exception:
                pass

        if debug:
            debug_print("─" * 60, level="title")
            debug_print(f"Starting annotation for image: {os.path.basename(image_path)}", level="title")
            debug_print(f"Debug: {on_off(debug)} | Debug-draw: {on_off(debug_draw)}", level="info")
            debug_print(f"Dimensions: {W}x{H}", level="info")
            debug_print("Initial stack: Root region '0-0'", level="step")
            debug_print("─" * 60, level="title")

        while stack:
            node_id, cur_bbox_global_norm, cur_image_path, [cur_W, cur_H], cur_region_info, is_leaf_flag = stack.pop()
            cur_level = int(node_id.split('-')[0])

            if debug:
                indent = "  " * cur_level
                debug_print(f"{indent}📂 Processing region {node_id}", level="step")
                debug_print(f"{indent}├─ Level: {cur_level}", level="info")
                debug_print(f"{indent}├─ BBox: [{cur_bbox_global_norm[0]:.3f}, {cur_bbox_global_norm[1]:.3f}, {cur_bbox_global_norm[2]:.3f}, {cur_bbox_global_norm[3]:.3f}]", level="info")
                debug_print(f"{indent}├─ Image size: {cur_W}x{cur_H}", level="info")
                if cur_region_info and 'functionality' in cur_region_info:
                    debug_print(f"{indent}├─ Functionality: {cur_region_info['functionality'][:50]}...", level="info")
                debug_print(f"{indent}└─ Stack remaining: {len(stack)} regions", level="info")

            cur_bbox_global_unnorm = [
                int(W * cur_bbox_global_norm[0]),
                int(H * cur_bbox_global_norm[1]),
                int(W * cur_bbox_global_norm[2]),
                int(H * cur_bbox_global_norm[3])
                ]

            if is_leaf_flag:
                result[node_id] = {
                    "root_image_path": image_path,
                    "node_image_path": cur_image_path,
                    "root_size(wxh)": [W, H],
                    "resized_size": [cur_W, cur_H],
                    "longest_imgsize": MAX_SIZE,
                    "bbox_global": cur_bbox_global_unnorm,
                    "bbox_global_norm": cur_bbox_global_norm,
                    "bbox_parent": [0, 0, 1, 1] if node_id == '0-0' else cur_region_info.get('bbox_parent_unnorm', [0, 0, 1, 1]),
                    "bbox_parent_norm": [0, 0, 1, 1] if node_id == '0-0' else cur_region_info.get('bbox_parent_norm', [0, 0, 1, 1]),
                    "type": cur_region_info.get('type', ''),
                    "dividable": cur_region_info.get('dividable', False),
                    "description": {
                        'with_context': cur_region_info.get('description', ''),
                        'wo_context': None},
                    # "description_zh": {
                    #     'with_context': cur_region_info.get('description_zh', ''),
                    #     'wo_context': None},
                    "functionality": {
                        'with_context': cur_region_info.get('functionality', ''),
                        'wo_context': None},
                    # "functionality_zh": {
                    #     'with_context': cur_region_info.get('functionality_zh', ''),
                    #     'wo_context': None},
                    "children": [], "raw_response": None,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                # Write cache for this leaf node
                self._write_node_cache(image_cache_dir, node_id, cur_image_path, result[node_id], result, stack)
            else:
                # Load image
                cur_image = cv2.imread(cur_image_path)
                if MAX_SIZE > 0 and max(cur_image.shape) > MAX_SIZE:
                    cur_image, _ = resize_image(cur_image, MAX_SIZE)
                image_base64 = image_to_base64(image_or_path=cur_image)

                # Prepare messages for LLM
                task_context = f" whose broader context is: {context.rstrip('.')}" if context is not None else ''

                if node_id != '0-0' and cur_region_info['dividable']:
                    prompt = ANNO_PROMPT_V2_EN.replace('{is_dividable}', 'dividable ')
                else: 
                    prompt = ANNO_PROMPT_V2_EN.replace('{is_dividable}', '')

                prompt = prompt.replace('{context}', task_context)
                messages = [{
                    'role': 'user',
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]

                # --- Verification configuration ---
                # If both conditions fail (low average completeness AND low boundedness sum),
                # we will re-run region annotation up to this many times.
                verification_max_retries = self.max_refine
                # Thresholds: completeness is in [0,3]; use 2.0 as a reasonable bar
                completeness_threshold = 2.5
                # Require at least this ratio of child regions to be bounded
                boundedness_threshold_ratio = 0.8
                refine_cnt = 0
                regions_backup, new_stack_backup, response_backup = [], [], []
                max_total_score, max_score_refine_idx = -1, 0

                next_level = cur_level + 1 
                if next_level not in level_cnt:
                    level_cnt[next_level] = 0

                temp_level_cnt = level_cnt[next_level]

                while refine_cnt < verification_max_retries:
                    # Get LLM response
                    repeat = 0
                    timeout = 300
                    temp_stack = []
                    regions = None  # Initialize to prevent "referenced before assignment" error

                    while repeat < 5:
                        repeat += 1
                        timeout *= 2
                        if debug:
                            indent = "  " * cur_level
                            debug_print(f"{indent}🔍 Getting LLM response (Model: {self.model.model}, attempt {repeat}/5, timeout: {timeout}s)", level="step")

                        query_start_time = time.time()
                        try:
                            # Count each LLM query attempt for annotation
                            FunctionalRegionAnnotator.annotation_query_count += 1
                            success, response, _ = self.model.get_model_response_with_prepared_messages(
                                messages, temperature=0.1 if repeat == 1 else 0.6, timeout = timeout
                            )
                            query_time = time.time() - query_start_time
                            if debug:
                                indent = "  " * cur_level
                                debug_print(f"{indent}⏱️ LLM query time: {query_time:.3f}s", level="info")
                        except Exception as e:
                            query_time = time.time() - query_start_time
                            error_type = type(e).__name__
                            error_msg = str(e)

                            # Check for specific OpenAI errors that shouldn't be retried
                            should_retry = True
                            if "PermissionDeniedError" in error_type or "403" in error_msg:
                                should_retry = False
                                if debug:
                                    indent = "  " * cur_level
                                    debug_print(f"{indent}🚫 Permission denied (likely quota exhausted) - not retrying", level="error")
                            elif "RateLimitError" in error_type or "429" in error_msg:
                                if debug:
                                    indent = "  " * cur_level
                                    debug_print(f"{indent}⏳ Rate limit hit - will retry with backoff", level="warn")
                            elif "AuthenticationError" in error_type or "401" in error_msg:
                                should_retry = False
                                if debug:
                                    indent = "  " * cur_level
                                    debug_print(f"{indent}🔐 Authentication failed - not retrying", level="error")
                            else:
                                if debug:
                                    indent = "  " * cur_level
                                    debug_print(f"{indent}❌ API call failed ({error_type}: {error_msg[:100]}...)", level="warn")

                            if debug and should_retry:
                                indent = "  " * cur_level
                                debug_print(f"{indent}🔄 Retrying (attempt {repeat}/5, {query_time:.2f}s)...", level="warn")

                            if not should_retry:
                                # Don't retry for permanent errors
                                break

                            continue

                        if not success or not isinstance(response, str):
                            if debug:
                                indent = "  " * cur_level
                                debug_print(f"{indent}❌ API call failed (attempt {repeat}/5, {query_time:.2f}s), retrying...", level="warn")
                            continue

                        # Extract functional regions from response
                        regions = extract_functional_regions_from_response(response)

                        if regions is None:
                            if debug:
                                indent = "  " * cur_level
                                debug_print(f"{indent}⚠️  Failed to parse response (attempt {repeat}/5, {query_time:.2f}s), retrying...", level="warn")
                            continue

                        if len(regions) > 10:
                            if debug:
                                indent = "  " * cur_level
                                debug_print(f"{indent}⚠️  Too many regions (attempt {repeat}/5, {query_time:.2f}s), retrying...", level="warn")
                            continue

                        if debug:
                            indent = "  " * cur_level
                            debug_print(f"{indent}✅ API response received (attempt {repeat}, {query_time:.2f}s)", level="success")
                            debug_print(f"{indent}├─ Regions found: {len(regions)}", level="info")
                            if regions:
                                debug_print(f"{indent}├─ Root functionality: {regions[0]['functionality'][:60]}...", level="info")
                                if len(regions) > 1:
                                    debug_print(f"{indent}└─ Child regions: {len(regions) - 1}", level="info")

                        break

                    # Check if all API retries failed
                    if regions is None:
                        if debug:
                            indent = "  " * cur_level
                            debug_print(f"{indent}💥 All API retries failed for this region - skipping to next refinement attempt", level="error")
                        # Skip to next refinement iteration instead of crashing
                        refine_cnt += 1
                        continue

                    children = []

                    # Current node's normalized width/height for converting child relative boxes
                    cur_bbox_W_global_norm = cur_bbox_global_norm[2] - cur_bbox_global_norm[0]
                    cur_bbox_H_global_norm = cur_bbox_global_norm[3] - cur_bbox_global_norm[1]

                    if debug and len(regions) > 1:
                        indent = "  " * cur_level
                        debug_print(f"{indent}🌱 Initial proposal has {len(regions) - 1} child regions", level="step")

                    regions_backup.append(regions)

                    for region in regions[1:]:
                        child_id = f"{next_level}-{temp_level_cnt}"
                        children.append(child_id)
                        temp_level_cnt += 1

                        if debug:
                            child_indent = "  " * (cur_level + 1)
                            debug_print(f"{child_indent}👶 Creating child {child_id}: {region['functionality'][:35]}...", level="info")

                        child_bbox_x1_norm, child_bbox_y1_norm, child_bbox_x2_norm, child_bbox_y2_norm = list(map(lambda p: p / 1000, region['bbox']))

                        # Calc node box coords relative to its parent
                        region['bbox_parent_unnorm'] = child_bbox_parent_unnorm = [
                            int(W * cur_bbox_W_global_norm * child_bbox_x1_norm),
                            int(H * cur_bbox_H_global_norm * child_bbox_y1_norm),
                            int(W * cur_bbox_W_global_norm * child_bbox_x2_norm),
                            int(H * cur_bbox_H_global_norm * child_bbox_y2_norm)
                        ]

                        child_bbox_global_unnorm = [
                            int(child_bbox_parent_unnorm[0] + cur_bbox_global_unnorm[0]),
                            int(child_bbox_parent_unnorm[1] + cur_bbox_global_unnorm[1]),
                            int(child_bbox_parent_unnorm[2] + cur_bbox_global_unnorm[0]),
                            int(child_bbox_parent_unnorm[3] + cur_bbox_global_unnorm[1])
                        ]

                        child_bbox_global_norm = [
                            child_bbox_global_unnorm[0] / W,
                            child_bbox_global_unnorm[1] / H,
                            child_bbox_global_unnorm[2] / W,
                            child_bbox_global_unnorm[3] / H
                        ]

                        is_leaf = next_level >= self.max_level or not region['dividable'] or (child_bbox_parent_unnorm[2] - child_bbox_parent_unnorm[0] <= 30 and child_bbox_parent_unnorm[3] - child_bbox_parent_unnorm[1] < 30)
                        # Persist parent bbox norms on the child metadata for later
                        region['bbox_parent_norm'] = [child_bbox_x1_norm, child_bbox_y1_norm, child_bbox_x2_norm, child_bbox_y2_norm]

                        if child_bbox_parent_unnorm[2] - child_bbox_parent_unnorm[0] >= 10 and child_bbox_parent_unnorm[3] - child_bbox_parent_unnorm[1] >= 10:
                            # Always crop and save for UI, even if very small
                            child_img_cropped = raw_img.crop(child_bbox_global_unnorm)
                            if image_cache_dir:
                                os.makedirs(os.path.join(image_cache_dir, 'nodes'), exist_ok=True)
                                child_img_path = os.path.join(image_cache_dir, 'nodes', f"{child_id}{ext}")
                            else:
                                child_img_path = f"{raw_img_wo_ext}_node{child_id}{ext}"
                            try:
                                child_img_cropped.save(child_img_path)
                            except Exception:
                                pass

                            # Check region completeness and boundedness
                            # Draw the box back onto the original GUI screenshot
                            marked_parent_image = cv2.rectangle(cur_image.copy(), (child_bbox_parent_unnorm[0], child_bbox_parent_unnorm[1]), (child_bbox_parent_unnorm[2], child_bbox_parent_unnorm[3]), (0, 0, 255), 8)

                            completeness_score, boundedness, check_region_completeness_response = self.check_region_completeness(child_img_path, whole_image_path=marked_parent_image, debug=debug)
                            region['completeness_info'] = {
                                'completeness_score': completeness_score,
                                'boundedness': boundedness,
                                'check_region_completeness_response': check_region_completeness_response
                            }
                        else:
                            region['completeness_info'] = {
                                'completeness_score': 3,
                                'boundedness': 'yes',
                                'check_region_completeness_response': 'The region is too small to be processed as it may be an individual element which is undividable.'
                            }

                        if debug:
                            debug_print(f"{child_indent}├─ Completeness: {completeness_score}/3, Bounded: {boundedness}", level="info")
                            debug_print(f"{child_indent}└─ Saved crop: {os.path.basename(child_img_path)}", level="success")

                        child_W, child_H = child_img_cropped.size

                        temp_stack.append([child_id, child_bbox_global_norm, child_img_path, [child_W, child_H], region, is_leaf])


                    new_stack_backup.append(temp_stack); response_backup.append(response)

                    # Robust aggregation: handle 0 children and convert boundedness to numeric ratio
                    num_children_eval = max(0, len(regions) - 1)
                    if num_children_eval == 0:
                        avg_completeness_score = 0.0
                        avg_boundedness_ratio = 0.0
                    else:
                        avg_completeness_score = sum(float(x['completeness_info']['completeness_score']) for x in regions[1:]) / num_children_eval
                        bounded_sum = 0
                        for x in regions[1:]:
                            b = x['completeness_info']['boundedness']
                            bounded_sum += 1 if (b is True or (isinstance(b, str) and 'yes' in str(b).lower())) else 0
                        avg_boundedness_ratio = bounded_sum / num_children_eval

                    if debug:
                        indent = "  " * cur_level
                        debug_print(f"{indent}🔎 Refine {refine_cnt + 1}/{verification_max_retries}: children={num_children_eval}, avg completeness={avg_completeness_score:.2f}, bounded={bounded_sum}/{num_children_eval} ({avg_boundedness_ratio:.2f}), thresholds: C≥{completeness_threshold}, B≥{boundedness_threshold_ratio}", level="info")

                    # If the average completeness score and boundedness score are both greater than the thresholds, we can stop refining
                    if avg_completeness_score >= completeness_threshold and avg_boundedness_ratio >= boundedness_threshold_ratio:
                        if debug:
                            indent = "  " * cur_level
                            debug_print(f"{indent}✅ Criteria met at refine {refine_cnt + 1}; accepting this proposal", level="success")
                        max_total_score, max_score_refine_idx = avg_completeness_score, refine_cnt
                        break
                    else:
                        this_total_score = avg_completeness_score + avg_boundedness_ratio
                        if this_total_score > max_total_score:
                            max_total_score, max_score_refine_idx = this_total_score, refine_cnt
                        if debug and refine_cnt + 1 < verification_max_retries:
                            indent = "  " * cur_level
                            debug_print(f"{indent}↻ Not meeting thresholds; attempting refinement {refine_cnt + 2}/{verification_max_retries}", level="warn")

                    refine_cnt += 1

                # Commit accepted children and ensure IDs/children metadata are consistent
                accepted_stack = new_stack_backup[max_score_refine_idx] if len(new_stack_backup) > 0 else []
                children = [entry[0] for entry in accepted_stack]
                stack.extend(accepted_stack)

                # Handle case where all API calls failed - provide fallback regions
                if len(regions_backup) > 0:
                    final_regions = regions_backup[max_score_refine_idx]
                elif regions is not None:
                    final_regions = regions
                else:
                    # All API calls failed - create a minimal fallback region
                    if debug:
                        indent = "  " * cur_level
                        debug_print(f"{indent}🚨 All API calls failed - using minimal fallback region", level="error")
                    final_regions = [{
                        'functionality': 'Unknown functionality (API failed)',
                        'description': 'Unable to analyze this region due to API failures',
                        'bbox': [0, 0, 1000, 1000]
                    }]
                # Advance the level counter to the next available suffix based on accepted children
                if children:
                    try:
                        last_suffix = int(children[-1].split('-')[-1]) + 1
                        level_cnt[next_level] = max(level_cnt[next_level], last_suffix)
                    except Exception:
                        # Fallback to previous behavior if parsing fails
                        level_cnt[next_level] = temp_level_cnt
                else:
                    # No children accepted; keep counter unchanged
                    pass

                if debug:
                    indent = "  " * cur_level
                    debug_print(f"{indent}📌 Selected refine attempt #{max_score_refine_idx + 1} with {len(children)} children (annotation queries: {FunctionalRegionAnnotator.annotation_query_count}, completeness queries: {FunctionalRegionAnnotator.completeness_query_count})", level="step")

                if context is None:
                    context = final_regions[0]['description']

                result[node_id] = {
                    "root_image_path": image_path,
                    "node_image_path": cur_image_path,
                    "root_size(wxh)": [W, H],
                    "resized_size": [cur_W, cur_H],
                    "longest_imgsize": MAX_SIZE,
                    "bbox_global": cur_bbox_global_unnorm,
                    "bbox_global_norm": cur_bbox_global_norm,
                    "bbox_parent": [0, 0, 1, 1] if node_id == '0-0' else cur_region_info.get('bbox_parent_unnorm', [0, 0, 1, 1]),
                    "bbox_parent_norm": [0, 0, 1, 1] if node_id == '0-0' else cur_region_info.get('bbox_parent_norm', [0, 0, 1, 1]),
                    "type": 'Entire GUI' if node_id == '0-0' else cur_region_info.get('type', 'Entire GUI'),
                    "dividable": True if node_id == '0-0' else cur_region_info.get('dividable', True),
                    "description": {
                        'with_context': (final_regions[0] if node_id == '0-0' else cur_region_info)['description'],
                        'wo_context': final_regions[0]['description']},
                    # "description_zh": {
                    #     'with_context': (final_regions[0] if node_id == '0-0' else cur_region_info)['description_zh'],
                    #     'wo_context': final_regions[0]['description_zh']},
                    "functionality": {
                        'with_context': (final_regions[0] if node_id == '0-0' else cur_region_info)['functionality'],
                        'wo_context': final_regions[0]['functionality']},
                    # "functionality_zh": {
                    #     'with_context': (final_regions[0] if node_id == '0-0' else cur_region_info)['functionality_zh'],
                    #     'wo_context': final_regions[0]['functionality_zh']},
                    "children": children,
                    "all_responses": response_backup,
                    "selected_idx": max_score_refine_idx,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Write cache for this node and update tree/stack snapshots
                self._write_node_cache(image_cache_dir, node_id, cur_image_path, result[node_id], result, stack)

                # Debug: draw regions on image
                if (debug or debug_draw) and debug_output_dir and final_regions:
                    debug_image = cv2.imread(cur_image_path)
                    debug_output_path = os.path.join(debug_output_dir, f"debug_{os.path.basename(cur_image_path)}")
                    draw_functional_regions(debug_image, final_regions, debug_output_path)

            if debug:
                indent = "  " * cur_level
                debug_print(f"{indent}✅ Completed region {node_id}" + (" (Leaf Node!)" if is_leaf_flag else ""), level="success")
                if children:
                    debug_print(f"{indent}├─ Added {len(children)} children to stack: {children}", level="info")
                    debug_print(f"{indent}└─ Stack now has {len(stack)} regions: {[x[0] for x in stack]}", level="info")
                else:
                    debug_print(f"{indent}└─ No children added (leaf region)", level="info")
                print()

        if debug:
            debug_print("=" * 60, level="title")
            debug_print("🎉 Annotation complete!", level="success")
            debug_print(f"📊 Total regions processed: {len(result)}", level="info")
            debug_print(f"🌳 Hierarchical levels: {max(level_cnt.keys()) + 1 if level_cnt else 1} (including the root node)", level="info")
            debug_print(f"📁 Results saved for image: {os.path.basename(image_path)}", level="info")
            debug_print("=" * 60, level="title")
            print()

        return result

def init_worker(base_url: str, api_key: str, model: str, max_retries: int, max_refine: int, max_level: int, debug: bool, debug_draw: bool, debug_output_dir: str, cache_dir: str, cache_namespace: str, checking_model: str = None):
    """Initialize worker with annotator instance"""
    global annotator_instance
    annotator_instance = FunctionalRegionAnnotator(base_url, api_key, model, max_retries, max_refine, max_level, cache_dir=cache_dir, cache_namespace=cache_namespace, checking_model=checking_model)
    global debug_flag, debug_draw_flag, debug_dir
    debug_flag = debug
    debug_draw_flag = debug_draw
    debug_dir = debug_output_dir
    # Mark that we are in parallel workers
    global parallel_mode
    parallel_mode = True

def process_image_with_timeout(args, timeout_seconds):
    """Process a single image with timeout protection"""
    image_path, worker_id = args

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing timeout after {timeout_seconds} seconds")

    # Set up signal handler for timeout
    import signal
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        result = process_image((image_path, worker_id))
        signal.alarm(0)  # Cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        return result
    except TimeoutError:
        print(f"[Worker {worker_id}] Timeout processing {os.path.basename(image_path)} after {timeout_seconds}s")
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise

def process_image(args) -> Dict:
    """Process a single image with error handling"""
    image_path, worker_id = args
    start_time = time.time()

    try:
        global annotator_instance, debug_flag, debug_draw_flag, debug_dir, parallel_mode
        # Set current worker id for debug filtering if needed
        global current_worker_id
        current_worker_id = worker_id

        print(f"[Worker {worker_id}] Processing image: {os.path.basename(image_path)}")
        # Only worker 1 should emit debug output when running in parallel
        effective_debug = bool(debug_flag) and (not parallel_mode or worker_id == 1)
        result = annotator_instance.annotate_image(image_path, effective_debug, debug_draw_flag, debug_dir)

        # Calculate processing time
        processing_time = time.time() - start_time
        sample = {
            "image_path": image_path,
            "result": result,
            "processing_time": processing_time,
            "annotation_queries": FunctionalRegionAnnotator.annotation_query_count,
            "completeness_queries": FunctionalRegionAnnotator.completeness_query_count
        }

        print(f"[Worker {worker_id}] Completed {os.path.basename(image_path)} with {len(result)} regions in {processing_time:.2f}s")
        return sample

    except Exception as e:
        print(f"[Worker {worker_id}] Error processing {os.path.basename(image_path)}: {e}")
        traceback.print_exc()
        return {
            "image_path": image_path,
            "error": str(e),
            "processing_time": time.time() - start_time
        }


def process_data_path(model_name: str, version: str, data_path: str, output_file: str = None, output_dir: str = '', debug: bool = False, random_sample: int = None):
    """Process a single data path"""
    if 'screenspot' in data_path.lower():
        bmk_name = 'screenspot_pro'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = ScreenSpotPro(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    elif 'osworld' in data_path.lower():
        bmk_name = 'osworld_g'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = OSWORLDG(data_path, cache_dir=img_cache_dir, debug=debug)
    elif 'mmbench' in data_path.lower():
        bmk_name = 'mmbenchgui'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = MMBenchGUI(data_path, cache_dir=img_cache_dir, debug=debug)
    elif 'agentnet' in data_path.lower():
        bmk_name = 'agentnet'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = AgentNet(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    elif 'androidcontrol' in data_path.lower():
        bmk_name = 'androidcontrol'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = AndroidControl(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    elif 'guiodyssey' in data_path.lower():
        bmk_name = 'guiodyssey'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = GUIOdyssey(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    elif 'amex' in data_path.lower():
        bmk_name = 'amex'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = AMEX(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    elif 'magic' in data_path.lower():
        bmk_name = 'magicui'
        img_cache_dir = os.path.join(output_dir, f'{bmk_name}/images')
        data_loader = MagicUI(data_path, cache_dir=img_cache_dir, debug=debug, random_sample=random_sample)
    else:
        print(f"Unknown dataset: {data_path}")
        return None, None

    if os.path.exists(data_path):
        print(f"Loading images from a local dir: {data_path}")
        if output_file is None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, bmk_name, model_name, version, f"functional_regions_{args.model}.json")        
    else:
        print(f"Loading images from a HF dataset: {data_path}")
        if output_file is None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, bmk_name, model_name, version, data_path.replace('/', '-') + '.json')

    return data_loader, output_file, bmk_name


def main(args):
    """Main processing function"""
    # Colorized run configuration banner
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("🎯 Functional Region Annotation - Run Configuration", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Data & Output Configuration
    debug_print("", level="info")  # Empty line for spacing
    debug_print("📁 DATA & OUTPUT CONFIGURATION", level="step")
    debug_print(f"   Data Path: {Fore.CYAN}{args.data_path}{Style.RESET_ALL}", level="info")
    debug_print(f"   Output Dir: {Fore.CYAN}{args.output_dir}{Style.RESET_ALL}", level="info")
    if args.output_file:
        debug_print(f"   Output File: {Fore.CYAN}{args.output_file}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Output File: {Fore.YELLOW}Auto-generated{Style.RESET_ALL}", level="info")

    # Model Configuration
    debug_print("", level="info")  # Empty line for spacing
    debug_print("🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Primary Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    if args.checking_model:
        debug_print(f"   Checking Model: {Fore.GREEN}{args.checking_model}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Checking Model: {Fore.YELLOW}Same as primary{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    debug_print(f"   Version: {Fore.MAGENTA}{args.version}{Style.RESET_ALL}", level="info")

    # Processing Configuration
    debug_print("", level="info")  # Empty line for spacing
    debug_print("⚙️  PROCESSING CONFIGURATION", level="step")
    mode_text = "SEQUENTIAL" if args.sequential else f"PARALLEL ({args.workers} workers)"
    mode_color = Fore.RED if args.sequential else Fore.GREEN
    debug_print(f"   Execution Mode: {mode_color}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Refine: {Fore.YELLOW}{args.max_refine}{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Level: {Fore.YELLOW}{args.max_level}{Style.RESET_ALL}", level="info")
    debug_print(f"   Task Timeout: {Fore.YELLOW}{args.task_timeout}s{Style.RESET_ALL}", level="info")

    # Debug & Visual Configuration
    debug_print("", level="info")  # Empty line for spacing
    debug_print("🔍 DEBUG & VISUAL CONFIGURATION", level="step")
    debug_print(f"   Debug Mode: {on_off(args.debug)}", level="info")
    debug_print(f"   Debug Draw: {on_off(args.debug_draw)}", level="info")
    debug_print(f"   Force Reprocess: {on_off(args.force)}", level="info")

    # Cache Configuration
    debug_print("", level="info")  # Empty line for spacing
    debug_print("💾 CACHE CONFIGURATION", level="step")
    cache_dir = args.cache_dir if args.cache_dir and args.cache_dir.strip() else f"{args.output_dir}/cache"
    debug_print(f"   Cache Directory: {Fore.BLUE}{cache_dir}{Style.RESET_ALL}", level="info")

    debug_print("", level="info")  # Empty line for spacing
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Set up output file path
    data_loader, output_file, bmk_name = process_data_path(args.model, args.version, args.data_path, args.output_file, args.output_dir, args.debug, args.random_sample)

    debug_print(f"📂 Final Output File: {Fore.CYAN}{output_file}{Style.RESET_ALL}", level="success")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Save experiment configuration alongside the annotation output
    try:
        exp_cfg_dir = os.path.dirname(output_file)
        os.makedirs(exp_cfg_dir, exist_ok=True)
        exp_cfg_path = os.path.join(exp_cfg_dir, 'exp_config.json')
        # Convert argparse Namespace to a plain dict
        exp_cfg = {k: v for k, v in vars(args).items()}
        with open(exp_cfg_path, 'w', encoding='utf-8') as f:
            json.dump(exp_cfg, f, indent=2, ensure_ascii=False)
        print(f"Saved experiment config to {exp_cfg_path}")
    except Exception as e:
        print(f"Failed to save experiment config: {e}")

    # Load existing results if available
    existing_results, metadata = load_checkpoint(output_file)

    # Filter out already processed images
    processed_images = set()
    for img_filename, result in existing_results.items():
        processed_images.add(img_filename)

    if not args.force:
        image_paths = [img for img in data_loader.image_paths if img not in processed_images and img.replace(args.output_dir + '/', '') not in processed_images]
        num_to_proc = len(data_loader.image_paths)
        print(f"{num_to_proc} to process. Filtered {num_to_proc} - {len(image_paths)} = {num_to_proc - len(image_paths)} already processed images")

    if not image_paths:
        print("No new images to process")
        return

    if args.sequential:
        print(f"Processing {len(image_paths)} images sequentially (debug mode)")
    else:
        print(f"Processing {len(image_paths)} images with {args.workers} workers")

    # Setup multiprocessing or sequential processing
    if args.sequential:
        results = {}
    else:
        manager = Manager()
        results = manager.dict()

    # Copy existing results
    for k, v in existing_results.items():
        results[k] = v

    if args.sequential:
        processed_count = len(existing_results)
        total_processing_time = 0.0
    else:
        processed_count = manager.Value('i', len(existing_results))
        total_processing_time = manager.Value('d', 0.0)

    start_time = time.time()

    # Setup debug directory
    debug_output_dir = None
    if args.debug:
        debug_output_dir = os.path.join(os.path.dirname(output_file), "debug_images")
        os.makedirs(debug_output_dir, exist_ok=True)

    # Setup cache directory (auto if not provided)
    if args.cache_dir is None or len(args.cache_dir.strip()) == 0:
        args.cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_namespace = os.path.join(bmk_name, args.model, args.version)

    if args.sequential:
        # Sequential processing for easier debugging
        print("🔧 Running in sequential mode - easier for debugging!")

        # Initialize single annotator instance
        global annotator_instance, debug_flag, debug_draw_flag, debug_dir
        annotator_instance = FunctionalRegionAnnotator(args.base_url, args.api_key, args.model, args.max_retries, args.max_refine, args.max_level, cache_dir=args.cache_dir, cache_namespace=cache_namespace, checking_model=args.checking_model)
        debug_flag = args.debug
        debug_draw_flag = args.debug_draw
        debug_dir = debug_output_dir

        with tqdm(total=len(image_paths), desc=f"Processing images for {args.output_dir}") as pbar:
            for i, img_path in enumerate(image_paths):
                worker_id = i % (args.workers if args.workers > 0 else 1)

                try:
                    print(f"[Worker {worker_id}] Processing image: {os.path.basename(img_path)}")
                    sample = process_image((img_path, worker_id))

                    # Store result
                    image_key = sample['image_path'].replace(args.output_dir + '/', '')
                    results[image_key] = sample

                    processed_count += 1
                    total_processing_time += sample.get('processing_time', 0)

                    # Update progress bar
                    pbar.update(1)

                    # Save checkpoint periodically (every 10 images)
                    if processed_count % 1 == 0:
                        checkpoint_metadata = {
                            "model": args.model,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "data_path": args.data_path,
                            "num_images_processed": processed_count,
                            "total_images": len(image_paths) + len(existing_results),
                            "total_processing_time": total_processing_time,
                            "avg_processing_time": total_processing_time / processed_count if processed_count > 0 else 0,
                            "processing_time_so_far": time.time() - start_time,
                            "total_annotation_queries": sum(int(v.get('annotation_queries', 0)) for v in results.values() if isinstance(v, dict)),
                            "total_completeness_queries": sum(int(v.get('completeness_queries', 0)) for v in results.values() if isinstance(v, dict))
                        }
                        save_checkpoint(results, output_file, checkpoint_metadata)
                        print(f"\nSaved checkpoint with {processed_count} processed images to {output_file}")

                except Exception as e:
                    print(f"Error processing {os.path.basename(img_path)}: {e}")
                    processed_count += 1

    else:
        # Initialize process pool with timeout protection
        pool_timeout = args.task_timeout  # Use user-specified timeout
        max_retries_per_task = 2  # Retry failed tasks

        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model, args.max_retries, args.max_refine, args.max_level, args.debug, args.debug_draw, debug_output_dir, args.cache_dir, cache_namespace, args.checking_model)
        ) as pool:
            try:
                # Prepare arguments with retry information
                task_queue = [(img_path, i % args.workers, 0) for i, img_path in enumerate(image_paths)]  # (path, worker_id, retry_count)

                # Process images with progress bar
                with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                    while task_queue:
                        # Get next task
                        current_tasks = task_queue[:args.workers]
                        task_queue = task_queue[args.workers:]

                        # Submit tasks with timeout
                        async_results = []
                        for task in current_tasks:
                            img_path, worker_id, retry_count = task
                            try:
                                result = pool.apply_async(process_image_with_timeout,
                                                        args=((img_path, worker_id), pool_timeout))
                                async_results.append((result, task))
                            except Exception as e:
                                print(f"Failed to submit task for {os.path.basename(img_path)}: {e}")
                                # Re-queue task if retries available
                                if retry_count < max_retries_per_task:
                                    task_queue.append((img_path, worker_id, retry_count + 1))

                        # Collect results with timeout
                        for async_result, (img_path, worker_id, retry_count) in async_results:
                            try:
                                sample = async_result.get(timeout=pool_timeout + 60)  # Extra minute for overhead

                                # Store result
                                image_key = sample['image_path'].replace(args.output_dir, '')
                                results[image_key] = sample

                                processed_count.value += 1
                                total_processing_time.value += sample.get('processing_time', 0)

                                # Update progress bar
                                pbar.update(1)

                                # Save checkpoint periodically (every 10 images)
                                if processed_count.value % 10 == 0:
                                    checkpoint_metadata = {
                                        "model": args.model,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "data_path": args.data_path,
                                        "num_images_processed": processed_count.value,
                                        "total_images": len(image_paths) + len(existing_results),
                                        "total_processing_time": total_processing_time.value,
                                        "avg_processing_time": total_processing_time.value / processed_count.value if processed_count.value > 0 else 0,
                                        "processing_time_so_far": time.time() - start_time,
                                        "total_annotation_queries": sum(int(v.get('annotation_queries', 0)) for v in results.values() if isinstance(v, dict)),
                                        "total_completeness_queries": sum(int(v.get('completeness_queries', 0)) for v in results.values() if isinstance(v, dict))
                                    }
                                    save_checkpoint(results, output_file, checkpoint_metadata)
                                    print(f"\nSaved checkpoint with {processed_count.value} processed images")

                            except Exception as e:
                                traceback.print_exc()
                                print(f"Task failed for {os.path.basename(img_path)}: {e}")
                                # Re-queue task if retries available
                                if retry_count < max_retries_per_task:
                                    task_queue.append((img_path, worker_id, retry_count + 1))
                                    print(f"Re-queuing {os.path.basename(img_path)} (attempt {retry_count + 1}/{max_retries_per_task + 1})")
                                else:
                                    print(f"Giving up on {os.path.basename(img_path)} after {max_retries_per_task + 1} attempts")

            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Terminating workers...")
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                print(f"Error during parallel processing: {e}")
                pool.terminate()
                pool.join()

    # Save final results
    final_count = processed_count if args.sequential else processed_count.value
    final_time = total_processing_time if args.sequential else total_processing_time.value

    final_metadata = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": args.data_path,
        "num_images_processed": final_count,
        "total_images": len(image_paths) + len(existing_results),
        "total_processing_time": final_time,
        "avg_processing_time": final_time / final_count if final_count > 0 else 0,
        "total_processing_time_wall": time.time() - start_time,
        "total_annotation_queries": sum(int(v.get('annotation_queries', 0)) for v in results.values() if isinstance(v, dict)),
        "total_completeness_queries": sum(int(v.get('completeness_queries', 0)) for v in results.values() if isinstance(v, dict))
    }
    save_checkpoint(results, output_file, final_metadata)

    print("\nProcessing complete.")
    print(f"Results saved to {output_file}")
    if args.debug:
        print(f"Debug images saved to {debug_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate functional regions in GUI images")
    parser.add_argument("--data-path", default=[
        "sujr/autogui-agentnet",
        "HongxinLi/ScreenSpot-Pro",
        "MMInstruction/OSWorld-G",
        "/mnt/vdb1/hongxin_li/MMBench-GUI/",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/androidcontrol/raw_image_path_filtered.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/guiodyssey/raw_image_path_subset_filtered.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/raw_image_path_filtered.json",
        "GUIAgent/Magic-RICH",
        ][2],
                       help="Text file containing list of data paths (one per line)")
    parser.add_argument("--random-sample", type=int, default=8000,
                       help="Random sample size")
    parser.add_argument("--output-file", default=None,
                       help="Output JSON file path")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "/mnt/vdb1/hongxin_li/AutoGUIv2"),
                       help="Output directory")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
                       help="OpenAI API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
                       help="OpenAI API base URL")
    parser.add_argument("--model", type=str, default=["gemini-2.5-pro-thinking"][-1],
                       help="Model to use for annotation")
    parser.add_argument("--checking-model", type=str, default=[None, "gemini-2.5-flash-lite-preview-06-17"][0],
                       help="Model to use for completeness/boundedness checking (defaults to --model if not set)")
    # v2: (1) Added division limit (less than 10) + (2) Add refinement mechanism (up to 3 times)
    parser.add_argument("--version", type=str, default=["v1", "v2", "v3"][2],
                       help="Version of the annotation")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls")
    parser.add_argument("--max-refine", type=int, default=3,
                       help="Maximum number of times to refine the region annotation")
    parser.add_argument("--max-level", type=int, default=3,
                       help="Maximum tree depth; do not add children when current level reaches this value (-1 for unlimited)")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of already processed images")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to draw bounding boxes on images")
    parser.add_argument("--debug-draw", action="store_true",
                       help="Enable debug mode to draw bounding boxes on images")
    parser.add_argument("--sequential", action="store_true",
                       help="Run in sequential mode (no multiprocessing) for easier debugging")
    parser.add_argument("--task-timeout", type=int, default=36000,
                       help="Timeout per task in seconds (default: 1800 = 30 minutes)")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Directory to cache node images and metadata for Web UI")

    args, _ = parser.parse_known_args()

    # Set multiprocessing start method for better compatibility
    multiprocessing.set_start_method('spawn', force=True)

    main(args)


def create_example_input_list(image_dir: str, output_file: str, extensions: List[str] = None):
    """Create an example input list file from a directory of images

    Args:
        image_dir: Directory containing images
        output_file: Output text file path
        extensions: List of image extensions to include (default: ['.png', '.jpg', '.jpeg'])
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    image_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))

    with open(output_file, 'w') as f:
        for path in sorted(image_paths):
            f.write(f"{path}\n")

    print(f"Created input list with {len(image_paths)} images: {output_file}")


# Example usage:
"""
# To create an input list from a directory of images:
create_example_input_list("/path/to/images", "image_list.txt")

# To run the annotation:
python annotate_functional_regions.py \
    --input-list image_list.txt \
    --model gpt-4o \
    --workers 4 \
    --debug

# The output will be saved to: image_list.txt_parent_dir/functional_regions/gpt-4o/functional_regions_gpt-4o.json
# Debug images will be saved to: image_list.txt_parent_dir/functional_regions/gpt-4o/debug_images/
"""

