"""
Evaluate VLMs on Functional Region Grounding tasks

This script evaluates vision-language models on functional REGION grounding,
where models need to locate GUI regions based on natural language queries.
"""

import os
import json
import uuid
import re
import time
import argparse
import multiprocessing
import glob
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
from multiprocessing import Pool, Manager
from functools import partial
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path before importing project modules
import sys
# __file__ = .../highres_autogui/utils/data_utils/autoguiv2/FuncElemQA_eval_gen/eval/eval_funcregion_mp.py
# Need to go up 6 levels to reach highres_autogui
sys.path.append('/'.join(__file__.split('/')[:-6]))

from utils.data_utils.misc import pred_2_point, get_image_dimensions, resize_pil_image
from utils.eval_utils.autoguiv2.misc import adjust_bbox

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
    """Colorized debug print"""
    level_to_color = {
        'info': Fore.CYAN,
        'step': Fore.BLUE,
        'inference_done': Fore.GREEN,
        'warn': Fore.YELLOW,
        'error': Fore.RED,
        'title': Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")

try:
    from datasets import load_dataset, load_from_disk
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    debug_print("⚠️  datasets library not available. HuggingFace dataset loading disabled.", level="warn")

from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen3vl import Qwen3VL
from utils.openai_utils.huggingface import HFEndpoint

# Optional imports for models that may have missing dependencies
try:
    from utils.openai_utils.doubao import DOUBAO
except ImportError:
    DOUBAO = None
    debug_print("⚠️  DOUBAO not available (missing volcenginesdkarkruntime)", level="warn")

try:
    from utils.openai_utils.stepfun import STEPFUN
except ImportError:
    STEPFUN = None
    debug_print("⚠️  STEPFUN not available", level="warn")

try:
    from utils.openai_utils.parasail import PARASAIL
except ImportError:
    PARASAIL = None
    debug_print("⚠️  PARASAIL not available", level="warn")

try:
    from utils.openai_utils.jedi import JEDI
except ImportError:
    JEDI = None
    debug_print("⚠️  JEDI not available", level="warn")
try:
    # For region type → 6-bucket mapping
    repo_root = "/mnt/nvme0n1p1/hongxin_li/highres_autogui"
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from utils.data_utils.autoguiv2.FuncElemQA_eval_gen.eval.count_region_types import resolve_parent_category  # type: ignore
except Exception:
    def resolve_parent_category(leaf_type: str) -> str:
        return "Others"

REF_TAGS = {
    'funcgnd': 'a question about locating',
    'descgnd': 'a description of',
    'intentgnd': 'an action intent about interacting with'
}

REF_PLACEHOLDER = {
    'funcgnd': 'Question',
    'descgnd': 'Description',
    'intentgnd': 'Action Intent'
}

# Normalize terminology in text fields
_ELEMENT_PATTERN = re.compile(r'\b(?:element|Element|elements|Elements)\b')

def replace_element_with_region(text: Optional[str]) -> Optional[str]:
    """Replace occurrences of 'element(s)' with 'region(s)' preserving capitalization."""
    if not text:
        return text

    def _repl(match: re.Match) -> str:
        token = match.group(0)
        replacements = {
            'element': 'region',
            'Element': 'Region',
            'elements': 'regions',
            'Elements': 'Regions',
        }
        return replacements.get(token, token)

    return _ELEMENT_PATTERN.sub(_repl, text)

# Grounding prompt template
GEMINI_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target region, which should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Now analyze the screenshot and provide the bounding box for the target element:"""

CLAUDE_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target region, which should be [xmin, ymin, xmax, ymax]. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target region:"""

# UI-Tars (https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py)
UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>'')

## User Instruction
{question}"""

# OS-ATLAS
OSATLAS_PROMPT = 'In this UI screenshot, what is the position of the region corresponding to the command "{question}" (with bbox)?'

# OpenCUA System Prompt
OPENCUA_SYSPROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)

# OpenCUA Bbox Prompt (absolute pixel coordinates, for funcgnd)
OPENCUA_BBOX_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target region, which should be [xmin, ymin, xmax, ymax] in absolute pixel coordinates. Note that the X-axis runs horizontally from left to right, and the Y-axis runs vertically from top to bottom.

{ref_placeholder}: {question}"""

# UGround Prompt
UGROUND_PROMPT = """Your task is to help the user identify the precise coordinates (x, y) of a specific region on the screen based on a description.

- Your response should point to the center or a representative point within the described region as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant region based on its likely context or purpose.
- Return a single pair of coordinates (x, y). Prefer normalized coordinates in [0, 1000).

Description: {instruction}

Answer:"""

# JEDI Prompt
JEDI_PROMPT = "Please complete the following tasks via mouse click or wait: {instruction}"

# HOLO Prompt
try:
    from pydantic import BaseModel, Field
    class ClickAbsoluteAction(BaseModel):
        """Click at absolute coordinates."""
        action: Literal["click_absolute"] = "click_absolute"
        x: int = Field(description="The x coordinate, number of pixels from the left edge.")
        y: int = Field(description="The y coordinate, number of pixels from the top edge.")
    
    HOLO_PROMPT = f"""Localize a region on the GUI image according to the provided target and output a click position.
         * You must output a valid JSON following the format: {ClickAbsoluteAction.model_json_schema()}"""
except:
    HOLO_PROMPT = """Localize a region on the GUI image according to the provided target and output a click position.
         * You must output a valid JSON with format: {{"action": "click_absolute", "x": <int>, "y": <int>}}"""

HOLO_BBOX_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target region, which should be [xmin, ymin, xmax, ymax]. Note that the X-axis runs horizontally from left to right, and the Y-axis runs vertically from top to bottom.

{ref_placeholder}: {question}"""

# InfiGUI-G1
INFIGUIG1_SYSPROMPT = 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.'
INFIGUIG1_PROMPT = '''The screen's resolution is {new_width}x{new_height}.
Locate the UI region(s) for "{instruction}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]
Note: Output absolute pixel coordinates based on the screen resolution.'''

# GUI-R1 (absolute pixel coordinates; image_first=True)
GUIR1_PROMPT = (
    "You are RUN1-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instruction}', with the action history being 'None'.\n"
    "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
    "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
    "<think> ... </think> <answer>[{{'action': enum['click'], 'point': [x, y], 'input_text': 'no input text [default]'}}]</answer>\n"
    "Example:\n"
    "[{{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}}]\n"
)

# UI-Venus (outputs bbox [x1,y1,x2,y2] in absolute pixel coordinates)
UIVENUS_PROMPT = "Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2]."

# GENERIC_PROMPT
GENERIC_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target region, which should be [xmin, ymin, xmax, ymax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

**IMPORTANT:** The output bbox must be tight and complete.
Now analyze the screenshot and provide the bounding box for the target region:"""


def _normalize_bbox_0_1(box: List[float]) -> List[float]:
    """Normalize bbox to 0-1 scale; supports inputs in 0-1 or 0-1000.
    Also supports 2-element point coordinates [x, y]."""
    if not isinstance(box, list):
        return [0.0, 0.0, 0.0, 0.0]
    
    # Handle point coordinates (2 elements)
    if len(box) == 2:
        if max(box) <= 1.0:
            return [float(box[0]), float(box[1])]
        return [box[0] / 1000, box[1] / 1000]
    
    # Handle bounding boxes (4 elements)
    if len(box) != 4:
        return [0.0, 0.0, 0.0, 0.0]
    if max(box) <= 1.0:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    return [box[0] / 1000, box[1] / 1000, box[2] / 1000, box[3] / 1000]


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes
    
    Args:
        bbox1: [x_min, y_min, x_max, y_max] normalized 0-1000
        bbox2: [x_min, y_min, x_max, y_max] normalized 0-1000
    
    Returns:
        IoU value between 0 and 1
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    # Normalize inputs to 0-1 scale (support both 0-1000 and 0-1 inputs)
    bbox1_norm = _normalize_bbox_0_1(bbox1)
    bbox2_norm = _normalize_bbox_0_1(bbox2)
    
    # Calculate intersection
    x1 = max(bbox1_norm[0], bbox2_norm[0])
    y1 = max(bbox1_norm[1], bbox2_norm[1])
    x2 = min(bbox1_norm[2], bbox2_norm[2])
    y2 = min(bbox1_norm[3], bbox2_norm[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1_norm[2] - bbox1_norm[0]) * (bbox1_norm[3] - bbox1_norm[1])
    area2 = (bbox2_norm[2] - bbox2_norm[0]) * (bbox2_norm[3] - bbox2_norm[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def load_dataset_from_json(questions_file: str) -> List[Dict[str, Any]]:
    """Load dataset from questions JSON file
    
    Args:
        questions_file: Path to questions JSON file or glob pattern
    
    Returns:
        List of dataset entries
    """
    debug_print(f"\n📂 Loading dataset from JSON: {questions_file}", level="step")
    
    if '/*' in questions_file or '*' in questions_file:
        questions_files = glob.glob(questions_file)
    else:
        questions_files = [questions_file]

    all_entries = []

    for file in questions_files:
        if not os.path.exists(file):
            debug_print(f"⚠️  File not found: {file}", level="warn")
            continue

        with open(file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        results = checkpoint.get('results', {})
        dataset_name = file.split('/')[-3] if len(file.split('/')) >= 3 else 'unknown'
        
        # Infer image_src_dir
        image_src_dir = os.path.join(os.path.dirname(os.path.dirname(file)), 'images')
        
        # Try to load attributes file
        attr_file = file.replace(".json", "_attributes.json")
        attr_data = {}
        if os.path.exists(attr_file):
            try:
                with open(attr_file, 'r', encoding='utf-8') as f:
                    attr_data = json.load(f)
            except Exception:
                pass
        
        for image_name, image_data in results.items():
            if 'error' in image_data:
                continue
            
            image_path = image_data.get('image_path', '')
            if not image_path:
                image_path = os.path.join(image_src_dir, image_name)
            
            if not os.path.exists(image_path):
                continue
            generated = image_data.get('generated', [])
            
            for group_data in generated:
                questions = group_data.get('questions', [])
                elements = group_data.get('elements', [])
                elements_by_id = {elem.get('id'): elem for elem in elements}
                num_similar_elements = len(elements)
                
                for q_data in questions:
                    target_elem_id = q_data.get('target_element_id', -1)
                    referring_expressions = q_data.get('referring_expressions', {})
                    
                    target_element = elements_by_id.get(target_elem_id, {})
                    target_bbox = target_element.get('revised bbox', [])
                    
                    if not target_bbox or len(target_bbox) != 4:
                        continue
                    
                    # Get density class from attributes if available
                    group_index = group_data.get('group_index', -1)
                    density_class = 'unknown'
                    if image_name in attr_data and str(group_index) in attr_data[image_name] and str(target_elem_id) in attr_data[image_name][str(group_index)]:
                        density_class = attr_data[image_name][str(group_index)][str(target_elem_id)].get('density_class', 'unknown')
                    
                    # Build a single entry per question (region task: no action_type)
                    for _, action_data in referring_expressions.items():
                        if not isinstance(action_data, dict):
                            continue
                        question = action_data.get('question', '')
                        if not question:
                            continue
                        question = replace_element_with_region(question)
                        # Try best-effort region_type retrieval
                        region_type = (
                            q_data.get('region_type') or
                            target_element.get('region_type') or
                            image_data.get('region_type') or
                            ''
                        )
                        region_parent = resolve_parent_category(str(region_type))
                        entry = {
                            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_{uuid.uuid4().hex[:8]}",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'question': question,
                            'gt_bbox': target_bbox,
                            'group_index': group_index,
                            'target_elem_id': target_elem_id,
                            'density_class': density_class,
                            'region_type': region_type,
                            'region_parent': region_parent,
                        }
                        all_entries.append(entry)
    
    debug_print(f"✅ Loaded {len(all_entries)} entries from JSON", level="success")
    return all_entries


def get_hf_cache_path(hf_dataset_id: str, split: str, task_type: str = 'funcgnd') -> tuple:
    """Get cache file path and image cache directory for HuggingFace dataset conversion
    
    Args:
        hf_dataset_id: HuggingFace dataset ID
        split: Dataset split
        task_type: Task type to evaluate (used for JSON cache filename only)
    Returns:
        Tuple of (cache_file_path, image_cache_dir)
    """
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use shared cache directory for all task types (images are shared)
    cache_dir = os.path.join(script_dir, 'elemgnd_hf_dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of dataset_id and split for cache (images are shared across task types)
    cache_key = f"{hf_dataset_id}_{split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # JSON cache file is task-specific (different questions/descriptions/intents)
    cache_filename = f"{cache_hash}_{task_type}.json"
    cache_file_path = os.path.join(cache_dir, cache_filename)
    
    # Image cache directory is shared across all task types (same images)
    image_cache_dir = os.path.join(cache_dir, 'images', cache_hash)
    os.makedirs(image_cache_dir, exist_ok=True)
    
    return cache_file_path, image_cache_dir


def load_hf_entries_from_cache(cache_path: str, image_cache_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Load converted entries from cache
    
    Args:
        cache_path: Path to cache file
        image_cache_dir: Directory where cached images are stored
    
    Returns:
        List of entries if cache exists and is valid, None otherwise
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        entries = cache_data.get('entries', [])
        metadata = cache_data.get('metadata', {})
        
        # Verify that cached entries are valid
        if not entries:
            return None
        
        # Check if image paths still exist (verify first few and random sample)
        valid_count = 0
        sample_size = min(20, len(entries))
        import random
        indices_to_check = list(range(min(10, len(entries)))) + random.sample(range(len(entries)), min(10, len(entries) - 10)) if len(entries) > 10 else []
        
        for idx in indices_to_check:
            entry = entries[idx]
            image_path = entry.get('image_path', '')
            if image_path and os.path.exists(image_path):
                valid_count += 1
        
        if valid_count == 0 and len(entries) > 0:
            debug_print(f"⚠️  Cached entries have invalid image paths, regenerating cache", level="warn")
            return None
        
        # Verify image cache directory exists
        if not os.path.exists(image_cache_dir):
            debug_print(f"⚠️  Image cache directory not found, regenerating cache", level="warn")
            return None
        
        debug_print(f"✅ Loaded {len(entries)} entries from cache (created: {metadata.get('timestamp', 'unknown')})", level="success")
        debug_print(f"📁 Image cache: {image_cache_dir}", level="info")
        return entries
    
    except Exception as e:
        debug_print(f"⚠️  Error loading cache: {e}, regenerating", level="warn")
        return None


def save_hf_entries_to_cache(entries: List[Dict[str, Any]], cache_path: str, 
                             hf_dataset_id: str, split: str, task_type: str = 'funcgnd'):
    """Save converted entries to cache
    
    Args:
        entries: List of converted entries
        cache_path: Path to cache file
        hf_dataset_id: HuggingFace dataset ID (for metadata)
        split: Dataset split (for metadata)
        task_type: Task type (for metadata)
    """
    cache_data = {
        'metadata': {
            'hf_dataset_id': hf_dataset_id,
            'split': split,
            'task_type': task_type,
            'num_entries': len(entries),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'entries': entries
    }
    
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        debug_print(f"💾 Saved {len(entries)} entries to cache: {cache_path}", level="info")
    except Exception as e:
        debug_print(f"⚠️  Failed to save cache: {e}", level="warn")


def load_dataset_from_hf(hf_dataset_id: str, split: str = 'test', cache_dir: Optional[str] = None, task_type: str = 'funcgnd') -> List[Dict[str, Any]]:
    """Load dataset from HuggingFace Hub with caching (including PIL Images)
    
    Args:
        hf_dataset_id: HuggingFace dataset ID (e.g., 'username/dataset-name')
        split: Dataset split to load (default: 'test')
        cache_dir: Optional cache directory for downloaded datasets (HF library cache)
        task_type: Task type to evaluate
    Returns:
        List of dataset entries
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset loading. Install with: pip install datasets")
    
    # Check cache first
    cache_path, image_cache_dir = get_hf_cache_path(hf_dataset_id, split, task_type)
    cached_entries = load_hf_entries_from_cache(cache_path, image_cache_dir)
    if cached_entries is not None:
        return cached_entries
    
    debug_print(f"\n📂 Loading dataset from HuggingFace: {hf_dataset_id} (split: {split})", level="step")
    
    try:
        if cache_dir:
            raw_dataset = load_from_disk(cache_dir)
            dataset = raw_dataset[split]
        else:
            dataset = load_dataset(hf_dataset_id, split=split)

        debug_print(f"✅ Loaded {len(dataset)} entries from HuggingFace", level="success")
    except Exception as e:
        debug_print(f"❌ Failed to load dataset from HuggingFace: {e}", level="error")
        raise
    
    all_entries = []
    
    debug_print(f"🔄 Converting HuggingFace dataset to entries (this may take a while)...", level="step")
    debug_print(f"📁 Caching images to: {image_cache_dir}", level="info")
    
    def _pick_index(candidate_indices: List[Any], length: int) -> Optional[int]:
        for cand in candidate_indices:
            if isinstance(cand, int) and 0 <= cand < length:
                return cand
            # Some ids may be strings of digits
            if isinstance(cand, str) and cand.isdigit():
                cand_int = int(cand)
                if 0 <= cand_int < length:
                    return cand_int
        return None
    
    def _select_from_variant(value: Any, candidate_indices: List[Any]) -> Any:
        # Handles list/dict selectors for option_* style storages
        if isinstance(value, list):
            if not value:
                return None
            idx = _pick_index(candidate_indices, len(value))
            if idx is None:
                # Heuristic: if there is only one item, return it; else None
                return value[0] if len(value) == 1 else None
            return value[idx]
        if isinstance(value, dict):
            for cand in candidate_indices:
                if cand in value:
                    return value[cand]
                if isinstance(cand, int) and str(cand) in value:
                    return value[str(cand)]
                if isinstance(cand, str) and cand.isdigit() and int(cand) in value:
                    return value[int(cand)]
            # If dict has a single value, return it
            if len(value) == 1:
                return next(iter(value.values()))
            return None
        return value
    
    def _extract_from_options(options: Any, keys: List[str], candidate_indices: List[Any]) -> Any:
        # options is typically a list[dict]; try to select by index and fetch key
        if isinstance(options, list) and options:
            idx = _pick_index(candidate_indices, len(options)) or 0
            opt = options[idx]
            if isinstance(opt, dict):
                for k in keys:
                    if k in opt:
                        return opt[k]
        return None
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Converting dataset"):
        # Handle image - can be PIL Image or path string
        image = item.get('image') or item.get('image_path') or item.get('image_file')
        if image is None:
            continue

        # Extract image_name early for use in both PIL and string cases
        image_name = item.get('image_name', f'image_{idx}')
        dataset_name = item.get('dataset_name', 'unknown')
        original_image_name = image_name

        # If image is a PIL Image, save it to persistent cache directory
        if isinstance(image, Image.Image):
            # Sanitize image name for filesystem
            if not image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_name += '.png'

            image_path = os.path.join(image_cache_dir, image_name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            if not os.path.exists(image_path):
                image.save(image_path)
        elif isinstance(image, str):
            image_path = image
            if not os.path.exists(image_path):
                debug_print(f"⚠️  Image path not found: {image_path}", level="warn")
                continue
            # Derive image_name from path if missing or placeholder
            if (not original_image_name) or original_image_name.startswith('image_'):
                image_name = os.path.basename(image_path)
            else:
                image_name = original_image_name
        else:
            debug_print(f"⚠️  Unsupported image type: {type(image)}", level="warn")
            continue

        # Extract required fields - use 'question' variable for all task types
        if task_type == 'funcgnd':
            question = item.get('question', '') or item.get('query', '') or item.get('ref', '')
        elif task_type == 'descgnd':
            # For descgnd, prefer selecting from option_descriptions via correct_option_idx
            correct_idx = item.get('correct_option_idx')
            if isinstance(correct_idx, str) and correct_idx.isdigit():
                correct_idx = int(correct_idx)
            option_descriptions = item.get('option_descriptions')
            if isinstance(option_descriptions, list):
                option_descriptions = [
                    replace_element_with_region(desc) if isinstance(desc, str) else desc
                    for desc in option_descriptions
                ]
            if isinstance(option_descriptions, list) and isinstance(correct_idx, int) and 0 <= correct_idx < len(option_descriptions):
                question = option_descriptions[correct_idx]
            else:
                question = item.get('description', '') or item.get('caption', '')
        else:
            question = item.get('question', '') or item.get('query', '')

        # BBox: strictly use correct_bbox for HF dataset
        bbox = item.get('correct_bbox', [])
        # Region attributes (fine-grained from option_region_types via correct_option_idx)
        correct_idx = item.get('correct_option_idx')
        if isinstance(correct_idx, str) and correct_idx.isdigit():
            correct_idx = int(correct_idx)
        option_region_types = item.get('option_region_types')
        region_type = ''
        if isinstance(option_region_types, list) and isinstance(correct_idx, int) and 0 <= correct_idx < len(option_region_types):
            region_type = option_region_types[correct_idx]
        region_parent = resolve_parent_category(str(region_type))
        group_index = item.get('group_index', -1) if 'group_index' in item else item.get('group_id', -1)
        target_elem_id = item.get('id', -1) if 'id' in item else item.get('target_elem_id', -1) or item.get('elem_id', -1)

        # Extract metadata fields
        dataset_name = item.get('dataset_name', 'unknown')
        # Use only correct_option_idx to index option_* fields
        # density_class: from option_density_classes
        density_class = 'unknown'
        option_density_classes = item.get('option_density_classes')
        if isinstance(option_density_classes, list) and isinstance(correct_idx, int) and 0 <= correct_idx < len(option_density_classes):
            density_class = option_density_classes[correct_idx]
        # num_similar_elements: from num_options
        num_similar_elements = item.get('num_options', -1)
        # area_class: from option_area_classes
        area_class = None
        option_area_classes = item.get('option_area_classes')
        if isinstance(option_area_classes, list) and isinstance(correct_idx, int) and 0 <= correct_idx < len(option_area_classes):
            area_class = option_area_classes[correct_idx]


        if not question:
            continue

        if not bbox or len(bbox) != 4:
            continue

        # Convert bbox to list of floats if needed
        bbox = [float(x) for x in bbox]

        # Create entry
        entry = {
            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_{idx}",
            'image_path': image_path,
            'image_name': image_name,
            'dataset_name': dataset_name,
            'question': question,
            'gt_bbox': bbox,
            'group_index': group_index,
            'target_elem_id': target_elem_id,
            'density_class': density_class,
            'num_similar_elements': num_similar_elements,
            'region_type': region_type,
            'region_parent': region_parent,
            'area_class': area_class,
        }
        all_entries.append(entry)
    
    debug_print(f"✅ Converted {len(all_entries)} entries from HuggingFace dataset", level="success")
    
    # Save to cache
    save_hf_entries_to_cache(all_entries, cache_path, hf_dataset_id, split, task_type)
    
    debug_print(f"💾 Cached {len(all_entries)} entries and images to persistent cache", level="success")
    
    return all_entries


def load_evaluation_dataset(source: str, questions_file: Optional[str] = None, hf_dataset_id: Optional[str] = None, hf_split: str = 'test', hf_cache_dir: Optional[str] = None, task_type: str = 'funcgnd') -> List[Dict[str, Any]]:
    """Load dataset from either JSON file or HuggingFace Hub
    
    Args:
        source: Source type - 'json' or 'hf'
        questions_file: Path to questions JSON file (required if source='json')
        hf_dataset_id: HuggingFace dataset ID (required if source='hf')
        hf_split: Dataset split for HF datasets (default: 'test')
        hf_cache_dir: Optional cache directory for HF datasets
        task_type: Task type to evaluate
    Returns:
        List of dataset entries
    """
    if source == 'hf':
        if not hf_dataset_id:
            raise ValueError("hf_dataset_id is required when source='hf'")
        return load_dataset_from_hf(hf_dataset_id, hf_split, hf_cache_dir, task_type)
    else:
        if not questions_file:
            raise ValueError("questions_file is required when source='json'")
        return load_dataset_from_json(questions_file)


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load evaluation checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint JSON file (can be checkpoint or full result file)
    
    Returns:
        Dictionary with processed entry IDs and results
        Note: Only successfully completed entries (inference_done=True) are included in processed_ids.
        Failed entries will be retried on resume.
    """
    if not os.path.exists(checkpoint_file):
        return {'processed_ids': set(), 'results': {}}

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize results to dict format for consistency
        results = data.get('results', {})
        if isinstance(results, list):
            # Convert list format to dict
            results = {r['entry_id']: r for r in results if 'entry_id' in r}
        elif not isinstance(results, dict):
            results = {}
        
        # Check if processed_ids exists in the checkpoint
        if 'processed_ids' in data:
            # Use existing processed_ids
            processed_ids = set(data.get('processed_ids', []))
            inferred = False
        else:
            # Backward compatibility: infer processed_ids from results
            # All entries in results are considered "processed" (attempted)
            processed_ids = set(results.keys())
            inferred = True
            if inferred:
                debug_print(f"📝 processed_ids not found in checkpoint, inferring from {len(processed_ids)} results", level="info")
        
        # Filter out failed entries from processed_ids so they can be retried
        # Only entries with inference_done=True and valid pred_bbox should be considered successfully processed
        successful_ids = set()
        failed_ids = set()
        
        for entry_id in processed_ids:
            result = results.get(entry_id)
            
            if result:
                # Check if inference was successful
                if result.get('inference_done', False) and result.get('pred_bbox') is not None:
                    successful_ids.add(entry_id)
                else:
                    failed_ids.add(entry_id)
            else:
                # If we can't find the result, assume it needs to be retried
                failed_ids.add(entry_id)

        # Update processed_ids to only include successful entries
        processed_ids = successful_ids

        if failed_ids:
            debug_print(f"⚠️  Found {len(failed_ids)} failed entries that will be retried", level="warn")
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} successful entries, {len(failed_ids)} failed entries to retry", level="success")
        else:
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} processed entries", level="success")

        return {'processed_ids': processed_ids, 'results': results}
    except Exception as e:
        debug_print(f"⚠️  Error loading checkpoint: {e}", level="warn")
        return {'processed_ids': set(), 'results': {}}


def save_checkpoint(results: Dict, processed_ids: set, checkpoint_file: str, metadata: Dict = None):
    """Save evaluation checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    
    checkpoint = {
        'metadata': metadata or {},
        'processed_ids': list(processed_ids),
        'results': results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def find_latest_checkpoint(eval_result_dir: str, model_name: str) -> Optional[str]:
    """Find the latest checkpoint file for a model
    
    Args:
        eval_result_dir: Directory containing evaluation results
        model_name: Model identifier
    
    Returns:
        Path to latest checkpoint or None
    """
    # Clean model name for filesystem
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    pattern = os.path.join(eval_result_dir, safe_model_name, '*.json')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


# Global worker model
worker_model = None


def init_worker(model_args: Dict):
    """Initialize worker with model"""
    global worker_model
    
    base_url = model_args['base_url']
    api_key = model_args['api_key']
    model = model_args['model']

    MAX_TOKENS = 8192
    
    # Check if using local vllm deployment (localhost base_url)
    # For local vllm, always use OpenAIModel regardless of model name
    is_local_vllm = base_url and ('localhost' in base_url or '127.0.0.1' in base_url)
    
    if is_local_vllm:
        # Local vllm deployment - use OpenAI-compatible API
        cloud_model_class = OpenAIModel
        api_key = api_key or "NOT_REQUIRED"
        MAX_TOKENS = 8192
    elif 'qwen' in model.lower():
        # Qwen models (including Qwen3-VL-8B-Instruct, qwen3-vl-32b-thinking, etc.)
        # Only for cloud API
        base_url = base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
        # Qwen models support longer outputs
        MAX_TOKENS = 8192
    elif any(x in model.lower() for x in ['atlas']):
        base_url = base_url or 'https://rvrvpi4zz7ispneo.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'jedi' in model.lower():
        if JEDI is None:
            raise ImportError("JEDI model requires additional dependencies. Please install them first.")
        base_url = 'https://afs3uxirrk48y8q5.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = JEDI
    elif 'uground' in model.lower():
        base_url = base_url or 'https://rs4m9o05rautq0ne.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'tars' in model.lower():
        if PARASAIL is None:
            raise ImportError("PARASAIL model requires additional dependencies. Please install them first.")
        base_url = 'https://api.parasail.io/v1'
        api_key = api_key or os.environ.get("PARASAIL_API_KEY", "EMPTY")
        cloud_model_class = PARASAIL
        MAX_TOKENS = 2048
    elif 'seed' in model.lower():
        if DOUBAO is None:
            raise ImportError("DOUBAO model requires volcenginesdkarkruntime. Please install it first.")
        base_url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
        api_key = api_key or os.environ.get("ARK_API_KEY", "EMPTY")
        cloud_model_class = DOUBAO
    elif 'step' in model.lower():
        if STEPFUN is None:
            raise ImportError("STEPFUN model requires additional dependencies. Please install them first.")
        base_url = 'https://api.stepfun.com/v1'
        api_key = api_key or os.environ.get("STEP_API_KEY", "EMPTY")
        cloud_model_class = STEPFUN
    elif 'glm' in model.lower():
        base_url = 'https://api.siliconflow.cn/v1'
        api_key = api_key or os.environ.get("SILICON_API_KEY", "EMPTY")
        cloud_model_class = OpenAIModel
    else:
        # Default: OpenAI-compatible API
        cloud_model_class = OpenAIModel
        base_url = base_url or os.environ.get("OPENAI_API_BASE_XIAOAI", "EMPTY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY_XIAOAI", "NOT_REQUIRED")

    worker_model = cloud_model_class(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=MAX_TOKENS
    )

def process_entry(entry: Dict, worker_id: int = 0, scale: int = 1000, task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Process a single dataset entry
    
    Args:
        entry: Dataset entry dictionary
        worker_id: Worker ID for logging
        scale: Scale for bbox normalization (default: 1000)
        task_type: Task type ('funcgnd', 'descgnd', 'intentgnd')
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    
    # Get the appropriate field based on task type
    question = entry.get('question', '')

    gt_bbox = entry['gt_bbox'] # x1, y1, x2, y2

    W, H = get_image_dimensions(image_path)

    gt_bbox = [gt_bbox[0] / W, gt_bbox[1]/H, gt_bbox[2]/W, gt_bbox[3]/H]

    

    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'question': question,
        'gt_bbox': gt_bbox,
        'pred_bbox': None,
        'iou': 0.0,
        'center_acc': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        # Preserve metadata for decomposed metrics
        'density_class': entry.get('density_class', 'unknown'),
        'num_similar_elements': entry.get('num_similar_elements', -1),
        'region_type': entry.get('region_type', ''),
        'region_parent': entry.get('region_parent', 'Others'),
        # area_class is provided by dataset; do not compute
        'area_class': entry.get('area_class'),
    }

    start_time = time.time()

    if not question:
        result['error'] = "No question found in entry"
        result['processing_time'] = time.time() - start_time
        return result

    retry = 0
    while retry < 4:
        try:
            retry += 1
            # Create prompt based on task type and model
            ref_tag = REF_TAGS.get(task_type, 'a question about locating')
            ref_placeholder = REF_PLACEHOLDER.get(task_type, 'Question')
            temp_img_path = image_path
            is_resized = False
            system_prompt = None
            orig_W, orig_H = get_image_dimensions(image_path)

            if 'gemini' in worker_model.model.lower():
                prompt = GEMINI_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                
            elif 'tars' in worker_model.model.lower():
                prompt = UI_TARS_PROMPT.format(question=question)
            elif 'atlas' in worker_model.model.lower():
                prompt = OSATLAS_PROMPT.format(question=question)
            elif 'uground' in worker_model.model.lower():
                prompt = UGROUND_PROMPT.format(instruction=question)
            elif 'jedi' in worker_model.model.lower():
                is_resized = True
                prompt = JEDI_PROMPT.format(instruction=('Click the region specified by this instruction: ' + question) if task_type in ['funcgnd', 'descgnd'] else question)
                temp_img_path = f'jedi_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=1080)
                resized_image.save(temp_img_path)
            elif 'holo' in worker_model.model.lower():
                # Use bbox prompt for Holo models to get bounding box output
                prompt = HOLO_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
            elif 'opencua' in worker_model.model.lower():
                # Use bbox prompt for OpenCUA (outputs absolute pixel coordinates)
                prompt = OPENCUA_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
            elif 'infigui-g1' in worker_model.model.lower():
                # Use Holo's bbox prompt for InfiGUI-G1 (outputs absolute coordinates)
                prompt = HOLO_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
            elif 'gui-r1' in worker_model.model.lower():
                # Use Holo-style bbox prompt for GUI-R1 (output [xmin, ymin, xmax, ymax])
                prompt = HOLO_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
            elif 'venus' in worker_model.model.lower():
                prompt = UIVENUS_PROMPT.format(instruction=question)
            elif any(x in worker_model.model.lower() for x in ['claude']):
                is_resized = True
                prompt = CLAUDE_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                temp_img_path = f'claude_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            elif 'qwen3' in worker_model.model.lower():
                prompt = GENERIC_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                image = Image.open(image_path).convert("RGB")
                if max(image.size) > 3500:
                    temp_img_path = f'qwen3_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                    resized_image, _ = resize_pil_image(image, max_size=3200)
                    resized_image.save(temp_img_path)
            else:
                prompt = GENERIC_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)

            W, H = get_image_dimensions(temp_img_path)

            # Pre-logging: Show query start information
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            print(f"[Worker {worker_id}] 🚀 Starting query{retry_info} | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Image: {image_name_short} | [{timestamp_start}]")

            try:
                # Get model response
                success, response, _ = worker_model.get_model_response(
                    prompt, 
                    [temp_img_path], 
                    use_img_url=True, 
                    temperature=0.0, 
                    timeout=360,
                    image_first=any(x in worker_model.model.lower() for x in ['opencua', 'holo', 'infigui-g1', 'gui-r1']),
                    sys_prompt=system_prompt if system_prompt else ''
                )
            except Exception as e:
                # Exception during API call - log and continue to next retry
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                # Clean up temp file if needed
                if is_resized:
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass
                # Concise error logging for exceptions
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(e)[:200]
                print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                # Continue to next retry (don't break)
                continue

            if not success:
                result['error'] = f"API call failed: {response}"
                result['processing_time'] = time.time() - start_time
                # Concise error logging
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(response)[:200] if response else "Unknown error"
                print(f"[Worker {worker_id}] ❌ Query FAILED | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Time: {result['processing_time']:.2f}s | [{timestamp_error}]")
                # Continue to next retry (don't return here, let the loop handle retries)
                continue

            result['prompt'], result['response'] = prompt, response

            # Parse bbox from response
            if '</think>' in response:
                thinking = response.split('</think>')[0].replace('<think>', '').strip()
                bbox_str = response.split('</think>')[1].strip()
            else:
                thinking, bbox_str = '', response.strip()

            result['thinking'], result['bbox_pred_str'] = thinking, bbox_str

            # Parsing
            raw_pred_bbox = None

            # Case 2: GLM-4.5. "The bounding box for the 'wall_95' entry in the Outliner list is <|begin_of_box|>[816, 162, 838, 172]<|end_of_box|>."
            if '<|begin_of_box|>' in bbox_str:
                raw_pred_bbox = bbox_str.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0].strip()
                raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale)
            # Case 4: UI-Tars: "Action: click(start_box='(1786,924)')"
            elif 'tars' in worker_model.model.lower():
                pass
            # Case 5: OS-Atlas: <|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|><|im_end|>
            # Or: Week 0 Functions(212,646),(324,676)
            elif 'atlas' in worker_model.model.lower():
                # Extract only the coordinate part to avoid parsing numbers from text
                # Find the last occurrence of coordinates pattern (x1,y1),(x2,y2)
                import re
                coord_pattern = re.compile(r'\((\d+),(\d+)\),\((\d+),(\d+)\)')
                match = coord_pattern.search(bbox_str)
                if match:
                    # Extract the matched coordinates and convert to bbox format
                    x1, y1, x2, y2 = map(int, match.groups())
                    raw_pred_bbox = [x1, y1, x2, y2]
                    raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale)
                else:
                    # Fallback: try to extract from the last parenthesis
                    if '(' in bbox_str and ')' in bbox_str:
                        # Find the last complete coordinate pair
                        last_paren_start = bbox_str.rfind('(')
                        bbox_str_trimmed = bbox_str[last_paren_start:]
                        raw_pred_bbox = pred_2_point(bbox_str_trimmed, scale=scale)
                    else:
                        raw_pred_bbox = None
            # Case 6: JEDI: '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [453, 258]}}\n</tool_call>'
            elif 'jedi' in worker_model.model.lower():
                bbox_str = bbox_str[bbox_str.rfind('['):bbox_str.rfind(']')+1]
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            # Case 7: Holo - now outputs bbox: "Box: [x1, y1, x2, y2]" or direct list
            elif 'holo' in worker_model.model.lower():
                # Holo now outputs bbox format, not point
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            # Case 8: OpenCUA: bbox prompt; model may output (x, y, width, height) instead of (xmin, ymin, xmax, ymax)
            elif 'opencua' in worker_model.model.lower():
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
                # If 4 values and xmax<xmin or ymax<ymin, treat as (x, y, w, h) and convert to (xmin, ymin, xmax, ymax)
                if (raw_pred_bbox is not None and isinstance(raw_pred_bbox, list) and len(raw_pred_bbox) == 4 and
                    (raw_pred_bbox[2] < raw_pred_bbox[0] or raw_pred_bbox[3] < raw_pred_bbox[1])):
                    raw_pred_bbox = [
                        raw_pred_bbox[0], raw_pred_bbox[1],
                        raw_pred_bbox[0] + raw_pred_bbox[2], raw_pred_bbox[1] + raw_pred_bbox[3]
                    ]
            # Case 9: InfiGUI-G1: Now uses Holo's bbox prompt, outputs bbox format like "[47, 358, 161, 391]"
            elif 'infigui-g1' in worker_model.model.lower():
                # Parse bbox string using pred_2_point (absolute coordinates, scale=-1)
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            # Case 10: GUI-R1: now uses Holo-style bbox prompt; parse [xmin, ymin, xmax, ymax] (absolute pixel)
            elif 'gui-r1' in worker_model.model.lower():
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            # Case 11: UI-Venus: outputs bbox "[x1,y1,x2,y2]" in absolute pixel coordinates
            elif 'venus' in worker_model.model.lower():
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            elif '```json' in bbox_str or bbox_str.startswith('{') and bbox_str.endswith('}'):
                    bbox_cleaned = bbox_str.split('```json')[1].split('```')[0].strip() if '```json' in bbox_str else bbox_str
                    bbox_parsed = json.loads(bbox_cleaned)

                    if isinstance(bbox_parsed, list):
                        item = bbox_parsed[0]
                        if isinstance(item, dict):
                            if 'box_2d' in item:
                                raw_pred_bbox = item['box_2d']
                        else:
                            raw_pred_bbox = bbox_parsed
                    elif isinstance(bbox_parsed, dict):
                        if 'box_2d' in bbox_parsed:
                            raw_pred_bbox = bbox_parsed['box_2d']
                    
                    if any([p > 1 for p in raw_pred_bbox]):
                        raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale, w=W, h=H)
            # Fall back to a general parsing method
            if raw_pred_bbox is None:
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            elif any([p > 1 for p in raw_pred_bbox]):
                raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale, w=W, h=H)

            # Adjust bbox format according to the model preferece
            pred_bbox = adjust_bbox(worker_model.model, raw_pred_bbox)

            if pred_bbox is None:
                # Failed to parse bbox - log and retry
                result['error'] = "Failed to parse bbox from response"
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] ❌ Query FAILED (parse error) | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: Could not parse bbox | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue

            # Only reach here if we successfully parsed a bbox
            result['pred_bbox'] = pred_bbox
            # Calculate IoU
            iou = calculate_iou(pred_bbox, gt_bbox)
            result['iou'] = iou

            # Calculate center accuracy
            pred_bbox_n = _normalize_bbox_0_1(pred_bbox) if isinstance(pred_bbox, list) else pred_bbox
            gt_bbox_n = _normalize_bbox_0_1(gt_bbox)
            if isinstance(pred_bbox_n, list) and len(pred_bbox_n) == 4:
                center = [(pred_bbox_n[0] + pred_bbox_n[2]) / 2, (pred_bbox_n[1] + pred_bbox_n[3]) / 2]
            elif isinstance(pred_bbox_n, list) and len(pred_bbox_n) == 2:
                # Point coordinates - use directly as center
                center = pred_bbox_n
            else:
                # Fallback for unexpected formats
                center = [0.0, 0.0]
            
            # Check if center point is within GT bbox
            if len(center) == 2 and len(gt_bbox_n) == 4:
                center_acc = gt_bbox_n[0] <= center[0] <= gt_bbox_n[2] and gt_bbox_n[1] <= center[1] <= gt_bbox_n[3]
            else:
                center_acc = False
            result['center_acc'] = center_acc
            result['inference_done'] = True

            # Only print "Query COMPLETE" if we actually got a valid bbox
            status = "✅" if center_acc else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | GT: {gt_bbox} <=> Pred: {bbox_str} -> {pred_bbox} -> {center} | "
                  f"IoU={iou:.3f} | CenterAcc={center_acc} | Time: {processing_time:.2f}s | [{timestamp}]")
            
            # Clean up temp file if needed
            if is_resized and temp_img_path != image_path:
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            # Clean up temp file if needed
            if is_resized and temp_img_path != image_path:
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
            # Concise error logging for exceptions
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            # Continue to next retry (don't break)
            continue
    else:
        result['error'] = "Failed to parse bbox from response"
        # Final error logging after all retries exhausted
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry_id} | "
              f"Model: {worker_model.model} | Error: Failed to parse bbox | "
              f"Total time: {time.time() - start_time:.2f}s | [{timestamp_error}]")

    result['processing_time'] = time.time() - start_time
    return result


def process_entries_with_multiprocessing(entries: List[Dict], model_args: Dict, 
                                        args, checkpoint: Dict) -> List[Dict]:
    """Process entries using multiprocessing"""
    manager = Manager()
    
    # Initialize manager.dict() properly - cannot initialize with nested dict directly
    results_dict = manager.dict()
    checkpoint_results = checkpoint.get('results', {})
    
    # Handle both dict and list formats from checkpoint
    if isinstance(checkpoint_results, dict):
        # Populate manager.dict() item by item
        for key, value in checkpoint_results.items():
            results_dict[key] = value
    elif isinstance(checkpoint_results, list):
        # Convert list format to dict
        for result in checkpoint_results:
            if 'entry_id' in result:
                results_dict[result['entry_id']] = result
    
    # Initialize processed_ids manager.dict()
    processed_ids = manager.dict()
    for entry_id in checkpoint.get('processed_ids', set()):
        processed_ids[entry_id] = True
    
    start_time = time.time()
    processed_count = manager.Value('i', len(processed_ids))
    lock = manager.Lock()
    
    def update_throughput():
        current_time = time.time()
        elapsed = current_time - start_time
        count = processed_count.value
        throughput = count / elapsed if elapsed > 0 else 0
        print(f"\rThroughput: {throughput:.2f} entries/s | "
              f"Processed: {count}/{len(entries)} | "
              f"Elapsed: {elapsed:.1f}s", end='', flush=True)
    
    if args.sample_limit is not None and args.sample_limit > 0:
        entries = entries[:args.sample_limit]
    
    # Determine the scale
    # The scale of Hcompany/Holo2-8B is 1000 while that of Hcompany/Holo1.5-7B is -1 (absolute coordinates)
    if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'jedi', 'holo1.5', 'opencua', 'infigui-g1', 'gui-r1', 'venus']):
        scale = -1
    else:
        scale = 1000

    # Filter entries to process
    entries_to_process = [
        (entry, i % args.max_workers, scale, args.task_type) 
        for i, entry in enumerate(entries) 
        if entry['entry_id'] not in processed_ids
    ]
    
    # Count retries (entries that failed previously)
    retry_count = 0
    checkpoint_results = checkpoint.get('results', {})
    if isinstance(checkpoint_results, dict):
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in checkpoint_results:
                result = checkpoint_results[entry_id]
                if not result.get('inference_done', False) or result.get('pred_bbox') is None:
                    retry_count += 1
    elif isinstance(checkpoint_results, list):
        # Handle list format (from full result files)
        results_by_id = {r.get('entry_id'): r for r in checkpoint_results if 'entry_id' in r}
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in results_by_id:
                result = results_by_id[entry_id]
                if not result.get('inference_done', False) or result.get('pred_bbox') is None:
                    retry_count += 1
    
    if retry_count > 0:
        debug_print(f"📋 Processing {len(entries_to_process)}/{len(entries)} entries ({retry_count} retries, {len(entries_to_process) - retry_count} new)", level="info")
    else:
        debug_print(f"📋 Processing {len(entries_to_process)}/{len(entries)} entries", level="info")

    results = []

    with Pool(processes=args.max_workers, initializer=init_worker, initargs=(model_args,)) as pool:
        for result in pool.starmap(process_entry, entries_to_process):
            results.append(result)
            results_dict[result['entry_id']] = result

            with lock:
                # Only mark as processed if inference was successful
                # Failed entries will be retried on next run
                if result.get('inference_done', False) and result.get('pred_bbox') is not None:
                    processed_ids[result['entry_id']] = True
                    processed_count.value += 1
                # Still save failed results to checkpoint for history, but don't mark as processed

            update_throughput()

            # Save checkpoint periodically (every result)
            # Include all results (successful and failed) but only mark successful ones as processed
            successful_processed_ids = {k for k, v in processed_ids.items() if v}
            save_checkpoint(
                dict(results_dict),
                successful_processed_ids,
                args.checkpoint_file,
                {'model': args.model, 'total_entries': len(entries)}
            )

    print()  # New line after throughput updates

    # Return all results (checkpoint results + newly processed results)
    # results_dict already contains both old and new results
    all_results = list(results_dict.values())

    return all_results


def calculate_metrics_for_subset(subset_results: List[Dict]) -> Dict[str, Any]:
    """Calculate metrics for a subset of results"""
    total = len(subset_results)
    successful = sum(1 for r in subset_results if r.get('inference_done', False))
    
    if successful == 0:
        return {
            'total': total,
            'successful': 0,
            'success_rate': 0.0,
            'avg_iou': 0.0,
            'iou_thresholds': {},
            'center_acc': 0.0
        }
    
    ious = [r.get('iou', 0.0) for r in subset_results if r.get('inference_done', False) and 'iou' in r]
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    iou_thresholds = {}
    for threshold in thresholds:
        count = sum(1 for iou in ious if iou >= threshold)
        iou_thresholds[f'iou@{threshold}'] = count / total if total > 0 else 0.0
    
    center_accs = [r.get('center_acc', False) for r in subset_results if r.get('inference_done', False) and 'center_acc' in r]
    center_acc = sum(center_accs) / len(center_accs) if center_accs else 0.0
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': successful / total if total > 0 else 0.0,
        'avg_iou': avg_iou,
        'iou_thresholds': iou_thresholds,
        'center_acc': center_acc
    }


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate evaluation metrics with decomposed breakdowns"""
    total = len(results)
    successful = sum(1 for r in results if r.get('inference_done', False))

    if successful == 0:
        return {
            'total': total,
            'successful': 0,
            'success_rate': 0.0,
            'avg_iou': 0.0,
            'iou_thresholds': {},
            'center_acc': 0.0,
            'decomposed': {}
        }

    # Get IoU values for successful predictions
    ious = [r.get('iou', 0.0) for r in results if r.get('inference_done', False) and 'iou' in r]
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    
    # Calculate accuracy at different IoU thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    iou_thresholds = {}
    for threshold in thresholds:
        count = sum(1 for iou in ious if iou >= threshold)
        iou_thresholds[f'iou@{threshold}'] = count / total if total > 0 else 0.0
    
    # Calculate center accuracy
    center_accs = [r.get('center_acc', False) for r in results if r.get('inference_done', False) and 'center_acc' in r]
    center_acc = sum(center_accs) / len(center_accs) if center_accs else 0.0

    # Area class breakdown (use dataset-provided area_class values only)
    area_breakdown = {}
    present_area_classes = sorted({r.get('area_class') for r in results if r.get('inference_done', False) and r.get('area_class')})
    for area_class in present_area_classes:
        subset = [r for r in results if r.get('inference_done', False) and r.get('area_class') == area_class]
        area_breakdown[area_class] = calculate_metrics_for_subset(subset) if subset else {
            'total': 0,
            'successful': 0,
            'success_rate': 0.0,
            'avg_iou': 0.0,
            'iou_thresholds': {f'iou@{t}': 0.0 for t in [0.1, 0.3, 0.5, 0.7, 0.9]},
            'center_acc': 0.0
        }

    # Region parent (6-bucket) breakdown
    region_parent_breakdown = {}
    for parent in ['Primary Interface Containers',
                   'Global Navigation & Structure',
                   'Content & Data Display',
                   'Interaction & Input',
                   'Contextual & Temporary Regions',
                   'Others']:
        subset = [r for r in results if r.get('inference_done', False) and r.get('region_parent', 'Others') == parent]
        region_parent_breakdown[parent] = calculate_metrics_for_subset(subset) if subset else {
            'total': 0,
            'successful': 0,
            'success_rate': 0.0,
            'avg_iou': 0.0,
            'iou_thresholds': {f'iou@{t}': 0.0 for t in [0.1, 0.3, 0.5, 0.7, 0.9]},
            'center_acc': 0.0
        }

    # Decomposed metrics by density class
    density_breakdown = {}
    for density in ['sparse', 'medium', 'dense', 'unknown']:
        subset = [r for r in results if r.get('density_class', 'unknown') == density]
        if subset:
            density_breakdown[density] = calculate_metrics_for_subset(subset)
    
    # Decomposed metrics by number of similar elements
    def get_num_elements_category(num_elements: int) -> str:
        if num_elements < 0:
            return 'unknown'
        elif num_elements <= 2:
            return '1-2'
        elif num_elements <= 4:
            return '3-4'
        elif num_elements <= 6:
            return '5-6'
        else:
            return '7+'
    
    num_elements_breakdown = {}
    for category in ['1-2', '3-4', '5-6', '7+', 'unknown']:
        subset = [r for r in results if get_num_elements_category(r.get('num_similar_elements', -1)) == category]
        # Always include category, even if empty, to show complete breakdown
        num_elements_breakdown[category] = calculate_metrics_for_subset(subset) if subset else {
            'total': 0,
            'successful': 0,
            'success_rate': 0.0,
            'avg_iou': 0.0,
            'iou_thresholds': {f'iou@{t}': 0.0 for t in [0.1, 0.3, 0.5, 0.7, 0.9]},
            'center_acc': 0.0
        }
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': successful / total if total > 0 else 0.0,
        'avg_iou': avg_iou,
        'iou_thresholds': iou_thresholds,
        'center_acc': center_acc,
        'decomposed': {
            'by_density': density_breakdown,
            'by_num_similar_elements': num_elements_breakdown,
            'by_region_parent': region_parent_breakdown,
            'by_area_class': area_breakdown,
        }
    }


def main(args):
    """Main evaluation function"""
    debug_print("═" * 60, level="title")
    debug_print("🔍 Functional Region Grounding Evaluation", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    if args.hf_dataset_id:
        debug_print(f"   Source: {Fore.GREEN}HuggingFace{Style.RESET_ALL}", level="info")
        debug_print(f"   Dataset ID: {Fore.CYAN}{args.hf_dataset_id}{Style.RESET_ALL}", level="info")
        debug_print(f"   Task Type: {Fore.CYAN}{args.task_type}{Style.RESET_ALL}", level="info")
        debug_print(f"   Split: {Fore.CYAN}{args.hf_split}{Style.RESET_ALL}", level="info")
        if args.hf_cache_dir:
            debug_print(f"   Cache Dir: {Fore.CYAN}{args.hf_cache_dir}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Source: {Fore.GREEN}JSON File{Style.RESET_ALL}", level="info")
        debug_print(f"   Questions File: {Fore.CYAN}{args.questions_file}{Style.RESET_ALL}", level="info")
    
    debug_print("\n🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    
    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    debug_print(f"   Workers: {Fore.YELLOW}{args.max_workers}{Style.RESET_ALL}", level="info")
    
    debug_print("\n💾 CHECKPOINT CONFIGURATION", level="step")
    if args.checkpoint_file:
        debug_print(f"   Checkpoint File: {Fore.CYAN}{args.checkpoint_file}{Style.RESET_ALL}", level="info")
    elif args.load_latest:
        debug_print(f"   Load Latest: {Fore.GREEN}YES{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Checkpoint: {Fore.YELLOW}Disabled{Style.RESET_ALL}", level="info")
    
    debug_print("\n" + "═" * 60, level="title")
    
    # Load dataset
    if args.hf_dataset_id:
        entries = load_evaluation_dataset('hf', hf_dataset_id=args.hf_dataset_id, 
                              hf_split=args.hf_split, hf_cache_dir=args.hf_cache_dir, task_type=args.task_type)
    else:
        entries = load_evaluation_dataset('json', questions_file=args.questions_file, task_type=args.task_type)
    
    if not entries:
        debug_print("❌ No entries found in dataset", level="error")
        return
    
    # If only printing samples, preview and exit early (no model/API usage)
    if getattr(args, 'print_samples', 0):
        limit = min(args.print_samples, len(entries))
        debug_print(f"\n👀 Previewing first {limit} normalized entries:", level="step")
        for idx, e in enumerate(entries[:limit], 1):
            debug_print(f"[{idx}] entry_id={e.get('entry_id')}, image={os.path.basename(e.get('image_path',''))}", level="info")
            debug_print(f"     question={e.get('question')}", level="info")
            debug_print(f"     region_type={e.get('region_type')} | region_parent={e.get('region_parent')}", level="info")
            debug_print(f"     density_class={e.get('density_class')} | area_class={e.get('area_class')} | num_similar_elements={e.get('num_similar_elements')}", level="info")
            debug_print(f"     gt_bbox={e.get('gt_bbox')}", level="info")
        return
    
    # Setup checkpoint and result file paths
    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results', args.task_type)
    os.makedirs(eval_result_dir, exist_ok=True)
    
    safe_model_name = args.model.replace('/', '_').replace('\\', '_')
    model_result_dir = os.path.join(eval_result_dir, safe_model_name)
    os.makedirs(model_result_dir, exist_ok=True)
    
    # Set checkpoint file: use specified file, or latest if load_latest, or create new timestamped file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.checkpoint_file:
        checkpoint_file = args.checkpoint_file
        # If checkpoint file is specified, use it as result file too (unless it's explicitly different)
        result_file = checkpoint_file
    elif args.load_latest:
        latest = find_latest_checkpoint(eval_result_dir, args.model)
        if latest:
            checkpoint_file = latest
            result_file = latest  # Resume to the same file
            debug_print(f"📂 Found latest checkpoint: {latest}", level="info")
        else:
            # No existing checkpoint, create new timestamped file
            result_file = os.path.join(model_result_dir, f"{timestamp}.json")
            checkpoint_file = result_file
            debug_print(f"📂 No existing checkpoint found, creating new: {checkpoint_file}", level="info")
    else:
        # Default: create new timestamped file for both checkpoint and result
        result_file = os.path.join(model_result_dir, f"{timestamp}.json")
        checkpoint_file = result_file
    
    # Load checkpoint
    checkpoint = {'processed_ids': set(), 'results': {}}
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file)
    
    # Set checkpoint_file in args for use in processing
    args.checkpoint_file = checkpoint_file
    
    # Prepare model arguments
    model_args = {
        'base_url': args.base_url,
        'api_key': args.api_key,
        'model': args.model
    }
    
    # Process entries
    debug_print(f"\n🚀 Starting evaluation...", level="step")
    start_time = time.time()

    results = process_entries_with_multiprocessing(entries, model_args, args, checkpoint)
    
    total_time = time.time() - start_time
    
    # Calculate metrics from all results
    debug_print(f"\n📊 Calculating metrics...", level="step")
    metrics = calculate_metrics(results)
    
    # Prepare final output
    output = {
        'metadata': {
            'model': args.model,
            'base_url': args.base_url,
            'questions_file': args.questions_file if not args.hf_dataset_id else None,
            'hf_dataset_id': args.hf_dataset_id if args.hf_dataset_id else None,
            'hf_split': args.hf_split if args.hf_dataset_id else None,
            'timestamp': timestamp,
            'total_time': total_time,
            'num_workers': args.max_workers,
            'sample_limit': args.sample_limit if hasattr(args, 'sample_limit') else None
        },
        'metrics': metrics,
        'results': results
    }
    
    # Save final result file
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # If checkpoint_file is different from result_file, also save checkpoint format
    if checkpoint_file != result_file:
        processed_ids = {r['entry_id'] for r in results}
        save_checkpoint(
            {r['entry_id']: r for r in results},
            processed_ids,
            checkpoint_file,
            {'model': args.model, 'total_entries': len(entries)}
        )
    # If they're the same, the result file already contains everything
    
    # Print summary with elegant tables
    console = Console() if RICH_AVAILABLE else None
    
    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Evaluation Complete!", level="success")
    
    if console:
        # Overall metrics table
        overall_table = Table(title="📊 Overall Metrics", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        overall_table.add_column("Metric", style="cyan", no_wrap=True)
        overall_table.add_column("Value", style="green", justify="right")
        
        overall_table.add_row("Total Entries", str(metrics['total']))
        overall_table.add_row("Successful", f"{metrics['successful']} ({metrics['success_rate']*100:.1f}%)")
        overall_table.add_row("Average IoU", f"{metrics['avg_iou']:.3f}")
        overall_table.add_row("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%")
        
        console.print(overall_table)
        
        # IoU thresholds table
        iou_table = Table(title="📈 Accuracy at IoU Thresholds", box=box.ROUNDED, show_header=True, header_style="bold blue")
        iou_table.add_column("Threshold", style="cyan", justify="center")
        iou_table.add_column("Accuracy", style="green", justify="right")
        
        for threshold, acc in sorted(metrics['iou_thresholds'].items(), key=lambda x: float(x[0].split('@')[1])):
            iou_table.add_row(threshold, f"{acc*100:.1f}%")
        
        console.print("\n")
        console.print(iou_table)
        
        # Region parent breakdown table
        decomposed = metrics.get('decomposed', {})
        if decomposed.get('by_region_parent'):
            parent_table = Table(title="🗂️ Metrics by Region Parent (6 classes)", box=box.ROUNDED, show_header=True, header_style="bold yellow")
            parent_table.add_column("Parent", style="cyan")
            parent_table.add_column("Total", style="white", justify="right")
            parent_table.add_column("Success Rate", style="green", justify="right")
            parent_table.add_column("Avg IoU", style="green", justify="right")
            parent_table.add_column("Center Acc", style="green", justify="right")
            parent_table.add_column("IoU@0.5", style="green", justify="right")
            order = ['Primary Interface Containers',
                     'Global Navigation & Structure',
                     'Content & Data Display',
                     'Interaction & Input',
                     'Contextual & Temporary Regions',
                     'Others']
            for parent in order:
                data = decomposed['by_region_parent'].get(parent, {})
                if not data:
                    continue
                parent_table.add_row(
                    parent,
                    str(data.get('total', 0)),
                    f"{data.get('success_rate', 0.0)*100:.1f}%",
                    f"{data.get('avg_iou', 0.0):.3f}",
                    f"{data.get('center_acc', 0.0)*100:.1f}%",
                    f"{data.get('iou_thresholds', {}).get('iou@0.5', 0.0)*100:.1f}%"
                )
            console.print("\n")
            console.print(parent_table)
        
        # Decomposed metrics tables
        decomposed = metrics.get('decomposed', {})

        # Density breakdown
        if decomposed.get('by_density'):
            density_table = Table(title="📊 Metrics by Density Class", box=box.ROUNDED, show_header=True, header_style="bold green")
            density_table.add_column("Density", style="cyan")
            density_table.add_column("Total", style="white", justify="right")
            density_table.add_column("Success Rate", style="green", justify="right")
            density_table.add_column("Avg IoU", style="green", justify="right")
            density_table.add_column("Center Acc", style="green", justify="right")
            density_table.add_column("IoU@0.5", style="green", justify="right")
            
            for density in ['sparse', 'medium', 'dense', 'unknown']:
                if density in decomposed['by_density']:
                    data = decomposed['by_density'][density]
                    density_table.add_row(
                        density.upper(),
                        str(data['total']),
                        f"{data['success_rate']*100:.1f}%",
                        f"{data['avg_iou']:.3f}",
                        f"{data.get('center_acc', 0.0)*100:.1f}%",
                        f"{data.get('iou_thresholds', {}).get('iou@0.5', 0.0)*100:.1f}%"
                    )
            
            console.print("\n")
            console.print(density_table)
        
        # Number of similar elements breakdown
        if decomposed.get('by_num_similar_elements'):
            num_elem_table = Table(title="🔢 Metrics by Number of Similar Elements", box=box.ROUNDED, show_header=True, header_style="bold yellow")
            num_elem_table.add_column("Num Elements", style="cyan")
            num_elem_table.add_column("Total", style="white", justify="right")
            num_elem_table.add_column("Success Rate", style="green", justify="right")
            num_elem_table.add_column("Avg IoU", style="green", justify="right")
            num_elem_table.add_column("Center Acc", style="green", justify="right")
            num_elem_table.add_column("IoU@0.5", style="green", justify="right")
            
            for category in ['1-2', '3-4', '5-6', '7+', 'unknown']:
                if category in decomposed['by_num_similar_elements']:
                    data = decomposed['by_num_similar_elements'][category]
                    num_elem_table.add_row(
                        category,
                        str(data['total']),
                        f"{data['success_rate']*100:.1f}%" if data['total'] > 0 else "N/A",
                        f"{data['avg_iou']:.3f}" if data['total'] > 0 else "N/A",
                        f"{data.get('center_acc', 0.0)*100:.1f}%" if data['total'] > 0 else "N/A",
                        f"{data.get('iou_thresholds', {}).get('iou@0.5', 0.0)*100:.1f}%" if data['total'] > 0 else "N/A"
                    )
            
            console.print("\n")
            console.print(num_elem_table)
        
        # Area class breakdown
        if decomposed.get('by_area_class'):
            area_table = Table(title="📐 Metrics by Area Class", box=box.ROUNDED, show_header=True, header_style="bold cyan")
            area_table.add_column("Area Class", style="cyan")
            area_table.add_column("Total", style="white", justify="right")
            area_table.add_column("Success Rate", style="green", justify="right")
            area_table.add_column("Avg IoU", style="green", justify="right")
            area_table.add_column("Center Acc", style="green", justify="right")
            area_table.add_column("IoU@0.5", style="green", justify="right")
            
            for area_class, data in decomposed['by_area_class'].items():
                area_table.add_row(
                    str(area_class),
                    str(data.get('total', 0)),
                    f"{data.get('success_rate', 0.0)*100:.1f}%" if data.get('total', 0) > 0 else "N/A",
                    f"{data.get('avg_iou', 0.0):.3f}" if data.get('total', 0) > 0 else "N/A",
                    f"{data.get('center_acc', 0.0)*100:.1f}%" if data.get('total', 0) > 0 else "N/A",
                    f"{data.get('iou_thresholds', {}).get('iou@0.5', 0.0)*100:.1f}%" if data.get('total', 0) > 0 else "N/A"
                )
            console.print("\n")
            console.print(area_table)
    else:
        # Fallback to simple printing if rich is not available
        debug_print(f"📊 Total Entries: {metrics['total']}", level="info")
        debug_print(f"✅ Successful: {metrics['successful']} ({metrics['success_rate']*100:.1f}%)", level="info")
        debug_print(f"📈 Average IoU: {metrics['avg_iou']:.3f}", level="info")
        debug_print(f"🎯 Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%", level="info")
        debug_print("\n📊 Accuracy at IoU Thresholds:", level="info")
        for threshold, acc in metrics['iou_thresholds'].items():
            debug_print(f"   {threshold}: {acc*100:.1f}%", level="info")
        
        if metrics.get('decomposed', {}).get('by_region_parent'):
            debug_print("\n📊 Metrics by Region Parent (6 classes):", level="info")
            for parent, data in metrics['decomposed']['by_region_parent'].items():
                debug_print(f"   {parent}: {data['success_rate']*100:.1f}% success, "
                           f"IoU={data['avg_iou']:.3f}, CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
        
        # Print decomposed metrics
        decomposed = metrics.get('decomposed', {})
        if decomposed.get('by_density'):
            debug_print("\n📊 Metrics by Density Class:", level="info")
            for density, data in decomposed['by_density'].items():
                debug_print(f"   {density.upper()}: {data['success_rate']*100:.1f}% success, "
                           f"IoU={data['avg_iou']:.3f}, CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
        
        if decomposed.get('by_num_similar_elements'):
            debug_print("\n🔢 Metrics by Number of Similar Elements:", level="info")
            for category, data in decomposed['by_num_similar_elements'].items():
                debug_print(f"   {category}: {data['success_rate']*100:.1f}% success, "
                           f"IoU={data['avg_iou']:.3f}, CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
        
        if decomposed.get('by_area_class'):
            debug_print("\n📐 Metrics by Area Class:", level="info")
            for area_class, data in decomposed['by_area_class'].items():
                debug_print(f"   {area_class.upper()}: {data['success_rate']*100:.1f}% success, "
                           f"IoU={data['avg_iou']:.3f}, CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
    
    debug_print(f"\n💾 Results saved to: {result_file}", level="info")
    if checkpoint_file != result_file:
        debug_print(f"💾 Checkpoint saved to: {checkpoint_file}", level="info")
    else:
        debug_print(f"💾 Checkpoint and results in same file: {result_file}", level="info")
    debug_print("═" * 60, level="title")

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Functional Region Grounding tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using JSON file:
  python eval_funcgnd_mp.py --questions-file /path/to/grounding_questions.json --model gpt-4o
  
  # Evaluate using HuggingFace dataset:
  python eval_funcgnd_mp.py --hf-dataset-id username/dataset-name --model gpt-4o --hf-split test
  
  # With checkpointing:
  python eval_funcgnd_mp.py --hf-dataset-id username/dataset-name --model gpt-4o --load-latest
        """
    )

    # Task selection
    parser.add_argument("--task-type", type=str, choices=['funcgnd', 'descgnd'], default='funcgnd',
                        help="Task type to evaluate (functional or description grounding)")
    # Data source - mutually exclusive: JSON vs HF dataset
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--questions-file", type=str, default=None,
                              help="Path to questions JSON file (from 2_generate_func_elemgnd_questions.py) or glob pattern")
    source_group.add_argument("--hf-dataset-id", type=str, default='HongxinLi/AutoGUIv2-FuncRegionGnd', help="HuggingFace dataset ID (e.g., 'username/dataset-name')")

    # HuggingFace specific arguments
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load from HuggingFace (default: 'test')")
    parser.add_argument("--hf-cache-dir", type=str, default='/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionGnd/',
                       help="Cache directory for HuggingFace datasets")

    # Model arguments
    parser.add_argument("--model", type=str, default=[
            'gemini-2.5-pro-thinking',
            'gpt-5',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-thinking',
            'qwen3-vl-32b-instruct',
            'qwen-vl-max-latest',
            'qwen2-vl-72b-instruct',
            'qwen3-vl-8b-instruct',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'OS-Copilot/OS-Atlas-Base-7B',
            'xlangai/OpenCUA-7B',
            'xlangai/Jedi-7B-1080p',
            'Hcompany/Holo2-8B',
            'Hcompany/Holo1.5-7B',
            'inclusionAI/UI-Venus-Ground-7B',
            'ritzzai/GUI-R1-7B',
            'InfiX-ai/InfiGUI-G1-7B',
            'step-3',
            'zai-org/GLM-4.5V'
        ][-6],
                       help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro-thinking', 'qwen3-vl-8b-instruct', 'xlangai/OpenCUA-7B', 'Hcompany/Holo2-8B')")
    parser.add_argument("--base-url", type=str, default=None,
                       help="API base URL (uses OPENAI_API_BASE env var if not provided)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses OPENAI_API_KEY env var if not provided)")

    # Processing arguments
    parser.add_argument("--max-workers", type=int, default=1,
                       help="Number of parallel workers")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-file", type=str, default=None,
                       help="Path to checkpoint file to load/save")
    parser.add_argument("--load-latest", action="store_true",
                       help="Load the latest checkpoint for this model")
    
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit the number of samples to process")
    parser.add_argument("--print-samples", type=int, default=0,
                       help="Print first N normalized entries and exit (no model calls)")
    
    args, _ = parser.parse_known_args()
    
    # Validate arguments
    if args.hf_dataset_id and not HF_AVAILABLE:
        parser.error("--hf-dataset-id requires the 'datasets' library. Install with: pip install datasets")
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)

