"""
Evaluate VLMs on Element Functionality Grounding tasks

This script evaluates vision-language models on the FuncElemGnd dataset,
where models need to locate GUI elements based on natural language questions.
"""

import os
import json
import uuid
import time
import argparse
import multiprocessing
import glob
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, Manager
from PIL import Image
from tqdm import tqdm
from utils.data_utils.misc import pred_2_point, get_image_dimensions, resize_pil_image
from utils.eval_utils.autoguiv2.misc import adjust_bbox, normalize_action_type
from utils.eval_utils.autoguiv2.prompt_lib import *

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

import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen3vl import Qwen3VL
from utils.openai_utils.doubao import DOUBAO
from utils.openai_utils.stepfun import STEPFUN
from utils.openai_utils.parasail import PARASAIL
from utils.openai_utils.huggingface import HFEndpoint
from utils.openai_utils.jedi import JEDI
from utils.openai_utils.misc import extract_step3_bounding_box

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

# Grounding prompt template
GEMINI_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Now analyze the screenshot and provide the bounding box for the target element:"""

CLAUDE_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax]. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""

# UI-Tars (https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/prompt.py)
UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>')

## User Instruction
{question}"""

# OS-ATLAS
OSATLAS_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command "{question}" (with bbox)?'

# JEDI Prompt
JEDI_PROMPT = "Please complete the following tasks via mouse click or wait: {instruction}"


# GENERIC_PROMPT
GENERIC_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""



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

    # Convert to 0-1 scale for calculation
    bbox1_norm = [bbox1[0] / 1000, bbox1[1] / 1000, bbox1[2] / 1000, bbox1[3] / 1000]
    bbox2_norm = [bbox2[0] / 1000, bbox2[1] / 1000, bbox2[2] / 1000, bbox2[3] / 1000]
    
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
                    
                    # Create entry for each action type
                    for raw_action_type, action_data in referring_expressions.items():
                        if not isinstance(action_data, dict):
                            continue

                        question = action_data.get('question', '')
                        if not question:
                            continue

                        action_type = normalize_action_type(raw_action_type)
                        entry = {
                            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_{action_type}",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'question': question,
                            'action_type': action_type,
                            'gt_bbox': target_bbox,
                            'group_index': group_index,
                            'target_elem_id': target_elem_id,
                            'density_class': density_class,
                            'num_similar_elements': num_similar_elements,
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
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Converting dataset"):
        # Handle image - can be PIL Image or path string
        image = item.get('image')
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
            image_name = original_image_name or os.path.basename(image_path)
        else:
            debug_print(f"⚠️  Unsupported image type: {type(image)}", level="warn")
            continue

        # Extract required fields - use 'question' variable for all task types
        if task_type == 'funcgnd':
            question = item.get('question', '')
        elif task_type == 'descgnd':
            question = item.get('description', '')
        elif task_type == 'intentgnd':
            question = item.get('action_intent', '')
        else:
            question = item.get('question', '')

        bbox = item.get('bbox', [])
        action_type = normalize_action_type(item.get('action_type', 'unknown'))
        group_index = item.get('group_index', -1)
        target_elem_id = item.get('target_elem_id') or item.get('id', -1)

        # Extract metadata fields
        dataset_name = item.get('dataset_name', 'unknown')
        density_class = item.get('density_class', 'unknown')
        num_similar_elements = item.get('num_similar_elements', -1)


        if not question:
            continue

        if not bbox or len(bbox) != 4:
            continue

        # Convert bbox to list of floats if needed
        bbox = [float(x) for x in bbox]

        # Create entry
        entry = {
            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_{action_type}_{idx}",
            'image_path': image_path,
            'image_name': image_name,
            'dataset_name': dataset_name,
            'question': question,
            'action_type': action_type,
            'gt_bbox': bbox,
            'group_index': group_index,
            'target_elem_id': target_elem_id,
            'density_class': density_class,
            'num_similar_elements': num_similar_elements,
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
is_oracle_mode = False


def init_worker(model_args: Dict):
    """Initialize worker with model"""
    global worker_model, is_oracle_mode

    model = model_args['model']

    # Check for oracle mode
    if model.lower() == 'oracle':
        is_oracle_mode = True
        worker_model = None
        return

    is_oracle_mode = False
    base_url = model_args['base_url']
    api_key = model_args['api_key']

    MAX_TOKENS = 8192
    if 'autoguiplus' in model.lower():
        cloud_model_class = OpenAIModel
    elif 'qwen' in model.lower():
        # Qwen models (including Qwen3-VL-8B-Instruct, qwen3-vl-32b-thinking, etc.)
        base_url = base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
        # Qwen models support longer outputs
        MAX_TOKENS = 8192
    elif any(x in model.lower() for x in ['atlas']):
        base_url = base_url or 'https://or3mlsxs95d7dgv6.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'jedi' in model.lower():
        base_url = base_url or 'https://afs3uxirrk48y8q5.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = JEDI
    elif 'uground' in model.lower():
        base_url = base_url or 'https://rs4m9o05rautq0ne.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'tars' in model.lower():
        base_url = base_url or 'https://api.parasail.io/v1'
        api_key = api_key or os.environ.get("PARASAIL_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint #PARASAIL
        MAX_TOKENS = 8192
    elif 'seed' in model.lower():
        base_url = base_url or 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
        api_key = api_key or os.environ.get("ARK_API_KEY", "EMPTY")
        cloud_model_class = DOUBAO
    elif 'step' in model.lower():
        base_url = base_url or 'https://api.stepfun.com/v1'
        api_key = api_key or os.environ.get("STEP_API_KEY", "EMPTY")
        cloud_model_class = STEPFUN
    elif 'glm' in model.lower():
        base_url = base_url or 'https://open.bigmodel.cn/api/paas/v4'
        api_key = api_key or os.environ.get("ZAI_API_KEY", "EMPTY")
        cloud_model_class = OpenAIModel
    else:
        cloud_model_class = OpenAIModel
        base_url = base_url or os.environ.get("OPENAI_API_BASE_XIAOAI", "EMPTY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY_XIAOAI", "EMPTY")

    # cloud_model_class = OpenAIModel
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
    global worker_model, is_oracle_mode

    entry_id = entry['entry_id']
    image_path = entry['image_path']

    # Get the appropriate field based on task type
    question = entry.get('question', '')

    gt_bbox = entry['gt_bbox']

    if any(p > 1 for p in gt_bbox):
        gt_bbox = [x / 1000 for x in gt_bbox]

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
        'action_type': normalize_action_type(entry.get('action_type', 'unknown')),
    }

    start_time = time.time()

    # Oracle mode: directly use gt_bbox as pred_bbox
    if is_oracle_mode:
        # Load image to verify it exists
        try:
            img = Image.open(image_path)
            img.close()
        except Exception as e:
            result['error'] = f"Failed to load image: {e}"
            result['processing_time'] = time.time() - start_time
            return result

        # Oracle returns gt_bbox as prediction
        result['pred_bbox'] = gt_bbox
        result['iou'] = 1.0
        result['center_acc'] = True
        result['inference_done'] = True
        result['response'] = f"[ORACLE] gt_bbox={gt_bbox}"
        result['processing_time'] = time.time() - start_time

        # Log oracle result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
        print(f"[Worker {worker_id}] ✅ ORACLE | Entry: {entry_id} | Image: {image_name_short} | "
              f"GT/Pred: {gt_bbox} | IoU=1.000 | CenterAcc=True | [{timestamp}]")
        return result

    if not question:
        result['error'] = "No question found in entry"
        result['processing_time'] = time.time() - start_time
        return result

    # Create prompt based on task type and model
    ref_tag = REF_TAGS.get(task_type, 'a question about locating')
    ref_placeholder = REF_PLACEHOLDER.get(task_type, 'Question')
    temp_img_path = image_path
    orig_W, orig_H = get_image_dimensions(image_path)
    
    is_resized = False
    
    system_prompt = ''
    if 'autoguiplus' in worker_model.model.lower():
        prompt = make_autoguiplus_prompt(question, worker_model.model.lower())
    elif 'gemini' in worker_model.model.lower():
        prompt = GEMINI_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
        
    elif 'tars' in worker_model.model.lower():
        prompt = UI_TARS_PROMPT.format(question=question)
    elif 'atlas' in worker_model.model.lower():
        prompt = OSATLAS_PROMPT.format(question=question)
    elif 'uground' in worker_model.model.lower():
        prompt = UGROUND_PROMPT.format(instruction=question)
    elif 'jedi' in worker_model.model.lower():
        is_resized = True
        prompt = JEDI_PROMPT.format(instruction=('Click the element specified by this instruction: ' + question) if task_type in ['funcgnd', 'descgnd'] else question)
        temp_img_path = f'jedi_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
        image = Image.open(image_path).convert("RGB")
        resized_image, _ = resize_pil_image(image, max_size=1080)
        resized_image.save(temp_img_path)
    elif 'holo' in worker_model.model.lower():
        prompt = HOLO_PROMPT + f"\nYour target is: {question}" # HOLO_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question) # 
    elif 'opencua' in worker_model.model.lower():
        system_prompt = OPENCUA_SYSPROMPT
        prompt = f"{question} Click on the target element"
    elif 'infigui-g1' in worker_model.model.lower():
        system_prompt = INFIGUIG1_SYSPROMPT
        prompt = INFIGUIG1_PROMPT.format(new_width=orig_W, new_height=orig_H, instruction=question)
    elif 'gui-r1' in worker_model.model.lower():
        prompt = GUIR1_PROMPT.replace('{instruction}', question) + ' Click on the target element'
    elif 'venus' in worker_model.model.lower():
        prompt = UIVENUS_PROMPT.format(instruction=question) + ' Click on the target element'
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

    image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))


    retry = 0
    while retry < 4:
        try:
            retry += 1

            # Pre-logging: Show query start information
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
                    image_first=any(x in worker_model.model.lower() for x in ['autoguiplus', 'opencua', 'holo', 'infigui-g1', 'gui-r1']),
                    sys_prompt=system_prompt
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

                # time.sleep(retry * 3)
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
            elif 'step' in worker_model.model.lower():
                bbox_str_proc = extract_step3_bounding_box(bbox_str)
                if bbox_str_proc is None:
                    1+1
                # bbox_str_proc = bbox_str.split('**')[-1].split('**')[0] if '**' in bbox_str else bbox_str
                raw_pred_bbox = pred_2_point(bbox_str_proc, scale=scale)
            elif 'tars' in worker_model.model.lower():
                pass
            # Case 5: OS-Atlas: <|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|><|im_end|>
            # In reality, the special tokens have been removed: 'close button(744,381),(756,400)' -> '(744,381),(756,400)'
            elif 'atlas' in worker_model.model.lower():
                if '(' in bbox_str and ')' in bbox_str:
                    bbox_str = bbox_str[bbox_str.find('(', 0, bbox_str.rfind('(')):]
                elif '[' in bbox_str and ']' in bbox_str:
                    bbox_str = bbox_str[bbox_str.rfind('['):]
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale)
            # Case 6: JEDI: '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [453, 258]}}\n</tool_call>'
            elif 'jedi' in worker_model.model.lower():
                bbox_str = bbox_str[bbox_str.rfind('['):bbox_str.rfind(']')+1]
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
            # Case 7: Holo: '{"action": "click_absolute", "x": 1043, "y": 502}'
            elif 'holo' in worker_model.model.lower():
                act_dict = json.loads(bbox_str)
                raw_pred_bbox = [act_dict['x'], act_dict['y']] # absolute coords
            # Case 8: OpenCUA: '## Code:\n```python\npyautogui.click(x=3167, y=360)\n```\n'
            elif 'opencua' in worker_model.model.lower():
                raw_pred_bbox = [
                    int(bbox_str.split('x=')[1].split(',')[0]),
                    int(bbox_str.split('y=')[1].split(')')[0])
                ]
            # Case 9: InfiGUI-G1: ''[{"point_2d": [1007, 924], "label": "UI element for \\"Go Here\\""}, {"point_2d": [1021, 74], "label": "UI element for \\"Go Here\\""}]''
            elif 'infigui-g1' in worker_model.model.lower():
                points = json.loads(bbox_str)
                raw_pred_bbox = points[0]['point_2d']
            # Case 10: GUI-R1: "<answer>[{'action': 'click', 'point': [2200, 354], 'input_text': 'no input text'}]</answer>"
            elif 'gui-r1' in worker_model.model.lower():
                act_dict = eval(bbox_str[bbox_str.find("{'action"):bbox_str.find('}]<')+1])
                raw_pred_bbox = act_dict['point']
            elif '```json' in bbox_str or bbox_str.startswith('{') and bbox_str.endswith('}'):
                    bbox_cleaned = bbox_str.split('```json')[1].split('```')[0].strip() if '```json' in bbox_str else bbox_str
                    bbox_parsed = json.loads(bbox_cleaned)

                    if isinstance(bbox_parsed, list):
                        item = bbox_parsed[0]
                        if isinstance(item, dict):
                            if 'box_2d' in item:
                                raw_pred_bbox = item['box_2d']
                        elif isinstance(item, list):
                            raw_pred_bbox = item
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
            if len(pred_bbox) == 4:
                center = [(pred_bbox[0]+pred_bbox[2])/2, (pred_bbox[1]+pred_bbox[3])/2]
            else:
                center = pred_bbox

            center_acc = gt_bbox[0] <= center[0] <= gt_bbox[2] and gt_bbox[1] <= center[1] <= gt_bbox[3]
            result['gt_bbox_norm'] = gt_bbox
            result['center_acc'] = center_acc
            result['inference_done'] = True

            # Only print "Query COMPLETE" if we actually got a valid bbox
            status = "✅" if center_acc else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | GT: {gt_bbox} <=> Pred: {bbox_str} -> {pred_bbox} -> {center} | "
                  f"IoU={iou:.3f} | CenterAcc={center_acc} | Time: {processing_time:.2f}s | [{timestamp}]")
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
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
    if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'jedi', 'holo1.5', 'opencua', 'infigui-g1', 'gui-r1', 'venus']): # Claude-Sonnet-4.5
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

    pool = Pool(processes=args.max_workers, initializer=init_worker, initargs=(model_args,))
    try:
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
    finally:
        pool.close()
        pool.join()

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

    # Action type breakdown
    action_types = {}
    for r in results:
        if not r.get('inference_done', False):
            continue
        action_type = r.get('action_type', 'unknown')
        if action_type == 'unknown':
            # Fallback to extracting from entry_id
            entry_id = r.get('entry_id', '')
            action_type = entry_id.split('_')[-1] if '_' in entry_id else 'unknown'
        if action_type not in action_types:
            action_types[action_type] = {'total': 0, 'successful': 0, 'ious': [], 'center_accs': []}
        action_types[action_type]['total'] += 1
        action_types[action_type]['successful'] += 1
        if 'iou' in r:
            action_types[action_type]['ious'].append(r['iou'])
        if 'center_acc' in r:
            action_types[action_type]['center_accs'].append(r['center_acc'])

    for action_type in action_types:
        data = action_types[action_type]
        data['success_rate'] = data['successful'] / data['total'] if data['total'] > 0 else 0.0
        data['avg_iou'] = sum(data['ious']) / len(data['ious']) if data['ious'] else 0.0
        data['center_acc'] = sum(data['center_accs']) / len(data['center_accs']) if data['center_accs'] else 0.0
        del data['ious']
        del data['center_accs']

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
        'action_types': action_types,
        'decomposed': {
            'by_density': density_breakdown,
            'by_num_similar_elements': num_elements_breakdown,
        }
    }


def main(args):
    """Main evaluation function"""
    debug_print("═" * 60, level="title")
    debug_print("🔍 Element Functionality Grounding Evaluation", level="title")
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
        
        # Action type breakdown table
        if metrics.get('action_types'):
            action_table = Table(title="🎯 Action Type Breakdown", box=box.ROUNDED, show_header=True, header_style="bold yellow")
            action_table.add_column("Action Type", style="cyan")
            action_table.add_column("Total", style="white", justify="right")
            action_table.add_column("Success Rate", style="green", justify="right")
            action_table.add_column("Avg IoU", style="green", justify="right")
            action_table.add_column("Center Acc", style="green", justify="right")
            
            for action_type, data in sorted(metrics['action_types'].items()):
                action_table.add_row(
                    action_type,
                    str(data['total']),
                    f"{data['success_rate']*100:.1f}%",
                    f"{data['avg_iou']:.3f}",
                    f"{data.get('center_acc', 0.0)*100:.1f}%"
                )
            
            console.print("\n")
            console.print(action_table)
        
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
    else:
        # Fallback to simple printing if rich is not available
        debug_print(f"📊 Total Entries: {metrics['total']}", level="info")
        debug_print(f"✅ Successful: {metrics['successful']} ({metrics['success_rate']*100:.1f}%)", level="info")
        debug_print(f"📈 Average IoU: {metrics['avg_iou']:.3f}", level="info")
        debug_print(f"🎯 Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%", level="info")
        debug_print("\n📊 Accuracy at IoU Thresholds:", level="info")
        for threshold, acc in metrics['iou_thresholds'].items():
            debug_print(f"   {threshold}: {acc*100:.1f}%", level="info")
        
        if metrics.get('action_types'):
            debug_print("\n📊 Action Type Breakdown:", level="info")
            for action_type, data in metrics['action_types'].items():
                debug_print(f"   {action_type}: {data['success_rate']*100:.1f}% success, "
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
    
    debug_print(f"\n💾 Results saved to: {result_file}", level="info")
    if checkpoint_file != result_file:
        debug_print(f"💾 Checkpoint saved to: {checkpoint_file}", level="info")
    else:
        debug_print(f"💾 Checkpoint and results in same file: {result_file}", level="info")
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Element Functionality Grounding tasks",
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

    # Input arguments - mutually exclusive (not required; defaults to HF dataset)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--task-type", type=str, default=['funcgnd', 'descgnd', 'intentgnd'][0],
                            help="Task type to evaluate")
    
    input_group.add_argument("--questions-file", type=str, default=None,
                            help="Path to questions JSON file (from 2_generate_func_elemgnd_questions.py) or glob pattern")
    input_group.add_argument("--hf-dataset-id", type=str, default='HongxinLi/AutoGUIv2-FuncElemGnd', help="HuggingFace dataset ID (e.g., 'username/dataset-name')")

    # HuggingFace specific arguments
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load from HuggingFace (default: 'test')")
    parser.add_argument("--hf-cache-dir", type=str, default='/volume/pt-coder/users/gji/data/gui_data/AutoGUIv2/hf_dataset_cache/FuncElemGnd/',
                       help="Cache directory for HuggingFace datasets")

    # Model arguments
    parser.add_argument("--model", type=str, default=[
            'gemini-2.5-pro-thinking',
            'gemini-2.5-pro',
            'gpt-5',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-instruct',
            'qwen3-vl-8b-instruct',
            'Qwen/Qwen2-VL-72B-Instruct',
            'Qwen/Qwen3-VL-8B-Instruct',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'OS-Copilot/OS-Atlas-Base-7B',
            'xlangai/Jedi-7B-1080p',
            'step-3',
            'zai-org/GLM-4.5V',
            'osunlp/UGround-V1-7B',
            'xlangai/OpenCUA-7B',
            'oracle'
        ][2],
                       help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro-thinking', 'Qwen/Qwen3-VL-8B-Instruct')")
    parser.add_argument("--base-url", type=str, default=None,
                       help="API base URL (uses OPENAI_API_BASE env var if not provided)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses OPENAI_API_KEY env var if not provided)")

    # Processing arguments
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel workers")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-file", type=str, default=None,
                       help="Path to checkpoint file to load/save")
    parser.add_argument("--load-latest", action="store_true",
                       help="Load the latest checkpoint for this model")
    
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit the number of samples to process")
    
    args, _ = parser.parse_known_args()
    
    # Validate arguments
    if args.hf_dataset_id and not HF_AVAILABLE:
        parser.error("--hf-dataset-id requires the 'datasets' library. Install with: pip install datasets")
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)

