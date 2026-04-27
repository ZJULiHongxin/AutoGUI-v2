"""
Evaluate VLMs on Element Functionality Grounding tasks

This script evaluates vision-language models on the FuncElemGnd dataset,
where models need to locate GUI elements based on natural language questions.
"""

import os
import hashlib
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
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, Manager
from functools import partial
from PIL import Image
from tqdm import tqdm
from utils.data_utils.misc import clip_coords, pred_2_point, get_image_dimensions, resize_pil_image
from utils.eval_utils.autoguiv2.misc import adjust_bbox
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
from utils.openai_utils.jedi import JEDI
from utils.openai_utils.parasail import PARASAIL
from utils.openai_utils.huggingface import HFEndpoint
from utils.openai_utils.intern import INTERN

# Grounding prompt template (https://github.com/xlang-ai/OSWorld-G/blob/main/evaluation/gemini_osworld_g.py)
GEMINI_GROUNDING_PROMPT = """Point to the element corresponding to the instruction: {action_intent}

The answer should follow the json format: [{{"point": <point>, "label": <label1>}}, ...] with no more than 1 items.
The points are in [y, x] format normalized to 0-1000.
"""

CLAUDE_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and an action intent, you need to identify the center coordinate of the target element, which should be [x_center, y_center] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

Action intent: {action_intent}

Output format:
Center: [x_center, y_center]

Now provide your answer:"""

# Qwen25_Prompt
QWEN3_PROMPT = "Point to the element corresponding to the instruction: {action_intent}. Output its coordinates in XML format <points x y>object</points>"

# Qwen3_Prompt
QWEN3_PROMPT = "Point to the element corresponding to the instruction: {action_intent}. Output its coordinates in XML format <points x y>object</points>"

# GENERIC_PROMPT
GENERIC_PROMPT = """You are a GUI expert. Given a screenshot and an action intent, you need to identify the center coordinate of the target element, which should be [x_center, y_center] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

Action intent: {action_intent}

Output format:
Center: [x_center, y_center]

Now provide your answer:"""



UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>'')

## User Instruction
{action_intent}"""

JEDI_PROMPT = "Please complete the following tasks via mouse click or wait: {instruction}"


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
                    target_bbox = target_element.get('revised bbox', [])

                    # Create entry for each action type
                    for action_type, action_data in referring_expressions.items():
                        if not isinstance(action_data, dict):
                            continue
                        
                        question = action_data.get('question', '')
                        if not question:
                            continue
                        
                        entry = {
                            'entry_id': f"{image_name}",
                            'func': func,
                            'gt_bbox': target_bbox
                        }
                        all_entries.append(entry)
    
    debug_print(f"✅ Loaded {len(all_entries)} entries from JSON", level="success")
    return all_entries


def get_hf_cache_path(hf_dataset_id: str, split: str) -> tuple:
    """Get cache file path and image cache directory for HuggingFace dataset conversion
    
    Args:
        hf_dataset_id: HuggingFace dataset ID
        split: Dataset split
    Returns:
        Tuple of (cache_file_path, image_cache_dir)
    """
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use shared cache directory for all task types (images are shared)
    cache_dir = os.path.join(script_dir, 'osworldg_hf_dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of dataset_id and split for cache (images are shared across task types)
    cache_key = f"{hf_dataset_id}_{split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # JSON cache file is task-specific (different questions/descriptions/intents)
    cache_filename = f"{cache_hash}.json"
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
                             hf_dataset_id: str, split: str):
    """Save converted entries to cache
    
    Args:
        entries: List of converted entries
        cache_path: Path to cache file
        hf_dataset_id: HuggingFace dataset ID (for metadata)
        split: Dataset split (for metadata)
    """
    cache_data = {
        'metadata': {
            'hf_dataset_id': hf_dataset_id,
            'split': split,
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


def load_dataset_from_hf(hf_dataset_id: str, split: str = 'test', cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load dataset from HuggingFace Hub with caching (including PIL Images)
    
    Args:
        hf_dataset_id: HuggingFace dataset ID (e.g., 'username/dataset-name')
        split: Dataset split to load (default: 'test')
        cache_dir: Optional cache directory for downloaded datasets (HF library cache)
    Returns:
        List of dataset entries
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset loading. Install with: pip install datasets")
    
    # Check cache first
    cache_path, image_cache_dir = get_hf_cache_path(hf_dataset_id, split)
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
        image_name = item['id']
        original_image_name = image_name

        # If image is a PIL Image, save it to persistent cache directory
        if isinstance(image, Image.Image):
            # Sanitize image name for filesystem
            if not image_name.endswith(('.png', '.jpg', '.jpeg')):
                image_name += '.png'
            
            image_path = os.path.join(image_cache_dir, image_name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            # Handle duplicate names
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

        action_intent = item['instruction']

        # Create entry
        x1, y1, w, h = item['box_coordinates']
        x2, y2 = x1 + w, y1 + h
        if item['box_type'] != 'bbox':
            1+1

        entry = {
            'entry_id': f"{image_name}",
            'image_path': image_path,
            'action_intent': action_intent,
            'gt_bbox': [x1, y1, x2, y2],
            'box_type': item['box_type']
        }
        
        # if False:
        #     img = cv2.imread(image_path)
        #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        #     cv2.imwrite('test.png', img)

        all_entries.append(entry)
    
    debug_print(f"✅ Converted {len(all_entries)} entries from HuggingFace dataset", level="success")
    
    # Save to cache
    save_hf_entries_to_cache(all_entries, cache_path, hf_dataset_id, split)
    
    debug_print(f"💾 Cached {len(all_entries)} entries and images to persistent cache", level="success")
    
    return all_entries


def load_evaluation_dataset(source: str, questions_file: Optional[str] = None, hf_dataset_id: Optional[str] = None, hf_split: str = 'test', hf_cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load dataset from either JSON file or HuggingFace Hub
    
    Args:
        source: Source type - 'json' or 'hf'
        questions_file: Path to questions JSON file (required if source='json')
        hf_dataset_id: HuggingFace dataset ID (required if source='hf')
        hf_split: Dataset split for HF datasets (default: 'test')
        hf_cache_dir: Optional cache directory for HF datasets
    Returns:
        List of dataset entries
    """
    if source == 'hf':
        if not hf_dataset_id:
            raise ValueError("hf_dataset_id is required when source='hf'")
        return load_dataset_from_hf(hf_dataset_id, hf_split, hf_cache_dir)
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
        # Only entries with inference_done=True and valid pred_center should be considered successfully processed
        successful_ids = set()
        failed_ids = set()

        for entry_id in processed_ids:
            result = results.get(entry_id)
            
            if result:
                # Check if inference was successful
                if result.get('inference_done', False) and result.get('pred_center') is not None:
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
    max_tokens = 8192

    if 'autoguiplus' in model.lower():
        cloud_model_class = OpenAIModel
    elif 'qwen' in model.lower():
        base_url = base_url or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
    elif 'tars' in model.lower():
        base_url = 'https://api.parasail.io/v1'
        api_key = api_key or os.environ.get("PARASAIL_API_KEY", "EMPTY")
        cloud_model_class = PARASAIL
        max_tokens = 2048
    elif 'jedi' in model.lower():
        base_url = 'https://afs3uxirrk48y8q5.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = JEDI
    elif 'uground' in model.lower():
        base_url = base_url or 'https://rs4m9o05rautq0ne.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'seed' in model.lower():
        base_url = 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'
        api_key = api_key or os.environ.get("ARK_API_KEY", "EMPTY")
        cloud_model_class = DOUBAO
    elif 'step' in model.lower():
        base_url = 'https://api.stepfun.com/v1'
        api_key = api_key or os.environ.get("STEP_API_KEY", "EMPTY")
        cloud_model_class = STEPFUN
    elif 'glm' in model.lower():
        base_url = 'https://api.siliconflow.cn/v1'
        api_key = api_key or os.environ.get("SILICON_API_KEY", "EMPTY")
        cloud_model_class = OpenAIModel
    elif 'internvl3' in model.lower():
        base_url = 'https://chat.intern-ai.org.cn/v1/messages'
        api_key = api_key or os.environ.get("INTERN_API_KEY", "EMPTY")
        cloud_model_class = INTERN
    else:
        base_url = base_url or os.environ.get("OPENAI_API_BASE_XIAOAI", "EMPTY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY_XIAOAI", "EMPTY")
        cloud_model_class = OpenAIModel

    cloud_model_class = OpenAIModel

    worker_model = cloud_model_class(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens
    )

def process_entry(entry: Dict, worker_id: int = 0, scale: int = 1000) -> Dict[str, Any]:
    """Process a single dataset entry
    
    Args:
        entry: Dataset entry dictionary
        worker_id: Worker ID for logging
        scale: Scale for bbox normalization (default: 1000)
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    
    gt_bbox = entry['gt_bbox']


    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'gt_bbox': gt_bbox,
        'pred_center': None,
        'center_acc': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        'elem_role': entry.get('elem_role', 'unknown'),
        'device': entry.get('device', 'unknown'),
    }

    # Calculate center accuracy
    W, H = get_image_dimensions(image_path)
    if any(p > 1 for p in gt_bbox):
        gt_bbox = [gt_bbox[0] / W, gt_bbox[1] / H, gt_bbox[2] / W, gt_bbox[3] / H]
                
    start_time = time.time()

    retry = 0
    while retry < 4:
        temp_img_path = image_path
        orig_W, orig_H = get_image_dimensions(image_path)

        is_resized = False
        try:
            retry += 1
            # Create prompt based on task type and model
            action_intent = entry.get('action_intent', '')
            system_prompt = ''

            if 'autoguiplus' in worker_model.model.lower():
                prompt = make_autoguiplus_prompt(action_intent, worker_model.model.lower())
            elif 'gemini' in worker_model.model.lower():
                prompt = GEMINI_GROUNDING_PROMPT.format(action_intent=action_intent)
            elif 'uground' in worker_model.model.lower():
                prompt = UGROUND_PROMPT.format(instruction=action_intent)
            elif 'qwen3' in worker_model.model.lower():
                prompt = QWEN3_PROMPT.format(action_intent=action_intent.strip('. '))
            elif 'tars' in worker_model.model.lower():
                prompt = UI_TARS_PROMPT.format(action_intent=action_intent)
            elif 'jedi' in worker_model.model.lower():
                prompt = JEDI_PROMPT.format(instruction=action_intent)
                is_resized = True
                temp_img_path = f"temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png"
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=1080)
                resized_image.save(temp_img_path)
            elif 'holo' in worker_model.model.lower():
                prompt = HOLO_PROMPT + f"\nYour target is: {action_intent}" # HOLO_BBOX_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question) # 
            elif 'opencua' in worker_model.model.lower():
                system_prompt = OPENCUA_SYSPROMPT
                prompt = f"{action_intent} Click on the target element"
            elif 'infigui-g1' in worker_model.model.lower():
                system_prompt = INFIGUIG1_SYSPROMPT
                prompt = INFIGUIG1_PROMPT.format(new_width=orig_W, new_height=orig_H, instruction=action_intent)
            elif 'gui-r1' in worker_model.model.lower():
                prompt = GUIR1_PROMPT.replace('{instruction}', action_intent) + ' Click on the target element'
            elif 'venus' in worker_model.model.lower():
                prompt = UIVENUS_PROMPT.format(instruction=action_intent) + ' Click on the target element'
            elif any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                prompt = CLAUDE_GROUNDING_PROMPT.format(action_intent=action_intent)
            else:
                prompt = GENERIC_PROMPT.format(action_intent=action_intent)

            W, H = get_image_dimensions(temp_img_path)

            # Get model response
            success, response, _ = worker_model.get_model_response(
                prompt, 
                [temp_img_path], 
                use_img_url=True, 
                temperature=0.0, 
                timeout=360,
                image_first=any(x in worker_model.model.lower() for x in ['autoguiplus', 'opencua', 'holo', 'infigui-g1', 'gui-r1', 'uground']),

            )

            if not success:
                result['error'] = f"API call failed: {response}"
                result['processing_time'] = time.time() - start_time
                return result

            result['prompt'], result['response'] = prompt, response

            # Parse bbox from response
            if '<think>' in response:
                thinking = response.split('<think>')[1].split('</think>')[0].strip()
                bbox_str = response.split('</think>')[1].strip()
            else:
                thinking, bbox_str = '', response.strip()

            result['thinking'], result['bbox_pred_str'] = thinking, bbox_str

            # Parsing
            raw_pred_center = None
            # Case 1: Gemini
            if 'gemini' in worker_model.model.lower() and ('point"' in bbox_str or "point'" in bbox_str):
                bbox_cleaned = bbox_str.replace('```json','').replace('```','')
                bbox_parsed = json.loads(bbox_cleaned)
                
                if isinstance(bbox_parsed, list):
                    bbox_parsed = bbox_parsed[0]

                raw_pred_center = bbox_parsed['point']
                raw_pred_center = pred_2_point(raw_pred_center, scale=scale)
            # Case 2: GLM-4.5. "The bounding box for the 'wall_95' entry in the Outliner list is <|begin_of_box|>[816, 162, 838, 172]<|end_of_box|>."
            elif '<|begin_of_box|>' in bbox_str:
                raw_pred_center = bbox_str.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0].strip()
                raw_pred_center = pred_2_point(raw_pred_center, scale=scale)
            # Case 3: Qwen3-VL. '<points x1="745" y1="312"></points>'
            elif '<point' in bbox_str:
                raw_pred_center = int(bbox_str.split('="')[1].split('"')[0].strip()), int(bbox_str.split('="')[-1].split('"')[0].strip())
                raw_pred_center = pred_2_point(raw_pred_center, scale=scale)
            elif 'tars' in worker_model.model.lower():
                pass
            elif 'jedi' in worker_model.model.lower() and '[' in bbox_str and ']' in bbox_str:
                coord_str = bbox_str[bbox_str.rfind('['):bbox_str.rfind(']')+1]
                raw_pred_center = pred_2_point(coord_str, keep_box=False, scale=scale, w=W, h=H)
            elif 'holo' in worker_model.model.lower():
                act_dict = json.loads(bbox_str)
                raw_pred_center = [act_dict['x'], act_dict['y']] # absolute coords
            # Case 8: OpenCUA: '## Code:\n```python\npyautogui.click(x=3167, y=360)\n```\n'
            elif 'opencua' in worker_model.model.lower():
                raw_pred_center = [
                    int(bbox_str.split('x=')[1].split(',')[0]),
                    int(bbox_str.split('y=')[1].split(')')[0])
                ]
            # Case 9: InfiGUI-G1: ''[{"point_2d": [1007, 924], "label": "UI element for \\"Go Here\\""}, {"point_2d": [1021, 74], "label": "UI element for \\"Go Here\\""}]''
            elif 'infigui-g1' in worker_model.model.lower():
                bbox_str_proc = bbox_str.split('```json')[1].split('```')[0].strip() if '```json' in bbox_str else bbox_str
                
                bbox_str_proc2 = '[' + bbox_str_proc[bbox_str_proc.rfind('{"'):bbox_str_proc.rfind(']')+1].strip('` ')
                try:
                    points = json.loads(bbox_str_proc2)
                    raw_pred_center = clip_coords(pred_2_point(points[0]['point_2d'], scale=scale, w=W, h=H))
                except Exception as e:
                    raw_pred_center = None
                    
            # Case 10: GUI-R1: "<answer>[{'action': 'click', 'point': [2200, 354], 'input_text': 'no input text'}]</answer>"
            elif 'gui-r1' in worker_model.model.lower():
                act_dict = eval(bbox_str[bbox_str.find("{'action"):bbox_str.find('}]<')+1])
                raw_pred_center = act_dict['point']
                raw_pred_center = pred_2_point(raw_pred_center, scale=scale, w=W, h=H)
            elif '```json' in bbox_str or bbox_str.startswith('{') and bbox_str.endswith('}'):
                    point_cleaned = bbox_str.split('```json')[1].split('```')[0].strip() if '```json' in bbox_str else bbox_str
                    point_parsed = json.loads(point_cleaned)

                    if isinstance(point_parsed, list):
                        item = point_parsed[0]
                        if isinstance(item, dict):
                            if 'point_2d' in item:
                                raw_pred_center = item['point_2d']
                        elif isinstance(item, list):
                            raw_pred_center = item
                        else:
                            raw_pred_center = point_parsed
                    elif isinstance(point_parsed, dict):
                        if 'point_2d' in point_parsed:
                            raw_pred_center = point_parsed['point_2d']
                    
                    if any([p > 1 for p in raw_pred_center]):
                        raw_pred_center = pred_2_point(raw_pred_center, scale=scale, w=W, h=H)

            # Fall back to a general parsing method
            if raw_pred_center is None or any(p > 1 for p in raw_pred_center):
                raw_pred_center = pred_2_point(bbox_str, scale=scale, w=W, h=H)

            # Adjust bbox format according to the model preferece
            pred_center = adjust_bbox(worker_model.model, raw_pred_center)

            if pred_center is None:
                continue

            result['pred_center'] = pred_center


            center_acc = gt_bbox[0] <= pred_center[0] <= gt_bbox[2] and gt_bbox[1] <= pred_center[1] <= gt_bbox[3]
            result['gt_bbox_norm'] = gt_bbox
            result['center_acc'] = center_acc
            result['inference_done'] = True

            # Log result
            status = "✅" if center_acc else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} GT: {gt_bbox} <=> Pred: {bbox_str} -> {pred_center} | CenterAcc={center_acc} | {image_path.split('images/')[-1]}  [{processing_time:.2f}s] [{timestamp}]")
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
        finally:
            if is_resized:
                try:
                    os.remove(temp_img_path)
                except Exception:
                    pass
    else:
        result['error'] = "Failed to parse bbox from response"

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
    if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'jedi', 'holo1.5', 'opencua', 'infigui-g1', 'gui-r1', 'venus']): # Claude-Sonnet-4.5
        scale = -1
    else:
        scale = 1000

    # Filter entries to process
    entries_to_process = [
        (entry, i % args.max_workers, scale) 
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
                if not result.get('inference_done', False) or result.get('pred_center') is None:
                    retry_count += 1
    elif isinstance(checkpoint_results, list):
        # Handle list format (from full result files)
        results_by_id = {r.get('entry_id'): r for r in checkpoint_results if 'entry_id' in r}
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in results_by_id:
                result = results_by_id[entry_id]
                if not result.get('inference_done', False) or result.get('pred_center') is None:
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
                if result.get('inference_done', False) and result.get('pred_center') is not None:
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
            'center_acc': 0.0
        }
    
    center_accs = [r.get('center_acc', False) for r in subset_results if r.get('inference_done', False) and 'center_acc' in r]
    center_acc = sum(center_accs) / len(center_accs) if center_accs else 0.0
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': successful / total if total > 0 else 0.0,
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
            'center_acc': 0.0,
            'decomposed': {}
        }
    
    # Calculate center accuracy
    center_accs = [r.get('center_acc', False) for r in results if r.get('inference_done', False) and 'center_acc' in r]
    center_acc = sum(center_accs) / len(center_accs) if center_accs else 0.0

    # Decomposed metrics by elem_role
    elem_role_breakdown = {}
    # Collect all unique elem_roles from results
    all_elem_roles = set(r.get('elem_role', 'unknown') for r in results)
    for elem_role in sorted(all_elem_roles):
        subset = [r for r in results if r.get('elem_role', 'unknown') == elem_role]
        if subset:
            elem_role_breakdown[elem_role] = calculate_metrics_for_subset(subset)
    
    # Decomposed metrics by device
    device_breakdown = {}
    # Collect all unique devices from results
    all_devices = set(r.get('device', 'unknown') for r in results)
    for device in sorted(all_devices):
        subset = [r for r in results if r.get('device', 'unknown') == device]
        if subset:
            device_breakdown[device] = calculate_metrics_for_subset(subset)
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': successful / total if total > 0 else 0.0,
        'center_acc': center_acc,
        'decomposed': {
            'by_elem_role': elem_role_breakdown,
            'by_device': device_breakdown,
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
                              hf_split=args.hf_split, hf_cache_dir=args.hf_cache_dir)
    else:
        entries = load_evaluation_dataset('json', questions_file=args.questions_file)
    
    if not entries:
        debug_print("❌ No entries found in dataset", level="error")
        return
    
    # Setup checkpoint and result file paths
    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results', 'osworldg')
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
        overall_table.add_row("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%")
        
        console.print(overall_table)
        
        console.print("\n")
        
        # Decomposed metrics tables
        decomposed = metrics.get('decomposed', {})

        # Element role breakdown
        if decomposed.get('by_elem_role'):
            elem_role_table = Table(title="🎯 Metrics by Element Role", box=box.ROUNDED, show_header=True, header_style="bold yellow")
            elem_role_table.add_column("Element Role", style="cyan")
            elem_role_table.add_column("Total", style="white", justify="right")
            elem_role_table.add_column("Success Rate", style="green", justify="right")
            elem_role_table.add_column("Center Acc", style="green", justify="right")
            
            for elem_role in sorted(decomposed['by_elem_role'].keys()):
                data = decomposed['by_elem_role'][elem_role]
                elem_role_table.add_row(
                    str(elem_role),
                    str(data['total']),
                    f"{data['success_rate']*100:.1f}%",
                    f"{data.get('center_acc', 0.0)*100:.1f}%"
                )
            
            console.print("\n")
            console.print(elem_role_table)
        
        # Device breakdown
        if decomposed.get('by_device'):
            device_table = Table(title="📱 Metrics by Device", box=box.ROUNDED, show_header=True, header_style="bold green")
            device_table.add_column("Device", style="cyan")
            device_table.add_column("Total", style="white", justify="right")
            device_table.add_column("Success Rate", style="green", justify="right")
            device_table.add_column("Center Acc", style="green", justify="right")
            
            for device in sorted(decomposed['by_device'].keys()):
                data = decomposed['by_device'][device]
                device_table.add_row(
                    str(device),
                    str(data['total']),
                    f"{data['success_rate']*100:.1f}%",
                    f"{data.get('center_acc', 0.0)*100:.1f}%"
                )
            
            console.print("\n")
            console.print(device_table)
    else:
        # Fallback to simple printing if rich is not available
        debug_print(f"📊 Total Entries: {metrics['total']}", level="info")
        debug_print(f"✅ Successful: {metrics['successful']} ({metrics['success_rate']*100:.1f}%)", level="info")
        debug_print(f"🎯 Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%", level="info")
        
        # Print decomposed metrics
        decomposed = metrics.get('decomposed', {})
        if decomposed.get('by_elem_role'):
            debug_print("\n🎯 Metrics by Element Role:", level="info")
            for elem_role, data in sorted(decomposed['by_elem_role'].items()):
                debug_print(f"   {elem_role}: {data['success_rate']*100:.1f}% success, "
                           f"CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
        
        if decomposed.get('by_device'):
            debug_print("\n📱 Metrics by Device:", level="info")
            for device, data in sorted(decomposed['by_device'].items()):
                debug_print(f"   {device}: {data['success_rate']*100:.1f}% success, "
                           f"CenterAcc={data.get('center_acc', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
    
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
    input_group.add_argument("--questions-file", type=str, default=None,
                            help="Path to questions JSON file (from 2_generate_func_elemgnd_questions.py) or glob pattern")
    input_group.add_argument("--hf-dataset-id", type=str, default='MMInstruction/OSWorld-G', help="HuggingFace dataset ID (e.g., 'username/dataset-name')")

    # HuggingFace specific arguments
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load from HuggingFace (default: 'test')")
    parser.add_argument("--hf-cache-dir", type=str, default='', help="Cache directory for HuggingFace datasets")

    # Model arguments
    parser.add_argument("--model", type=str, default=[
            'gemini-2.5-pro-thinking',
            'gpt-5',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-thinking',
            'qwen3-vl-32b-thinking',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'xlangai/Jedi-7B-1080p',
            'step-3',
            'zai-org/GLM-4.5V',
            'internvl3-78b',
            'UGround-7b'
        ][-1],
                       help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro-thinking')")
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
    
    args, _ = parser.parse_known_args()
    
    # Validate arguments
    if args.hf_dataset_id and not HF_AVAILABLE:
        parser.error("--hf-dataset-id requires the 'datasets' library. Install with: pip install datasets")
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)

