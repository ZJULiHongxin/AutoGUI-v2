"""
Evaluate VLMs on multi-choice state prediction questions (functional captioning)

This script evaluates vision-language models on multiple-choice questions that
ask about the outcome of interacting with GUI elements. It supports:
 - Multiprocessing evaluation
 - Elegant logging
 - Checkpointing (resume from previous results)
 - HuggingFace dataset loading with persistent caching (entries and images)
"""

import os
import re
import json
import time
import glob
import hashlib
import argparse
import multiprocessing
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, Manager

from PIL import Image, ImageDraw
from tqdm import tqdm
from utils.data_utils.misc import resize_pil_image
from utils.eval_utils.autoguiv2.misc import normalize_action_type

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

EVAL_PROMPT = """You are a GUI expert. Read the question and options, analyze the given GUI screenshot, and then choose the single best answer.

Answer with a JSON object only:
{{"answer": "A/B/C/D/E"}}

Question:
{question}

Options:
{options_block}

Now provide your answer:"""


def get_hf_cache_paths(hf_dataset_id: str, split: str, task_type: str = 'func') -> tuple:
    """Get cache file path and image cache directory for HuggingFace dataset conversion
    
    Args:
        hf_dataset_id: HuggingFace dataset ID
        split: Dataset split
        task_type: Task type to evaluate ('func' for functionality, 'desc' for description)
    Returns:
        Tuple of (cache_file_path, image_cache_dir)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use shared cache directory for all task types (images are shared)
    cache_dir = os.path.join(script_dir, 'elemcap_hf_dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash of dataset_id and split for cache (images are shared across task types)
    cache_key = f"{hf_dataset_id}_{split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    # JSON cache file is task-specific
    cache_filename = f"{cache_hash}_{task_type}.json"
    entries_cache = os.path.join(cache_dir, cache_filename)
    
    # Image cache directory is shared across all task types (same images)
    images_cache_dir = os.path.join(cache_dir, 'images', cache_hash)
    os.makedirs(images_cache_dir, exist_ok=True)
    return entries_cache, images_cache_dir


def load_entries_cache(cache_file: str, image_cache_dir: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(cache_file):
        return None
    try:
        data = json.load(open(cache_file, 'r', encoding='utf-8'))
        entries = data.get('entries', [])
        if not entries:
            return None
        if not os.path.exists(image_cache_dir):
            return None
        # quick validation of paths
        valid = any(os.path.exists(e.get('image_path', '')) for e in entries[:20])
        return entries if valid else None
    except Exception:
        return None


def save_entries_cache(entries: List[Dict[str, Any]], cache_file: str, hf_dataset_id: str, split: str, task_type: str = 'func'):
    payload = {
        'metadata': {
            'hf_dataset_id': hf_dataset_id,
            'split': split,
            'task_type': task_type,
            'num_entries': len(entries),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        'entries': entries,
    }
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def normalize_choices(raw_choices: Any) -> Dict[str, str]:
    # raw choices might be dict or JSON string
    try:
        if isinstance(raw_choices, str):
            raw_choices = json.loads(raw_choices)
    except Exception:
        return {}
    if not isinstance(raw_choices, dict):
        return {}
    # Map letters to text, strip any decorations like "(hard_neg_x)" or "(correct)"
    norm = {}
    for key, text in raw_choices.items():
        if not isinstance(text, str):
            continue
        letter = key.split()[0].strip()  # "A (correct)" -> "A"
        cleaned = re.sub(r"\s*\([^)]*\)$", "", text).strip()
        norm[letter] = cleaned
    return norm


def detect_correct_letter(raw_choices: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    # Find key that contains (correct)
    for key in raw_choices.keys() if isinstance(raw_choices, dict) else []:
        if '(correct' in key or '(correct)' in key:
            return key.split()[0]
    return fallback


def load_hf_statepred(hf_dataset_id: str, split: str = 'test', cache_dir: Optional[str] = None, task_type: str = 'func') -> List[Dict[str, Any]]:
    """Load HuggingFace dataset for state prediction evaluation
    
    Args:
        hf_dataset_id: HuggingFace dataset ID
        split: Dataset split
        cache_dir: Optional cache directory for downloaded datasets
        task_type: Task type - 'func' for functionality-based choices, 'desc' for description-based choices
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset loading. Install with: pip install datasets")

    cache_file, image_cache_dir = get_hf_cache_paths(hf_dataset_id, split, task_type)
    cached = load_entries_cache(cache_file, image_cache_dir)
    if cached is not None:
        debug_print(f"✅ Loaded {len(cached)} entries from cache", level="success")
        return cached

    debug_print(f"\n📂 Loading HF dataset: {hf_dataset_id} (split: {split}, task_type: {task_type})", level="step")
    try:
        if cache_dir:
            raw_dataset = load_from_disk(cache_dir)
            ds = raw_dataset[split]
        else:
            ds = load_dataset(hf_dataset_id, split=split)
    except Exception as e:
        debug_print(f"❌ Failed to load dataset from HuggingFace: {e}", level="error")
        raise
    entries: List[Dict[str, Any]] = []

    debug_print(f"🔄 Converting dataset to entries and caching images...", level="step")
    for idx, item in tqdm(enumerate(ds), total=len(ds), desc=f"Converting HF {task_type}"):
        image = item.get('image')
        if image is None:
            continue

        # Determine image file path (cache PIL to disk)
        # Extract image_name early for use in both PIL and string cases
        image_name = item.get('image_name', f'image_{idx}')
        original_image_name = image_name
        
        # Extract target element bbox
        target_element = item.get('target_element', {})
        if isinstance(target_element, str):
            try:
                target_element = json.loads(target_element)
            except Exception:
                target_element = {}
        target_bbox = target_element.get('bbox', [])
        
        if isinstance(image, Image.Image):
            # Sanitize image name for filesystem (preserve '/' for subdirectories)
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
                continue
            image_name = original_image_name or os.path.basename(image_path)
        else:
            continue

        question = item.get('question', '')
        if not question:
            continue

        # Extract choices based on task type and candidate type information
        choices = {}
        choices_metadata = {}  # Store candidate_type for each choice letter
        
        # First, try to get choices from JSON to extract metadata
        raw_choices = item.get('choices', {})
        if isinstance(raw_choices, str):
            try:
                raw_choices = json.loads(raw_choices)
            except Exception:
                pass
        
        if isinstance(raw_choices, dict):
            # Extract text/description and candidate_type from choices JSON
            for letter, choice_data in raw_choices.items():
                if isinstance(choice_data, dict):
                    if task_type == 'desc':
                        text = choice_data.get('description', '')
                    else:
                        text = choice_data.get('text', '')
                    if text:
                        choices[letter] = text
                        # Store candidate type (target, hard_negative, easy_negative)
                        candidate_type = choice_data.get('candidate_type', 'unknown')
                        choices_metadata[letter] = candidate_type
                elif isinstance(choice_data, str):
                    # Legacy format: direct text
                    choices[letter] = choice_data
                    choices_metadata[letter] = 'unknown'
        
        # Fallback: parse from string format if JSON parsing didn't work
        if not choices:
            if task_type == 'desc':
                # Use description-based choices
                desc_candidates = item.get('description_candidates_string', '')
                if desc_candidates:
                    # Parse "A: description text\nB: description text\n..." format
                    for line in desc_candidates.strip().split('\n'):
                        if ':' in line:
                            letter, text = line.split(':', 1)
                            letter = letter.strip()
                            text = text.strip()
                            if letter and text:
                                choices[letter] = text
                                choices_metadata[letter] = 'unknown'
            else:
                # Default to functionality-based choices (task_type == 'func')
                statepred_candidates = item.get('statepred_candidates_string', '')
                if statepred_candidates:
                    # Parse "A: functionality text\nB: functionality text\n..." format
                    for line in statepred_candidates.strip().split('\n'):
                        if ':' in line:
                            letter, text = line.split(':', 1)
                            letter = letter.strip()
                            text = text.strip()
                            if letter and text:
                                choices[letter] = text
                                choices_metadata[letter] = 'unknown'
        
        if not choices:
            continue

        # Get correct answer (same for both task types since choices are shuffled together)
        correct = item.get('correct_answer', None)
        if correct is None:
            # Fallback: try to detect from raw choices
            raw_choices = item.get('choices', {})
            correct = detect_correct_letter(raw_choices, fallback=item.get('answer'))
        if correct is None:
            # Heuristic: If no explicit correct marker, assume 'A' is correct per generation spec
            correct = 'A' if 'A' in choices else None
        if correct is None:
            continue

        # Extract metadata fields
        dataset_name = item.get('dataset_name', 'unknown')
        action_type = normalize_action_type(item.get('interaction_type', item.get('action_type', 'unknown')))

        entry = {
            'entry_id': f"{image_name}_{idx}_{task_type}",
            'image_path': image_path,
            'image_name': image_name,
            'dataset_name': dataset_name,
            'question': question,
            'choices': choices,      # dict letter -> text
            'choices_metadata': choices_metadata,  # dict letter -> candidate_type
            'correct': correct,      # letter
            'action_type': action_type,
            'target_bbox': target_bbox,  # Target element bounding box (normalized 0-1000)
        }
        entries.append(entry)

    debug_print(f"✅ Prepared {len(entries)} entries", level="success")
    save_entries_cache(entries, cache_file, hf_dataset_id, split, task_type)
    return entries


def is_failed_result(result: Dict[str, Any]) -> bool:
    """Check if a result represents a failed query
    
    A result is considered failed if:
    - pred is None/null, OR
    - error is present and not empty, OR
    - response is None/null and error is present
    """
    if result is None:
        return True
    pred = result.get('pred')
    error = result.get('error')
    response = result.get('response')
    
    # Failed if prediction is missing
    if pred is None:
        return True
    
    # Failed if there's an error
    if error and error.strip():
        return True
    
    # Failed if response is missing but error exists
    if response is None and error:
        return True
    
    return False


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load evaluation checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint JSON file (can be checkpoint or full result file)
    
    Returns:
        Dictionary with processed entry IDs, results, and failed IDs
        Note: Only successfully completed entries are included in processed_ids.
        Failed entries will be retried on resume.
    """
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        return {'processed_ids': set(), 'results': {}, 'failed_ids': set()}
    
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
        # Only entries with valid pred (not None) and no error should be considered successfully processed
        successful_ids = set()
        failed_ids = set()
        
        for entry_id in processed_ids:
            result = results.get(entry_id)
            
            if result:
                # Check if inference was successful
                if not is_failed_result(result):
                    successful_ids.add(entry_id)
                else:
                    failed_ids.add(entry_id)
            else:
                # If we can't find the result, assume it needs to be retried
                failed_ids.add(entry_id)
        
        # Also check all results for failed entries (in case some weren't in processed_ids)
        for entry_id, result in results.items():
            if entry_id not in successful_ids and entry_id not in failed_ids:
                if is_failed_result(result):
                    failed_ids.add(entry_id)
                else:
                    # This is a successful result that wasn't in processed_ids
                    # (might be from an old checkpoint format)
                    successful_ids.add(entry_id)
        
        # Update processed_ids to only include successful entries
        processed_ids = successful_ids
        
        if failed_ids:
            debug_print(f"⚠️  Found {len(failed_ids)} failed entries that will be retried", level="warn")
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} successful entries, {len(failed_ids)} failed entries to retry", level="success")
        else:
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} processed entries", level="success")
        
        return {
            'processed_ids': processed_ids,
            'results': results,
            'failed_ids': failed_ids
        }
    except Exception as e:
        debug_print(f"⚠️  Error loading checkpoint: {e}", level="warn")
        return {'processed_ids': set(), 'results': {}, 'failed_ids': set()}


def save_checkpoint(results: Dict[str, Any], processed_ids: set, checkpoint_file: str, metadata: Dict[str, Any]):
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    payload = {
        'metadata': metadata,
        'processed_ids': list(processed_ids),
        'results': results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# Global worker model
worker_model = None


def init_worker(model_args: Dict[str, Any]):
    """Initialize worker with model"""
    global worker_model
    
    base_url = model_args['base_url']
    api_key = model_args['api_key']
    model = model_args['model']

    if 'qwen' in model.lower():
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
    elif 'tars' in model.lower():
        base_url = 'https://api.parasail.io/v1'
        api_key = api_key or os.environ.get("PARASAIL_API_KEY", "EMPTY")
        cloud_model_class = PARASAIL
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
    else:
        cloud_model_class = OpenAIModel
        base_url = base_url or os.environ.get("OPENAI_API_BASE_XIAOAI", "EMPTY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY_XIAOAI", "EMPTY")

    worker_model = cloud_model_class(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0.0,
        max_tokens=4096
    )

    debug_print(f"   Model: {model} | Cloud Class: {cloud_model_class.__name__} | Base URL: {base_url} | API Key: {api_key}", level="info")



def parse_answer_from_response(response: str) -> Optional[str]:
    try:
        m = re.search(r'\{\s*"answer"\s*:\s*"([A-E])"\s*\}', response)
        if m:
            return m.group(1)
        # fallback: single letter on its own
        m2 = re.search(r'\b([A-E])\b', response)
        if m2:
            return m2.group(1)
    except Exception:
        pass
    return None


def process_entry(entry: Dict[str, Any], worker_id: int = 0, task_type: str = 'func') -> Dict[str, Any]:
    """Process a single dataset entry
    
    Args:
        entry: Dataset entry dictionary
        worker_id: Worker ID for logging
        task_type: Task type ('func' for functionality, 'desc' for description)
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    start = time.time()
    result = {
        'entry_id': entry['entry_id'],
        'image_path': entry['image_path'],
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'question': entry['question'],
        'correct': entry['correct'],
        'pred': None,
        'is_correct': False,
        'response': None,
        'error': None,
        'processing_time': 0.0,
        # Preserve metadata for decomposed metrics
        'action_type': normalize_action_type(entry.get('action_type', 'unknown')),
        'target_bbox': entry.get('target_bbox', []),
    }

    retry = 0
    while retry < 4:
        try:
            retry += 1
            options_lines = [f"{letter}) {text}" for letter, text in sorted(entry['choices'].items())]

            # Draw bbox on image only for 'desc' task type
            image_path_to_use = entry['image_path']
            temp_file_created = False
            temp_img_path = None

            # Handle image resizing for Claude and seed models
            if any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                # Resize image for Claude/seed models
                image = Image.open(entry['image_path']).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                resized_image.save(temp_img_path)
                image_path_to_use = temp_img_path
                temp_file_created = True

            if task_type in ['func-w-bbox', 'desc']:
                if task_type == 'func-w-bbox':
                    action_type = entry.get('action_type', 'unknown')
                    prompt = EVAL_PROMPT.format(
                        question=f"What will happen to the given GUI if I {action_type} on the target element highlighted with a red rectangle?",
                        options_block='\n'.join(options_lines),
                    )
                elif task_type == 'desc':
                    prompt += " (The target element has been highlighted with a red rectangle.)"

                target_bbox = entry.get('target_bbox', [])
                if target_bbox and len(target_bbox) == 4:
                    # Load the image (either original or resized)
                    if temp_file_created and temp_img_path:
                        # If we already resized, use the resized image
                        image = Image.open(temp_img_path)
                    else:
                        # Load the original image
                        image = Image.open(entry['image_path'])
                    
                    # resize for certain models
                    
                    if 'qwen3' in worker_model.model.lower() and max(image.size) > 3500:
                        image, _ = resize_pil_image(image, max_size=3200)                    
                    
                    # Create a copy to draw on
                    image_with_bbox = image.copy()
                    draw = ImageDraw.Draw(image_with_bbox)
                    
                    # Convert normalized bbox (0-1000) to pixel coordinates
                    W, H = image_with_bbox.size
                    x1 = int(target_bbox[0] * W / 1000)
                    y1 = int(target_bbox[1] * H / 1000)
                    x2 = int(target_bbox[2] * W / 1000)
                    y2 = int(target_bbox[3] * H / 1000)
                    
                    # Draw rectangle (red color, width 3)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    
                    # Save to temporary file (overwrite temp_img_path if it exists, or create new)
                    if temp_img_path:
                        # Overwrite the existing temp file
                        image_with_bbox.save(temp_img_path)
                    else:
                        # Create new temp file
                        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                        os.close(temp_fd)
                        image_with_bbox.save(temp_path)
                        temp_img_path = temp_path
                        image_path_to_use = temp_path
                        temp_file_created = True
            else:
                prompt = EVAL_PROMPT.format(
                    question=entry['question'] + " Please choose the best option.",
                    options_block='\n'.join(options_lines),
                )

            # Pre-logging: Show query start information
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            cloud_class = worker_model.__class__.__name__
            base_url = worker_model.base_url
            api_key_masked = worker_model.api_key[:8] + '...' + worker_model.api_key[-4:]
            print(f"[Worker {worker_id}] 🚀 Starting query{retry_info} | Entry: {entry['entry_id']} | "
                  f"Model: {worker_model.model} | Class: {cloud_class} | "
                  f"BaseURL: {base_url} | APIKey: {api_key_masked} | "
                  f"Image: {image_name_short} | [{timestamp_start}]")

            try:
                success, response, _ = worker_model.get_model_response(
                    prompt, [image_path_to_use], use_img_url=True, temperature=0.0, timeout=360
                )
            except Exception as e:
                # Exception during API call - log and continue to next retry
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                # Concise error logging for exceptions
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(e)[:200]
                print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry['entry_id']} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                # Continue to next retry (don't break)
                continue
            finally:
                # Clean up temporary file if created
                if temp_file_created and temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass

            if not success:
                result['error'] = f"API call failed: {response}"
                result['processing_time'] = time.time() - start
                # Concise error logging
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(response)[:200] if response else "Unknown error"
                print(f"[Worker {worker_id}] ❌ Query FAILED | Entry: {entry['entry_id']} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Time: {result['processing_time']:.2f}s | [{timestamp_error}]")
                # Continue to next retry (don't return here, let the loop handle retries)
                continue

            # Only reach here if API call succeeded
            result['prompt'], result['response'] = prompt, response
            pred = parse_answer_from_response(response)
            result['pred'] = pred
            result['is_correct'] = (pred == entry['correct'])

            # Track candidate type of predicted answer (for error analysis)
            choices_metadata = entry.get('choices_metadata', {})
            if pred and pred in choices_metadata:
                result['predicted_candidate_type'] = choices_metadata[pred]
            else:
                result['predicted_candidate_type'] = 'unknown'

            # Only print "Query COMPLETE" if we actually got a valid prediction
            if pred is not None:
                status = "✅" if result['is_correct'] else "❌"
                processing_time = time.time() - start
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry['entry_id']} -> {pred} (gt={entry['correct']}) | "
                      f"Time: {processing_time:.2f}s | [{timestamp}]")
                break
            else:
                # Failed to parse answer - treat as failure and retry
                result['error'] = "Failed to parse answer from response"
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] ❌ Query FAILED (parse error) | Entry: {entry['entry_id']} | "
                      f"Model: {worker_model.model} | Error: Could not parse answer | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue
        except Exception as e:
            # Exception outside API call (e.g., during image processing)
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            # Concise error logging for exceptions
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry['entry_id']} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            # Continue to next retry (don't break)
            continue
    else:
        result['error'] = "Failed after 4 retries"
        # Final error logging after all retries exhausted
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry['entry_id']} | "
              f"Model: {worker_model.model} | Total time: {time.time() - start:.2f}s | [{timestamp_error}]")

    result['processing_time'] = time.time() - start
    return result


def process_entries_parallel(entries: List[Dict[str, Any]], model_args: Dict[str, Any], args, checkpoint: Dict) -> List[Dict[str, Any]]:
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

    # Only mark successfully processed entries (exclude failed ones)
    processed_ids_all = checkpoint.get('processed_ids', set())
    failed_ids = checkpoint.get('failed_ids', set())
    processed_ids_successful = processed_ids_all - failed_ids

    # Initialize processed_ids manager.dict()
    processed_ids = manager.dict()
    for entry_id in processed_ids_successful:
        processed_ids[entry_id] = True

    start_time = time.time()
    processed_count = manager.Value('i', len(processed_ids_successful))
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
    
    # Filter entries to process (exclude only successfully processed ones)
    entries_to_process = [
        (entry, i % args.max_workers, args.task_type) 
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
                if is_failed_result(result):
                    retry_count += 1
    elif isinstance(checkpoint_results, list):
        # Handle list format (from full result files)
        results_by_id = {r.get('entry_id'): r for r in checkpoint_results if 'entry_id' in r}
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in results_by_id:
                result = results_by_id[entry_id]
                if is_failed_result(result):
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
                # Only increment count and mark as processed if result is successful
                if not is_failed_result(result):
                    processed_ids[result['entry_id']] = True
                    processed_count.value += 1
                # Still update count for failed entries to track progress, but don't mark as processed
                else:
                    # Don't increment processed_count for failed entries
                    pass

            update_throughput()

            # Save checkpoint periodically
            if processed_count.value % 1 == 0:
                # processed_ids already only contains successful entries (set above)
                save_checkpoint(
                    dict(results_dict),
                    set(processed_ids.keys()),
                    args.checkpoint_file,
                    {'model': args.model, 'total_entries': len(entries)}
                )
    finally:
        pool.close()
        pool.join()

    print()  # New line after throughput updates

    # Return all results (checkpoint results + newly processed results)
    all_results = list(results_dict.values())

    return all_results


def calculate_metrics_for_subset(subset_results: List[Dict]) -> Dict[str, Any]:
    """Calculate metrics for a subset of results"""
    total = len(subset_results)
    successful = sum(1 for r in subset_results if not is_failed_result(r))
    correct = sum(1 for r in subset_results if r.get('is_correct', False))
    acc = correct / total if total > 0 else 0.0
    success_rate = successful / total if total > 0 else 0.0
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': success_rate,
        'correct': correct,
        'accuracy': acc,
    }


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation metrics with decomposed breakdowns"""
    total = len(results)
    successful = sum(1 for r in results if not is_failed_result(r))
    correct = sum(1 for r in results if r.get('is_correct', False))
    acc = correct / total if total > 0 else 0.0
    success_rate = successful / total if total > 0 else 0.0
    
    # Action type breakdown
    action_types = {}
    for r in results:
        action_type = r.get('action_type', 'unknown')
        if action_type not in action_types:
            action_types[action_type] = {'total': 0, 'successful': 0, 'correct': 0}
        action_types[action_type]['total'] += 1
        if not is_failed_result(r):
            action_types[action_type]['successful'] += 1
        if r.get('is_correct', False):
            action_types[action_type]['correct'] += 1
    
    for action_type in action_types:
        data = action_types[action_type]
        data['success_rate'] = data['successful'] / data['total'] if data['total'] > 0 else 0.0
        data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0.0
    
    # Candidate type breakdown for incorrect answers
    incorrect_results = [r for r in results if not r.get('is_correct', False)]
    total_errors = len(incorrect_results)
    
    candidate_type_errors = {
        'hard_negative': 0,
        'easy_negative': 0,
        'target': 0,  # Should be rare (model chose correct but marked as wrong)
        'unknown': 0
    }
    
    for r in incorrect_results:
        candidate_type = r.get('predicted_candidate_type', 'unknown')
        if candidate_type in candidate_type_errors:
            candidate_type_errors[candidate_type] += 1
        else:
            candidate_type_errors['unknown'] += 1
    
    # Calculate proportions
    candidate_type_proportions = {}
    if total_errors > 0:
        for candidate_type, count in candidate_type_errors.items():
            candidate_type_proportions[candidate_type] = count / total_errors
    else:
        for candidate_type in candidate_type_errors:
            candidate_type_proportions[candidate_type] = 0.0
    
    candidate_type_breakdown = {
        'total_errors': total_errors,
        'error_counts': candidate_type_errors,
        'error_proportions': candidate_type_proportions
    }
    
    return {
        'total': total,
        'successful': successful,
        'success_rate': success_rate,
        'correct': correct,
        'accuracy': acc,
        'action_types': action_types,
        'candidate_type_breakdown': candidate_type_breakdown,
        'decomposed': {
            'by_action_type': action_types,
            'by_candidate_type': candidate_type_breakdown,
        }
    }


def main(args):
    debug_print("═" * 60, level="title")
    task_type_display = "Functionality" if args.task_type == 'func' else "Description"
    debug_print(f"📘 State Prediction (Multi-Choice) Evaluation - {task_type_display}", level="title")
    debug_print("═" * 60, level="title")

    # Inputs
    if args.hf_dataset_id:
        debug_print(f"   Source: HF dataset {args.hf_dataset_id} [{args.hf_split}]", level="info")
    else:
        debug_print(f"   Source: JSON file {args.questions_file}", level="info")
    debug_print(f"   Task Type: {args.task_type} ({task_type_display})", level="info")

    # Prepare dataset
    entries = load_hf_statepred(args.hf_dataset_id, args.hf_split, args.hf_cache_dir or None, args.task_type)
    
    if not entries:
        debug_print("❌ JSON source for state-pred not implemented in this script. Use --hf-dataset-id.", level="error")
        return

    if not entries:
        debug_print("❌ No entries to evaluate", level="error")
        return

    # Checkpoint setup
    eval_dir = os.path.join(os.path.dirname(__file__), 'eval_results', 'AutoGUI-v2-ElemCap', args.task_type)
    os.makedirs(eval_dir, exist_ok=True)
    safe_model = args.model.replace('/', '_').replace('\\', '_')
    model_dir = os.path.join(eval_dir, safe_model)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.checkpoint_file:
        checkpoint_file = args.checkpoint_file
        result_file = checkpoint_file
    elif args.load_latest:
        # Find latest checkpoint
        pattern = os.path.join(model_dir, '*.json')
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            checkpoint_file = files[0]
            result_file = checkpoint_file
            debug_print(f"📂 Found latest checkpoint: {checkpoint_file}", level="info")
        else:
            result_file = os.path.join(model_dir, f"{timestamp}.json")
            checkpoint_file = result_file
            debug_print(f"📂 No existing checkpoint found, creating new: {checkpoint_file}", level="info")
    else:
        result_file = os.path.join(model_dir, f"{timestamp}.json")
        checkpoint_file = result_file
    
    ckpt = load_checkpoint(checkpoint_file)
    args.checkpoint_file = checkpoint_file

    # Filter remaining entries if resuming
    processed_ids = ckpt['processed_ids']
    failed_ids = ckpt.get('failed_ids', set())
    
    # Remove failed entries from processed_ids so they get re-evaluated
    processed_ids_excluding_failed = processed_ids - failed_ids
    
    if processed_ids or failed_ids:
        # Include failed entries for re-evaluation, exclude successfully processed ones
        entries = [
            e for e in entries 
            if e['entry_id'] not in processed_ids_excluding_failed
        ]
        num_failed = len(failed_ids)
        num_successful = len(processed_ids_excluding_failed)
        debug_print(f"🔁 Resuming: {num_successful} successfully processed, {num_failed} failed entries detected", level="info")
        if num_failed > 0:
            debug_print(f"🔄 Re-evaluating {num_failed} failed entries", level="warn")
        debug_print(f"📋 {len(entries)} entries to process ({num_failed} failed + {len(entries) - num_failed} new)", level="info")

    # Model args
    model_args = {
        'base_url': args.base_url,
        'api_key': args.api_key,
        'model': args.model,
    }

    # Evaluate
    debug_print("\n🚀 Starting evaluation...", level="step")
    start = time.time()
    if args.max_workers > 1:
        results = process_entries_parallel(entries, model_args, args, ckpt)
    else:
        init_worker(model_args)
        results = []
        # Start with checkpoint results (convert to list format)
        checkpoint_results = ckpt.get('results', {})
        if isinstance(checkpoint_results, dict):
            checkpoint_results_list = list(checkpoint_results.values())
        elif isinstance(checkpoint_results, list):
            checkpoint_results_list = checkpoint_results
        else:
            checkpoint_results_list = []
        
        # Filter out entries that are already successfully processed
        processed_ids_excluding_failed = processed_ids - failed_ids
        for i, e in enumerate(entries):
            res = process_entry(e, i, args.task_type)
            results.append(res)
            if (i + 1) % 20 == 0 and checkpoint_file:
                # Merge checkpoint results with new results
                merged_dict = {r['entry_id']: r for r in checkpoint_results_list}
                for r in results:
                    merged_dict[r['entry_id']] = r
                # Only mark successful entries as processed
                successful_ids = {r['entry_id'] for r in results if not is_failed_result(r)}
                save_checkpoint(merged_dict, processed_ids_excluding_failed | successful_ids, checkpoint_file, {'model': args.model, 'total_entries': len(entries)})
        
        # Merge checkpoint results with newly processed results for final metrics
        # Convert checkpoint results to dict for easy merging
        checkpoint_results_dict = {}
        if isinstance(checkpoint_results, dict):
            checkpoint_results_dict = checkpoint_results
        elif isinstance(checkpoint_results, list):
            checkpoint_results_dict = {r['entry_id']: r for r in checkpoint_results if 'entry_id' in r}
        
        # Merge: checkpoint results + newly processed results (new results override old ones)
        merged_results_dict = {**checkpoint_results_dict}
        for r in results:
            merged_results_dict[r['entry_id']] = r
        
        # Convert back to list for consistency with parallel path
        results = list(merged_results_dict.values())

    total_time = time.time() - start

    # Calculate metrics from all results
    # Note: results already includes both checkpoint results and newly processed results
    debug_print(f"\n📊 Calculating metrics...", level="step")
    metrics = calculate_metrics(results)

    # Save final outputs
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'model': args.model,
                'base_url': args.base_url,
                'hf_dataset_id': args.hf_dataset_id,
                'hf_split': args.hf_split,
                'task_type': args.task_type,
                'num_workers': args.max_workers,
                'total_time': total_time,
                'timestamp': timestamp,
            },
            'metrics': metrics,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    if checkpoint_file != result_file:
        # Only mark successful entries as processed
        # Convert results list to dict for checkpoint saving
        results_dict = {r['entry_id']: r for r in results}
        successful_entry_ids = {entry_id for entry_id, result in results_dict.items() if not is_failed_result(result)}
        save_checkpoint(results_dict, successful_entry_ids, checkpoint_file, {'model': args.model, 'total_entries': len(results)})

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
        overall_table.add_row("Correct", f"{metrics['correct']} ({metrics['accuracy']*100:.2f}%)")
        
        console.print(overall_table)
        
        # Action type breakdown table
        if metrics.get('action_types'):
            action_table = Table(title="🎯 Action Type Breakdown", box=box.ROUNDED, show_header=True, header_style="bold yellow")
            action_table.add_column("Action Type", style="cyan")
            action_table.add_column("Total", style="white", justify="right")
            action_table.add_column("Success Rate", style="green", justify="right")
            action_table.add_column("Correct", style="green", justify="right")
            action_table.add_column("Accuracy", style="green", justify="right")
            
            for action_type, data in sorted(metrics['action_types'].items()):
                action_table.add_row(
                    action_type,
                    str(data['total']),
                    f"{data.get('success_rate', 0.0)*100:.1f}%",
                    str(data['correct']),
                    f"{data['accuracy']*100:.2f}%"
                )
            
            console.print("\n")
            console.print(action_table)
        
        # Candidate type breakdown table
        if metrics.get('candidate_type_breakdown'):
            candidate_table = Table(title="🔍 Error Analysis by Candidate Type", box=box.ROUNDED, show_header=True, header_style="bold blue")
            candidate_table.add_column("Candidate Type", style="cyan")
            candidate_table.add_column("Error Count", style="white", justify="right")
            candidate_table.add_column("Proportion", style="green", justify="right")
            
            breakdown = metrics['candidate_type_breakdown']
            error_counts = breakdown.get('error_counts', {})
            error_proportions = breakdown.get('error_proportions', {})
            
            # Display in order: hard_negative, easy_negative, target, unknown
            for candidate_type in ['hard_negative', 'easy_negative', 'target', 'unknown']:
                if candidate_type in error_counts:
                    count = error_counts[candidate_type]
                    proportion = error_proportions.get(candidate_type, 0.0)
                    candidate_table.add_row(
                        candidate_type.replace('_', ' ').title(),
                        str(count),
                        f"{proportion*100:.1f}%"
                    )
            
            console.print("\n")
            console.print(candidate_table)
    else:
        debug_print(f"📊 Total Entries: {metrics['total']}", level="info")
        debug_print(f"✅ Successful: {metrics['successful']} ({metrics['success_rate']*100:.1f}%)", level="info")
        debug_print(f"✅ Correct: {metrics['correct']} ({metrics['accuracy']*100:.2f}%)", level="info")
        
        if metrics.get('action_types'):
            debug_print("\n📊 Action Type Breakdown:", level="info")
            for action_type, data in metrics['action_types'].items():
                debug_print(f"   {action_type}: {data.get('success_rate', 0.0)*100:.1f}% success, "
                           f"{data['accuracy']*100:.2f}% accuracy ({data['correct']}/{data['total']})", level="info")
        
        # Candidate type breakdown
        if metrics.get('candidate_type_breakdown'):
            debug_print("\n🔍 Error Analysis by Candidate Type:", level="info")
            breakdown = metrics['candidate_type_breakdown']
            error_counts = breakdown.get('error_counts', {})
            error_proportions = breakdown.get('error_proportions', {})
            total_errors = breakdown.get('total_errors', 0)
            debug_print(f"   Total Errors: {total_errors}", level="info")
            for candidate_type in ['hard_negative', 'easy_negative', 'target', 'unknown']:
                if candidate_type in error_counts:
                    count = error_counts[candidate_type]
                    proportion = error_proportions.get(candidate_type, 0.0)
                    debug_print(f"   {candidate_type.replace('_', ' ').title()}: {count} ({proportion*100:.1f}%)", level="info")
    
    debug_print(f"\n💾 Results saved to: {result_file}", level="info")
    if checkpoint_file != result_file:
        debug_print(f"💾 Checkpoint saved to: {checkpoint_file}", level="info")
    else:
        debug_print(f"💾 Checkpoint and results in same file: {result_file}", level="info")
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs on multi-choice state prediction questions")
    parser.add_argument("--task-type", type=str, default='func-w-bbox', choices=['func-w-bbox', 'func-w-ques', 'desc'], help="Task type: 'func' for functionality-based choices, 'desc' for description-based choices")

    # Input (prefer HF dataset)
    parser.add_argument("--hf-dataset-id", type=str, default=['HongxinLi/AutoGUIv2-FuncElemCap', ''][-1], help="HF dataset ID")
    parser.add_argument("--hf-split", type=str, default='test', help="HF split")
    parser.add_argument("--hf-cache-dir", type=str, default=['/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemCap/' ,'/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/AutoGUIv2-FuncElemCap-0125/'][-1], help="HF cache dir")
    parser.add_argument("--questions-file", type=str, default=None, help="(Optional) JSON source - not implemented")

    # Model
    parser.add_argument("--model", type=str, default=[
            'gemini-3-flash-preview-thinking',
            'gemini-2.5-pro-thinking',
            'gpt-5-2025-08-07',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-thinking',
            'qwen3-vl-8b-instruct',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'step-3',
            'zai-org/GLM-4.5V',
            'xlangai/OpenCUA-7B'
        ][-1],
                       help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro-thinking')")
    parser.add_argument("--base-url", type=str, default=None, help="Model API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="Model API key")

    # Processing
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit the number of samples to process")

    # Checkpointing
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Path to checkpoint JSON")
    parser.add_argument("--load-latest", action="store_true", help="Load the latest checkpoint for this model")

    args, _ = parser.parse_known_args()

    if args.hf_dataset_id and not HF_AVAILABLE:
        parser.error("--hf-dataset-id requires the 'datasets' library. Install with: pip install datasets")

    multiprocessing.set_start_method('spawn', force=True)
    main(args)