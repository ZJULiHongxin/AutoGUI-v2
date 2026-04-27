"""
Evaluate VLMs on Functional Region Captioning (Funccap) tasks

This script evaluates vision-language models on functional REGION captioning,
where models need to identify the functionality of circled UI elements from multiple choice options.
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
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, Manager
from functools import partial
from PIL import Image
from tqdm import tqdm
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

import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen3vl import Qwen3VL
from utils.openai_utils.doubao import DOUBAO
from utils.openai_utils.stepfun import STEPFUN
from utils.openai_utils.parasail import PARASAIL
from utils.openai_utils.huggingface import HFEndpoint
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

# Grounding prompt template
GEMINI_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, you need to identify the bounding box of the target element, which should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

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
click(point='<point>x1 y1</point>'')

## User Instruction
{question}"""

# OS-ATLAS
OSATLAS_PROMPT = 'In this UI screenshot, what is the position of the element corresponding to the command "{question}" (with bbox)?'

# GENERIC_PROMPT
GENERIC_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""

# OpenCUA System Prompt
OPENCUA_SYSPROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)

# Multiple choice prompt for funccap task
FUNCCAP_PROMPT = """You are a GUI expert. Given an image with a circled UI region, you need to identify the functionality of that region.

Question: {question}

Options:
{options}

Return ONLY the option letters separated by commas (e.g., "A,C,E" for multiple answers, or "A" for a single answer). Do not include any explanation."""

FUNCCAP_DEFAULT_QUESTION = "Which options accurately describe the functionality of the region marked with a red rectangle? (Select all that apply)"

def _normalize_correct_indices(raw_value: Any, options_len: int, option_labels: Optional[List[str]] = None,
                               option_texts: Optional[List[str]] = None) -> List[int]:
    """Normalize various correct-index representations into a sorted unique list."""
    indices: List[int] = []

    def add_idx(idx: Any) -> None:
        if isinstance(idx, int) and 0 <= idx < options_len and idx not in indices:
            indices.append(idx)

    def add_letter(letter: str) -> None:
        letter = letter.strip().upper()
        if not letter or len(letter) != 1:
            return
        # Try match option labels first
        if option_labels:
            for i, label in enumerate(option_labels):
                if label and label.strip().upper() == letter:
                    add_idx(i)
                    return
        # Fallback to A=0 mapping
        idx = ord(letter) - ord('A')
        add_idx(idx)

    def handle_value(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, int):
            add_idx(value)
            return
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return
            if stripped.isdigit():
                add_idx(int(stripped))
                return
            # Extract letter labels like A,C or "Option B"
            letters = re.findall(r'[A-H]', stripped, flags=re.IGNORECASE)
            for letter in letters:
                add_letter(letter)
            # Extract numeric tokens
            for token in re.findall(r'\b\d+\b', stripped):
                try:
                    add_idx(int(token))
                except ValueError:
                    pass
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                handle_value(item)
            return
        if isinstance(value, dict):
            for item in value.keys():
                handle_value(item)
            return

    handle_value(raw_value)

    # Optional fallback: match by option text (exact, normalized)
    if not indices and option_texts and raw_value is not None:
        if isinstance(raw_value, str):
            parts = [p.strip() for p in re.split(r'(?:,|;|/|\||、| and |以及|及|和)+', raw_value.lower()) if p.strip()]
        elif isinstance(raw_value, (list, tuple, set)):
            parts = [str(p).strip().lower() for p in raw_value if str(p).strip()]
        else:
            parts = []
        if parts:
            normalized_options = [re.sub(r'[.\s]+$', '', str(opt).strip().lower()) for opt in option_texts]
            for part in parts:
                part_norm = re.sub(r'[.\s]+$', '', part)
                for i, opt_norm in enumerate(normalized_options):
                    if part_norm == opt_norm:
                        add_idx(i)

    indices.sort()
    return indices


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
    """Load dataset from questions JSON file (supports both caption & legacy formats)
    
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

    all_entries: List[Dict[str, Any]] = []

    def _resolve_image_path(candidate: str, base_dirs: List[str]) -> Optional[str]:
        if not candidate:
            return None
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        for base in base_dirs:
            if not base:
                continue
            trial = os.path.join(base, candidate)
            if os.path.exists(trial):
                return trial
        return None

    def _load_caption_format(file_path: str, payload: Any, attr_data: Dict[str, Any]) -> None:
        nonlocal all_entries

        if isinstance(payload, dict):
            if 'result' in payload and isinstance(payload['result'], dict):
                questions = payload['result'].get('questions', [])
            else:
                questions = payload.get('questions', []) or payload.get('entries', [])
            dataset_name = payload.get('metadata', {}).get('dataset_name')
        elif isinstance(payload, list):
            questions = payload
            dataset_name = None
        else:
            questions = []
            dataset_name = None

        if not dataset_name:
            dataset_name = file_path.split('/')[-3] if len(file_path.split('/')) >= 3 else 'unknown'

        if not questions:
            debug_print(f"⚠️  No caption questions found in {file_path}", level="warn")
            return

        file_dir = os.path.dirname(file_path)
        parent_dir = os.path.dirname(file_dir)
        candidate_dirs = [
            file_dir,
            os.path.join(file_dir, 'images'),
            os.path.join(file_dir, 'annotated_images'),
            parent_dir,
            os.path.join(parent_dir, 'images'),
            os.path.join(parent_dir, 'annotated_images'),
        ]

        for q_data in questions:
            if not isinstance(q_data, dict):
                continue

            options = q_data.get('options', [])
            if not options:
                continue

            option_labels = [str(opt.get('label', '')).strip() for opt in options]
            option_functionalities = []
            option_region_types = []
            option_area_classes = []
            option_density_classes = []

            for opt in options:
                functionality = opt.get('option_context') or opt.get('functionality') or opt.get('description') or ''
                option_functionalities.append(functionality)

                metrics = opt.get('metrics') or {}
                area_info = metrics.get('area') or {}
                density_info = metrics.get('density') or {}

                option_region_types.append(opt.get('region_type', ''))
                option_area_classes.append(area_info.get('area_class') or opt.get('area_class') or None)
                option_density_classes.append(density_info.get('density_class') or opt.get('density_class') or 'unknown')

            correct_answer_label = q_data.get('correct_answer', '')
            correct_answer_labels = q_data.get('correct_answers', None)
            raw_correct = (q_data.get('correct_option_indices') or q_data.get('correct_option_idx') or
                           q_data.get('correct_indices') or q_data.get('correct_index'))
            correct_option_indices = _normalize_correct_indices(
                raw_correct, len(options), option_labels=option_labels, option_texts=option_functionalities
            )
            if not correct_option_indices:
                # Fallback to labels / answers if indices are not provided
                label_source = correct_answer_labels if correct_answer_labels is not None else correct_answer_label
                correct_option_indices = _normalize_correct_indices(
                    label_source, len(options), option_labels=option_labels, option_texts=option_functionalities
                )
            if not correct_option_indices:
                # Final fallback: derive from per-option is_correct flags
                correct_option_indices = [
                    idx for idx, opt in enumerate(options)
                    if isinstance(opt, dict) and opt.get('is_correct') is True
                ]

            if not correct_option_indices:
                debug_print(f"⚠️  Unable to determine correct options for entry in {file_path}", level="warn")
                continue

            correct_option_idx = correct_option_indices[0]

            annotated_image_path = q_data.get('annotated_image_path') or q_data.get('image_path') or ''
            image_path = _resolve_image_path(annotated_image_path, candidate_dirs)

            if not image_path:
                debug_print(f"⚠️  Annotated image not found for question in {file_path}", level="warn")
                continue

            image_name = os.path.basename(image_path)
            group_id = q_data.get('group_id', q_data.get('group_index', -1))
            target_region_id = q_data.get('target_region_id', '')
            density_class = 'unknown'
            area_class = None
            region_type = ''

            if correct_option_idx < len(option_density_classes):
                density_class = option_density_classes[correct_option_idx] or 'unknown'
            if correct_option_idx < len(option_area_classes):
                area_class = option_area_classes[correct_option_idx]
            if correct_option_idx < len(option_region_types):
                region_type = option_region_types[correct_option_idx] or ''

            if attr_data and image_name in attr_data:
                attr_entry = attr_data.get(image_name, {})
                region_key = str(target_region_id) if target_region_id is not None else ''
                if isinstance(attr_entry, dict):
                    if region_key in attr_entry and isinstance(attr_entry[region_key], dict):
                        density_class = attr_entry[region_key].get('density_class', density_class)
                    else:
                        for sub_entry in attr_entry.values():
                            if isinstance(sub_entry, dict) and region_key in sub_entry and isinstance(sub_entry[region_key], dict):
                                density_class = sub_entry[region_key].get('density_class', density_class)
                                break

            region_parent = resolve_parent_category(str(region_type))
            question_text = q_data.get('question', '')
            if not question_text:
                question_text = FUNCCAP_DEFAULT_QUESTION
            num_similar_elements = len(options)

            correct_answers = [option_functionalities[i] for i in correct_option_indices if i < len(option_functionalities)]
            
            # Use existing entry_id from JSON if available, otherwise generate deterministic one
            existing_entry_id = q_data.get('entry_id')
            if existing_entry_id:
                entry_id = existing_entry_id
            else:
                # Generate deterministic entry_id (no random uuid for consistency across runs)
                entry_id = f"{image_name}_{group_id}_{target_region_id}"
            
            entry = {
                'entry_id': entry_id,
                'image_path': image_path,
                'image_name': image_name,
                'dataset_name': dataset_name,
                'question': question_text,
                'options': option_functionalities,
                'correct_option_idx': correct_option_idx,
                'correct_option_indices': correct_option_indices,
                'correct_answer': option_functionalities[correct_option_idx] if correct_option_idx < len(option_functionalities) else '',
                'correct_answers': correct_answers,
                'group_index': group_id,
                'target_elem_id': target_region_id,
                'density_class': density_class or 'unknown',
                'num_similar_elements': num_similar_elements,
                'region_type': region_type,
                'region_parent': region_parent,
                'area_class': area_class,
            }
            all_entries.append(entry)

    def _load_legacy_format(file_path: str, payload: Dict[str, Any], attr_data: Dict[str, Any]) -> None:
        nonlocal all_entries

        results = payload.get('results', {})
        dataset_name = file_path.split('/')[-3] if len(file_path.split('/')) >= 3 else 'unknown'
        image_src_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'images')

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
                    
                    group_index = group_data.get('group_index', -1)
                    density_class = 'unknown'
                    if image_name in attr_data and str(group_index) in attr_data[image_name] and str(target_elem_id) in attr_data[image_name][str(group_index)]:
                        density_class = attr_data[image_name][str(group_index)][str(target_elem_id)].get('density_class', 'unknown')
                    
                    for _, action_data in referring_expressions.items():
                        if not isinstance(action_data, dict):
                            continue
                        question = action_data.get('question', '')
                        if not question:
                            continue
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

    for file in questions_files:
        if not os.path.exists(file):
            debug_print(f"⚠️  File not found: {file}", level="warn")
            continue

        with open(file, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        attr_file = file.replace(".json", "_attributes.json")
        attr_data: Dict[str, Any] = {}
        if os.path.exists(attr_file):
            try:
                with open(attr_file, 'r', encoding='utf-8') as f:
                    attr_data = json.load(f)
            except Exception:
                attr_data = {}

        if isinstance(payload, dict) and 'results' in payload:
            _load_legacy_format(file, payload, attr_data)
        else:
            _load_caption_format(file, payload, attr_data)
    
    debug_print(f"✅ Loaded {len(all_entries)} entries from JSON", level="success")
    return all_entries


def load_easy_funccap_dataset(questions_file: str) -> List[Dict[str, Any]]:
    """Load Easy Funccap dataset from simplified JSON format.

    The dataset contains pre-annotated options and annotated image paths.
    """
    debug_print(f"\n📂 Loading Easy Funccap dataset: {questions_file}", level="step")

    if not questions_file:
        debug_print("❌ Easy Funccap requires a questions file path", level="error")
        return []

    if not os.path.exists(questions_file):
        debug_print(f"❌ Easy Funccap file not found: {questions_file}", level="error")
        return []

    with open(questions_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entries_data = payload.get("entries", [])
    if not entries_data:
        debug_print(f"⚠️  No entries in Easy Funccap dataset: {questions_file}", level="warn")
        return []

    dataset_dir = os.path.dirname(os.path.abspath(questions_file))
    repo_dir = globals().get('repo_root')
    image_cache: Dict[str, Optional[str]] = {}

    def _resolve_image_path(path_str: str) -> Optional[str]:
        if not path_str:
            return None
        if path_str in image_cache:
            return image_cache[path_str]

        candidate = os.path.expanduser(path_str)
        resolved: Optional[str] = None
        if candidate and os.path.exists(candidate):
            resolved = candidate
        else:
            basename = os.path.basename(candidate)
            search_dirs = [
                dataset_dir,
                os.path.join(dataset_dir, "images"),
                os.path.join(dataset_dir, "annotated_images"),
            ]
            if repo_dir:
                search_dirs.extend([
                    os.path.join(repo_dir, "utils", "data_utils", "autoguiv2"),
                    os.path.join(repo_dir, "testdata"),
                ])
            for base in search_dirs:
                if not base:
                    continue
                trial = os.path.join(base, basename)
                if os.path.exists(trial):
                    resolved = trial
                    break

        if not resolved:
            debug_print(f"⚠️  Easy Funccap image not found: {path_str}", level="warn")

        image_cache[path_str] = resolved
        return resolved

    all_entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries_data):
        annotated_image_path = entry.get("annotated_image_path", "")
        options = entry.get("options", [])
        correct_index = entry.get("correct_index", -1)
        correct_indices = entry.get("correct_indices", None) or entry.get("correct_option_indices", None)

        if not isinstance(options, list) or not options:
            debug_print(f"⚠️  Skipping entry {idx}: missing options", level="warn")
            continue

        correct_option_indices = _normalize_correct_indices(
            correct_indices if correct_indices is not None else correct_index,
            len(options),
            option_texts=options
        )

        if not correct_option_indices:
            debug_print(f"⚠️  Skipping entry {idx}: invalid correct_index {correct_index}", level="warn")
            continue

        correct_index = correct_option_indices[0]

        image_path = _resolve_image_path(annotated_image_path)
        if not image_path:
            debug_print(f"⚠️  Skipping entry {idx}: image not found", level="warn")
            continue

        image_name = os.path.basename(image_path)
        target_region_id = entry.get("target_region_id", "")
        target_region_type = entry.get("target_region_type", "") or ""
        region_parent = resolve_parent_category(str(target_region_type))
        option_region_ids = entry.get("option_region_ids", [])
        group_match = re.search(r"group(\d+)", os.path.splitext(os.path.basename(annotated_image_path))[0])
        group_index = int(group_match.group(1)) if group_match else -1

        dataset_name = payload.get("metadata", {}).get("dataset_name") if isinstance(payload.get("metadata"), dict) else None
        if not dataset_name:
            dataset_name = payload.get("dataset_name", "easy_funccap")

        correct_answers = [options[i] for i in correct_option_indices if i < len(options)]
        easy_entry = {
            "entry_id": f"easy_{idx:05d}_{image_name}_{target_region_id}",
            "image_path": image_path,
            "image_name": image_name,
            "dataset_name": dataset_name,
            "question": FUNCCAP_DEFAULT_QUESTION,
            "options": options,
            "correct_option_idx": correct_index,
            "correct_option_indices": correct_option_indices,
            "correct_answer": options[correct_index],
            "correct_answers": correct_answers,
            "group_index": group_index,
            "target_elem_id": target_region_id,
            "density_class": "unknown",
            "num_similar_elements": len(options),
            "region_type": target_region_type,
            "region_parent": region_parent,
            "area_class": None,
            "option_region_ids": option_region_ids,
            "source_annotated_image_path": annotated_image_path,
        }
        all_entries.append(easy_entry)

    debug_print(f"✅ Loaded {len(all_entries)} Easy Funccap entries", level="success")
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
    preferred_dir_name = 'elemgnd_hf_dataset_cache'
    if task_type in ('funccap', 'easy_funccap'):
        preferred_dir_name = 'regioncap_hf_dataset_cache'
    cache_dir = os.path.join(script_dir, preferred_dir_name)
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
        # Handle image - for funccap, use annotated_image
        if task_type in ('funccap', 'easy_funccap'):
            image = item.get('annotated_image') or item.get('annotated_image_path')
        else:
            image = item.get('image') or item.get('image_path') or item.get('image_file')
        
        if image is None:
            continue

        # Extract image_name early for use in both PIL and string cases
        if task_type in ('funccap', 'easy_funccap'):
            image_name = item.get('annotated_image_name', f'annotated_image_{idx}')
        else:
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
            if (not original_image_name) or original_image_name.startswith('image_') or original_image_name.startswith('annotated_image_'):
                image_name = os.path.basename(image_path)
            else:
                image_name = original_image_name
        else:
            debug_print(f"⚠️  Unsupported image type: {type(image)}", level="warn")
            continue

        # Extract required fields based on task type
        if task_type in ('funccap', 'easy_funccap'):
            # For funccap, use fixed question and get options from dataset
            # Prefer option_contexts, fallback to option_functionalities
            question = FUNCCAP_DEFAULT_QUESTION
            option_contexts = item.get('option_contexts', [])
            if not option_contexts:
                option_contexts = item.get('option_functionalities', [])
            raw_correct = (item.get('correct_option_indices') or item.get('correct_option_idx') or
                           item.get('correct_indices') or item.get('correct_index'))
            correct_option_indices = _normalize_correct_indices(
                raw_correct, len(option_contexts), option_texts=option_contexts
            )

            if not option_contexts or not isinstance(option_contexts, list) or len(option_contexts) == 0:
                continue
            if not correct_option_indices:
                continue
            correct_option_idx = correct_option_indices[0]
        elif task_type == 'funcgnd':
            question = item.get('question', '') or item.get('query', '') or item.get('ref', '')
        elif task_type == 'descgnd':
            # For descgnd, prefer selecting from option_descriptions via correct_option_idx
            correct_idx = item.get('correct_option_idx')
            if correct_idx is None:
                correct_idx = item.get('correct_option_indices')
            if isinstance(correct_idx, list):
                correct_idx = correct_idx[0] if correct_idx else -1
            if isinstance(correct_idx, str) and correct_idx.isdigit():
                correct_idx = int(correct_idx)
            option_descriptions = item.get('option_descriptions')
            if isinstance(option_descriptions, list) and isinstance(correct_idx, int) and 0 <= correct_idx < len(option_descriptions):
                question = option_descriptions[correct_idx]
            else:
                question = item.get('description', '') or item.get('caption', '')
        else:
            question = item.get('question', '') or item.get('query', '')

        # Region attributes (fine-grained from option_region_types via correct_option_idx)
        correct_idx = item.get('correct_option_idx')
        if correct_idx is None:
            correct_idx = item.get('correct_option_indices')
        if isinstance(correct_idx, list):
            correct_idx = correct_idx[0] if correct_idx else -1
        if isinstance(correct_idx, str) and correct_idx.isdigit():
            correct_idx = int(correct_idx)
        if not isinstance(correct_idx, int) and task_type in ('funccap', 'easy_funccap'):
            correct_idx = correct_option_idx
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

        # For funccap-style tasks, create entry with options and correct answer
        if task_type in ('funccap', 'easy_funccap'):
            correct_answers = [option_contexts[i] for i in correct_option_indices if i < len(option_contexts)]
            entry = {
                'entry_id': f"{image_name}_{group_index}_{target_elem_id}_{idx}",
                'image_path': image_path,
                'image_name': image_name,
                'dataset_name': dataset_name,
                'question': question,
                'options': option_contexts,
                'correct_option_idx': correct_option_idx,
                'correct_option_indices': correct_option_indices,
                'correct_answer': option_contexts[correct_option_idx] if correct_option_idx >= 0 else '',
                'correct_answers': correct_answers,
                'group_index': group_index,
                'target_elem_id': target_elem_id,
                'density_class': density_class,
                'num_similar_elements': num_similar_elements,
                'region_type': region_type,
                'region_parent': region_parent,
                'area_class': area_class,
            }
            all_entries.append(entry)
        else:
            # For grounding tasks, require bbox
            bbox = item.get('correct_bbox', [])
            if not bbox or len(bbox) != 4:
                continue
            # Convert bbox to list of floats if needed
            bbox = [float(x) for x in bbox]
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


def load_checkpoint(checkpoint_file: str, task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Load evaluation checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint JSON file (can be checkpoint or full result file)
        task_type: Task type ('funcgnd', 'descgnd', 'intentgnd', 'funccap')
    
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
        # Success criteria depends on task type
        successful_ids = set()
        failed_ids = set()
        
        for entry_id in processed_ids:
            result = results.get(entry_id)
            
            if result:
                # Check if inference was successful based on task type
                if task_type in ('funccap', 'easy_funccap'):
                    # For funccap, only check inference_done (is_correct is for metrics, not for retry)
                    if result.get('inference_done', False):
                        successful_ids.add(entry_id)
                    else:
                        failed_ids.add(entry_id)
                else:
                    # For grounding tasks, check inference_done and pred_bbox
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
        # Only set default base_url if not provided by user
        if not base_url:
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
    elif any(x in model.lower() for x in ['atlas']):
        base_url = 'https://rvrvpi4zz7ispneo.us-east-1.aws.endpoints.huggingface.cloud/v1/'
        api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
        cloud_model_class = HFEndpoint
    elif 'opencua' in model.lower():
        # OpenCUA model - Use XIAOAI API proxy (same as eval_elemgnd_mp.py)
        base_url = base_url or os.environ.get("OPENAI_API_BASE_XIAOAI", "EMPTY")
        api_key = api_key or os.environ.get("OPENAI_API_KEY_XIAOAI", "EMPTY")
        cloud_model_class = OpenAIModel
    elif 'tars' in model.lower():
        base_url = 'https://api.parasail.io/v1'
        api_key = api_key or os.environ.get("PARASAIL_API_KEY", "EMPTY")
        cloud_model_class = PARASAIL
        MAX_TOKENS = 2048
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
        max_tokens=MAX_TOKENS
    )

def _score_multi_select(pred_indices: List[int], correct_indices: List[int]) -> float:
    """Score multi-select answers.

    - Exact match: 1.0
    - Partial correct (strict subset): score = num_correct_selected / total_correct_options
    - Any wrong option or empty prediction: 0.0
    
    Examples:
        - Correct: [a,b,c], Pred: [a,b,c] → 1.0 (exact match)
        - Correct: [a,b,c], Pred: [a,b] → 2/3 = 0.667 (selected 2 out of 3)
        - Correct: [a,b,c], Pred: [a] → 1/3 = 0.333 (selected 1 out of 3)
        - Correct: [a,b], Pred: [a,b,c] → 0.0 (selected wrong option c)
        - Correct: [a,b], Pred: [] → 0.0 (empty prediction)
    """
    pred_set = set(pred_indices)
    correct_set = set(correct_indices)
    
    # Empty prediction gets 0
    if not pred_set:
        return 0.0
    
    # Exact match gets full score
    if pred_set == correct_set:
        return 1.0
    
    # If prediction is a strict subset of correct answers, score by proportion
    if pred_set.issubset(correct_set):
        return len(pred_set) / len(correct_set)
    
    # Any wrong option gets 0
    return 0.0


def parse_multiple_choice_answer(response: str, options: List[str], correct_option_indices: Any) -> tuple:
    """Parse multiple choice answer(s) from model response.

    This parser prioritizes explicit final answers (letters or option text)
    and avoids matching option text that only appears in earlier reasoning.

    Args:
        response: Model response string
        options: List of option strings
        correct_option_indices: Index or list of indices for correct answers

    Returns:
        Tuple of (predicted_indices, score, is_correct)
    """
    normalized_correct = _normalize_correct_indices(correct_option_indices, len(options), option_texts=options)
    if response is None:
        return [], 0.0, False

    def letter_to_index(letter: str) -> Optional[int]:
        letter = letter.upper()
        if len(letter) != 1 or letter < 'A':
            return None
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return idx
        return None

    def extract_indices_from_text(text: str) -> List[int]:
        indices: List[int] = []
        # Match standalone option letters (e.g., "C", "Answer: C", "A,C")
        # More strict pattern: allow commas, spaces, or end-of-string after letters
        letter_matches = re.findall(r'(?:^|[^a-zA-Z])([A-H])(?:[^a-zA-Z]|$)', text, flags=re.IGNORECASE)
        for letter in letter_matches:
            idx = letter_to_index(letter)
            if idx is not None and idx not in indices:
                indices.append(idx)

        # Match numeric answers (e.g., "Answer: 2, 3")
        if not indices:
            for token in re.findall(r'\b\d+\b', text):
                try:
                    idx = int(token)
                    if 0 <= idx < len(options) and idx not in indices:
                        indices.append(idx)
                except ValueError:
                    pass

        if indices:
            return sorted(indices)

        # Match when the model repeats the option text (possibly multiple) as the final line
        normalized_line = re.sub(r'[.\s]+$', '', text.strip().lower())
        if not normalized_line:
            return []
        parts = [p.strip() for p in re.split(r'(?:,|;|/|\||、| and |以及|及|和)+', normalized_line) if p.strip()]
        if not parts:
            parts = [normalized_line]
        matched: List[int] = []
        for part in parts:
            for idx, option in enumerate(options):
                option_normalized = re.sub(r'[.\s]+$', '', str(option).strip().lower())
                if part == option_normalized and idx not in matched:
                    matched.append(idx)
        return sorted(matched)

    def evaluate_lines(lines: List[str]) -> Optional[List[int]]:
        """Inspect lines (from bottom to top) to find an answer signal."""
        for raw_line in reversed(lines):
            line = raw_line.strip()
            if not line:
                continue
            if re.fullmatch(r'<\s*/?\s*think\s*>', line, flags=re.IGNORECASE):
                continue

            indices = extract_indices_from_text(line)
            if indices:
                return indices
        return None

    response_str = response.strip()
    if not response_str:
        return [], 0.0, False

    # Prefer the portion after </think> if present
    sections = []
    think_split = re.split(r'</think>', response, flags=re.IGNORECASE)
    if len(think_split) > 1:
        sections.append(think_split[-1])
    sections.append(response_str)

    for section in sections:
        lines = [line for line in section.splitlines() if line.strip()]
        indices = evaluate_lines(lines)
        if indices is not None:
            score = _score_multi_select(indices, normalized_correct)
            return indices, score, score == 1.0

    # If no match found, return empty and False
    return [], 0.0, False

def process_funccap_entry(entry: Dict, worker_id: int = 0) -> Dict[str, Any]:
    """Process a single funccap dataset entry (multiple choice task)
    
    Args:
        entry: Dataset entry dictionary with options and correct_answer
        worker_id: Worker ID for logging
    
    Returns:
        Result dictionary with accuracy metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    question = entry.get('question', FUNCCAP_DEFAULT_QUESTION)
    options = entry.get('options', [])
    correct_option_indices = entry.get('correct_option_indices', None)
    correct_option_idx = entry.get('correct_option_idx', -1)
    correct_answer = entry.get('correct_answer', '')
    correct_answers = entry.get('correct_answers', None)
    
    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'question': question,
        'options': options,
        'correct_option_idx': correct_option_idx,
        'correct_option_indices': correct_option_indices,
        'correct_answer': correct_answer,
        'correct_answers': correct_answers,
        'pred_option_idx': None,
        'pred_option_indices': None,
        'pred_answer': None,
        'pred_answers': None,
        'score': 0.0,
        'is_correct': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        # Preserve metadata for decomposed metrics
        'density_class': entry.get('density_class', 'unknown'),
        'num_similar_elements': entry.get('num_similar_elements', -1),
        'region_type': entry.get('region_type', ''),
        'region_parent': entry.get('region_parent', 'Others'),
        'area_class': entry.get('area_class'),
    }
    
    start_time = time.time()
    
    if not options or len(options) == 0:
        result['error'] = "No options found in entry"
        result['processing_time'] = time.time() - start_time
        return result
    
    normalized_correct_indices = _normalize_correct_indices(
        correct_option_indices if correct_option_indices is not None else correct_option_idx,
        len(options),
        option_texts=options
    )
    if not normalized_correct_indices:
        result['error'] = f"Invalid correct_option_indices: {correct_option_indices}"
        result['processing_time'] = time.time() - start_time
        return result
    correct_option_idx = normalized_correct_indices[0]
    correct_answers_list = [options[i] for i in normalized_correct_indices if i < len(options)]
    result['correct_option_indices'] = normalized_correct_indices
    result['correct_option_idx'] = correct_option_idx
    result['correct_answers'] = correct_answers_list
    result['correct_answer'] = options[correct_option_idx] if 0 <= correct_option_idx < len(options) else result.get('correct_answer')
    
    retry = 0
    while retry < 4:
        try:
            retry += 1
            
            # Format options as A, B, C, D, etc.
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            formatted_options = []
            for idx, opt in enumerate(options):
                if idx < len(option_letters):
                    formatted_options.append(f"{option_letters[idx]}. {opt}")
                else:
                    formatted_options.append(f"{idx + 1}. {opt}")
            
            options_text = "\n".join(formatted_options)
            
            # Initialize system_prompt (None for most models)
            system_prompt = None
            
            # Handle model-specific prompts
            if 'opencua' in worker_model.model.lower():
                system_prompt = OPENCUA_SYSPROMPT
                # For OpenCUA, use a simpler action-based prompt
                prompt = f"{question}\n\nOptions:\n{options_text}\n\nSelect the correct option(s)."
            elif 'holo' in worker_model.model.lower():
                # Holo models can use standard funccap prompt
                prompt = FUNCCAP_PROMPT.format(question=question, options=options_text)
            elif any(x in worker_model.model.lower() for x in ['infigui-g1', 'gui-r1', 'venus']):
                # These models use standard prompt
                prompt = FUNCCAP_PROMPT.format(question=question, options=options_text)
            else:
                prompt = FUNCCAP_PROMPT.format(question=question, options=options_text)
            # print(f"[Worker {worker_id}] 📝 Funccap prompt follows:\n{prompt}\n")
            
            temp_img_path = image_path
            is_resized = False
            if any(x in worker_model.model.lower() for x in ['claude']):
                is_resized = True
                temp_img_path = f'claude_temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            
            # Pre-logging: Show query start information
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            print(f"[Worker {worker_id}] 🚀 Starting funccap query{retry_info} | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Image: {image_name_short} | [{timestamp_start}]")
            
            try:
                # Get model response (use *_ to handle variable return values for thinking models)
                # OpenAIModel supports image_first parameter for OpenCUA, Holo, InfiGUI-G1
                # PARASAIL and other models don't support it, so we need to check the model type
                base_kwargs = {
                    'use_img_url': True,
                    'temperature': 0.0,
                    'timeout': 360,
                    'sys_prompt': system_prompt if system_prompt else ''
                }
                
                # Only add image_first parameter for OpenAIModel
                if isinstance(worker_model, OpenAIModel):
                    # OpenCUA, Holo, InfiGUI-G1 need image_first=True
                    base_kwargs['image_first'] = any(x in worker_model.model.lower() for x in ['opencua', 'holo', 'infigui-g1'])
                
                success, response, *_ = worker_model.get_model_response(
                    prompt,
                    [temp_img_path],
                    **base_kwargs
                )
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
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(e)[:200]
                print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue
            
            if not success:
                result['error'] = f"API call failed: {response}"
                result['processing_time'] = time.time() - start_time
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg_short = str(response)[:200] if response else "Unknown error"
                print(f"[Worker {worker_id}] ❌ Query FAILED | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: {error_msg_short} | "
                      f"Time: {result['processing_time']:.2f}s | [{timestamp_error}]")
                continue
            
            result['prompt'] = prompt
            result['response'] = response
            # print(f"[Worker {worker_id}] 🧠 Model raw response:\n{response}\n")
            
            # Parse answer from response
            pred_indices, score, is_correct = parse_multiple_choice_answer(response, options, normalized_correct_indices)
            
            result['pred_option_indices'] = pred_indices
            result['pred_option_idx'] = pred_indices[0] if len(pred_indices) == 1 else None
            result['pred_answers'] = [options[i] for i in pred_indices if 0 <= i < len(options)]
            result['pred_answer'] = options[pred_indices[0]] if len(pred_indices) == 1 else None
            result['score'] = score
            result['is_correct'] = is_correct
            result['inference_done'] = True
            
            # Logging
            status = "✅" if score == 1.0 else ("⚠️" if score > 0.0 else "❌")
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pred_str = "N/A"
            if pred_indices:
                pred_str = f"Options {pred_indices} ({[options[i] for i in pred_indices if 0 <= i < len(options)]})"
            correct_str = f"Options {normalized_correct_indices} ({correct_answers_list})"
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | "
                  f"GT: {correct_str} <=> Pred: {pred_str} | "
                  f"Score: {score:.3f} | Time: {processing_time:.2f}s | [{timestamp}]")
            
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
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            continue
    else:
        result['error'] = "Failed to get valid response after all retries"
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry_id} | "
              f"Model: {worker_model.model} | Error: Failed to get valid response | "
              f"Total time: {time.time() - start_time:.2f}s | [{timestamp_error}]")
    
    result['processing_time'] = time.time() - start_time
    return result

def process_entry(entry: Dict, worker_id: int = 0, scale: int = 1000, task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Process a single dataset entry
    
    Args:
        entry: Dataset entry dictionary
        worker_id: Worker ID for logging
        scale: Scale for bbox normalization (default: 1000, not used for funccap)
        task_type: Task type ('funcgnd', 'descgnd', 'intentgnd', 'funccap')
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    
    # Handle funccap task differently
    if task_type in ('funccap', 'easy_funccap'):
        return process_funccap_entry(entry, worker_id)
    
    # Original grounding task logic
    # Get the appropriate field based on task type
    question = entry.get('question', '')

    gt_bbox = entry['gt_bbox']
    
    if gt_bbox[0] > 1:
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
            system_prompt = None

            if 'gemini' in worker_model.model.lower():
                prompt = GEMINI_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                
            elif 'tars' in worker_model.model.lower():
                prompt = UI_TARS_PROMPT.format(question=question)
            elif 'atlas' in worker_model.model.lower():
                prompt = OSATLAS_PROMPT.format(question=question)
            elif 'opencua' in worker_model.model.lower():
                system_prompt = OPENCUA_SYSPROMPT
                prompt = f"{question} Click on the target element"
            elif any(x in worker_model.model.lower() for x in ['claude']):
                prompt = CLAUDE_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
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
                # Get model response (use *_ to handle variable return values for thinking models)
                # OpenAIModel supports image_first parameter for OpenCUA, Holo, InfiGUI-G1
                # PARASAIL and other models don't support it, so we need to check the model type
                base_kwargs = {
                    'use_img_url': True,
                    'temperature': 0.0,
                    'timeout': 360,
                    'sys_prompt': system_prompt if system_prompt else ''
                }
                
                # Only add image_first parameter for OpenAIModel
                if isinstance(worker_model, OpenAIModel):
                    # OpenCUA, Holo, InfiGUI-G1 need image_first=True
                    base_kwargs['image_first'] = any(x in worker_model.model.lower() for x in ['opencua', 'holo', 'infigui-g1'])
                
                success, response, *_ = worker_model.get_model_response(
                    prompt, 
                    [temp_img_path], 
                    **base_kwargs
                )
            except Exception as e:
                # Exception during API call - log and continue to next retry
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
            if '<think>' in response:
                thinking = response.split('<think>')[1].split('</think>')[0].strip()
                bbox_str = response.split('</think>')[1].strip()
            else:
                thinking, bbox_str = '', response.strip()

            result['thinking'], result['bbox_pred_str'] = thinking, bbox_str

            # Parsing

            
            # Case 1: Gemini
            raw_pred_bbox = None
            if '```json' in bbox_str or bbox_str.startswith('{') and bbox_str.endswith('}'):
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
            # Case 2: GLM-4.5. "The bounding box for the 'wall_95' entry in the Outliner list is <|begin_of_box|>[816, 162, 838, 172]<|end_of_box|>."
            elif '<|begin_of_box|>' in bbox_str:
                raw_pred_bbox = bbox_str.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0].strip()
                raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale)
            # Case 4: UI-Tars: "Action: click(start_box='(1786,924)')"
            elif 'tars' in worker_model.model.lower():
                pass
            # Case 5: OS-Atlas: <|object_ref_start|>language switch<|object_ref_end|><|box_start|>(576,12),(592,42)<|box_end|><|im_end|>
            elif 'atlas' in worker_model.model.lower():
                if '(' in bbox_str and ')' in bbox_str:
                    bbox_str = bbox_str[bbox_str.rfind('('):]
                elif '[' in bbox_str and ']' in bbox_str:
                    bbox_str = bbox_str[bbox_str.rfind('['):]
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale)
            # Case 6: OpenCUA: '## Code:\n```python\npyautogui.click(x=3167, y=360)\n```\n'
            elif 'opencua' in worker_model.model.lower():
                if 'x=' in bbox_str and 'y=' in bbox_str:
                    raw_pred_bbox = [
                        int(bbox_str.split('x=')[1].split(',')[0]),
                        int(bbox_str.split('y=')[1].split(')')[0])
                    ]

            # Fall back to a general parsing method
            if raw_pred_bbox is None or any([p > 1 for p in raw_pred_bbox]):
                raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)

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
    # OpenCUA outputs absolute coordinates, so use scale = -1
    if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'opencua']):
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
                if args.task_type in ('funccap', 'easy_funccap'):
                    if not result.get('inference_done', False):
                        retry_count += 1
                else:
                    if not result.get('inference_done', False) or result.get('pred_bbox') is None:
                        retry_count += 1
    elif isinstance(checkpoint_results, list):
        # Handle list format (from full result files)
        results_by_id = {r.get('entry_id'): r for r in checkpoint_results if 'entry_id' in r}
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in results_by_id:
                result = results_by_id[entry_id]
                if args.task_type in ('funccap', 'easy_funccap'):
                    if not result.get('inference_done', False):
                        retry_count += 1
                else:
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
                if args.task_type in ('funccap', 'easy_funccap'):
                    # For funccap, check if inference_done is True
                    if result.get('inference_done', False):
                        processed_ids[result['entry_id']] = True
                        processed_count.value += 1
                else:
                    # For grounding tasks, check pred_bbox
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

    # Return only results that correspond to the current entries list
    # This ensures that if the dataset changes between runs, we only return current entries
    current_entry_ids = {entry['entry_id'] for entry in entries}
    all_results = [results_dict[entry_id] for entry_id in current_entry_ids if entry_id in results_dict]
    
    # Add warning if some entries are missing from results
    missing_count = len(current_entry_ids) - len(all_results)
    if missing_count > 0:
        debug_print(f"⚠️  Warning: {missing_count} entries were not processed and have no results", level="warn")

    return all_results


def calculate_metrics_for_subset(subset_results: List[Dict], task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Calculate metrics for a subset of results"""
    total = len(subset_results)
    successful = sum(1 for r in subset_results if r.get('inference_done', False))
    
    if successful == 0:
        if task_type in ('funccap', 'easy_funccap'):
            return {
                'total': total,
                'successful': 0,
                'success_rate': 0.0,
                'accuracy': 0.0
            }
        else:
            return {
                'total': total,
                'successful': 0,
                'success_rate': 0.0,
                'avg_iou': 0.0,
                'iou_thresholds': {},
                'center_acc': 0.0
            }
    
    if task_type in ('funccap', 'easy_funccap'):
        # For funccap, calculate average score (proportional scoring for partial matches)
        score_sum = sum(r.get('score', 0.0) for r in subset_results if r.get('inference_done', False))
        accuracy = score_sum / total if total > 0 else 0.0
        
        # Additional metrics for multi-select evaluation
        exact_matches = sum(1 for r in subset_results if r.get('inference_done', False) and r.get('score', 0.0) == 1.0)
        partial_matches = sum(1 for r in subset_results if r.get('inference_done', False) and 0.0 < r.get('score', 0.0) < 1.0)
        
        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'accuracy': accuracy,
            'exact_match_rate': exact_matches / total if total > 0 else 0.0,
            'partial_match_rate': partial_matches / total if total > 0 else 0.0
        }
    else:
        # For grounding tasks, calculate IoU metrics
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


def calculate_metrics(results: List[Dict], task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Calculate evaluation metrics with decomposed breakdowns"""
    total = len(results)
    successful = sum(1 for r in results if r.get('inference_done', False))

    if task_type in ('funccap', 'easy_funccap'):
        # For funccap, calculate accuracy
        if successful == 0:
            return {
                'total': total,
                'successful': 0,
                'success_rate': 0.0,
                'accuracy': 0.0,
                'decomposed': {}
            }
        
        score_sum = sum(r.get('score', 0.0) for r in results if r.get('inference_done', False))
        accuracy = score_sum / total if total > 0 else 0.0
        
        # Additional metrics for multi-select evaluation
        exact_matches = sum(1 for r in results if r.get('inference_done', False) and r.get('score', 0.0) == 1.0)
        partial_matches = sum(1 for r in results if r.get('inference_done', False) and 0.0 < r.get('score', 0.0) < 1.0)
        
        # Region parent (6-bucket) breakdown
        region_parent_breakdown = {}
        for parent in ['Primary Interface Containers',
                       'Global Navigation & Structure',
                       'Content & Data Display',
                       'Interaction & Input',
                       'Contextual & Temporary Regions',
                       'Others']:
            subset = [r for r in results if r.get('inference_done', False) and r.get('region_parent', 'Others') == parent]
            region_parent_breakdown[parent] = calculate_metrics_for_subset(subset, task_type) if subset else {
                'total': 0,
                'successful': 0,
                'success_rate': 0.0,
                'accuracy': 0.0,
                'exact_match_rate': 0.0,
                'partial_match_rate': 0.0
            }
        
        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'accuracy': accuracy,
            'exact_match_rate': exact_matches / total if total > 0 else 0.0,
            'partial_match_rate': partial_matches / total if total > 0 else 0.0,
            'decomposed': {
                'by_region_parent': region_parent_breakdown,
            }
        }
    else:
        # For grounding tasks, calculate IoU metrics
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

        # Region parent (6-bucket) breakdown
        region_parent_breakdown = {}
        for parent in ['Primary Interface Containers',
                       'Global Navigation & Structure',
                       'Content & Data Display',
                       'Interaction & Input',
                       'Contextual & Temporary Regions',
                       'Others']:
            subset = [r for r in results if r.get('inference_done', False) and r.get('region_parent', 'Others') == parent]
            region_parent_breakdown[parent] = calculate_metrics_for_subset(subset, task_type) if subset else {
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
                'by_region_parent': region_parent_breakdown,
            }
        }


def main(args):
    """Main evaluation function"""
    debug_print("═" * 60, level="title")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.task_type == 'easy_funccap':
        if not args.questions_file:
            args.questions_file = os.path.join(script_dir, 'easy_func_dataset_sample.json')
        # Easy Funccap always uses local JSON; ignore HF configuration
        args.hf_dataset_id = None
    if args.task_type in ('funccap', 'easy_funccap'):
        title = "🔍 Functional Region Captioning (Funccap) Evaluation"
        if args.task_type == 'easy_funccap':
            title = "🔍 Easy Functional Region Captioning Evaluation"
        debug_print(title, level="title")
    else:
        debug_print("🔍 Functional Region Grounding Evaluation", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    if args.task_type == 'easy_funccap':
        debug_print(f"   Source: {Fore.GREEN}Easy Funccap JSON File{Style.RESET_ALL}", level="info")
        debug_print(f"   Questions File: {Fore.CYAN}{args.questions_file}{Style.RESET_ALL}", level="info")
    elif args.hf_dataset_id:
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
    # Priority: questions_file > hf_dataset_id (even if hf_dataset_id has default value)
    if args.task_type == 'easy_funccap':
        entries = load_easy_funccap_dataset(args.questions_file)
    elif args.questions_file:
        # If questions_file is explicitly provided, use it (ignore hf_dataset_id default)
        entries = load_evaluation_dataset('json', questions_file=args.questions_file, task_type=args.task_type)
    elif args.hf_dataset_id:
        entries = load_evaluation_dataset('hf', hf_dataset_id=args.hf_dataset_id, 
                              hf_split=args.hf_split, hf_cache_dir=args.hf_cache_dir, task_type=args.task_type)
    else:
        entries = load_evaluation_dataset('json', questions_file=args.questions_file, task_type=args.task_type)
    
    if not entries:
        debug_print("❌ No entries found in dataset", level="error")
        return
    
    # Override question format if force_multi_select is enabled
    if getattr(args, 'force_multi_select', False):
        debug_print(f"\n🔄 Forcing multi-select question format for all {len(entries)} entries...", level="step")
        for entry in entries:
            entry['question'] = FUNCCAP_DEFAULT_QUESTION
        debug_print(f"   ✅ All questions updated to: {FUNCCAP_DEFAULT_QUESTION}", level="info")
    
    # If only printing samples, preview and exit early (no model/API usage)
    if getattr(args, 'print_samples', 0):
        limit = min(args.print_samples, len(entries))
        debug_print(f"\n👀 Previewing first {limit} normalized entries:", level="step")
        for idx, e in enumerate(entries[:limit], 1):
            debug_print(f"[{idx}] entry_id={e.get('entry_id')}, image={os.path.basename(e.get('image_path',''))}", level="info")
            debug_print(f"     question={e.get('question')}", level="info")
            debug_print(f"     region_type={e.get('region_type')} | region_parent={e.get('region_parent')}", level="info")
            debug_print(f"     density_class={e.get('density_class')} | area_class={e.get('area_class')} | num_similar_elements={e.get('num_similar_elements')}", level="info")
            if args.task_type in ('funccap', 'easy_funccap'):
                options = e.get('options', [])
                correct_idx = e.get('correct_option_idx', -1)
                correct_indices = e.get('correct_option_indices', None)
                if correct_indices is None:
                    correct_indices = [correct_idx] if isinstance(correct_idx, int) and correct_idx >= 0 else []
                debug_print(f"     options={options}", level="info")
                debug_print(f"     correct_option_indices={correct_indices}", level="info")
                if correct_indices:
                    debug_print(f"     correct_answers={[options[i] for i in correct_indices if 0 <= i < len(options)]}", level="info")
            else:
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
        checkpoint = load_checkpoint(checkpoint_file, args.task_type)
    
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
    metrics = calculate_metrics(results, args.task_type)
    
    # Prepare final output
    output = {
        'metadata': {
            'model': args.model,
            'base_url': args.base_url,
            'questions_file': args.questions_file if args.questions_file else None,
            'hf_dataset_id': args.hf_dataset_id if (args.hf_dataset_id and not args.questions_file) else None,
            'hf_split': args.hf_split if (args.hf_dataset_id and not args.questions_file) else None,
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
        if args.task_type in ('funccap', 'easy_funccap'):
            overall_table.add_row("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.1f}%")
        else:
            overall_table.add_row("Average IoU", f"{metrics['avg_iou']:.3f}")
            overall_table.add_row("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%")
        
        console.print(overall_table)
        
        # IoU thresholds table (only for grounding tasks)
        if args.task_type not in ('funccap', 'easy_funccap') and 'iou_thresholds' in metrics:
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
            if args.task_type in ('funccap', 'easy_funccap'):
                parent_table = Table(title="🗂️ Metrics by Region Parent (6 classes)", box=box.ROUNDED, show_header=True, header_style="bold yellow")
                parent_table.add_column("Parent", style="cyan")
                parent_table.add_column("Total", style="white", justify="right")
                parent_table.add_column("Success Rate", style="green", justify="right")
                parent_table.add_column("Accuracy", style="green", justify="right")
            else:
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
                if args.task_type in ('funccap', 'easy_funccap'):
                    parent_table.add_row(
                        parent,
                        str(data.get('total', 0)),
                        f"{data.get('success_rate', 0.0)*100:.1f}%",
                        f"{data.get('accuracy', 0.0)*100:.1f}%"
                    )
                else:
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
    else:
        # Fallback to simple printing if rich is not available
        debug_print(f"📊 Total Entries: {metrics['total']}", level="info")
        debug_print(f"✅ Successful: {metrics['successful']} ({metrics['success_rate']*100:.1f}%)", level="info")
        if args.task_type in ('funccap', 'easy_funccap'):
            debug_print(f"🎯 Accuracy: {metrics.get('accuracy', 0.0)*100:.1f}%", level="info")
        else:
            debug_print(f"📈 Average IoU: {metrics['avg_iou']:.3f}", level="info")
            debug_print(f"🎯 Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%", level="info")
            debug_print("\n📊 Accuracy at IoU Thresholds:", level="info")
            for threshold, acc in metrics.get('iou_thresholds', {}).items():
                debug_print(f"   {threshold}: {acc*100:.1f}%", level="info")
        
        if metrics.get('decomposed', {}).get('by_region_parent'):
            debug_print("\n📊 Metrics by Region Parent (6 classes):", level="info")
            for parent, data in metrics['decomposed']['by_region_parent'].items():
                if args.task_type in ('funccap', 'easy_funccap'):
                    debug_print(f"   {parent}: {data['success_rate']*100:.1f}% success, "
                               f"Accuracy={data.get('accuracy', 0.0)*100:.1f}% ({data['successful']}/{data['total']})", level="info")
                else:
                    debug_print(f"   {parent}: {data['success_rate']*100:.1f}% success, "
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
    parser.add_argument("--task-type", type=str, choices=['funcgnd', 'descgnd', 'funccap', 'easy_funccap'], default='funccap',
                        help="Task type to evaluate (functional grounding, description grounding, functionality captioning, or Easy Funccap)")
    # Data source - mutually exclusive: JSON vs HF dataset
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument("--questions-file", type=str, default=None,
                              help="Path to questions JSON file (from 2_generate_func_elemgnd_questions.py) or glob pattern")
    # Default dataset ID based on task type
    args_pre, _ = parser.parse_known_args()
    task_type_pre = getattr(args_pre, 'task_type', 'funccap')
    if task_type_pre == 'funccap':
        default_dataset_id = 'HongxinLi/AutoGUIv2-FuncRegionCap'
    elif task_type_pre == 'easy_funccap':
        default_dataset_id = None
    else:
        default_dataset_id = 'HongxinLi/AutoGUIv2-FuncRegionGnd'
    source_group.add_argument("--hf-dataset-id", type=str, default=default_dataset_id, help="HuggingFace dataset ID (e.g., 'username/dataset-name')")

    # HuggingFace specific arguments
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load from HuggingFace (default: 'test')")
    parser.add_argument("--hf-cache-dir", type=str, default='/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionCap/',
                       help="Cache directory for HuggingFace datasets")

    # Model arguments
    parser.add_argument("--model", type=str, default=[
            'gemini-2.5-pro-thinking',
            'gpt-5',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-thinking',
            'qwen3-vl-32b-instruct',
            'qwen3-vl-8b-instruct',
            'qwen2-vl-72b-instruct',
            'qwen-vl-max-latest',
            'Qwen/Qwen3-VL-32B-Instruct',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'OS-Copilot/OS-Atlas-Base-7B',
            'xlangai/OpenCUA-7B',
            'Hcompany/Holo2-8B',
            'Hcompany/Holo1.5-7B',
            'inclusionAI/UI-Venus-Ground-7B',
            'ritzzai/GUI-R1-7B',
            'InfiX-ai/InfiGUI-G1-7B',
            'step-3',
            'zai-org/GLM-4.5V'
        ][-7],
                       help="Model name (e.g., 'gpt-4o', 'gemini-2.5-pro-thinking', 'Qwen/Qwen3-VL-32B-Instruct', 'Hcompany/Holo2-8B', 'xlangai/OpenCUA-7B')")
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
    parser.add_argument("--force-multi-select", action="store_true",
                       help="Force multi-select question format (override question from data)")
    
    args, _ = parser.parse_known_args()
    
    # Validate arguments
    if args.hf_dataset_id and not HF_AVAILABLE:
        parser.error("--hf-dataset-id requires the 'datasets' library. Install with: pip install datasets")
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)
