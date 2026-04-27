"""
Evaluate VLMs on Element Functionality Grounding tasks

This script evaluates vision-language models on the FuncElemGnd dataset,
where models need to locate GUI elements based on natural language questions.
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

REF_TAGS = {
    'funcgnd': 'a question about locating',
    'descgnd': 'a description of',
    'desccap': 'a description of',
    'funccap': 'a question about locating'
}

REF_PLACEHOLDER = {
    'funcgnd': 'Question',
    'descgnd': 'Description',
    'desccap': 'Description',
    'funccap': 'Question'
}

# Fixed question template for desc gnd
DESC_GND_QUESTION_TEMPLATE = "Which element matches the following visual description: {description}?"

# Grounding prompt template
GEMINI_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [ymin, xmin, ymax, xmax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Now analyze the screenshot and provide the bounding box for the target element:"""

CLAUDE_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax]. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""

# UI-Tars
UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format

Action: ...


## Action Space
click(point='<point>x1 y1</point>'')

## User Instruction
{question}"""

# GENERIC_PROMPT
GENERIC_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax] normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""

# Multiple Choice Grounding prompts (given question, select from options with bboxes)
GEMINI_MC_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to select the correct element from the provided options. Each option has a bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000.

{ref_placeholder}: {question}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the correct element:"""

CLAUDE_MC_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to select the correct element from the provided options. Each option has a bounding box [xmin, ymin, xmax, ymax] normalized to 0-1000.

{ref_placeholder}: {question}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the correct element:"""

GENERIC_MC_GROUNDING_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to select the correct element from the provided options. Each option has a bounding box [xmin, ymin, xmax, ymax] normalized to 0-1000.

{ref_placeholder}: {question}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the correct element:"""

# Description Captioning prompt (given bbox, select the best matching description)
DESC_CAP_PROMPT_GEMINI = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the visual description that best matches the highlighted element from the provided options.

The bounding box coordinates are [ymin, xmin, ymax, xmax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching description:"""

DESC_CAP_PROMPT_CLAUDE = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the visual description that best matches the highlighted element from the provided options.

The bounding box coordinates are [xmin, ymin, xmax, ymax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching description:"""

DESC_CAP_PROMPT_GENERIC = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the visual description that best matches the highlighted element from the provided options.

The bounding box coordinates are [xmin, ymin, xmax, ymax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching description:"""

# Functionality Captioning prompt (given bbox, select the best matching functionality question)
FUNC_CAP_PROMPT_GEMINI = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the functionality question that best matches the highlighted element from the provided options.

The bounding box coordinates are [ymin, xmin, ymax, xmax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching functionality question:"""

FUNC_CAP_PROMPT_CLAUDE = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the functionality question that best matches the highlighted element from the provided options.

The bounding box coordinates are [xmin, ymin, xmax, ymax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching functionality question:"""

FUNC_CAP_PROMPT_GENERIC = """You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, you need to select the functionality question that best matches the highlighted element from the provided options.

The bounding box coordinates are [xmin, ymin, xmax, ymax] normalized to 0-1000: {bbox}

Options:
{options}

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching functionality question:"""


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

def load_dataset_from_json(questions_file: str, field_type: str = 'functionality', task_type: str = 'funcgnd') -> List[Dict[str, Any]]:
    """Load dataset from questions JSON file
    
    Args:
        questions_file: Path to questions JSON file or glob pattern
        field_type: 'functionality' or 'description'
        task_type: 'funcgnd', 'descgnd', 'desccap', etc.
    
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
        
        # Support two formats:
        # 1. Old format: {"results": {"image_name": {...}}}
        # 2. New format: {"result": {...}, "image_key": "..."}
        results = checkpoint.get('results', {})
        result_data = checkpoint.get('result', None)
        
        # If new format detected, convert to old format structure
        if result_data is not None and not results:
            image_key = checkpoint.get('image_key', '')
            # Extract image name from image_key or file path
            if image_key:
                image_name = os.path.basename(image_key).replace('.png', '').replace('.jpg', '')
            else:
                image_name = os.path.basename(file).replace('_result.json', '').replace('.json', '')
            results = {image_name: result_data}
        
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
            
            # Get image size if available
            image_size = image_data.get('image_size', None)
            if not image_size:
                # Try to get from image dimensions
                try:
                    img = Image.open(image_path)
                    image_size = [img.height, img.width]  # [height, width]
                except Exception:
                    image_size = None
            
            # Support new format: questions directly in result.questions
            questions_direct = image_data.get('questions', [])
            generated = image_data.get('generated', [])
            
            # If new format (questions directly in result), process them
            if questions_direct and not generated:
                # New format: process questions directly
                # Group questions by group_id for funccap task
                questions_by_group = {}
                for q_data in questions_direct:
                    group_id = q_data.get('group_id', -1)
                    if group_id not in questions_by_group:
                        questions_by_group[group_id] = []
                    questions_by_group[group_id].append(q_data)
                
                for q_idx, q_data in enumerate(questions_direct):
                    question_text = q_data.get('question', '')
                    options = q_data.get('options', [])
                    correct_answer = q_data.get('correct_answer', '')
                    target_region_id = q_data.get('target_region_id', '')
                    group_id = q_data.get('group_id', q_idx)
                    
                    if not question_text or not options:
                        continue
                    
                    if task_type in ['desccap', 'funccap']:
                        # Captioning task: given bbox, select best option
                        # Find the target option (correct answer)
                        target_option = None
                        correct_option_idx = -1
                        for idx, opt in enumerate(options):
                            if opt.get('label') == correct_answer or opt.get('region_id') == target_region_id:
                                target_option = opt
                                correct_option_idx = idx
                                break
                        
                        if target_option is None or correct_option_idx < 0:
                            continue
                        
                        # Get bbox from target option (for grounding mode) or from first option with bbox
                        gt_bbox = target_option.get('bbox', [])
                        if not gt_bbox:
                            # Try to find bbox from any option
                            for opt in options:
                                if opt.get('bbox'):
                                    gt_bbox = opt['bbox']
                                    break
                        
                        # If still no bbox (captioning mode), try to get from grounding data
                        if not gt_bbox or len(gt_bbox) != 4:
                            # Try to load corresponding grounding file
                            grounding_file = file.replace('captioning_mode', 'grounding_mode')
                            if os.path.exists(grounding_file):
                                try:
                                    with open(grounding_file, 'r', encoding='utf-8') as gf:
                                        gnd_data = json.load(gf)
                                    gnd_result = gnd_data.get('result', {})
                                    gnd_questions = gnd_result.get('questions', [])
                                    # Find matching question by target_region_id
                                    for gq in gnd_questions:
                                        gnd_options = gq.get('options', [])
                                        for gnd_opt in gnd_options:
                                            if gnd_opt.get('region_id') == target_region_id:
                                                gt_bbox = gnd_opt.get('bbox', [])
                                                if gt_bbox and len(gt_bbox) == 4:
                                                    break
                                        if gt_bbox and len(gt_bbox) == 4:
                                            break
                                except Exception:
                                    pass
                        
                        if not gt_bbox or len(gt_bbox) != 4:
                            continue
                        
                        # Prepare options for captioning
                        caption_options = []
                        if task_type == 'desccap':
                            # Description captioning: use description field from options
                            for idx, opt in enumerate(options):
                                desc = opt.get('description', '')
                                if isinstance(desc, dict):
                                    desc = desc.get('revised description') or desc.get('with_context') or desc.get('wo_context') or ''
                                if desc:
                                    caption_options.append({
                                        'label': opt.get('label', chr(65 + idx)),
                                        'description': desc,
                                        'region_id': opt.get('region_id', '')
                                    })
                        elif task_type == 'funccap':
                            # Functionality captioning: collect all questions from the same group
                            # Each question in the group becomes an option
                            group_questions = questions_by_group.get(group_id, [])
                            option_idx = 0
                            for gq in group_questions:
                                gq_question = gq.get('question', '')
                                gq_target_region = gq.get('target_region_id', '')
                                if gq_question:
                                    caption_options.append({
                                        'label': chr(65 + option_idx),  # A, B, C, ...
                                        'question': gq_question,
                                        'region_id': gq_target_region
                                    })
                                    # Check if this is the correct answer
                                    if gq_target_region == target_region_id or gq == q_data:
                                        correct_option_idx = option_idx
                                    option_idx += 1
                        
                        if len(caption_options) < 2:
                            continue
                        
                        # Get metrics from target option
                        metrics = target_option.get('metrics', {})
                        density_class = metrics.get('density', {}).get('density_class', 'unknown') if isinstance(metrics, dict) else 'unknown'
                        
                        entry = {
                            'entry_id': f"{image_name}_{group_id}_{target_region_id}_{task_type}",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'gt_bbox': gt_bbox,
                            'group_index': group_id,
                            'target_elem_id': target_region_id,
                            'density_class': density_class,
                            'num_similar_elements': len(options),
                            'options': caption_options,
                            'correct_option_idx': correct_option_idx,
                            'task_type': task_type,
                        }
                        all_entries.append(entry)
                    
                    elif task_type in ['funcgnd', 'descgnd']:
                        # Grounding task: given question, predict bbox
                        # Find the target option (correct answer)
                        target_option = None
                        for opt in options:
                            if opt.get('label') == correct_answer or opt.get('region_id') == target_region_id:
                                target_option = opt
                                break
                        
                        if target_option is None:
                            continue
                        
                        gt_bbox = target_option.get('bbox', [])
                        if not gt_bbox or len(gt_bbox) != 4:
                            continue
                        
                        # Get metrics
                        metrics = target_option.get('metrics', {})
                        density_class = metrics.get('density', {}).get('density_class', 'unknown') if isinstance(metrics, dict) else 'unknown'
                        
                        # For descgnd, use description from target option
                        if task_type == 'descgnd':
                            desc = target_option.get('description', '')
                            if isinstance(desc, dict):
                                desc = desc.get('revised description') or desc.get('with_context') or desc.get('wo_context') or ''
                            if not desc:
                                continue
                            question_text = DESC_GND_QUESTION_TEMPLATE.format(description=desc)
                        
                        entry = {
                            'entry_id': f"{image_name}_{group_id}_{target_region_id}_{task_type}",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'question': question_text,
                            'action_type': task_type,
                            'gt_bbox': gt_bbox,
                            'group_index': group_id,
                            'target_elem_id': target_region_id,
                            'density_class': density_class,
                            'num_similar_elements': len(options),
                            'options': options,  # Save options for multiple choice format
                            'correct_answer': correct_answer,  # Save correct answer label
                            'image_size': image_size,  # Save image size for bbox normalization
                        }
                        if task_type == 'descgnd':
                            entry['description'] = desc
                        all_entries.append(entry)
                
                continue  # Skip old format processing for this file
            
            # Old format: process generated groups
            for group_data in generated:
                questions = group_data.get('questions', [])
                elements = group_data.get('elements', [])
                elements_by_id = {elem.get('id'): elem for elem in elements}
                num_similar_elements = len(elements)
                
                # Handle different field types and task types
                if field_type == 'description' and task_type == 'descgnd':
                    # For desc gnd: use fixed question template with description field
                    for target_elem_id, target_element in elements_by_id.items():
                        target_bbox = target_element.get('revised bbox', [])
                        if not target_bbox or len(target_bbox) != 4:
                            continue
                        
                        description = target_element.get('description', '')
                        # Handle dict format description
                        if isinstance(description, dict):
                            description = description.get('revised description') or description.get('with_context') or description.get('wo_context') or ''
                        if not description or not isinstance(description, str):
                            continue
                        
                        # Use fixed question template
                        question = DESC_GND_QUESTION_TEMPLATE.format(description=description)
                        
                        # Get density class from attributes if available
                        group_index = group_data.get('group_index', -1)
                        density_class = 'unknown'
                        if image_name in attr_data and str(group_index) in attr_data[image_name] and str(target_elem_id) in attr_data[image_name][str(group_index)]:
                            density_class = attr_data[image_name][str(group_index)][str(target_elem_id)].get('density_class', 'unknown')
                        
                        entry = {
                            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_descgnd",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'question': question,
                            'action_type': 'descgnd',
                            'gt_bbox': target_bbox,
                            'group_index': group_index,
                            'target_elem_id': target_elem_id,
                            'density_class': density_class,
                            'num_similar_elements': num_similar_elements,
                            'description': description,
                        }
                        all_entries.append(entry)
                
                elif field_type == 'description' and task_type == 'desccap':
                    # For desc cap: given bbox, select best matching description from group
                    for target_elem_id, target_element in elements_by_id.items():
                        target_bbox = target_element.get('revised bbox', [])
                        if not target_bbox or len(target_bbox) != 4:
                            continue
                        
                        # Collect all descriptions from the group as options
                        options = []
                        correct_option_idx = -1
                        for idx, (elem_id, elem) in enumerate(elements_by_id.items()):
                            desc = elem.get('description', '')
                            # Handle dict format description
                            if isinstance(desc, dict):
                                desc = desc.get('revised description') or desc.get('with_context') or desc.get('wo_context') or ''
                            if desc and isinstance(desc, str):
                                options.append({
                                    'label': chr(65 + idx),  # A, B, C, ...
                                    'description': desc,
                                    'elem_id': elem_id
                                })
                                if elem_id == target_elem_id:
                                    correct_option_idx = len(options) - 1
                        
                        if len(options) < 2 or correct_option_idx < 0:
                            continue  # Need at least 2 options
                        
                        # Get density class from attributes if available
                        group_index = group_data.get('group_index', -1)
                        density_class = 'unknown'
                        if image_name in attr_data and str(group_index) in attr_data[image_name] and str(target_elem_id) in attr_data[image_name][str(group_index)]:
                            density_class = attr_data[image_name][str(group_index)][str(target_elem_id)].get('density_class', 'unknown')
                        
                        entry = {
                            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_desccap",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'gt_bbox': target_bbox,
                            'group_index': group_index,
                            'target_elem_id': target_elem_id,
                            'density_class': density_class,
                            'num_similar_elements': num_similar_elements,
                            'options': options,
                            'correct_option_idx': correct_option_idx,
                            'task_type': 'desccap',
                        }
                        all_entries.append(entry)
                
                elif field_type == 'functionality' and task_type == 'funccap':
                    # For func cap: given bbox, select best matching functionality question from group
                    for target_elem_id, target_element in elements_by_id.items():
                        target_bbox = target_element.get('revised bbox', [])
                        if not target_bbox or len(target_bbox) != 4:
                            continue
                        
                        # Collect all functionality questions from the group as options
                        options = []
                        correct_option_idx = -1
                        option_idx = 0
                        
                        # Iterate through questions to find all functionality questions for elements in this group
                        for q_data in questions:
                            elem_id = q_data.get('target_element_id', -1)
                            if elem_id not in elements_by_id:
                                continue
                            
                            referring_expressions = q_data.get('referring_expressions', {})
                            # Get questions from all action types
                            for action_type, action_data in referring_expressions.items():
                                if not isinstance(action_data, dict):
                                    continue
                                question = action_data.get('question', '')
                                if question:
                                    options.append({
                                        'label': chr(65 + option_idx),  # A, B, C, ...
                                        'question': question,
                                        'elem_id': elem_id,
                                        'action_type': action_type
                                    })
                                    if elem_id == target_elem_id:
                                        correct_option_idx = option_idx
                                    option_idx += 1
                        
                        if len(options) < 2 or correct_option_idx < 0:
                            continue  # Need at least 2 options
                        
                        # Get density class from attributes if available
                        group_index = group_data.get('group_index', -1)
                        density_class = 'unknown'
                        if image_name in attr_data and str(group_index) in attr_data[image_name] and str(target_elem_id) in attr_data[image_name][str(group_index)]:
                            density_class = attr_data[image_name][str(group_index)][str(target_elem_id)].get('density_class', 'unknown')
                        
                        entry = {
                            'entry_id': f"{image_name}_{group_index}_{target_elem_id}_funccap",
                            'image_path': image_path,
                            'image_name': image_name,
                            'dataset_name': dataset_name,
                            'gt_bbox': target_bbox,
                            'group_index': group_index,
                            'target_elem_id': target_elem_id,
                            'density_class': density_class,
                            'num_similar_elements': num_similar_elements,
                            'options': options,
                            'correct_option_idx': correct_option_idx,
                            'task_type': 'funccap',
                        }
                        all_entries.append(entry)
                
                else:
                    # Original functionality-based logic (funcgnd)
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
                        for action_type, action_data in referring_expressions.items():
                            if not isinstance(action_data, dict):
                                continue
                            
                            question = action_data.get('question', '')
                            if not question:
                                continue
                            
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


def load_dataset_from_hf(hf_dataset_id: str, split: str = 'test', cache_dir: Optional[str] = None, task_type: str = 'funcgnd', field_type: str = 'functionality') -> List[Dict[str, Any]]:
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

            # Handle duplicate names
            if os.path.exists(image_path):
                base, ext = os.path.splitext(image_name)
                image_name = f"{base}_{idx}{ext}"
                image_path = os.path.join(image_cache_dir, image_name)
            
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

        bbox = item.get('bbox', [])
        if not bbox or len(bbox) != 4:
            continue
        
        # Convert bbox to list of floats if needed
        bbox = [float(x) for x in bbox]
        
        action_type = item.get('action_type', 'unknown')
        group_index = item.get('group_index', -1)
        target_elem_id = item.get('id', -1)
        
        # Extract metadata fields
        dataset_name = item.get('dataset_name', 'unknown')
        density_class = item.get('density_class', 'unknown')
        num_similar_elements = item.get('num_similar_elements', -1)
        
        # Handle different task types
        if task_type == 'desccap' or task_type == 'funccap':
            # For desc cap / func cap: need options from similar elements
            # This requires grouping by group_index, which may not be available in HF dataset
            # For now, we'll skip cap tasks for HF datasets or require special format
            # TODO: Implement cap tasks for HF datasets if needed
            debug_print(f"⚠️  {task_type} task not yet fully supported for HF datasets, skipping entry {idx}", level="warn")
            continue
        
        # Extract required fields - use 'question' variable for all task types
        if task_type == 'funcgnd':
            question = item.get('question', '')
        elif task_type == 'descgnd':
            if field_type == 'description':
                # Use fixed question template for desc gnd
                description = item.get('description', '')
                if not description:
                    continue
                question = DESC_GND_QUESTION_TEMPLATE.format(description=description)
            else:
                question = item.get('description', '')
        else:
            question = item.get('question', '')

        if not question:
            continue

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
        if task_type == 'descgnd' and field_type == 'description':
            entry['description'] = item.get('description', '')
        all_entries.append(entry)
    
    debug_print(f"✅ Converted {len(all_entries)} entries from HuggingFace dataset", level="success")
    
    # Save to cache
    save_hf_entries_to_cache(all_entries, cache_path, hf_dataset_id, split, task_type)
    
    debug_print(f"💾 Cached {len(all_entries)} entries and images to persistent cache", level="success")
    
    return all_entries


def load_evaluation_dataset(source: str, questions_file: Optional[str] = None, hf_dataset_id: Optional[str] = None, hf_split: str = 'test', hf_cache_dir: Optional[str] = None, task_type: str = 'funcgnd', field_type: str = 'functionality') -> List[Dict[str, Any]]:
    """Load dataset from either JSON file or HuggingFace Hub
    
    Args:
        source: Source type - 'json' or 'hf'
        questions_file: Path to questions JSON file (required if source='json')
        hf_dataset_id: HuggingFace dataset ID (required if source='hf')
        hf_split: Dataset split for HF datasets (default: 'test')
        hf_cache_dir: Optional cache directory for HF datasets
        task_type: Task type to evaluate ('funcgnd', 'descgnd', 'desccap', etc.)
        field_type: 'functionality' or 'description'
    Returns:
        List of dataset entries
    """
    if source == 'hf':
        if not hf_dataset_id:
            raise ValueError("hf_dataset_id is required when source='hf'")
        return load_dataset_from_hf(hf_dataset_id, hf_split, hf_cache_dir, task_type, field_type)
    else:
        if not questions_file:
            raise ValueError("questions_file is required when source='json'")
        return load_dataset_from_json(questions_file, field_type, task_type)


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

    if 'qwen' in model.lower():
        base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        cloud_model_class = Qwen3VL
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
        max_tokens=8192
    )

def process_entry_desccap(entry: Dict, worker_id: int = 0) -> Dict[str, Any]:
    """Process a desc cap entry (given bbox, select best matching description)
    
    Args:
        entry: Dataset entry dictionary with 'options' and 'correct_option_idx'
        worker_id: Worker ID for logging
        task_type: Task type ('desccap')
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    gt_bbox = entry['gt_bbox']
    options = entry.get('options', [])
    correct_option_idx = entry.get('correct_option_idx', -1)
    
    # Convert bbox to 0-1000 scale if needed
    if gt_bbox[0] <= 1:
        gt_bbox = [x * 1000 for x in gt_bbox]
    
    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'gt_bbox': gt_bbox,
        'options': options,
        'correct_option_idx': correct_option_idx,
        'pred_option': None,
        'correct': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        'density_class': entry.get('density_class', 'unknown'),
        'num_similar_elements': entry.get('num_similar_elements', -1),
        'task_type': 'desccap',
    }
    
    start_time = time.time()
    
    if not options or correct_option_idx < 0 or correct_option_idx >= len(options):
        result['error'] = "Invalid options or correct_option_idx"
        result['processing_time'] = time.time() - start_time
        return result
    
    # Format options string
    options_str = '\n'.join([f"{opt['label']}: {opt['description']}" for opt in options])
    
    retry = 0
    while retry < 4:
        try:
            retry += 1
            
            # Create prompt based on model
            if 'gemini' in worker_model.model.lower():
                prompt = DESC_CAP_PROMPT_GEMINI.format(bbox=gt_bbox, options=options_str)
                temp_img_path = image_path
            elif any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                prompt = DESC_CAP_PROMPT_CLAUDE.format(bbox=gt_bbox, options=options_str)
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            else:
                prompt = DESC_CAP_PROMPT_GENERIC.format(bbox=gt_bbox, options=options_str)
                temp_img_path = image_path
            
            # Pre-logging
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            print(f"[Worker {worker_id}] 🚀 Starting desc cap query{retry_info} | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Image: {image_name_short} | [{timestamp_start}]")
            
            try:
                # Get model response
                success, response, _ = worker_model.get_model_response(
                    prompt,
                    [temp_img_path],
                    use_img_url=True,
                    temperature=0.0,
                    timeout=360
                )
            except Exception as e:
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                if any(x in worker_model.model.lower() for x in ['claude', 'seed']) and os.path.exists(temp_img_path):
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
            
            if any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                os.remove(temp_img_path)
            
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
            
            # Parse answer from response
            pred_option = None
            # Try to extract answer label (A, B, C, etc.)
            answer_match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
            if answer_match:
                pred_option = answer_match.group(1).upper()
            else:
                # Try to find any single letter at the start of a line
                answer_match = re.search(r'^([A-Z]):', response, re.MULTILINE | re.IGNORECASE)
                if answer_match:
                    pred_option = answer_match.group(1).upper()
            
            if not pred_option:
                result['error'] = "Failed to parse answer from response"
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] ❌ Query FAILED (parse error) | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: Could not parse answer | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue
            
            result['pred_option'] = pred_option
            correct_label = options[correct_option_idx]['label']
            result['correct'] = (pred_option.upper() == correct_label.upper())
            result['inference_done'] = True
            
            status = "✅" if result['correct'] else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | "
                  f"Pred: {pred_option} | Correct: {correct_label} | Time: {processing_time:.2f}s | [{timestamp}]")
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            continue
    else:
        result['error'] = "Failed to parse answer from response"
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry_id} | "
              f"Model: {worker_model.model} | Error: Failed to parse answer | "
              f"Total time: {time.time() - start_time:.2f}s | [{timestamp_error}]")
    
    result['processing_time'] = time.time() - start_time
    return result

def process_entry_funcap(entry: Dict, worker_id: int = 0) -> Dict[str, Any]:
    """Process a func cap entry (given bbox, select best matching functionality question)
    
    Args:
        entry: Dataset entry dictionary with 'options' and 'correct_option_idx'
        worker_id: Worker ID for logging
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    gt_bbox = entry['gt_bbox']
    options = entry.get('options', [])
    correct_option_idx = entry.get('correct_option_idx', -1)
    
    # Convert bbox to 0-1000 scale if needed
    if gt_bbox[0] <= 1:
        gt_bbox = [x * 1000 for x in gt_bbox]
    
    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'gt_bbox': gt_bbox,
        'options': options,
        'correct_option_idx': correct_option_idx,
        'pred_option': None,
        'correct': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        'density_class': entry.get('density_class', 'unknown'),
        'num_similar_elements': entry.get('num_similar_elements', -1),
        'task_type': 'funccap',
    }
    
    start_time = time.time()
    
    if not options or correct_option_idx < 0 or correct_option_idx >= len(options):
        result['error'] = "Invalid options or correct_option_idx"
        result['processing_time'] = time.time() - start_time
        return result
    
    # Format options string
    options_str = '\n'.join([f"{opt['label']}: {opt['question']}" for opt in options])
    
    retry = 0
    while retry < 4:
        try:
            retry += 1
            
            # Create prompt based on model
            if 'gemini' in worker_model.model.lower():
                prompt = FUNC_CAP_PROMPT_GEMINI.format(bbox=gt_bbox, options=options_str)
                temp_img_path = image_path
            elif any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                prompt = FUNC_CAP_PROMPT_CLAUDE.format(bbox=gt_bbox, options=options_str)
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            else:
                prompt = FUNC_CAP_PROMPT_GENERIC.format(bbox=gt_bbox, options=options_str)
                temp_img_path = image_path
            
            # Pre-logging
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            print(f"[Worker {worker_id}] 🚀 Starting func cap query{retry_info} | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Image: {image_name_short} | [{timestamp_start}]")
            
            try:
                # Get model response
                success, response, _ = worker_model.get_model_response(
                    prompt,
                    [temp_img_path],
                    use_img_url=True,
                    temperature=0.0,
                    timeout=360
                )
            except Exception as e:
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                if any(x in worker_model.model.lower() for x in ['claude', 'seed']) and os.path.exists(temp_img_path):
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
            
            if any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                os.remove(temp_img_path)
            
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
            
            # Parse answer from response
            pred_option = None
            # Try to extract answer label (A, B, C, etc.)
            answer_match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
            if answer_match:
                pred_option = answer_match.group(1).upper()
            else:
                # Try to find any single letter at the start of a line
                answer_match = re.search(r'^([A-Z]):', response, re.MULTILINE | re.IGNORECASE)
                if answer_match:
                    pred_option = answer_match.group(1).upper()
            
            if not pred_option:
                result['error'] = "Failed to parse answer from response"
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] ❌ Query FAILED (parse error) | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: Could not parse answer | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue
            
            result['pred_option'] = pred_option
            correct_label = options[correct_option_idx]['label']
            result['correct'] = (pred_option.upper() == correct_label.upper())
            result['inference_done'] = True
            
            status = "✅" if result['correct'] else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | "
                  f"Pred: {pred_option} | Correct: {correct_label} | Time: {processing_time:.2f}s | [{timestamp}]")
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            continue
    else:
        result['error'] = "Failed to parse answer from response"
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry_id} | "
              f"Model: {worker_model.model} | Error: Failed to parse answer | "
              f"Total time: {time.time() - start_time:.2f}s | [{timestamp_error}]")
    
    result['processing_time'] = time.time() - start_time
    return result

def process_entry_mc_grounding(entry: Dict, worker_id: int = 0, task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Process a multiple choice grounding entry (given question, select from options with bboxes)
    
    Args:
        entry: Dataset entry dictionary with 'options' and 'correct_answer'
        worker_id: Worker ID for logging
        task_type: Task type ('funcgnd', 'descgnd')
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    question = entry.get('question', '')
    options = entry.get('options', [])
    correct_answer = entry.get('correct_answer', '')
    gt_bbox = entry['gt_bbox']
    image_size = entry.get('image_size', None)  # [height, width]
    
    # Normalize gt_bbox to 0-1000 if needed
    if gt_bbox[0] > 1:
        # Already in pixel coordinates, need to normalize
        if image_size and len(image_size) == 2:
            H, W = image_size[0], image_size[1]
            gt_bbox = [
                gt_bbox[0] * 1000 / W,  # x_min
                gt_bbox[1] * 1000 / H,  # y_min
                gt_bbox[2] * 1000 / W,  # x_max
                gt_bbox[3] * 1000 / H   # y_max
            ]
        else:
            # Fallback: assume already normalized or use image dimensions
            gt_bbox = [x / 1000 if x > 1 else x for x in gt_bbox]
    else:
        # Already normalized, but ensure in 0-1000 range
        gt_bbox = [x * 1000 if x <= 1 else x for x in gt_bbox]
    
    # Find correct option index
    correct_option_idx = -1
    for idx, opt in enumerate(options):
        if opt.get('label') == correct_answer:
            correct_option_idx = idx
            break
    
    result = {
        'entry_id': entry_id,
        'image_path': image_path,
        'image_name': entry.get('image_name', ''),
        'dataset_name': entry.get('dataset_name', 'unknown'),
        'question': question,
        'gt_bbox': gt_bbox,
        'options': options,
        'correct_answer': correct_answer,
        'correct_option_idx': correct_option_idx,
        'pred_option': None,
        'pred_bbox': None,
        'iou': 0.0,
        'center_acc': False,
        'correct': False,
        'inference_done': False,
        'error': None,
        'response': None,
        'processing_time': 0.0,
        'density_class': entry.get('density_class', 'unknown'),
        'num_similar_elements': entry.get('num_similar_elements', -1),
        'action_type': entry.get('action_type', task_type),
        'task_type': task_type,
    }
    
    start_time = time.time()
    
    if not question or not options or correct_option_idx < 0:
        result['error'] = "Invalid question, options, or correct_answer"
        result['processing_time'] = time.time() - start_time
        return result
    
    # Format options string with bbox information
    options_str_list = []
    for opt in options:
        label = opt.get('label', '')
        bbox = opt.get('bbox', [])
        if bbox and len(bbox) == 4:
            # Normalize bbox if needed
            if bbox[0] > 1 and image_size and len(image_size) == 2:
                H, W = image_size[0], image_size[1]
                normalized_bbox = [
                    bbox[0] * 1000 / W,
                    bbox[1] * 1000 / H,
                    bbox[2] * 1000 / W,
                    bbox[3] * 1000 / H
                ]
            else:
                normalized_bbox = [x * 1000 if x <= 1 else x for x in bbox]
            
            # Format bbox based on model type (will be set later)
            options_str_list.append(f"{label}: bbox {normalized_bbox}")
        else:
            options_str_list.append(f"{label}: (no bbox)")
    
    options_str = '\n'.join(options_str_list)
    
    retry = 0
    while retry < 4:
        try:
            retry += 1
            
            # Create prompt based on model
            ref_tag = REF_TAGS.get(task_type, 'a question about locating')
            ref_placeholder = REF_PLACEHOLDER.get(task_type, 'Question')
            
            if 'gemini' in worker_model.model.lower():
                prompt = GEMINI_MC_GROUNDING_PROMPT.format(
                    ref_tag=ref_tag,
                    ref_placeholder=ref_placeholder,
                    question=question,
                    options=options_str
                )
                temp_img_path = image_path
            elif any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                prompt = CLAUDE_MC_GROUNDING_PROMPT.format(
                    ref_tag=ref_tag,
                    ref_placeholder=ref_placeholder,
                    question=question,
                    options=options_str
                )
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            else:
                prompt = GENERIC_MC_GROUNDING_PROMPT.format(
                    ref_tag=ref_tag,
                    ref_placeholder=ref_placeholder,
                    question=question,
                    options=options_str
                )
                temp_img_path = image_path
            
            # Pre-logging
            image_name_short = os.path.basename(entry.get('image_name', entry.get('image_path', 'unknown')))
            timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_info = f" (retry {retry}/4)" if retry > 1 else ""
            print(f"[Worker {worker_id}] 🚀 Starting MC grounding query{retry_info} | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Image: {image_name_short} | [{timestamp_start}]")
            
            try:
                # Get model response
                success, response, _ = worker_model.get_model_response(
                    prompt,
                    [temp_img_path],
                    use_img_url=True,
                    temperature=0.0,
                    timeout=360
                )
            except Exception as e:
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                if any(x in worker_model.model.lower() for x in ['claude', 'seed']) and os.path.exists(temp_img_path):
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
            
            if any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                os.remove(temp_img_path)
            
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
            
            # Parse answer from response
            pred_option = None
            # Try to extract answer label (A, B, C, etc.)
            answer_match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
            if answer_match:
                pred_option = answer_match.group(1).upper()
            else:
                # Try to find any single letter at the start of a line
                answer_match = re.search(r'^([A-Z]):', response, re.MULTILINE | re.IGNORECASE)
                if answer_match:
                    pred_option = answer_match.group(1).upper()
            
            if not pred_option:
                result['error'] = "Failed to parse answer from response"
                timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[Worker {worker_id}] ❌ Query FAILED (parse error) | Entry: {entry_id} | "
                      f"Model: {worker_model.model} | Error: Could not parse answer | "
                      f"Retry: {retry}/4 | [{timestamp_error}]")
                continue
            
            result['pred_option'] = pred_option
            
            # Find the bbox for the predicted option
            pred_bbox = None
            for opt in options:
                if opt.get('label') == pred_option:
                    bbox = opt.get('bbox', [])
                    if bbox and len(bbox) == 4:
                        # Normalize bbox
                        if bbox[0] > 1 and image_size and len(image_size) == 2:
                            H, W = image_size[0], image_size[1]
                            pred_bbox = [
                                bbox[0] * 1000 / W,
                                bbox[1] * 1000 / H,
                                bbox[2] * 1000 / W,
                                bbox[3] * 1000 / H
                            ]
                        else:
                            pred_bbox = [x * 1000 if x <= 1 else x for x in bbox]
                    break
            
            if pred_bbox:
                result['pred_bbox'] = pred_bbox
                # Calculate IoU
                iou = calculate_iou(pred_bbox, gt_bbox)
                result['iou'] = iou
                
                # Calculate center accuracy
                center = [(pred_bbox[0]+pred_bbox[2])/2, (pred_bbox[1]+pred_bbox[3])/2]
                center_acc = gt_bbox[0] <= center[0] <= gt_bbox[2] and gt_bbox[1] <= center[1] <= gt_bbox[3]
                result['center_acc'] = center_acc
            
            # Check if answer is correct
            correct_label = options[correct_option_idx]['label']
            result['correct'] = (pred_option.upper() == correct_label.upper())
            result['inference_done'] = True
            
            status = "✅" if result['correct'] else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            iou_str = f" | IoU={result['iou']:.3f}" if result.get('pred_bbox') else ""
            center_acc_str = f" | CenterAcc={result['center_acc']}" if result.get('pred_bbox') else ""
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | "
                  f"Pred: {pred_option} | Correct: {correct_label}{iou_str}{center_acc_str} | "
                  f"Time: {processing_time:.2f}s | [{timestamp}]")
            
            break
        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg_short = str(e)[:200]
            print(f"[Worker {worker_id}] ❌ Query EXCEPTION | Entry: {entry_id} | "
                  f"Model: {worker_model.model} | Error: {error_msg_short} | "
                  f"Retry: {retry}/4 | [{timestamp_error}]")
            continue
    else:
        result['error'] = "Failed to parse answer from response"
        timestamp_error = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Worker {worker_id}] ❌ Query FAILED (all retries exhausted) | Entry: {entry_id} | "
              f"Model: {worker_model.model} | Error: Failed to parse answer | "
              f"Total time: {time.time() - start_time:.2f}s | [{timestamp_error}]")
    
    result['processing_time'] = time.time() - start_time
    return result

def process_entry(entry: Dict, worker_id: int = 0, scale: int = 1000, task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Process a single dataset entry
    
    Args:
        entry: Dataset entry dictionary
        worker_id: Worker ID for logging
        scale: Scale for bbox normalization (default: 1000)
        task_type: Task type ('funcgnd', 'descgnd', 'desccap', 'funccap')
    
    Returns:
        Result dictionary with metrics
    """
    global worker_model
    
    entry_id = entry['entry_id']
    image_path = entry['image_path']
    
    # Handle cap tasks differently
    if task_type == 'desccap':
        return process_entry_desccap(entry, worker_id)
    elif task_type == 'funccap':
        return process_entry_funcap(entry, worker_id)
    
    # Check if this is a multiple choice grounding task
    options = entry.get('options', [])
    correct_answer = entry.get('correct_answer', '')
    
    # If we have options and correct_answer, treat as multiple choice task
    if options and correct_answer and len(options) >= 2:
        return process_entry_mc_grounding(entry, worker_id, task_type)
    
    # Otherwise, use traditional bbox prediction format
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
        'action_type': entry.get('action_type', 'unknown'),
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

            if 'gemini' in worker_model.model.lower():
                prompt = GEMINI_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                temp_img_path = image_path
            elif 'tars' in worker_model.model.lower():
                prompt = UI_TARS_PROMPT.format(question=question)
                temp_img_path = image_path
            elif any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                prompt = CLAUDE_GROUNDING_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                temp_img_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
                image = Image.open(image_path).convert("RGB")
                resized_image, _ = resize_pil_image(image, max_size=2560)
                resized_image.save(temp_img_path)
            else:
                prompt = GENERIC_PROMPT.format(ref_tag=ref_tag, ref_placeholder=ref_placeholder, question=question)
                temp_img_path = image_path

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
                    timeout=360
                )
            except Exception as e:
                # Exception during API call - log and continue to next retry
                result['error'] = str(e)
                import traceback
                result['traceback'] = traceback.format_exc()
                # Clean up temp file if needed
                if any(x in worker_model.model.lower() for x in ['claude', 'seed']) and os.path.exists(temp_img_path):
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

            if any(x in worker_model.model.lower() for x in ['claude', 'seed']):
                os.remove(temp_img_path)

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

            # Fall back to a general parsing method
            if raw_pred_bbox is None:
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
            center = [(pred_bbox[0]+pred_bbox[2])/2, (pred_bbox[1]+pred_bbox[3])/2]
            center_acc = gt_bbox[0] <= center[0] <= gt_bbox[2] and gt_bbox[1] <= center[1] <= gt_bbox[3]
            result['center_acc'] = center_acc
            result['inference_done'] = True

            # Only print "Query COMPLETE" if we actually got a valid bbox
            status = "✅" if center_acc else "❌"
            processing_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Worker {worker_id}] {status} Query COMPLETE | Entry: {entry_id} | "
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
    if 'claude' in model_args['model'].lower(): # Claude-Sonnet-4.5
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
                is_failed = False
                if args.task_type in ['desccap', 'funccap']:
                    is_failed = not result.get('inference_done', False) or 'pred_option' not in result
                else:
                    is_failed = not result.get('inference_done', False) or result.get('pred_bbox') is None
                if is_failed:
                    retry_count += 1
    elif isinstance(checkpoint_results, list):
        # Handle list format (from full result files)
        results_by_id = {r.get('entry_id'): r for r in checkpoint_results if 'entry_id' in r}
        for entry in entries_to_process:
            entry_id = entry[0]['entry_id']
            if entry_id in results_by_id:
                result = results_by_id[entry_id]
                is_failed = False
                if args.task_type in ['desccap', 'funccap']:
                    is_failed = not result.get('inference_done', False) or 'pred_option' not in result
                else:
                    is_failed = not result.get('inference_done', False) or result.get('pred_bbox') is None
                if is_failed:
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
                is_successful = False
                if args.task_type in ['desccap', 'funccap']:
                    # For cap tasks, check for 'pred_option' field
                    is_successful = result.get('inference_done', False) and 'pred_option' in result
                else:
                    # For grounding tasks, check for 'pred_bbox'
                    is_successful = result.get('inference_done', False) and result.get('pred_bbox') is not None
                
                if is_successful:
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


def calculate_metrics(results: List[Dict], task_type: str = 'funcgnd') -> Dict[str, Any]:
    """Calculate evaluation metrics with decomposed breakdowns"""
    total = len(results)
    successful = sum(1 for r in results if r.get('inference_done', False))

    if successful == 0:
        if task_type == 'desccap' or task_type == 'funccap':
            return {
                'total': total,
                'successful': 0,
                'success_rate': 0.0,
                'accuracy': 0.0,
                'decomposed': {}
            }
        else:
            return {
                'total': total,
                'successful': 0,
                'success_rate': 0.0,
                'avg_iou': 0.0,
                'iou_thresholds': {},
                'center_acc': 0.0,
                'decomposed': {}
            }
    
    # Handle cap tasks differently
    if task_type == 'desccap' or task_type == 'funccap':
        correct = sum(1 for r in results if r.get('inference_done', False) and r.get('correct', False))
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'accuracy': accuracy,
            'correct': correct,
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
    field_type = getattr(args, 'field_type', 'functionality')
    if args.hf_dataset_id:
        debug_print(f"   Source: {Fore.GREEN}HuggingFace{Style.RESET_ALL}", level="info")
        debug_print(f"   Dataset ID: {Fore.CYAN}{args.hf_dataset_id}{Style.RESET_ALL}", level="info")
        debug_print(f"   Task Type: {Fore.CYAN}{args.task_type}{Style.RESET_ALL}", level="info")
        debug_print(f"   Field Type: {Fore.CYAN}{field_type}{Style.RESET_ALL}", level="info")
        debug_print(f"   Split: {Fore.CYAN}{args.hf_split}{Style.RESET_ALL}", level="info")
        if args.hf_cache_dir:
            debug_print(f"   Cache Dir: {Fore.CYAN}{args.hf_cache_dir}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Source: {Fore.GREEN}JSON File{Style.RESET_ALL}", level="info")
        debug_print(f"   Questions File: {Fore.CYAN}{args.questions_file}{Style.RESET_ALL}", level="info")
        debug_print(f"   Task Type: {Fore.CYAN}{args.task_type}{Style.RESET_ALL}", level="info")
        debug_print(f"   Field Type: {Fore.CYAN}{field_type}{Style.RESET_ALL}", level="info")
    
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
                              hf_split=args.hf_split, hf_cache_dir=args.hf_cache_dir, 
                              task_type=args.task_type, field_type=field_type)
    else:
        entries = load_evaluation_dataset('json', questions_file=args.questions_file, 
                              task_type=args.task_type, field_type=field_type)
    
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
    metrics = calculate_metrics(results, task_type=args.task_type)
    
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
        
        if args.task_type == 'desccap' or args.task_type == 'funccap':
            overall_table.add_row("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.1f}%")
            overall_table.add_row("Correct", f"{metrics.get('correct', 0)}/{metrics['total']}")
        else:
            overall_table.add_row("Average IoU", f"{metrics.get('avg_iou', 0.0):.3f}")
            overall_table.add_row("Center Accuracy", f"{metrics.get('center_acc', 0.0)*100:.1f}%")
        
        console.print(overall_table)
        
        # IoU thresholds table (only for grounding tasks)
        if args.task_type not in ['desccap', 'funccap'] and 'iou_thresholds' in metrics:
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
        if args.task_type == 'desccap' or args.task_type == 'funccap':
            debug_print(f"🎯 Accuracy: {metrics.get('accuracy', 0.0)*100:.1f}%", level="info")
            debug_print(f"✅ Correct: {metrics.get('correct', 0)}/{metrics['total']}", level="info")
        else:
            debug_print(f"📈 Average IoU: {metrics.get('avg_iou', 0.0):.3f}", level="info")
            debug_print(f"🎯 Center Accuracy: {metrics.get('center_acc', 0.0)*100:.1f}%", level="info")
            if 'iou_thresholds' in metrics:
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
    input_group.add_argument("--task-type", type=str, default='funcgnd',
                            choices=['funcgnd', 'descgnd', 'desccap', 'funccap'],
                            help="Task type to evaluate: 'funcgnd' (functionality grounding), 'descgnd' (description grounding), 'desccap' (description captioning), 'funccap' (functionality captioning)")
    
    parser.add_argument("--field-type", type=str, default='functionality',
                       choices=['functionality', 'description'],
                       help="Field type to use: 'functionality' or 'description' (default: 'functionality')")
    
    input_group.add_argument("--questions-file", type=str, default=None,
                            help="Path to questions JSON file (from 2_generate_func_elemgnd_questions.py) or glob pattern")
    input_group.add_argument("--hf-dataset-id", type=str, default='HongxinLi/AutoGUIv2-FuncElemGnd', help="HuggingFace dataset ID (e.g., 'username/dataset-name')")

    # HuggingFace specific arguments
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load from HuggingFace (default: 'test')")
    parser.add_argument("--hf-cache-dir", type=str, default='/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemGnd/',
                       help="Cache directory for HuggingFace datasets")

    # Model arguments
    parser.add_argument("--model", type=str, default=[
            'gemini-2.5-pro-thinking',
            'gpt-5',
            'claude-sonnet-4-5-20250929-thinking',
            'o3',
            'qwen3-vl-32b-thinking',
            'ByteDance-Seed/UI-TARS-1.5-7B',
            'step-3',
            'zai-org/GLM-4.5V'
        ][-2],
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

