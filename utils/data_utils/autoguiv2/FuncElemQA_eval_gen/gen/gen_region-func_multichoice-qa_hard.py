#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate hard functional region captioning questions with advanced adversarial strategies.

This script:
1. Loads questions from 4 captioning_mode directories
2. Refines correct options using Gemini (abstract intent+outcome style, minimize hints)
3. Normalizes and adds model prediction errors as distractors
4. Generates three types of adversarial distractors:
   - Minimal-pair distractors: 70%+ vocabulary overlap, only 1 semantic slot changed
   - Contrastive-pair distractors: Mirror-image opposites (increases/decreases, includes/excludes)
   - Normal distractors: Same outcome different target/scope (prioritized confusion strategies)
5. Reranks 16 candidate distractors to select the hardest 5
6. Ensures correct answers are evenly distributed across A/B/C/D/E/F (6-option questions)

Key Features:
- Intent+outcome focused descriptions (avoid UI control names and locations)
- Multi-stage adversarial generation with quality control
- No generic fallback distractors (maintains high quality or skips question)
"""

import os
import json
import glob
import argparse
import random
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

# Import utilities
import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel

# Colorized output support
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

# Set random seed for reproducibility
random.seed(42)

# Source directories
SOURCE_DIRS = [
    "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode"
]

# Eval results directory (load all JSON files from this directory)
EVAL_RESULTS_DIR = "/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/data_utils/autoguiv2/FuncElemQA_eval_gen/eval/eval_results/funccap/gemini-2.5-pro-thinking/"

# Unified question text
UNIFIED_QUESTION = "Which option most accurately describes the functionality of the region marked with a red rectangle?"

# Prompts for Gemini
REFINE_CORRECT_OPTION_PROMPT = """You are a GUI expert. Your task is to refine the description of a UI region's functionality to make it more abstract and outcome-focused.

**Original description:**
{original_option_context}

**Full functionality (for reference):**
{functionality}

**Task:**
Rewrite the description to focus on **user intent and outcome** rather than specific UI actions or paths.

**HARD Constraints (must follow):**
1. **Minimize UI control names**: Avoid or minimize mentioning specific UI elements like "button", "menu", "tab", "sidebar", "dropdown", "icon", etc.
2. **Minimize location/positioning words**: Avoid spatial descriptors like "top-left", "toolbar", "navigation bar", "header", "footer", etc.
3. **Focus on outcomes**: Describe WHAT will result or happen, not HOW to do it
   - ✅ Good: "updates the content shown", "changes the view mode", "modifies the display settings"
   - ❌ Bad: "opens menu", "clicks button", "navigates to tab"
4. **Describe intent + result**: Focus on user goal and end state
   - ✅ Good: "Enables filtering content by specific criteria"
   - ❌ Bad: "Opens the filter menu to select options"

**Additional Requirements:**
- Keep it to 1-2 sentences maximum
- Use abstract, functional language
- Describe the capability or transformation, not the interaction path

**Output format (JSON):**
{{
  "refined_option_context": "Your refined description here"
}}

Please provide the refined description now:"""

NORMALIZE_PRED_ANSWER_PROMPT = """You are a GUI expert. Your task is to normalize a model's incorrect prediction to match the style of the correct answer.

**Correct option (ground truth):**
{correct_option_context}

**Model's incorrect prediction (raw):**
{pred_answer}

**Full functionality (for reference):**
{functionality}

**Task:**
Perform a **minimal edit** on the model's prediction to:
1. Match the style and format of the correct option (abstract, intent+outcome focused)
2. Keep the INCORRECT nature of the prediction (don't make it correct!)
3. Make it sound plausible and similar to the correct option
4. Ensure similar length and conciseness

**Requirements:**
- Preserve the core incorrect meaning from the model's prediction
- Adjust language to be abstract and outcome-focused (like the correct option)
- Remove any overly specific UI control names or location words if present
- Make minimal changes - keep as much of the original prediction as possible
- The result should be a plausible but wrong alternative that fits the question style

**Output format (JSON):**
{{
  "normalized_pred_answer": "Your normalized version of the model's prediction here"
}}

Please provide the normalized prediction now:"""

GENERATE_MINIMAL_PAIR_DISTRACTORS_PROMPT = """You are a GUI expert. Your task is to generate {num_distractors} **minimal-pair** distractor options - these should be EXTREMELY similar to the correct option but with only ONE semantic slot changed.

**Correct option (ground truth):**
{correct_option_context}

**Full functionality (for reference):**
{functionality}

**Context:**
This is the UI region marked with a red rectangle in the provided screenshot.

**Task:**
Generate {num_distractors} minimal-pair distractors where each distractor:

**STRICT Requirements:**
1. **70%+ vocabulary overlap**: Must share at least 70% of words/phrases with the correct option
2. **Change ONLY ONE semantic slot**: Replace exactly one element from these categories:
   - Target object (e.g., "messages" → "notifications")
   - Condition/criteria (e.g., "by date" → "by type")
   - Scope/range (e.g., "current page" → "entire document")
   - Action type (e.g., "filtering" → "sorting", "ordering" → "grouping")
   - Target group (e.g., "active users" → "all users")
   - Time scope (e.g., "recent" → "archived")
3. **Keep everything else identical**: Same structure, same outcome verb, same style

**Examples of minimal-pair changes:**
- Correct: "Enables filtering items by time range"
  Distractor: "Enables filtering items by file type" (only change: time range → file type)
- Correct: "Modifies the ordering of displayed items"
  Distractor: "Modifies the grouping of displayed items" (only change: ordering → grouping)
- Correct: "Changes how messages are displayed"
  Distractor: "Changes how notifications are displayed" (only change: messages → notifications)

**Style Consistency:**
- Use outcome-focused language like: "enables...", "modifies...", "changes...", "updates..."
- Match the exact grammatical structure of the correct option

**Output format (JSON):**
{{
  "distractors": [
    "First minimal-pair distractor (70%+ overlap, 1 slot changed)",
    "Second minimal-pair distractor (70%+ overlap, 1 slot changed)"
  ]
}}

Please generate {num_distractors} minimal-pair distractors now:"""

GENERATE_CONTRASTIVE_PAIR_DISTRACTORS_PROMPT = """You are a GUI expert. Your task is to generate {num_pairs} **pairs** of contrastive distractor options - distractors that are opposite or mirror images of each other.

**Correct option (ground truth):**
{correct_option_context}

**Full functionality (for reference):**
{functionality}

**Context:**
This is the UI region marked with a red rectangle in the provided screenshot.

**Task:**
Generate {num_pairs} PAIRS of distractors where each pair consists of two options that are opposite/contrastive to each other.

**Contrastive Patterns (use these):**
1. **Opposite modifiers**: increases ↔ decreases, expands ↔ collapses, shows ↔ hides
2. **Opposite inclusivity**: includes ↔ excludes, adds ↔ removes, enables ↔ disables
3. **Opposite persistence**: temporary ↔ permanent, preview ↔ apply, draft ↔ published
4. **Opposite direction**: ascending ↔ descending, forward ↔ backward, next ↔ previous
5. **Opposite scope**: current ↔ all, selected ↔ unselected, visible ↔ hidden

**Requirements:**
- Each pair should use the same base structure and vocabulary
- Only the modifier/direction should be opposite
- Both options in the pair should be plausible but incorrect
- Maintain the abstract, outcome-focused style

**Example pair:**
- "Increases the visibility of inactive elements"
- "Decreases the visibility of inactive elements"

**Output format (JSON):**
{{
  "contrastive_pairs": [
    {{
      "pair_1_option_a": "First option of first pair",
      "pair_1_option_b": "Opposite/contrastive option of first pair"
    }},
    {{
      "pair_2_option_a": "First option of second pair",
      "pair_2_option_b": "Opposite/contrastive option of second pair"
    }}
  ]
}}

Please generate {num_pairs} contrastive pairs now:"""

GENERATE_DISTRACTORS_PROMPT = """You are a GUI expert. Your task is to generate {num_distractors} highly confusing distractor options for a multiple-choice question about UI functionality.

**Correct option (ground truth):**
- Option text: {correct_option_context}
- Functionality: {functionality}

**Context:**
This is the UI region marked with a red rectangle in the provided screenshot.

**Task:**
Generate {num_distractors} plausible but INCORRECT descriptions using these confusion strategies **IN ORDER OF PRIORITY**:

**Confusion Strategies (PRIORITIZED - apply in this order):**
1. **Same outcome, different target/scope** [HIGHEST PRIORITY]: Keep the outcome identical, only change the object or scope
   - Example: "Changes how messages are displayed" → "Changes how notifications are displayed"
   - Example: "Updates the content shown based on a condition" → "Updates the content shown based on a different scope"
2. **Same target, different condition**: Keep the object same, change the criteria/condition
   - Example: "Filters items by date" → "Filters items by status"
3. **Same action, different persistence**: Keep the action same, change temporal/persistence aspect
   - Example: "Temporarily hides elements" → "Permanently removes elements"
4. **Same intent, different outcome**: Same user goal but different result (lower priority now)

**CRITICAL Requirements:**
- **Each distractor MUST reuse at least 1-2 key terms/phrases from the correct option** to maximize confusion
- **Follow the same abstract style**: Focus on intent + outcome, avoid specific UI control names (button/menu/tab) and location words (toolbar/top-left)
- **Sound equally plausible**: Match the conciseness and sophistication of the correct option
- Avoid obviously wrong or unrelated functionalities
- Keep descriptions to 1-2 sentences

**Style Consistency:**
- Use outcome-focused language like: "enables...", "updates...", "modifies...", "changes...", "provides access to..."
- Avoid action-path language like: "opens...", "clicks...", "navigates to..."

**Output format (JSON):**
{{
  "distractors": [
    "First distractor description (preferably same outcome, different target/scope)",
    "Second distractor description (preferably same outcome, different target/scope)",
    "Third distractor description (use prioritized strategies)"
  ]
}}

Please generate {num_distractors} confusing distractors now:"""

RERANK_DISTRACTORS_PROMPT = """You are a GUI expert evaluator. Your task is to rank and filter distractor options by their **hardness** (how likely they are to fool a model or human).

**Correct option (ground truth):**
{correct_option_context}

**Full functionality (for reference):**
{functionality}

**Candidate distractors (to be ranked):**
{candidate_distractors}

**Context:**
This is the UI region marked with a red rectangle in the provided screenshot.

**Task:**
Evaluate each candidate distractor and select the {num_to_select} HARDEST ones that:
1. **Most likely to fool models/humans**: Sound extremely plausible and similar to the correct answer
2. **Semantically confusing**: Share key terminology with the correct option but describe different functionality
3. **Subtle differences**: The distinction from the correct answer should be non-obvious
4. **Stylistically consistent**: Match the abstract, intent+outcome style of the correct option

**Ranking Criteria (in order of importance):**
1. **Confusion potential**: How easy is it to mistake this for the correct answer?
2. **Keyword overlap**: Does it reuse key terms from the correct option effectively?
3. **Plausibility**: Does it sound like a reasonable functionality for a similar UI element?
4. **Subtlety**: Are the differences subtle enough to require careful reading?

**Filter out distractors that:**
- Are obviously wrong or unrelated
- Have completely different terminology
- Are too easy to eliminate

**Output format (JSON):**
{{
  "selected_distractors": [
    {{
      "distractor": "First hardest distractor text",
      "rank": 1,
      "reasoning": "Brief explanation of why this is confusing"
    }},
    {{
      "distractor": "Second hardest distractor text",
      "rank": 2,
      "reasoning": "Brief explanation of why this is confusing"
    }}
  ]
}}

Please rank and select the {num_to_select} hardest distractors now:"""


def debug_print(message: str, level: str = "info") -> None:
    """Colorized debug print."""
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


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 data URL."""
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    
    ext = Path(image_path).suffix.lower()
    with open(image_path, "rb") as f:
        binary_data = f.read()
    base64_data = base64.b64encode(binary_data).decode("utf-8")
    return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"


def parse_json_response(response: str, debug: bool = False) -> Any:
    """Parse JSON response from LLM."""
    try:
        # Remove <think> tags
        if '</think>' in response:
            response = response.split('</think>')[-1]
        
        # Remove markdown code blocks
        import re
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Find JSON content
        bracket_idx = response.find('[')
        brace_idx = response.find('{')
        
        if bracket_idx == -1 and brace_idx == -1:
            if debug:
                debug_print(f"No JSON found in response", level="error")
            return None
        elif bracket_idx == -1:
            start_idx = brace_idx
            is_array = False
        elif brace_idx == -1:
            start_idx = bracket_idx
            is_array = True
        else:
            if bracket_idx < brace_idx:
                start_idx = bracket_idx
                is_array = True
            else:
                start_idx = brace_idx
                is_array = False
        
        end_idx = response.rfind(']' if is_array else '}')
        if end_idx == -1:
            if debug:
                debug_print(f"No closing bracket found", level="error")
            return None
        
        json_str = response[start_idx:end_idx+1]
        data = json.loads(json_str)
        return data
        
    except json.JSONDecodeError as e:
        if debug:
            debug_print(f"JSON parsing failed: {e}", level="error")
        return None
    except Exception as e:
        if debug:
            debug_print(f"Unexpected error during parsing: {e}", level="error")
        return None


def load_all_questions(source_dirs: List[str], debug: bool = False) -> List[Dict]:
    """Load all questions from source directories."""
    all_questions = []
    
    for source_dir in source_dirs:
        dataset_name = Path(source_dir).parent.parent.name
        
        if not os.path.exists(source_dir):
            debug_print(f"Warning: Directory not found: {source_dir}", level="warn")
            continue
        
        # Find all *_result.json files
        result_files = glob.glob(os.path.join(source_dir, "*_result.json"))
        
        if debug:
            debug_print(f"Loading from {dataset_name}: {len(result_files)} files", level="info")
        
        for result_file in result_files:
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract questions
                questions = data.get('result', {}).get('questions', [])
                
                for question in questions:
                    # Add source information
                    question['source_dataset'] = dataset_name
                    question['source_file'] = result_file
                    
                    # Generate entry_id from annotated_image_path
                    annotated_image_path = question.get('annotated_image_path', '')
                    if annotated_image_path:
                        image_filename = os.path.basename(annotated_image_path)
                        # Remove extension
                        image_stem = os.path.splitext(image_filename)[0]
                        question['entry_id'] = f"{dataset_name}_{image_stem}"
                    
                    all_questions.append(question)
            
            except Exception as e:
                debug_print(f"Error loading {result_file}: {e}", level="error")
                continue
    
    if debug:
        debug_print(f"Total questions loaded: {len(all_questions)}", level="success")
    
    return all_questions


def load_eval_results(eval_dir: str, debug: bool = False) -> Dict[str, Dict]:
    """
    Load evaluation results from all JSON files in directory and build entry_id -> result mapping.
    For each entry_id, collect all unique pred_answers from different eval files.
    """
    if not os.path.exists(eval_dir):
        debug_print(f"Error: Eval results directory not found: {eval_dir}", level="error")
        return {}
    
    if not os.path.isdir(eval_dir):
        debug_print(f"Error: Path is not a directory: {eval_dir}", level="error")
        return {}
    
    try:
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(eval_dir, "*.json"))
        
        if len(json_files) == 0:
            debug_print(f"Warning: No JSON files found in {eval_dir}", level="warn")
            return {}
        
        if debug:
            debug_print(f"Found {len(json_files)} eval result files to load", level="info")
        
        # Build mapping: simplified_entry_id -> result with all pred_answers
        eval_mapping = {}
        total_results = 0
        total_errors = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                total_results += len(results)
                
                for result in results:
                    entry_id = result.get('entry_id', '')
                    image_name = result.get('image_name', '')
                    image_path = result.get('image_path', '')
                    is_correct = result.get('is_correct', True)
                    pred_answer = result.get('pred_answer', '')
                    
                    if not is_correct:
                        total_errors += 1
                    
                    # Extract dataset name from image_path
                    # e.g., "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/..." -> "osworld_g"
                    dataset_name = None
                    if image_path:
                        path_parts = image_path.split('/')
                        # Find AutoGUIv2 and get the next part
                        try:
                            autogui_idx = path_parts.index('AutoGUIv2')
                            if autogui_idx + 1 < len(path_parts):
                                dataset_name = path_parts[autogui_idx + 1]
                        except (ValueError, IndexError):
                            pass
                    
                    if not dataset_name:
                        # Fallback: use dataset_name from result
                        dataset_name = result.get('dataset_name', '')
                    
                    # The meaningful part is the image_name (e.g., "group2_2-7_hBrUZN5ZUo.png")
                    # Remove extension to get stem
                    image_stem = os.path.splitext(image_name)[0]
                    
                    # Create simplified key: dataset_name_image_stem
                    simplified_key = f"{dataset_name}_{image_stem}"
                    
                    # If this entry doesn't exist yet, create it
                    if simplified_key not in eval_mapping:
                        eval_mapping[simplified_key] = {
                            'entry_id': entry_id,
                            'image_name': image_name,
                            'image_path': image_path,
                            'dataset_name': dataset_name,
                            'pred_answers': [],  # Collect all unique pred_answers
                            'is_correct': is_correct,
                            'correct_answer': result.get('correct_answer', '')
                        }
                    else:
                        # If entry exists and current result is incorrect, update is_correct to False
                        # (Once incorrect in any eval file, mark as incorrect)
                        if not is_correct:
                            eval_mapping[simplified_key]['is_correct'] = False
                    
                    # Add pred_answer if it's incorrect and not already in the list
                    if not is_correct and pred_answer:
                        if pred_answer not in eval_mapping[simplified_key]['pred_answers']:
                            eval_mapping[simplified_key]['pred_answers'].append(pred_answer)
            
            except Exception as e:
                if debug:
                    debug_print(f"Error loading {os.path.basename(json_file)}: {e}", level="error")
                continue
        
        if debug:
            debug_print(f"Loaded {len(eval_mapping)} unique entries from {len(json_files)} files", level="success")
            debug_print(f"  - Total predictions: {total_results}", level="info")
            debug_print(f"  - Incorrect predictions: {total_errors}", level="info")
            
            # Count entries with multiple pred_answers
            multi_pred_count = sum(1 for v in eval_mapping.values() if len(v['pred_answers']) > 1)
            if multi_pred_count > 0:
                debug_print(f"  - Entries with multiple pred_answers: {multi_pred_count}", level="info")
        
        return eval_mapping
    
    except Exception as e:
        debug_print(f"Error loading eval results: {e}", level="error")
        return {}


def refine_correct_option(model: OpenAIModel, option_context: str, functionality: str,
                          annotated_image_path: str, max_retries: int = 3,
                          debug: bool = False) -> str:
    """Use Gemini to refine the correct option description (reduce hints)."""
    
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return option_context  # Return original if image not found
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(annotated_image_path)
        
        # Prepare prompt
        prompt = REFINE_CORRECT_OPTION_PROMPT.format(
            original_option_context=option_context,
            functionality=functionality
        )
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                # Use *_ to handle variable return values (thinking models return 4 values)
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3, timeout=60
                )
                
                if not success:
                    if debug and attempt == max_retries - 1:
                        debug_print(f"API call failed: {response}", level="warn")
                    continue
                
                # Parse response
                result = parse_json_response(response, debug=debug)
                
                if result and 'refined_option_context' in result:
                    refined_text = result['refined_option_context']
                    if refined_text and len(refined_text) > 0:
                        return refined_text
            
            except Exception as e:
                if debug and attempt == max_retries - 1:
                    debug_print(f"Exception in refine_correct_option: {e}", level="error")
                continue
        
        # If all retries fail, return original
        if debug:
            debug_print(f"Failed to refine option, using original", level="warn")
        return option_context
    
    except Exception as e:
        if debug:
            debug_print(f"Error in refine_correct_option: {e}", level="error")
        return option_context


def normalize_pred_answer(model: OpenAIModel, pred_answer: str, 
                         correct_option_context: str, functionality: str,
                         annotated_image_path: str, max_retries: int = 3,
                         debug: bool = False) -> str:
    """Use Gemini to normalize model's prediction to match correct option style."""
    
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return pred_answer  # Return original if image not found
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(annotated_image_path)
        
        # Prepare prompt
        prompt = NORMALIZE_PRED_ANSWER_PROMPT.format(
            correct_option_context=correct_option_context,
            pred_answer=pred_answer,
            functionality=functionality
        )
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                # Use *_ to handle variable return values (thinking models return 4 values)
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3, timeout=60
                )
                
                if not success:
                    if debug and attempt == max_retries - 1:
                        debug_print(f"API call failed: {response}", level="warn")
                    continue
                
                # Parse response
                result = parse_json_response(response, debug=debug)
                
                if result and 'normalized_pred_answer' in result:
                    normalized_text = result['normalized_pred_answer']
                    if normalized_text and len(normalized_text) > 0:
                        return normalized_text
            
            except Exception as e:
                if debug and attempt == max_retries - 1:
                    debug_print(f"Exception in normalize_pred_answer: {e}", level="error")
                continue
        
        # If all retries fail, return original
        if debug:
            debug_print(f"Failed to normalize pred_answer, using original", level="warn")
        return pred_answer
    
    except Exception as e:
        if debug:
            debug_print(f"Error in normalize_pred_answer: {e}", level="error")
        return pred_answer


def rerank_distractors(model: OpenAIModel, candidate_distractors: List[str],
                      correct_option_context: str, functionality: str,
                      annotated_image_path: str, num_to_select: int = 5,
                      max_retries: int = 3, debug: bool = False) -> List[str]:
    """Use Gemini to rank and filter distractors by hardness."""
    
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return candidate_distractors[:num_to_select]  # Return first N if image not found
    
    if len(candidate_distractors) <= num_to_select:
        # No need to rerank if we don't have more candidates than needed
        return candidate_distractors
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(annotated_image_path)
        
        # Format candidate distractors for prompt
        candidate_list = "\n".join([f"{i+1}. {d}" for i, d in enumerate(candidate_distractors)])
        
        # Prepare prompt
        prompt = RERANK_DISTRACTORS_PROMPT.format(
            correct_option_context=correct_option_context,
            functionality=functionality,
            candidate_distractors=candidate_list,
            num_to_select=num_to_select
        )
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(max_retries):
            try:
                # Use *_ to handle variable return values (thinking models return 4 values)
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3, timeout=60
                )
                
                if not success:
                    if debug and attempt == max_retries - 1:
                        debug_print(f"API call failed: {response}", level="warn")
                    continue
                
                # Parse response
                result = parse_json_response(response, debug=debug)
                
                if result and 'selected_distractors' in result:
                    selected = result['selected_distractors']
                    if isinstance(selected, list) and len(selected) > 0:
                        # Extract distractor texts
                        reranked_distractors = [item['distractor'] for item in selected if 'distractor' in item]
                        
                        if len(reranked_distractors) >= num_to_select:
                            if debug:
                                debug_print(f"  Reranked and selected {len(reranked_distractors[:num_to_select])} distractors", level="success")
                            return reranked_distractors[:num_to_select]
            
            except Exception as e:
                if debug and attempt == max_retries - 1:
                    debug_print(f"Exception in rerank_distractors: {e}", level="error")
                continue
        
        # If all retries fail, return first N candidates
        if debug:
            debug_print(f"Failed to rerank distractors, using first {num_to_select}", level="warn")
        return candidate_distractors[:num_to_select]
    
    except Exception as e:
        if debug:
            debug_print(f"Error in rerank_distractors: {e}", level="error")
        return candidate_distractors[:num_to_select]


def generate_minimal_pair_distractors(model: OpenAIModel, correct_option_context: str,
                                     functionality: str, annotated_image_path: str,
                                     num_distractors: int = 8, max_total_attempts: int = 40,
                                     debug: bool = False) -> List[str]:
    """
    Generate minimal-pair distractors: 70%+ vocabulary overlap, only 1 semantic slot changed.
    """
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return []
    
    try:
        image_base64 = image_to_base64(annotated_image_path)
        collected_distractors = []
        attempt_count = 0
        
        while len(collected_distractors) < num_distractors and attempt_count < max_total_attempts:
            attempt_count += 1
            remaining_needed = num_distractors - len(collected_distractors)
            
            prompt = GENERATE_MINIMAL_PAIR_DISTRACTORS_PROMPT.format(
                num_distractors=remaining_needed,
                correct_option_context=correct_option_context,
                functionality=functionality
            )
            
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': image_base64}},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            
            try:
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.5, timeout=60
                )
                
                if not success:
                    if debug:
                        debug_print(f"  Minimal-pair API call {attempt_count} failed", level="warn")
                    continue
                
                result = parse_json_response(response, debug=debug)
                
                if result and 'distractors' in result:
                    distractors = result['distractors']
                    if isinstance(distractors, list) and len(distractors) >= 1:
                        for d in distractors:
                            if d not in collected_distractors:
                                collected_distractors.append(d)
                                if len(collected_distractors) >= num_distractors:
                                    break
                        
                        if debug:
                            debug_print(f"  Minimal-pair attempt {attempt_count}: collected {len(distractors)} distractors, total: {len(collected_distractors)}/{num_distractors}", level="info")
            
            except Exception as e:
                if debug:
                    debug_print(f"  Exception in minimal-pair attempt {attempt_count}: {e}", level="error")
                continue
        
        if len(collected_distractors) < num_distractors and debug:
            debug_print(f"  Warning: Only collected {len(collected_distractors)}/{num_distractors} minimal-pair distractors after {attempt_count} attempts", level="warn")
        
        return collected_distractors[:num_distractors]
    
    except Exception as e:
        if debug:
            debug_print(f"Error in generate_minimal_pair_distractors: {e}", level="error")
        return []


def generate_contrastive_pair_distractors(model: OpenAIModel, correct_option_context: str,
                                         functionality: str, annotated_image_path: str,
                                         num_pairs: int = 2, max_total_attempts: int = 20,
                                         debug: bool = False) -> List[str]:
    """
    Generate contrastive pairs of distractors: pairs that are opposite/mirror images.
    Returns a flat list of distractors (unpacked from pairs).
    """
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return []
    
    try:
        image_base64 = image_to_base64(annotated_image_path)
        collected_distractors = []
        attempt_count = 0
        
        while len(collected_distractors) < num_pairs * 2 and attempt_count < max_total_attempts:
            attempt_count += 1
            remaining_pairs_needed = (num_pairs * 2 - len(collected_distractors) + 1) // 2
            
            prompt = GENERATE_CONTRASTIVE_PAIR_DISTRACTORS_PROMPT.format(
                num_pairs=remaining_pairs_needed,
                correct_option_context=correct_option_context,
                functionality=functionality
            )
            
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': image_base64}},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            
            try:
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.5, timeout=60
                )
                
                if not success:
                    if debug:
                        debug_print(f"  Contrastive-pair API call {attempt_count} failed", level="warn")
                    continue
                
                result = parse_json_response(response, debug=debug)
                
                if result and 'contrastive_pairs' in result:
                    pairs = result['contrastive_pairs']
                    if isinstance(pairs, list) and len(pairs) >= 1:
                        for pair in pairs:
                            # Extract both options from each pair
                            option_a = None
                            option_b = None
                            
                            # Handle different possible key names
                            for key in pair.keys():
                                if 'option_a' in key.lower() or key.startswith('pair_') and key.endswith('_a'):
                                    option_a = pair[key]
                                elif 'option_b' in key.lower() or key.startswith('pair_') and key.endswith('_b'):
                                    option_b = pair[key]
                            
                            if option_a and option_a not in collected_distractors:
                                collected_distractors.append(option_a)
                            if option_b and option_b not in collected_distractors:
                                collected_distractors.append(option_b)
                            
                            if len(collected_distractors) >= num_pairs * 2:
                                break
                        
                        if debug:
                            debug_print(f"  Contrastive-pair attempt {attempt_count}: collected {len(pairs)} pairs, total distractors: {len(collected_distractors)}/{num_pairs*2}", level="info")
            
            except Exception as e:
                if debug:
                    debug_print(f"  Exception in contrastive-pair attempt {attempt_count}: {e}", level="error")
                continue
        
        if len(collected_distractors) < num_pairs * 2 and debug:
            debug_print(f"  Warning: Only collected {len(collected_distractors)}/{num_pairs*2} contrastive-pair distractors after {attempt_count} attempts", level="warn")
        
        return collected_distractors[:num_pairs * 2]
    
    except Exception as e:
        if debug:
            debug_print(f"Error in generate_contrastive_pair_distractors: {e}", level="error")
        return []


def generate_distractors(model: OpenAIModel, correct_option_context: str,
                        functionality: str, region_type: str,
                        annotated_image_path: str, num_distractors: int = 3,
                        max_total_attempts: int = 40, debug: bool = False) -> List[str]:
    """
    Use Gemini to generate confusing distractor options (normal strategy).
    
    Strategy: Collect distractors across multiple API calls until we have enough.
    Accept any response with ≥1 distractor and keep calling until we reach num_distractors
    or hit max_total_attempts.
    """
    
    if not os.path.exists(annotated_image_path):
        if debug:
            debug_print(f"Warning: Image not found: {annotated_image_path}", level="warn")
        return []
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(annotated_image_path)
        
        # Collect distractors across multiple attempts
        collected_distractors = []
        attempt_count = 0
        
        while len(collected_distractors) < num_distractors and attempt_count < max_total_attempts:
            attempt_count += 1
            
            # Calculate how many more we need
            remaining_needed = num_distractors - len(collected_distractors)
            
            # Prepare prompt
            prompt = GENERATE_DISTRACTORS_PROMPT.format(
                num_distractors=remaining_needed,
                correct_option_context=correct_option_context,
                functionality=functionality,
                region_type=region_type
            )
            
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': image_base64}},
                    {'type': 'text', 'text': prompt}
                ]
            }]
            
            try:
                # Use *_ to handle variable return values (thinking models return 4 values)
                success, response, *_ = model.get_model_response_with_prepared_messages(
                    messages, temperature=0.5, timeout=60
                )
                
                if not success:
                    if debug:
                        debug_print(f"  API call {attempt_count} failed: {response}", level="warn")
                    continue
                
                # Parse response
                result = parse_json_response(response, debug=debug)
                
                if result and 'distractors' in result:
                    distractors = result['distractors']
                    if isinstance(distractors, list) and len(distractors) >= 1:
                        # Accept any response with at least 1 distractor
                        # Add unique distractors to our collection
                        for d in distractors:
                            if d not in collected_distractors:
                                collected_distractors.append(d)
                                if len(collected_distractors) >= num_distractors:
                                    break
                        
                        if debug:
                            debug_print(f"  Attempt {attempt_count}: collected {len(distractors)} distractors, total now: {len(collected_distractors)}/{num_distractors}", level="info")
            
            except Exception as e:
                if debug:
                    debug_print(f"  Exception in attempt {attempt_count}: {e}", level="error")
                continue
        
        # Check if we got enough distractors
        if len(collected_distractors) < num_distractors:
            if debug:
                debug_print(f"  Warning: Only collected {len(collected_distractors)}/{num_distractors} distractors after {attempt_count} attempts", level="warn")
        
        return collected_distractors[:num_distractors]
    
    except Exception as e:
        if debug:
            debug_print(f"Error in generate_distractors: {e}", level="error")
        return []


def process_question(question: Dict, eval_results: Dict[str, Dict],
                    model: OpenAIModel, debug: bool = False) -> Dict:
    """Process a single question to create hard version."""
    
    entry_id = question.get('entry_id', '')
    source_dataset = question.get('source_dataset', '')
    annotated_image_path = question.get('annotated_image_path', '')
    group_id = question.get('group_id', -1)
    
    # Get correct answer
    correct_answer_label = question.get('correct_answer', '')
    options = question.get('options', [])
    
    # Find correct option
    correct_option = None
    for opt in options:
        if opt.get('label', '') == correct_answer_label:
            correct_option = opt
            break
    
    if not correct_option:
        if debug:
            debug_print(f"Warning: No correct option found for {entry_id}", level="warn")
        return None
    
    # Extract correct option information
    region_id = correct_option.get('region_id', '')
    region_type = correct_option.get('region_type', 'Unknown')
    functionality = correct_option.get('functionality', '')
    original_option_context = correct_option.get('option_context', '')
    metrics = correct_option.get('metrics', {})
    
    # Step 1: Refine correct option using Gemini
    if debug:
        debug_print(f"  Refining correct option for {entry_id}...", level="step")
    
    refined_option_context = refine_correct_option(
        model, original_option_context, functionality,
        annotated_image_path, debug=debug
    )
    
    # Step 2: Check if there are prediction errors for this entry and normalize them
    eval_result = eval_results.get(entry_id)
    pred_answer_distractors = []  # List to store all normalized pred_answers
    
    # Check if we have pred_answers (more robust than checking is_correct flag)
    if eval_result:
        # Get all pred_answers for this entry (may be multiple from different eval files)
        pred_answers = eval_result.get('pred_answers', [])
        
        if len(pred_answers) > 0:
            if debug:
                debug_print(f"  Found {len(pred_answers)} prediction error(s) from eval files", level="info")
            
            # Normalize each unique pred_answer
            for pred_answer in pred_answers:
                if pred_answer and pred_answer != original_option_context:
                    if debug:
                        debug_print(f"    Normalizing pred_answer: {pred_answer[:50]}...", level="info")
                    
                    # Normalize the prediction to match the style of refined correct option
                    normalized_pred = normalize_pred_answer(
                        model, pred_answer, refined_option_context, functionality,
                        annotated_image_path, debug=debug
                    )
                    
                    # Add to list if not already present (additional deduplication after normalization)
                    if normalized_pred and normalized_pred not in pred_answer_distractors:
                        pred_answer_distractors.append(normalized_pred)
                        
                        if debug:
                            debug_print(f"    Normalized to: {normalized_pred[:50]}...", level="info")
            
            if debug:
                debug_print(f"  Collected {len(pred_answer_distractors)} unique normalized pred_answer(s)", level="success")
    
    # Step 3: Generate additional distractors using Gemini (adversarial generation with multiple strategies)
    # Changed from 3 to 5 distractors to support 6-option questions
    # Use mixed adversarial strategy: 
    #   - N normalized pred_answers (from multiple eval files)
    #   - 8 minimal-pair distractors (70%+ overlap, 1 slot changed)
    #   - 4 contrastive-pair distractors (2 pairs of opposites)
    #   - 4 normal distractors (related but wrong)
    # Total: N+16 candidates, then rerank to select top 5
    num_pred_distractors = len(pred_answer_distractors)
    num_needed_distractors = 5 - num_pred_distractors  # Adjust based on how many pred_answers we have
    
    # Ensure we need at least 1 more distractor
    if num_needed_distractors <= 0:
        num_needed_distractors = 1  # Always generate some, we'll rerank everything
    
    if debug:
        debug_print(f"  Step 3a: Generating mixed candidate distractors for {entry_id}...", level="step")
    
    # Step 3a-1: Generate 8 minimal-pair distractors
    if debug:
        debug_print(f"    - Generating 8 minimal-pair distractors (70%+ overlap, 1 slot change)...", level="info")
    minimal_pair_distractors = generate_minimal_pair_distractors(
        model, refined_option_context, functionality,
        annotated_image_path, num_distractors=8, debug=debug
    )
    if debug:
        debug_print(f"    - Collected {len(minimal_pair_distractors)} minimal-pair distractors", level="info")
    
    # Step 3a-2: Generate 2 pairs (4 distractors) of contrastive opposites
    if debug:
        debug_print(f"    - Generating 2 contrastive pairs (4 opposite distractors)...", level="info")
    contrastive_pair_distractors = generate_contrastive_pair_distractors(
        model, refined_option_context, functionality,
        annotated_image_path, num_pairs=2, debug=debug
    )
    if debug:
        debug_print(f"    - Collected {len(contrastive_pair_distractors)} contrastive-pair distractors", level="info")
    
    # Step 3a-3: Generate 4 normal distractors (prioritized strategies)
    if debug:
        debug_print(f"    - Generating 4 normal distractors (same outcome/different target)...", level="info")
    normal_distractors = generate_distractors(
        model, refined_option_context, functionality, region_type,
        annotated_image_path, num_distractors=4, debug=debug
    )
    if debug:
        debug_print(f"    - Collected {len(normal_distractors)} normal distractors", level="info")
    
    # Combine all candidate distractors (including pred_answers + generated ones)
    # Pred_answers are already normalized, add them to the candidate pool
    candidate_distractors = (
        pred_answer_distractors +  # Normalized pred_answers from multiple eval files
        minimal_pair_distractors + 
        contrastive_pair_distractors + 
        normal_distractors
    )
    
    if debug:
        debug_print(f"  Total candidate pool: {len(candidate_distractors)} distractors", level="info")
        debug_print(f"    - {len(pred_answer_distractors)} from normalized pred_answers", level="info")
        debug_print(f"    - {len(minimal_pair_distractors)} minimal-pair", level="info")
        debug_print(f"    - {len(contrastive_pair_distractors)} contrastive-pair", level="info")
        debug_print(f"    - {len(normal_distractors)} normal", level="info")
    
    # Step 3b: Rerank and select the hardest 5 distractors from all candidates
    if debug:
        debug_print(f"  Step 3b: Reranking to select top 5 hardest distractors from {len(candidate_distractors)} candidates...", level="step")
    
    # Always select top 5 (regardless of how many pred_answers we have)
    generated_distractors = rerank_distractors(
        model, candidate_distractors, refined_option_context, functionality,
        annotated_image_path, num_to_select=5,
        debug=debug
    )
    
    if debug:
        debug_print(f"  Selected {len(generated_distractors)} hardest distractors after reranking", level="success")
    
    # Build distractor list from reranked distractors
    # The reranked list already includes pred_answers (if any), so we need to identify source
    all_distractors = []
    
    # Convert pred_answer_distractors to set for quick lookup
    pred_answer_set = set(pred_answer_distractors)
    
    # Add reranked distractors with appropriate source labels
    for distractor_text in generated_distractors:
        if distractor_text in pred_answer_set:
            # This distractor came from normalized pred_answer
            all_distractors.append({
                'option_context': distractor_text,
                'is_correct': False,
                'source': 'reranked_pred_answer_normalized'
            })
        else:
            # This distractor came from Gemini generation
            all_distractors.append({
                'option_context': distractor_text,
                'is_correct': False,
                'source': 'reranked_gemini_generated'
            })
    
    # Ensure we have exactly 5 distractors
    if len(all_distractors) < 5:
        if debug:
            debug_print(f"  Warning: Only {len(all_distractors)} distractors for {entry_id}, attempting to fill from candidates", level="warn")
        
        # Extract already used distractor texts
        used_distractor_texts = set([d['option_context'] for d in all_distractors])
        
        # Find unused candidates from the original candidate pool
        unused_candidates = [
            d for d in candidate_distractors 
            if d not in used_distractor_texts
        ]
        
        if len(unused_candidates) > 0:
            # Randomly select from unused candidates to fill up to 5
            needed = 5 - len(all_distractors)
            random.shuffle(unused_candidates)
            selected_backups = unused_candidates[:needed]
            
            for backup_text in selected_backups:
                all_distractors.append({
                    'option_context': backup_text,
                    'is_correct': False,
                    'source': 'candidate_backup'
                })
            
            if debug:
                debug_print(f"  Filled {len(selected_backups)} distractors from unused candidates", level="info")
        
        # If still not enough, log error and skip this question
        if len(all_distractors) < 5:
            if debug:
                debug_print(f"  ERROR: Cannot fill 5 distractors for {entry_id}, only have {len(all_distractors)}. Skipping this question.", level="error")
            return None
    
    # Take first 5 distractors if we have more
    all_distractors = all_distractors[:5]
    
    # Build final question
    # Create correct option dict
    correct_option_dict = {
        'option_context': refined_option_context,
        'is_correct': True,
        'region_id': region_id,
        'region_type': region_type,
        'functionality': functionality,
        'metrics': metrics,
        'source': 'ground_truth_refined'
    }
    
    # Combine all options (1 correct + 5 distractors)
    all_options = [correct_option_dict] + all_distractors
    
    # Shuffle and assign labels (but track which is correct)
    random.shuffle(all_options)
    
    # Assign labels A, B, C, D, E, F
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, opt in enumerate(all_options):
        opt['label'] = option_labels[i]
        if opt.get('is_correct', False):
            final_correct_answer = option_labels[i]
    
    # Build final question dict
    final_question = {
        'entry_id': entry_id,
        'source_dataset': source_dataset,
        'source_file': question.get('source_file', ''),
        'annotated_image_path': annotated_image_path,
        'question': UNIFIED_QUESTION,
        'options': all_options,
        'correct_answer': final_correct_answer,
        'group_id': group_id,
        'original_question_data': {
            'original_correct_option': original_option_context
        }
    }
    
    return final_question


def balance_correct_answers(questions: List[Dict]) -> List[Dict]:
    """Ensure correct answers are evenly distributed across A/B/C/D/E/F."""
    
    # Count current distribution
    answer_counts = defaultdict(int)
    for q in questions:
        answer_counts[q['correct_answer']] += 1
    
    debug_print(f"Original distribution: {dict(answer_counts)}", level="info")
    
    # Target distribution (as even as possible)
    total = len(questions)
    target_per_answer = total // 6
    remainder = total % 6
    
    targets = {
        'A': target_per_answer + (1 if remainder > 0 else 0),
        'B': target_per_answer + (1 if remainder > 1 else 0),
        'C': target_per_answer + (1 if remainder > 2 else 0),
        'D': target_per_answer + (1 if remainder > 3 else 0),
        'E': target_per_answer + (1 if remainder > 4 else 0),
        'F': target_per_answer
    }
    
    debug_print(f"Target distribution: {targets}", level="info")
    
    # Re-shuffle options to balance distribution
    balanced_questions = []
    answer_assignments = ['A'] * targets['A'] + ['B'] * targets['B'] + ['C'] * targets['C'] + ['D'] * targets['D'] + ['E'] * targets['E'] + ['F'] * targets['F']
    random.shuffle(answer_assignments)
    
    for i, question in enumerate(questions):
        target_answer = answer_assignments[i]
        
        # Find which option is currently correct
        current_correct_idx = None
        for idx, opt in enumerate(question['options']):
            if opt.get('is_correct', False):
                current_correct_idx = idx
                break
        
        # If we need to change the correct answer position
        if question['correct_answer'] != target_answer:
            # Find target position
            target_idx = ['A', 'B', 'C', 'D', 'E', 'F'].index(target_answer)
            
            # Swap options
            if current_correct_idx is not None and target_idx < len(question['options']):
                options = question['options'][:]
                options[current_correct_idx], options[target_idx] = options[target_idx], options[current_correct_idx]
                
                # Update labels
                for j, opt in enumerate(options):
                    opt['label'] = ['A', 'B', 'C', 'D', 'E', 'F'][j]
                
                question['options'] = options
                question['correct_answer'] = target_answer
        
        balanced_questions.append(question)
    
    # Verify new distribution
    new_counts = defaultdict(int)
    for q in balanced_questions:
        new_counts[q['correct_answer']] += 1
    
    debug_print(f"New distribution: {dict(new_counts)}", level="success")
    
    return balanced_questions



def save_checkpoint(output_file: str, processed_questions: List[Dict], 
                   source_dirs: List[str], model_name: str, eval_dir: str) -> None:
    """Save current progress to output file."""
    output_data = {
        'metadata': {
            'total_questions': len(processed_questions),
            'generation_date': datetime.now().strftime("%Y-%m-%d"),
            'description': 'Hard functional region captioning questions with reduced hints',
            'source_datasets': [Path(d).parent.parent.name for d in source_dirs],
            'gemini_model': model_name,
            'eval_results_dir': eval_dir,
            'checkpoint': True,
            'is_checkpoint': True  # Flag to indicate this is a checkpoint
        },
        'questions': processed_questions
    }
    
    # Create backup if file exists
    if os.path.exists(output_file):
        backup_file = output_file + '.backup'
        if os.path.exists(backup_file):
            os.remove(backup_file)
        import shutil
        shutil.copy2(output_file, backup_file)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def load_checkpoint(output_file: str, debug: bool = False) -> Tuple[List[Dict], set]:
    """Load checkpoint from output file if exists."""
    if not os.path.exists(output_file):
        return [], set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', [])
        processed_entry_ids = {q.get('entry_id', '') for q in questions}
        
        if debug:
            debug_print(f"Loaded checkpoint: {len(questions)} questions already processed", level="success")
        
        return questions, processed_entry_ids
    
    except Exception as e:
        if debug:
            debug_print(f"Error loading checkpoint: {e}", level="error")
        return [], set()


def main(args):
    """Main processing function."""
    
    debug_print("=" * 60, level="title")
    debug_print("Hard Functional Region Captioning QA Generation", level="title")
    debug_print("=" * 60, level="title")
    debug_print("", level="info")
    
    # Load checkpoint if resume mode is enabled
    processed_questions = []
    processed_entry_ids = set()
    
    if args.resume:
        debug_print("Resume mode enabled, loading checkpoint...", level="step")
        processed_questions, processed_entry_ids = load_checkpoint(args.output_file, debug=args.debug)
        if processed_entry_ids:
            debug_print(f"Resuming from checkpoint: {len(processed_questions)} questions already processed", level="success")
        else:
            debug_print("No checkpoint found, starting from scratch", level="info")
        debug_print("", level="info")
    
    # Initialize model
    debug_print("Initializing Gemini model...", level="step")
    model = OpenAIModel(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=0.3,
        max_tokens=4096
    )
    debug_print(f"Model initialized: {args.model}", level="success")
    debug_print("", level="info")
    
    # Load eval results
    debug_print("Loading evaluation results...", level="step")
    eval_results = load_eval_results(EVAL_RESULTS_DIR, debug=args.debug)
    debug_print("", level="info")
    
    # Load all questions
    debug_print("Loading questions from source directories...", level="step")
    all_questions = load_all_questions(SOURCE_DIRS, debug=args.debug)
    
    if not all_questions:
        debug_print("Error: No questions loaded", level="error")
        return
    
    # Filter out already processed questions
    if processed_entry_ids:
        questions_to_process = [q for q in all_questions if q.get('entry_id', '') not in processed_entry_ids]
        debug_print(f"Total questions: {len(all_questions)}, Already processed: {len(processed_entry_ids)}, To process: {len(questions_to_process)}", level="info")
    else:
        questions_to_process = all_questions
        debug_print(f"Total questions to process: {len(questions_to_process)}", level="info")
    
    debug_print("", level="info")
    
    # Process each question
    debug_print("Processing questions...", level="step")
    checkpoint_interval = 10  # Save every 10 questions
    
    with tqdm(total=len(questions_to_process), desc="Processing") as pbar:
        for idx, question in enumerate(questions_to_process):
            try:
                processed = process_question(
                    question, eval_results, model, debug=args.debug
                )
                
                if processed:
                    processed_questions.append(processed)
                
                pbar.update(1)
                
                # Save checkpoint periodically
                if args.resume and (idx + 1) % checkpoint_interval == 0:
                    if args.debug:
                        debug_print(f"Saving checkpoint at {idx + 1}/{len(questions_to_process)}", level="info")
                    save_checkpoint(args.output_file, processed_questions, SOURCE_DIRS, args.model, EVAL_RESULTS_DIR)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            except Exception as e:
                debug_print(f"Error processing question: {e}", level="error")
                pbar.update(1)
                continue
    
    debug_print(f"Successfully processed {len(processed_questions)} questions", level="success")
    debug_print("", level="info")
    
    # Balance correct answers
    debug_print("Balancing correct answer distribution...", level="step")
    balanced_questions = balance_correct_answers(processed_questions)
    debug_print("", level="info")
    
    # Build output
    output_data = {
        'metadata': {
            'total_questions': len(balanced_questions),
            'generation_date': datetime.now().strftime("%Y-%m-%d"),
            'description': 'Hard functional region captioning questions with reduced hints',
            'source_datasets': [Path(d).parent.parent.name for d in SOURCE_DIRS],
            'gemini_model': args.model,
            'eval_results_dir': EVAL_RESULTS_DIR
        },
        'questions': balanced_questions
    }
    
    # Save output
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    debug_print("=" * 60, level="title")
    debug_print("Processing complete!", level="success")
    debug_print(f"Output saved to: {output_file}", level="success")
    debug_print(f"Total questions: {len(balanced_questions)}", level="info")
    debug_print("=" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate hard functional region captioning QA",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--output-file", type=str,
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/func_region_cap_hard.json",
                       help="Output JSON file path")
    parser.add_argument("--api-key", type=str, required=True,
                       help="Gemini API key")
    parser.add_argument("--base-url", type=str,
                       default="https://xiaoai.plus/v1",
                       help="API base URL")
    parser.add_argument("--model", type=str,
                       default="gemini-2.5-pro-thinking",
                       help="Model name")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint (if output file exists)")
    
    args = parser.parse_args()
    
    main(args)
