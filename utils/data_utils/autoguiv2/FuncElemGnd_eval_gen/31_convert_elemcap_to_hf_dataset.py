"""
Convert generated captioning questions to Hugging Face dataset format and upload
to both Hugging Face Hub and ModelScope Hub.

This script reads the output of 3_generate_func_captioning_questions.py and converts
it to a Hugging Face dataset, including images, questions, choices, and full metadata
for all candidate elements.

Extended to support ModelScope dataset upload.
"""

import os
import cv2
import glob
import json
import random
import re
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

import immutabledict
ACTION_MAPPING = immutabledict.immutabledict({
    "clicking|click|click_at": "click",
    "hovering|hover|hover_at": "hover",
    "dragging|drag|drag_at": "drag",
    "scrolling|scroll|scroll_at": "scroll",
    "double-clicking|double-click|double_click|double_click_at|double clicking": "double_click",
    "right-clicking|right-click|right_click|right_click_at": "right_click",
    "middle-clicking|middle-click|middle_click|middle_click_at|middle clicking": "middle_click",
    "long pressing|long-pressing|long_pressing|long_press|long_press_at": "long_press",
    "typing|type|type_text_at": "type",
    "selecting|select|select_option|select_option_at": "select",
    "swiping|swipe|swipe_at": "swipe",
    "pressing|press|press_key|press_key_at": "press_key",
    "pressing|press|press_key": "press_key"
    })

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence
    from huggingface_hub import HfApi
except ImportError:
    print("Please install required packages: pip install datasets huggingface_hub")
    exit(1)

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

# Optional ModelScope imports
MODELSCOPE_AVAILABLE = False
try:
    from modelscope.hub.api import HubApi as MSHubApi
    from modelscope.hub.api import ModelScopeConfig
    from modelscope import push_to_hub as ms_push_to_hub
    MODELSCOPE_AVAILABLE = True
except ImportError:
    pass

from utils.data_utils.misc import get_image_dimensions
import numpy as np

random.seed(999)


def normalize_action_type(action_type: str) -> str:
    for pattern, norm_name in ACTION_MAPPING.items():
        if re.match(pattern.lower(), action_type.lower()):
            return norm_name
    return action_type

def sample_entries_by_interaction_type(entries: List[Dict[str, Any]], max_per_action: int = 200) -> List[Dict[str, Any]]:
    """Sample entries to limit each interaction type to max_per_action samples

    Args:
        entries: List of all dataset entries
        max_per_action: Maximum number of samples per interaction type

    Returns:
        Sampled list of entries with at most max_per_action per interaction type
    """
    debug_print(f"\n🎲 Sampling entries (max {max_per_action} per interaction type)...", level="step")

    # Group entries by interaction type
    entries_by_action = defaultdict(list)
    for entry in entries:
        interaction_type = entry.get('interaction_type', 'other')
        entries_by_action[interaction_type].append(entry)

    # Sample from each interaction type
    sampled_entries = []
    for interaction_type, action_entries in entries_by_action.items():
        if len(action_entries) > max_per_action:
            sampled = random.sample(action_entries, max_per_action)
            debug_print(f"   {interaction_type}: {len(action_entries)} -> {max_per_action} (sampled)", level="info")
        else:
            sampled = action_entries
            debug_print(f"   {interaction_type}: {len(action_entries)} (kept all)", level="info")
        sampled_entries.extend(sampled)

    # Shuffle the final list
    random.shuffle(sampled_entries)

    debug_print(f"✅ Sampled {len(sampled_entries)} entries from {len(entries)} total", level="success")

    return sampled_entries


def resolve_cache_reference_path(questions_file: str) -> str:
    """Pick a concrete questions file path for cache derivation."""
    if '*' in questions_file:
        matches = glob.glob(questions_file)
        if matches:
            return matches[0]
    return questions_file


def derive_cache_dir(questions_file: str, repo_id: Optional[str]) -> str:
    """Derive the cache directory path that mirrors the previous logic."""
    reference_path = resolve_cache_reference_path(questions_file)
    segments = reference_path.split('/')

    if len(segments) > 3:
        base_dir = '/'.join(segments[:-3])
    else:
        base_dir = os.path.dirname(reference_path)

    base_dir = base_dir or '.'
    repo_suffix = repo_id.split('/')[-1] if repo_id else Path(reference_path).stem or 'hf_dataset'
    cache_dir = os.path.join(base_dir, 'hf_dataset_cache', repo_suffix)
    return cache_dir


def load_dataset_from_cache(cache_dir: str) -> Optional[DatasetDict]:
    """Load a cached dataset if it already exists."""
    if not os.path.isdir(cache_dir):
        return None

    try:
        dataset = DatasetDict.load_from_disk(cache_dir)
        return dataset
    except Exception as e:
        debug_print(f"⚠️  Failed to load cached dataset at {cache_dir}: {e}", level="warn")
        return None


def compute_dataset_distributions(dataset_dict: DatasetDict) -> tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Compute summary statistics from the dataset for reporting."""
    interaction_types = defaultdict(int)
    dataset_names = defaultdict(int)
    correct_answers = defaultdict(int)

    test_split = dataset_dict.get('test')
    if test_split is None:
        return interaction_types, dataset_names, correct_answers

    for entry in test_split:
        interaction_types[entry.get('interaction_type', 'unknown')] += 1
        dataset_names[entry.get('dataset_name', 'unknown')] += 1
        correct_answers[entry.get('correct_answer', 'unknown')] += 1

    return interaction_types, dataset_names, correct_answers

def debug_print(message: str, level: str = "info") -> None:
    """Colorized debug print"""
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


def load_detection_data(detection_file: str) -> Dict:
    """Load detection data from script 1 output"""
    try:
        with open(detection_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        return checkpoint.get('results', {})
    except Exception as e:
        debug_print(f"❌ Failed to load detection file: {e}", level="error")
        return {}


def get_element_by_id(elements: List[Dict], elem_id: Any) -> Dict:
    """Find element in list by ID"""
    for elem in elements:
        if elem.get('id') == elem_id:
            return elem
    return {}


def load_omniparser_data(image_path: str) -> tuple:
    """Load OmniParser embeddings and elements for an image
    
    Args:
        image_path: Full path to the image file
        
    Returns:
        Tuple of (similar_groups, omniparser_elements) or (None, None) if not found
    """
    try:
        # Derive base_dir and stem from image_path
        if 'images/' in image_path:
            base_dir_str, stem = image_path.rsplit('images/', 1)
            base_dir = Path(base_dir_str)
            stem = stem.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        else:
            # Fallback: assume image_path is relative to some base
            base_dir = Path(image_path).parent.parent
            stem = Path(image_path).stem
        
        # Load embeddings npz file
        npz_path = base_dir / 'omniparser_embeddings' / f'{stem}.npz'
        if not npz_path.exists():
            return None, None
        
        data = np.load(str(npz_path), allow_pickle=True)
        similar_groups = data.get('similar_groups')
        if similar_groups is not None:
            similar_groups = similar_groups.tolist() if hasattr(similar_groups, 'tolist') else similar_groups
        
        # Load omniparser json file
        omniparser_json = base_dir / 'omniparser' / f'{stem}.json'
        if not omniparser_json.exists():
            # Try recursive search
            search_results = glob.glob(str(base_dir / 'omniparser' / '**' / f'{stem}.json'), recursive=True)
            if search_results:
                omniparser_json = Path(search_results[0])
            else:
                return None, None
        
        with open(omniparser_json, 'r', encoding='utf-8') as f:
            omniparser_elements = json.load(f)
        
        return similar_groups, omniparser_elements
    
    except Exception as e:
        debug_print(f"⚠️  Failed to load OmniParser data for {image_path}: {e}", level="warn")
        return None, None


def get_original_omniparser_bbox(element_id: Any, group_index: Any, 
                                  similar_groups: List, 
                                  omniparser_elements: List[Dict]) -> List[float]:
    """Get original OmniParser bbox for an element
    
    Args:
        element_id: Element ID within the group (local index)
        group_index: Group index in similar_groups
        similar_groups: List of similar groups from embeddings
        omniparser_elements: List of OmniParser elements
        
    Returns:
        Original bbox in normalized format [0-1] or empty list if not found
    """
    if similar_groups is None or omniparser_elements is None:
        return []
    
    try:
        # Convert group_index to int if it's a string
        # Note: group_index from detection data is 1-indexed, but similar_groups is 0-indexed
        if isinstance(group_index, str):
            # Try to convert string to int
            try:
                group_idx = int(group_index)
            except ValueError:
                return []
        elif isinstance(group_index, int):
            group_idx = group_index
        else:
            return []
        
        # Convert from 1-indexed to 0-indexed (groups in detection data are 1-indexed)
        if group_idx > 0:
            group_idx = group_idx - 1
        
        # Get the similar group
        if group_idx < 0 or group_idx >= len(similar_groups):
            return []
        
        group = similar_groups[group_idx]
        if not isinstance(group, (list, np.ndarray)):
            return []
        
        # Convert element_id to int
        if isinstance(element_id, str):
            try:
                elem_id = int(element_id) - 1
            except ValueError:
                return []
        elif isinstance(element_id, int):
            elem_id = element_id - 1
        else:
            return []
        
        # Get the OmniParser index from the group
        if elem_id < 0 or elem_id >= len(group):
            return []
        
        omniparser_idx = int(group[elem_id])
        
        # Get the original element
        if omniparser_idx < 0 or omniparser_idx >= len(omniparser_elements):
            return []
        
        omniparser_elem = omniparser_elements[omniparser_idx]
        original_bbox_by_omniparser = omniparser_elem.get('bbox', [])
        
        # Convert from [0-1] normalized to [0-1000] format to match revised bbox format
        if len(original_bbox_by_omniparser) == 4:
            return [round(coord * 1000) for coord in original_bbox_by_omniparser]
        
        return []
    
    except Exception as e:
        debug_print(f"⚠️  Failed to get original bbox: {e}", level="warn")
        return []


def map_candidate_to_element(candidate_info: Dict, img_data: Dict) -> Dict:
    """Map a candidate reference to full element metadata
    
    Args:
        candidate_info: Dict with 'element_id' and 'group_index'
        detection_data: Full detection data from script 1
        current_image_key: Key for current image in detection data
    
    Returns:
        Full element metadata including bbox, description, functionality, etc.
    """
    if not img_data:
        return {}

    elem_id = candidate_info.get('element_id')
    group_idx = candidate_info.get('group_index')

    # Get the image's detection data
    similar_groups = img_data.get('similar_groups', {})

    # Find the group
    if isinstance(similar_groups, dict):
        group = similar_groups.get(str(group_idx), {})
    else:
        # Handle list format
        if isinstance(group_idx, int) and 0 <= group_idx < len(similar_groups):
            group = similar_groups[group_idx]
        else:
            return {}
    
    # Find the element in the group
    elements = group.get('elements', [])
    element = get_element_by_id(elements, elem_id)
    
    return element


def convert_to_dataset_format(questions_file: str) -> List[Dict[str, Any]]:
    """Convert captioning questions JSON to dataset format
    
    Args:
        questions_file: Path to questions JSON from script 3
    
    Returns:
        List of dictionaries, one per question, with all relevant metadata
    """
    debug_print("\n📂 Loading questions and detection files...", level="step")

    # Support glob patterns
    if '/*' in questions_file or '*' in os.path.basename(questions_file):
        questions_files = glob.glob(questions_file)
    else:
        questions_files = [questions_file]

    dataset_entries = []

    for file_idx, q_file in enumerate(questions_files, start=1):
        with open(q_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        # Load detection data
        detection_file = os.path.join(os.path.dirname(q_file), 'similar_elements_anno.json')
        detection_data = load_detection_data(detection_file)
        if not detection_data:
            debug_print("❌ Failed to load detection data", level="error")
            return []

        debug_print(f"✅ Loaded detection data for {len(detection_data)} images", level="success")

        # Load extra attributes
        attr_file = q_file.replace(".json", "_attributes.json")
        if os.path.exists(attr_file):
            with open(attr_file, 'r', encoding='utf-8') as f:
                attr_data = json.load(f)
        else:
            attr_data = {}

        dataset_name = q_file.split('/')[-3] if len(q_file.split('/')) >= 3 else 'unknown'

        # Infer image_src_dir
        image_src_dir = os.path.join(os.path.dirname(os.path.dirname(q_file)), 'images')

        results = checkpoint.get('results', {})
        debug_print(f"✅ Loaded questions for {len(results)} images from {dataset_name}", level="success")

        debug_print(f"\n🔄 Converting to dataset format [{file_idx}/{len(questions_files)}]...", level="step")

        cnt = 0
        for image_key, image_data in tqdm(results.items(), desc=f"Processing {dataset_name}"):
            # Skip entries with errors
            if 'error' in image_data:
                continue

            image_path = image_data.get('image_path', '')
            if not image_path:
                # Try to construct from image_src_dir
                image_path = os.path.join(image_src_dir, image_key)

            # Check if image exists
            if not os.path.exists(image_path):
                debug_print(f"⚠️  Image not found: {image_path}", level="warn")
                continue
            
            # Get image dimension
            W, H = get_image_dimensions(image_path)
            
            # Load OmniParser data for this image (to get original bboxes)
            similar_groups, omniparser_elements = load_omniparser_data(image_path)
            
            # Process each group's generated questions
            generated = image_data.get('generated', [])

            for group_data in generated:
                group_index = group_data.get('group_index', -1)
                visual_similarity = group_data.get('visual_similarity', '')
                elements_in_group = group_data.get('elements_in_group', [])

                # Get question metadata
                question_meta = group_data.get('question_meta', {})
                if not question_meta:
                    continue

                questions_dict = question_meta.get('questions', {})
                candidate_mapping = question_meta.get('candidate_mapping', {})

                # Get target element info
                target_info = candidate_mapping.get('target', {})
                target_elem_id = target_info.get('element_id')

                if image_key in attr_data:
                    group_idx_str, elem_id_str = str(group_index), str(target_elem_id)
                    if group_idx_str not in attr_data[image_key] or elem_id_str not in attr_data[image_key][group_idx_str]:
                        continue
                    extra_attrs = attr_data[image_key][group_idx_str][elem_id_str]
                else:
                    extra_attrs = {}

                # Map target element to full metadata
                image_det_data = detection_data.get(image_key, detection_data.get(image_path, {}))
                target_element = map_candidate_to_element(target_info, image_det_data)

                if not target_element:
                    debug_print(f"⚠️  Could not find target element {target_elem_id} in detection data", level="warn")
                    continue
                
                # Get original OmniParser bbox for target element
                target_original_bbox = get_original_omniparser_bbox(
                    target_elem_id, group_index, similar_groups, omniparser_elements
                )
                
                # if not target_original_bbox:
                #     continue

                target_element['original_bbox_by_omniparser'] = target_original_bbox

                # Process each interaction type question
                for interaction_type, question_data in questions_dict.items():
                    if not isinstance(question_data, dict):
                        continue

                    question_text = question_data.get('question', '')
                    choices = question_data.get('choices', {})
                    explanation = question_data.get('explanation', '')
                    choice_mapping = question_data.get('choice_mapping', {})
                    
                    # The annotating model (i.e., Gemini) probably come up with hallucinated candidates. This causes a situation where the hallucinated candidate does not own a group index, nor its corresponding place in choice_mapping. Therefore, we skip these questions.
                    if len(choice_mapping) != len(choices):
                        continue

                    if not question_text or not choices:
                        continue
                    
                    # Map all choices to their full element metadata
                    choices_list = []
                    correct_choice_data = None
                    
                    for choice_key, choice_text in choices.items():
                        choice_letter = choice_key.split()[0]  # Extract 'A', 'B', etc.
                        
                        # Get candidate info from choice_mapping
                        candidate_info = choice_mapping.get(choice_letter, {})
                        candidate_type = candidate_info.get('type', 'unknown')
                        
                        # Map to full element metadata
                        if candidate_type == 'target':
                            elem_metadata = target_element
                        elif candidate_type == 'generated_negative':
                            elem_metadata = {}
                        else:
                            image_det_data = detection_data.get(image_key, detection_data.get(image_path, {}))
                            elem_metadata = map_candidate_to_element(candidate_info, image_det_data)
                        
                        # Get original OmniParser bbox for this choice element
                        choice_elem_id = candidate_info.get('element_id', -1)
                        choice_group_idx = candidate_info.get('group_index', -1)
                        
                        if candidate_type == 'generated_negative':
                            choice_original_bbox = []
                        else:
                            if not str(choice_group_idx).isdigit():
                                continue

                            choice_original_bbox = get_original_omniparser_bbox(
                                choice_elem_id, choice_group_idx, similar_groups, omniparser_elements
                            )
                            
                            if not choice_original_bbox:
                                continue

                        choice_data = {
                            'text': choice_text,
                            'candidate_type': candidate_type,
                            'element_id': choice_elem_id,
                            'group_index': choice_group_idx,
                            'bbox': elem_metadata.get('revised bbox', []),
                            'original_bbox_by_omniparser': choice_original_bbox,
                            'description': elem_metadata.get('detailed desctiption', 'N/A' if candidate_type == 'generated_negative' else ''),
                            'functionality': elem_metadata.get('unique functionality', 'N/A' if candidate_type == 'generated_negative' else ''),
                            'interaction_outcomes': elem_metadata.get('interaction outcomes', {})
                        }
                        
                        # Compare the two bboxes
                        if False:
                            img = cv2.imread(image_path)
                            x1, y1, x2, y2 = choice_data['original_bbox_by_omniparser']
                            x1, y1, x2, y2 = round(x1 / 1000 * W), round(y1 / 1000 * H), round(x2 / 1000 * W), round(y2 / 1000 * H)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img, 'Original', (x1-15, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            x1, y1, x2, y2 = choice_data['bbox']
                            x1, y1, x2, y2 = round(x1 / 1000 * W), round(y1 / 1000 * H), round(x2 / 1000 * W), round(y2 / 1000 * H)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, 'Revised', (x1-15, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.imwrite('test.png', img)
                            
                        # Track the correct answer (originally 'A')
                        if choice_letter == 'A':
                            correct_choice_data = choice_data
                        
                        choices_list.append(choice_data)
                    
                    # Use rolling mechanism to ensure balanced distribution
                    # Find the index of the correct answer
                    correct_answer_idx_in_list = None
                    for idx, choice_data in enumerate(choices_list):
                        if choice_data == correct_choice_data:
                            correct_answer_idx_in_list = idx
                            break
                    
                    # Skip if correct answer not found
                    if correct_answer_idx_in_list is None or correct_choice_data is None:
                        continue
                    
                    # Calculate target position using modulo for balanced distribution
                    num_choices = len(choices_list)
                    
                    # Skip if no valid choices
                    if num_choices == 0:
                        continue
                    
                    target_position = cnt % num_choices
                    
                    # Rotate the list so correct answer ends up at target_position
                    # Remove correct answer from its current position
                    correct_choice = choices_list.pop(correct_answer_idx_in_list)
                    # Insert it at the target position
                    choices_list.insert(target_position, correct_choice)
                    
                    # Reconstruct choices_with_metadata with new order and find correct answer
                    # Use only as many letters as there are choices (support < 5 choices)
                    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F'][:num_choices]
                    choices_with_metadata = {}
                    correct_answer = choice_letters[target_position]  # Correct answer is now at target_position

                    if num_choices < 5:
                        1+1
                    correct_answer_idx = target_position
                    for idx, choice_data in enumerate(choices_list):
                        if idx >= num_choices:
                            break
                        letter = choice_letters[idx]
                        choices_with_metadata[letter] = choice_data

                    # Debug Draw
                    DEBUG_DRAW = False
                    if DEBUG_DRAW:
                        img = cv2.imread(image_path)
                        print(f"Question: {question_text}")
                        for idx, choice_data in enumerate(choices_list):
                            if idx >= len(choice_letters):
                                break
                            cand_str = f"{choice_letters[idx]}: {choice_data['text']}"
                            if idx == correct_answer_idx:
                                color = (0, 0, 255)
                                cand_str = "(Correct ->) " + cand_str
                            else:
                                color = (0, 255, 0)
                            print(cand_str)
                            x1, y1, x2, y2 = choice_data['bbox']
                            # rescale back
                            x1, y1, x2, y2 = round(x1 / 1000 * W), round(y1 / 1000 * H), round(x2 / 1000 * W), round(y2 / 1000 * H)
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, choice_letters[idx], (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.imwrite('test.png', img)
                        1+1

                    # Create dataset entry
                    entry = {
                        'dataset_name': dataset_name,
                        'image_path': image_path,
                        'image_name': image_path.split("images/")[-1],
                        'question': question_text,
                        'statepred_candidates_string': "\n".join([f"{choice_letters[idx]}: {choice_data['text']}" for idx, choice_data in enumerate(choices_list) if idx < len(choice_letters)]),
                        'description_candidates_string': "\n".join([f"{choice_letters[idx]}: {choice_data['description']}" for idx, choice_data in enumerate(choices_list) if idx < len(choice_letters)]),
                        'correct_answer': correct_answer,  # Dynamically determined after shuffling
                        'interaction_type': normalize_action_type(interaction_type),
                        'choices': choices_with_metadata,
                        'explanation': explanation,
                        'group_index': group_index,
                        'visual_similarity': visual_similarity,
                        'target_element': {
                            'id': target_elem_id,
                            'bbox': target_element.get('revised bbox', []),
                            'original_bbox_by_omniparser': target_element.get('original_bbox_by_omniparser', []),
                            'description': target_element.get('detailed desctiption', ''),
                            'functionality': target_element.get('unique functionality', ''),
                            'interaction_outcomes': target_element.get('interaction outcomes', {})
                        },
                        'num_elements_in_group': len(elements_in_group),
                    } | extra_attrs
                    
                    dataset_entries.append(entry)
                    cnt += 1
        
        debug_print(f"✅ Created {cnt} dataset entries for [{file_idx}/{len(questions_files)}] {dataset_name}", level="success")
    
    # Shuffle for better distribution
    random.shuffle(dataset_entries)
    
    return dataset_entries


def create_hf_dataset(entries: List[Dict[str, Any]], include_images: bool = True) -> DatasetDict:
    """Create Hugging Face Dataset from entries

    Args:
        entries: List of dataset entry dictionaries
        include_images: Whether to include actual image data (vs just paths)
    """
    debug_print("\n📦 Creating Hugging Face dataset...", level="step")

    # Define features schema
    features = Features({
        'dataset_name': Value('string'),
        'image': HFImage() if include_images else Value('string'),
        'image_name': Value('string'),
        'image_size': Sequence(Value('int32')),
        'question': Value('string'),
        'statepred_candidates_string': Value('string'),
        'description_candidates_string': Value('string'),
        'interaction_type': Value('string'),
        'choices': Value('string'),  # JSON string with all choice metadata
        'correct_answer': Value('string'),
        'explanation': Value('string'),
        'group_index': Value('int32'),
        'visual_similarity': Value('string'),
        'target_element': Value('string'),  # JSON string
        'num_elements_in_group': Value('int32'),
        'density_class': Value('string'),
    })

    # Prepare data for dataset
    dataset_dict = {
        'dataset_name': [],
        'image': [],
        'image_name': [],
        'image_size': [],
        'question': [],
        'statepred_candidates_string': [],
        'description_candidates_string': [],
        'interaction_type': [],
        'choices': [],
        'correct_answer': [],
        'explanation': [],
        'group_index': [],
        'visual_similarity': [],
        'target_element': [],
        'num_elements_in_group': [],
        'density_class': []
    }

    for entry in tqdm(entries, desc="Preparing dataset"):
        if not isinstance(entry.get('group_index'), int):
            continue

        image_path = entry['image_path']

        # Load image if requested
        if include_images:
            try:
                img = Image.open(image_path).convert('RGB')
                dataset_dict['image'].append(img)
                dataset_dict['image_size'].append(list(img.size))
            except Exception as e:
                debug_print(f"⚠️  Failed to load image {image_path}: {e}", level="warn")
                continue
        else:
            dataset_dict['image'].append(image_path)
            dataset_dict['image_size'].append([0, 0])

        dataset_dict['dataset_name'].append(entry['dataset_name'])
        dataset_dict['statepred_candidates_string'].append(entry['statepred_candidates_string'])
        dataset_dict['description_candidates_string'].append(entry['description_candidates_string'])
        dataset_dict['image_name'].append(entry['image_name'])
        dataset_dict['question'].append(entry['question'])
        dataset_dict['interaction_type'].append(entry['interaction_type'])
        dataset_dict['correct_answer'].append(entry['correct_answer'])
        dataset_dict['explanation'].append(entry['explanation'])
        dataset_dict['group_index'].append(entry['group_index'])
        dataset_dict['visual_similarity'].append(entry['visual_similarity'])
        dataset_dict['num_elements_in_group'].append(entry['num_elements_in_group'])
        dataset_dict['density_class'].append(entry.get('density_class', 'unknown'))
        # Convert dicts to JSON strings for storage
        dataset_dict['choices'].append(
            json.dumps(entry['choices'], ensure_ascii=False)
        )
        dataset_dict['target_element'].append(
            json.dumps(entry['target_element'], ensure_ascii=False)
        )
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)
    
    # Create DatasetDict with "test" split
    dataset_dict_obj = DatasetDict({"test": dataset})
    
    debug_print(f"✅ Created dataset with {len(dataset)} entries", level="success")
    return dataset_dict_obj


def push_to_hub(dataset: DatasetDict, repo_id: str, token: str = None, private: bool = False):
    """Push dataset to Hugging Face Hub
    
    Args:
        dataset: The dataset to push
        repo_id: Repository ID (e.g., 'username/dataset-name')
        token: HF token (if None, uses HF_TOKEN env var or huggingface-cli login)
        private: Whether to make the dataset private
    """
    debug_print(f"\n🚀 Pushing dataset to Hugging Face Hub: {repo_id}", level="step")

    try:
        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private
        )
        debug_print(f"✅ Successfully pushed to {repo_id}", level="success")
        debug_print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}", level="info")

    except Exception as e:
        debug_print(f"❌ Failed to push to Hub: {e}", level="error")
        raise


def push_to_modelscope(dataset: DatasetDict, repo_id: str, token: str = None, private: bool = False):
    """Push dataset to ModelScope Hub
    
    This function exports the HuggingFace dataset to Parquet format and uploads
    it to ModelScope using the ModelScope Hub API.
    
    Args:
        dataset: The HuggingFace DatasetDict to push
        repo_id: Repository ID on ModelScope (e.g., 'username/dataset-name')
        token: ModelScope token (if None, uses MODELSCOPE_API_KEY env var)
        private: Whether to make the dataset private
    """
    if not MODELSCOPE_AVAILABLE:
        debug_print("❌ ModelScope library not available. Install with: pip install modelscope", level="error")
        return

    debug_print(f"\n🚀 Pushing dataset to ModelScope Hub: {repo_id}", level="step")

    # Create a temporary directory for the dataset files
    temp_dir = tempfile.mkdtemp(prefix="modelscope_upload_")

    try:
        # Login to ModelScope
        api = MSHubApi()
        if token:
            api.login(token)
            debug_print("✅ Logged in to ModelScope", level="success")
        else:
            debug_print("⚠️  No token provided, using cached credentials", level="warn")

        # Export dataset to parquet format (HuggingFace standard format that ModelScope supports)
        debug_print("📦 Exporting dataset to Parquet format...", level="step")

        # Create data directory structure
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Export each split to parquet
        for split_name, split_data in dataset.items():
            parquet_path = os.path.join(data_dir, f"{split_name}.parquet")
            split_data.to_parquet(parquet_path)
            debug_print(f"   Exported {split_name} split to {parquet_path}", level="info")

        # Create a README.md file
        readme_content = f"""# {repo_id.split('/')[-1]}

This dataset was automatically converted from HuggingFace format and uploaded to ModelScope.

## Dataset Description

A GUI element captioning dataset for training and evaluating vision-language models on functional element understanding tasks.

## Usage

```python
from modelscope.msdatasets import MsDataset

# Load the dataset
dataset = MsDataset.load('{repo_id}', split='test')

# Or use with datasets library
from datasets import load_dataset
dataset = load_dataset('{repo_id}', split='test')
```

## Dataset Structure

- **test**: Test split containing evaluation examples

## Fields

- `image`: The screenshot image
- `question`: The question about the GUI element
- `choices`: Multiple choice options (JSON string)
- `correct_answer`: The correct answer letter
- `interaction_type`: Type of interaction (click, hover, etc.)
- And more metadata fields...
"""
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create dataset_infos.json for compatibility
        dataset_info = {
            "default": {
                "description": "GUI element captioning dataset",
                "features": {
                    col: str(dataset['test'].features[col]) 
                    for col in dataset['test'].column_names
                },
                "splits": {
                    split_name: {"num_examples": len(split_data)}
                    for split_name, split_data in dataset.items()
                }
            }
        }
        
        info_path = os.path.join(temp_dir, "dataset_infos.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Try to create dataset repository (may already exist)
        try:
            visibility = 1 if not private else 0  # 1=public, 0=private
            api.create_dataset(repo_id.split('/')[-1], visibility=visibility)
            debug_print(f"✅ Created dataset repository: {repo_id}", level="success")
        except Exception as e:
            debug_print(f"ℹ️  Repository may already exist: {e}", level="info")
        
        # Push to ModelScope using push_to_hub
        debug_print("📤 Uploading to ModelScope...", level="step")
        ms_push_to_hub(
            repo_name=repo_id,
            output_dir=temp_dir,
            token=token,
            private=private,
            commit_message="Upload dataset from HuggingFace format"
        )
        
        debug_print(f"✅ Successfully pushed to ModelScope: {repo_id}", level="success")
        debug_print(f"🔗 View at: https://modelscope.cn/datasets/{repo_id}", level="info")
        
    except Exception as e:
        debug_print(f"❌ Failed to push to ModelScope: {e}", level="error")
        raise
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def push_to_modelscope_git(dataset: DatasetDict, repo_id: str, token: str = None, private: bool = False):
    """Alternative method: Push dataset to ModelScope using git-based approach
    
    This is a more robust method that uses git to push the dataset,
    similar to how HuggingFace Hub works.
    
    Args:
        dataset: The HuggingFace DatasetDict to push
        repo_id: Repository ID on ModelScope (e.g., 'username/dataset-name')
        token: ModelScope token
        private: Whether to make the dataset private
    """
    if not MODELSCOPE_AVAILABLE:
        debug_print("❌ ModelScope library not available. Install with: pip install modelscope", level="error")
        return
    
    debug_print(f"\n🚀 Pushing dataset to ModelScope Hub (git method): {repo_id}", level="step")
    
    # Create a temporary directory for the dataset files
    temp_dir = tempfile.mkdtemp(prefix="modelscope_git_upload_")
    
    try:
        from modelscope.hub.repository import DatasetRepository
        
        # Login
        api = MSHubApi()
        if token:
            api.login(token)
        
        # Get namespace
        namespace, _ = ModelScopeConfig.get_user_info()
        if '/' in repo_id:
            namespace = repo_id.split('/')[0]
            dataset_name = repo_id.split('/')[1]
        else:
            dataset_name = repo_id
        
        full_repo_id = f"{namespace}/{dataset_name}"
        
        # Try to create the dataset
        try:
            visibility = 1 if not private else 0
            api.create_dataset(dataset_name, visibility=visibility)
            debug_print(f"✅ Created dataset repository: {full_repo_id}", level="success")
        except Exception as e:
            debug_print(f"ℹ️  Repository may already exist or creation failed: {e}", level="info")
        
        # Clone/init the repository
        repo = DatasetRepository(
            local_dir=temp_dir,
            clone_from=f"https://www.modelscope.cn/datasets/{full_repo_id}.git"
        )
        
        # Export dataset to parquet format
        debug_print("📦 Exporting dataset to Parquet format...", level="step")
        
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for split_name, split_data in dataset.items():
            parquet_path = os.path.join(data_dir, f"{split_name}.parquet")
            split_data.to_parquet(parquet_path)
            debug_print(f"   Exported {split_name} split", level="info")
        
        # Create README
        readme_content = f"""# {dataset_name}

GUI Element Captioning Dataset

## Usage

```python
from datasets import load_dataset
dataset = load_dataset('{full_repo_id}')
```
"""
        with open(os.path.join(temp_dir, "README.md"), 'w') as f:
            f.write(readme_content)
        
        # Git add, commit, and push
        repo.git_add()
        repo.git_commit("Upload dataset")
        repo.git_push()
        
        debug_print(f"✅ Successfully pushed to ModelScope: {full_repo_id}", level="success")
        debug_print(f"🔗 View at: https://modelscope.cn/datasets/{full_repo_id}", level="info")
        
    except Exception as e:
        debug_print(f"❌ Failed to push to ModelScope: {e}", level="error")
        # Try fallback method
        debug_print("🔄 Trying fallback upload method...", level="warn")
        try:
            push_to_modelscope_simple(dataset, repo_id, token, private, temp_dir)
        except Exception as e2:
            debug_print(f"❌ Fallback also failed: {e2}", level="error")
            raise
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def push_to_modelscope_simple(dataset: DatasetDict, repo_id: str, token: str = None, 
                               private: bool = False, temp_dir: str = None):
    """Simplest method: Export to parquet and use ModelScope's file upload
    
    This method exports the dataset to parquet files and uses basic file upload.
    Most compatible approach.
    
    Args:
        dataset: The HuggingFace DatasetDict to push
        repo_id: Repository ID on ModelScope
        token: ModelScope token
        private: Whether to make the dataset private
        temp_dir: Optional existing temp directory with exported data
    """
    if not MODELSCOPE_AVAILABLE:
        debug_print("❌ ModelScope library not available.", level="error")
        return
    
    debug_print(f"\n🚀 Pushing dataset to ModelScope (simple method): {repo_id}", level="step")
    
    cleanup_temp = temp_dir is None
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="modelscope_simple_")
    
    try:
        # Export to parquet if not already done
        data_dir = os.path.join(temp_dir, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
            for split_name, split_data in dataset.items():
                parquet_path = os.path.join(data_dir, f"{split_name}.parquet")
                split_data.to_parquet(parquet_path)
                debug_print(f"   Exported {split_name} split", level="info")
        
        # Use push_to_hub which handles the upload
        ms_push_to_hub(
            repo_name=repo_id,
            output_dir=temp_dir,
            token=token,
            private=private,
            commit_message="Upload dataset"
        )
        
        debug_print(f"✅ Successfully pushed to ModelScope: {repo_id}", level="success")
        
    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def save_local(dataset: DatasetDict, output_dir: str):
    """Save dataset locally
    
    Args:
        dataset: The dataset to save
        output_dir: Directory to save the dataset
    """
    debug_print(f"\n💾 Saving dataset locally to: {output_dir}", level="step")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset.save_to_disk(output_dir)
        debug_print(f"✅ Successfully saved to {output_dir}", level="success")
    except Exception as e:
        debug_print(f"❌ Failed to save locally: {e}", level="error")
        raise


def main(args):
    """Main conversion and upload function"""
    
    debug_print("═" * 60, level="title")
    debug_print("🔄 Convert Captioning Questions to Hugging Face Dataset", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    debug_print(f"   Questions File: {Fore.CYAN}{args.questions_file}{Style.RESET_ALL}", level="info")
   
    debug_print("\n📤 OUTPUT CONFIGURATION", level="step")
    if args.repo_id:
        debug_print(f"   Push to HuggingFace Hub: {Fore.GREEN}YES{Style.RESET_ALL}", level="info")
        debug_print(f"   HuggingFace Repository: {Fore.CYAN}{args.repo_id}{Style.RESET_ALL}", level="info")
    
    if args.modelscope_repo_id:
        debug_print(f"   Push to ModelScope Hub: {Fore.GREEN}YES{Style.RESET_ALL}", level="info")
        debug_print(f"   ModelScope Repository: {Fore.CYAN}{args.modelscope_repo_id}{Style.RESET_ALL}", level="info")
    
    debug_print(f"   Private: {Fore.YELLOW}{args.private}{Style.RESET_ALL}", level="info")
    
    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    
    debug_print("\n" + "═" * 60, level="title")
    
    cache_dir = derive_cache_dir(args.questions_file, args.repo_id or args.modelscope_repo_id)
    debug_print(f"\n🔍 Checking for cached dataset at: {cache_dir}", level="step")

    dataset = load_dataset_from_cache(cache_dir)
    if dataset is not None:
        debug_print(f"✅ Loaded cached dataset with {len(dataset['test'])} entries", level="success")
    else:
        # Convert to dataset format
        entries = convert_to_dataset_format(args.questions_file)
        if not entries:
            debug_print("❌ No valid entries found", level="error")
            return

        # Sample entries to limit each interaction type to max 200
        entries = sample_entries_by_interaction_type(entries, max_per_action=200)

        # Create HF dataset and save to cache
        dataset = create_hf_dataset(entries)
        debug_print(f"\n💾 Saving dataset to cache: {cache_dir}", level="info")
        save_local(dataset, cache_dir)
        debug_print(f"\n💾 Successfully saved to cache: {cache_dir}", level="info")

    # Push to ModelScope Hub if modelscope_repo_id provided
    if args.modelscope_repo_id:
        if not MODELSCOPE_AVAILABLE:
            debug_print("⚠️  ModelScope library not installed. Install with: pip install modelscope", level="warn")
        else:
            push_to_modelscope(dataset, args.modelscope_repo_id, args.modelscope_token, args.private)
    else:
        debug_print("ℹ️  Skipping push to ModelScope Hub (no --modelscope-repo-id provided)", level="info")

    # Push to HuggingFace Hub if repo_id provided   
    if args.repo_id:
        push_to_hub(dataset, args.repo_id, args.hf_token, args.private)
    else:
        debug_print("ℹ️  Skipping push to HuggingFace Hub (no --repo-id provided)", level="info")



    # Print summary
    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Conversion Complete!", level="success")
    debug_print(f"📊 Total entries: {len(dataset['test'])}", level="info")
    
    # Print statistics
    interaction_types, dataset_names, correct_answers = compute_dataset_distributions(dataset)
    
    debug_print("\n📈 Interaction Type Distribution:", level="info")
    for interaction_type, count in sorted(interaction_types.items(), key=lambda x: x[1], reverse=True):
        debug_print(f"   {interaction_type}: {count}", level="info")
    
    debug_print("\n📈 Dataset Distribution:", level="info")
    for dataset_name, count in sorted(dataset_names.items(), key=lambda x: x[1], reverse=True):
        debug_print(f"   {dataset_name}: {count}", level="info")
    
    debug_print("\n📈 Correct Answer Distribution (after shuffling):", level="info")
    for answer, count in sorted(correct_answers.items()):
        debug_print(f"   {answer}: {count}", level="info")
    
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert generated captioning questions to Hugging Face dataset format and optionally upload to HuggingFace and/or ModelScope"
    )

    # Input arguments
    parser.add_argument("--questions-file", 
                       default=[
                           "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncElemGnd/captioning_questions.json",
                           "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncElemGnd/captioning_questions_gemini-3-flash-preview-thinking.json"][-1],
                       help="Path to questions JSON from 3_generate_func_captioning_questions.py (supports glob patterns)")

    # HuggingFace output arguments
    parser.add_argument("--repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncElemCap-0125",
                       help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("LHX_HF_KEY"),
                       help="Hugging Face token (uses LHX_HF_KEY env var if not provided)")
    
    # ModelScope output arguments
    parser.add_argument("--modelscope-repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncElemCap-0125",
                       help="ModelScope repository ID (e.g., 'username/dataset-name'). If not provided, skips ModelScope upload.")
    parser.add_argument("--modelscope-token", type=str, default=os.environ.get("MODELSCOPE_API_KEY"),
                       help="ModelScope token (uses MODELSCOPE_API_KEY env var if not provided)")
    
    # Common arguments
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private on both platforms")

    args, _ = parser.parse_known_args()

    main(args)