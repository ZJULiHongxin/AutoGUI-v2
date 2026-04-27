"""
Convert generated questions to Hugging Face dataset format and upload

This script reads the output of 2_generate_func_elemgnd_questions.py and converts
it to a Hugging Face dataset, including images, bounding boxes, questions, and
OmniParser metadata.
"""

import os
import cv2
import re
import glob
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
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
    "right-clicking|right-click|right clicking|right_click|right_click_at": "right_click",
    "middle-clicking|middle-click|middle_click|middle_click_at|middle clicking": "middle_click",
    "long clicking|long pressing|long-pressing|long_pressing|long_press|long_press_at": "long_press",
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

def normalize_action_type(action_type: str) -> str:
    for pattern, norm_name in ACTION_MAPPING.items():
        if re.match(pattern.lower(), action_type.lower()):
            return norm_name
    return action_type

def sample_entries_by_action_type(entries: List[Dict[str, Any]], max_per_action: int = 200) -> List[Dict[str, Any]]:
    """Sample entries to limit each action type to max_per_action samples
    
    Args:
        entries: List of all dataset entries
        max_per_action: Maximum number of samples per action type
    
    Returns:
        Sampled list of entries with at most max_per_action per action type
    """
    debug_print(f"\n🎲 Sampling entries (max {max_per_action} per action type)...", level="step")
    
    # Group entries by action type
    entries_by_action = defaultdict(list)
    for entry in entries:
        action_type = entry.get('action_type', 'other')
        entries_by_action[action_type].append(entry)

    # Sample from each action type
    sampled_entries = []
    for action_type, action_entries in entries_by_action.items():
        if len(action_entries) > max_per_action:
            sampled = random.sample(action_entries, max_per_action)
            debug_print(f"   {action_type}: {len(action_entries)} -> {max_per_action} (sampled)", level="info")
        else:
            sampled = action_entries
            debug_print(f"   {action_type}: {len(action_entries)} (kept all)", level="info")
        sampled_entries.extend(sampled)
    
    # Shuffle the final list
    random.shuffle(sampled_entries)
    
    debug_print(f"✅ Sampled {len(sampled_entries)} entries from {len(entries)} total", level="success")
    
    return sampled_entries

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


def load_corrections(dataset_root: str, dataset_name: str) -> Dict[str, Any]:
    """Load corrections from grounding_questions_corrections.json for a dataset"""
    corrections_path = "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncElemGnd/grounding_questions_corrections.json"
    if not os.path.exists(corrections_path):
        debug_print(f"   No corrections file found: {corrections_path}", level="info")
        return {}

    try:
        with open(corrections_path, 'r', encoding='utf-8') as f:
            corrections = json.load(f)
        debug_print(f"   Loaded corrections from: {corrections_path} ({len(corrections)} entries)", level="info")
        return corrections
    except Exception as e:
        debug_print(f"   Warning: Could not load corrections file {corrections_path}: {e}", level="warn")
        return {}


def load_omniparser_data(image_path: str) -> List[Dict]:
    """Load OmniParser results for an image"""
    try:
        # Derive OmniParser path from image path
        p = Path(image_path).resolve()
        stem = p.stem
        parts = list(p.parts)
        base_dir = p.parent

        # Find base directory (before 'images')
        if 'images' in parts:
            idx = parts.index('images')
            base_dir = Path(*parts[:idx])
        
        omniparser_file = base_dir / 'omniparser' / f'{stem}.json'

        if not omniparser_file.exists():
            return []

        with open(omniparser_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Data should be a list of elements
        if isinstance(data, list):
            return data
        return []

    except Exception as e:
        print(f"Warning: Could not load OmniParser data for {image_path}: {e}")
        return []


def find_matching_omniparser_element(target_bbox: List[float], omniparser_data: List[Dict]) -> Dict:
    """Find the OmniParser element that best matches the target bbox
    
    Args:
        target_bbox: Normalized bbox [x_min, y_min, x_max, y_max] in 0-1000 scale
        omniparser_data: List of OmniParser elements with bbox in 0-1 scale
    
    Returns:
        Best matching OmniParser element or empty dict
    """
    if not omniparser_data or not target_bbox or len(target_bbox) != 4:
        return {}

    # Convert target bbox from 0-1000 to 0-1 scale
    target_norm = [target_bbox[0] / 1000, target_bbox[1] / 1000, 
                   target_bbox[2] / 1000, target_bbox[3] / 1000]

    best_match = None
    best_iou = 0.0

    for elem in omniparser_data:
        elem_bbox = elem.get('bbox', [])
        if len(elem_bbox) != 4:
            continue

        # Calculate IoU
        x1 = max(target_norm[0], elem_bbox[0])
        y1 = max(target_norm[1], elem_bbox[1])
        x2 = min(target_norm[2], elem_bbox[2])
        y2 = min(target_norm[3], elem_bbox[3])

        if x2 < x1 or y2 < y1:
            continue

        intersection = (x2 - x1) * (y2 - y1)
        target_area = (target_norm[2] - target_norm[0]) * (target_norm[3] - target_norm[1])
        elem_area = (elem_bbox[2] - elem_bbox[0]) * (elem_bbox[3] - elem_bbox[1])
        union = target_area + elem_area - intersection

        if union > 0:
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_match = elem

    # Return match if IoU is reasonably high
    if best_iou > 0.3:
        return best_match

    return {}


def convert_to_dataset_format(questions_file: str) -> List[Dict[str, Any]]:
    """Convert questions JSON to dataset format
    
    Returns:
        List of dictionaries, one per question, with all relevant metadata
    """
    debug_print("\n📂 Loading questions file...", level="step")

    if '/*' in questions_file:
        questions_files = glob.glob(questions_file)
    else:
        questions_files = [questions_file]

    dataset_entries = []

    for idx, file in enumerate(questions_files, start=1):
        with open(file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        # Load extra attributes
        attr_file = file.replace(".json", "_attributes.json")
        if os.path.exists(attr_file):
            with open(attr_file, 'r', encoding='utf-8') as f:
                attr_data = json.load(f)
        else:
            attr_data = {}

        # Load corrections
        dataset_name = file.split('/')[-3]
        dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(file)))  # Go up from dataset/FuncElemGnd/
        corrections = load_corrections(dataset_root, dataset_name)

        # infer image_src_dir
        image_src_dir = os.path.join(os.path.dirname(os.path.dirname(file)), 'images')

        results = checkpoint.get('results', {})
        debug_print(f"✅ Loaded data for {len(results)} images", level="success")

        debug_print("\n🔄 Converting to dataset format...", level="step")

        cnt = 0
        for image_name, image_data in tqdm(results.items(), desc=f"Processing images for {idx}/{len(questions_files)} {dataset_name}"):
            # Skip entries with errors
            if 'error' in image_data:
                continue

            image_path = image_data.get('image_path', '')
            if not image_path:
                # Try to construct from image_src_dir
                image_path = os.path.join(image_src_dir, image_name)

            # Check if image exists
            if not os.path.exists(image_path):
                debug_print(f"⚠️  Image not found: {image_path}", level="warn")
                continue

            # Load OmniParser data for this image
            omniparser_data = load_omniparser_data(image_path)

            # Process each group
            generated = image_data.get('generated', [])

            for group_data in generated:
                group_index = group_data.get('group_index', -1)
                visual_similarity = group_data.get('visual_similarity', '')

                if not isinstance(group_index, int):
                    continue
                    
                elements = group_data.get('elements', [])
                questions = group_data.get('questions', [])

                # Create a lookup for elements by ID
                elements_by_id = {elem.get('id'): elem for elem in elements}

                # Process each question
                for q_idx, q_data in enumerate(questions):
                    target_elem_id = q_data.get('target_element_id', -1)
                    referring_expressions = q_data.get('referring_expressions', {})

                    # Check corrections for this sample
                    correction_key = f"{image_name}__{group_index-1}__{q_idx}"
                    correction_entry = corrections.get(correction_key, {})

                    # Skip abandoned samples
                    if len(corrections) and (not correction_entry or correction_entry.get("abandoned", False)):
                        debug_print(f"   Skipping abandoned sample: {correction_key}", level="info")
                        continue

                    # Get target element info and apply corrections
                    target_element = elements_by_id.get(target_elem_id, {})
                    original_bbox = target_element.get('revised bbox', [])
                    target_bbox = original_bbox

                    # Apply modified bbox from corrections if available
                    if "modified_bbox" in correction_entry:
                        modified_bbox = correction_entry["modified_bbox"]
                        if isinstance(modified_bbox, list) and len(modified_bbox) == 4:
                            target_bbox = modified_bbox
                            debug_print(f"   Applied modified bbox for {correction_key}", level="info")

                        if False and isinstance(original_bbox, list) and len(original_bbox) == 4:
                            img = cv2.imread(image_path)
                            H, W = img.shape[:2]

                            # draw the original box
                            ox1, oy1, ox2, oy2 = original_bbox
                            ox1, oy1, ox2, oy2 = (
                                round(ox1 / 1000 * W),
                                round(oy1 / 1000 * H),
                                round(ox2 / 1000 * W),
                                round(oy2 / 1000 * H),
                            )
                            cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                            # draw the revised box
                            mx1, my1, mx2, my2 = modified_bbox
                            mx1, my1, mx2, my2 = (
                                round(mx1 / 1000 * W),
                                round(my1 / 1000 * H),
                                round(mx2 / 1000 * W),
                                round(my2 / 1000 * H),
                            )
                            cv2.rectangle(img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
                            cv2.imwrite('test.png', img)
                            1+1

                    # Get extra attributes for this entry
                    if image_name in attr_data:
                        group_idx_str, elem_id_str = str(group_index), str(target_elem_id)
                        if group_idx_str in attr_data[image_name] and elem_id_str in attr_data[image_name][group_idx_str]:
                            extra_attrs = attr_data[image_name][group_idx_str][elem_id_str]
                        else:
                            extra_attrs = {}
                    else:
                        extra_attrs = {}

                    # Create entries for each action type (clicking, hovering, etc.)
                    for action_type, action_data in referring_expressions.items():
                        if not isinstance(action_data, dict):
                            continue

                        # Get original question and action intent
                        question = action_data.get('question', '')
                        action_intent = action_data.get('action_intent', '')

                        # Apply modified questions from corrections if available
                        modified_questions_by_action = correction_entry.get("modified_questions_by_action", {})
                        if action_type in modified_questions_by_action:
                            modified_question = modified_questions_by_action[action_type]
                            if isinstance(modified_question, str) and modified_question.strip():
                                question = modified_question
                                debug_print(f"   Applied modified question for {correction_key} action {action_type}", level="info")
                                # For modified questions, we keep the original action_intent since it's not modified

                        if not question or not action_intent:
                            continue

                        # Prepare all similar elements info for context
                        similar_elements_info = []
                        for elem in elements:
                            elem_info = {
                                'id': elem.get('id', -1),
                                'bbox': elem.get('revised bbox', []),
                                'description': elem.get('detailed desctiption', ''),  # Note: typo from script 1
                                'functionality': elem.get('unique functionality', ''),
                                'interaction_outcomes': elem.get('interaction outcomes', {})
                            }
                            similar_elements_info.append(elem_info)

                        # Create dataset entry
                        entry = {
                            'dataset_name': dataset_name,
                            'image_path': image_path,
                            'image_name': image_name,
                            'question': question,
                            'action_intent': action_intent,
                            'action_type': normalize_action_type(action_type),
                            'group_index': group_index,
                            'visual_similarity': visual_similarity,
                            'id': target_elem_id,
                            'bbox': target_bbox,  # Normalized 0-1000
                            'description': target_element.get('detailed desctiption', ''),
                            'functionality': target_element.get('unique functionality', ''),
                            'interaction_outcomes': target_element.get('interaction outcomes', {}),
                            'similar_elements': similar_elements_info,
                            'num_similar_elements': len(elements),
                        } | extra_attrs

                        dataset_entries.append(entry); cnt += 1

        debug_print(f"✅ Created {cnt} dataset entries for [{idx}/{len(questions_files)} {dataset_name}]", level="success")

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
        'image': HFImage() if include_images else Value('string'),
        'image_name': Value('string'),
        'dataset_name': Value('string'),
        'image_size': Sequence(Value('int32')),
        'question': Value('string'),
        'action_intent': Value('string'),
        'action_type': Value('string'),
        'group_index': Value('int32'),
        'visual_similarity': Value('string'),
        'id': Value('int32'),
        'bbox': Sequence(Value('float32')),  # [x_min, y_min, x_max, y_max] 0-1000
        'description': Value('string'),
        'functionality': Value('string'),
        'interaction_outcomes': Value('string'),  # JSON string
        'similar_elements': Value('string'),  # JSON string for simplicity
        'num_similar_elements': Value('int32'),
        'density_class': Value('string'),
    })

    # Prepare data for dataset
    dataset_dict = {
        'image': [],
        'image_name': [],
        'dataset_name': [],
        'image_size': [],
        'question': [],
        'action_intent': [],
        'action_type': [],
        'group_index': [],
        'visual_similarity': [],
        'id': [],
        'bbox': [],
        'description': [],
        'functionality': [],
        'interaction_outcomes': [],
        'similar_elements': [],
        'num_similar_elements': [],
        'density_class': []
    }

    for entry in tqdm(entries, total=len(entries), desc="Preparing dataset"):
       # if len(dataset_dict['image_name']) >= 10: break
        if not isinstance(entry['group_index'], int) or not isinstance(entry['id'], int) or len(entry['bbox']) != 4 or not isinstance(entry['num_similar_elements'], int):
            continue
        
        
        image_path = entry['image_path']

        # Load image if requested
        try:
            img = Image.open(image_path).convert('RGB')
            dataset_dict['image'].append(img)
        except Exception as e:
            debug_print(f"⚠️  Failed to load image {image_path}: {e}", level="warn")
            continue

        dataset_dict['dataset_name'].append(entry['dataset_name'])
        dataset_dict['image_name'].append(entry['image_name'])
        dataset_dict['image_size'].append(list(img.size))
        dataset_dict['question'].append(entry['question'])
        dataset_dict['action_intent'].append(entry['action_intent'])
        dataset_dict['action_type'].append(entry['action_type'])
        dataset_dict['group_index'].append(entry['group_index'])
        dataset_dict['visual_similarity'].append(entry['visual_similarity'])
        dataset_dict['id'].append(entry['id'])
        dataset_dict['bbox'].append(entry['bbox'])
        dataset_dict['description'].append(entry['description'])
        dataset_dict['functionality'].append(entry['functionality'])


        # Convert dicts to JSON strings for storage
        dataset_dict['interaction_outcomes'].append(
            json.dumps(entry['interaction_outcomes'], ensure_ascii=False)
        )
        dataset_dict['similar_elements'].append(
            json.dumps(entry['similar_elements'], ensure_ascii=False)
        )
        
        dataset_dict['num_similar_elements'].append(entry['num_similar_elements'])
        dataset_dict['density_class'].append(entry.get('density_class', 'unknown'))

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
    debug_print("🔄 Convert to Hugging Face Dataset", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    debug_print(f"   Image Source Dir: {Fore.YELLOW}Will be inferred from questions file{Style.RESET_ALL}", level="info")
    debug_print(f"   Questions File: {Fore.CYAN}{args.questions_file}{Style.RESET_ALL}", level="info")
    
    debug_print("\n📤 OUTPUT CONFIGURATION", level="step")

    upload_status = f"{Fore.GREEN}YES{Style.RESET_ALL}" if args.upload else f"{Fore.YELLOW}NO (use --upload to enable){Style.RESET_ALL}"
    debug_print(f"   Push to Hub: {upload_status}", level="info")
    if args.upload:
        debug_print(f"   Repository: {Fore.CYAN}{args.repo_id}{Style.RESET_ALL}", level="info")
        debug_print(f"   Private: {Fore.YELLOW}{args.private}{Style.RESET_ALL}", level="info")
    debug_print(f"   Local Cache: {Fore.BLUE}YES{Style.RESET_ALL} (saved to /mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemGnd)", level="info")

    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    debug_print(f"   Include Images: {Fore.YELLOW}{args.include_images}{Style.RESET_ALL}", level="info")

    debug_print("\n" + "═" * 60, level="title")

    # Convert to dataset format
    entries = convert_to_dataset_format(args.questions_file)

    if not entries:
        debug_print("❌ No valid entries found", level="error")
        return

    # Sample entries to limit each action type to max 200
    entries = sample_entries_by_action_type(entries, max_per_action=200)

    # Create HF dataset
    dataset = create_hf_dataset(entries, include_images=args.include_images)

    # Use fixed cache location as requested
    cache_dir = "/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncElemGnd"

    # Always save locally first as cache
    debug_print(f"\n💾 Saving dataset to cache: {cache_dir}", level="info")
    save_local(dataset, cache_dir)

    # Push to hub (only if --upload flag is set)
    if args.upload:
        if not args.repo_id:
            debug_print("❌ --repo-id is required when using --upload", level="error")
            return
        push_to_hub(dataset, args.repo_id, args.hf_token, args.private)
    else:
        debug_print("\n💡 Tip: Use --upload flag to push dataset to Hugging Face Hub", level="info")
    
    # Print summary
    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Conversion Complete!", level="success")
    debug_print(f"📊 Total entries: {len(dataset['test'])}", level="info")
    
    # Print some statistics
    action_types = {}
    for entry in entries:
        action_type = entry['action_type']
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    debug_print("\n📈 Action Type Distribution:", level="info")
    for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
        debug_print(f"   {action_type}: {count}", level="info")
    
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert generated questions to Hugging Face dataset format and optionally upload"
    )
    
    # Input arguments
    parser.add_argument("--questions-file", default="/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncElemGnd/grounding_questions.json",
                       help="Path to questions JSON from 2_generate_func_elemgnd_questions.py")
    
    # Output arguments
    parser.add_argument("--upload", action="store_true",
                       help="Upload dataset to Hugging Face Hub (default: False, only save locally)")
    parser.add_argument("--repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncElemGnd",
                       help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("LHX_HF_KEY"),
                       help="Hugging Face token (uses HF_TOKEN env var if not provided)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private on Hugging Face")
    
    # Processing arguments
    parser.add_argument("--include-images", action="store_true",
                       help="Include actual image data in dataset (otherwise just paths)")
    
    args, _ = parser.parse_known_args()

    main(args)

