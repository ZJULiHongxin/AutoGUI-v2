"""
Convert FuncElemQA grounding mode questions to Hugging Face dataset format and upload

This script reads the output of gen_region-*_multichoice-qa_aliyun.py (grounding_mode)
and converts it to a Hugging Face dataset, including images, bounding boxes, questions,
options, and metadata.
"""

import os
import glob
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from PIL import Image

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence
except ImportError:
    print("Please install required packages: pip install datasets huggingface_hub")
    exit(1)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except (ImportError, OSError):
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
        'success': Fore.GREEN,
        'warn': Fore.YELLOW,
        'error': Fore.RED,
        'title': Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def convert_to_dataset_format(grounding_dir: str) -> List[Dict[str, Any]]:
    """Convert grounding mode JSON files to dataset format
    
    Args:
        grounding_dir: Directory containing grounding_mode result JSON files
    
    Returns:
        List of dictionaries, one per question, with all relevant metadata
    """
    debug_print("\n📂 Loading grounding mode files...", level="step")

    # Find all result JSON files in grounding_mode directory
    pattern = os.path.join(grounding_dir, "*_result.json")
    result_files = glob.glob(pattern)
    
    if not result_files:
        debug_print(f"⚠️  No result files found in {grounding_dir}", level="warn")
        return []

    debug_print(f"✅ Found {len(result_files)} result files", level="success")
    debug_print("\n🔄 Converting to dataset format...", level="step")

    dataset_entries = []
    cnt = 0

    for idx, file in enumerate(result_files, start=1):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            debug_print(f"⚠️  Failed to load {file}: {e}", level="warn")
            continue

        # Extract result data
        result = data.get('result', {})
        if not result:
            debug_print(f"⚠️  No result data in {file}", level="warn")
            continue

        # Get base image info from result level
        base_image_path = result.get('image_path', '')
        num_regions = result.get('num_regions', 0)
        questions = result.get('questions', [])

        # Extract dataset name from path
        # Assuming path structure: /path/to/AutoGUIv2/dataset_name/...
        path_parts = Path(base_image_path).parts if base_image_path else Path(file).parts
        dataset_name = 'unknown'
        if 'AutoGUIv2' in path_parts:
            idx_autogui = path_parts.index('AutoGUIv2')
            if idx_autogui + 1 < len(path_parts):
                dataset_name = path_parts[idx_autogui + 1]

        # Process each question
        for q_data in questions:
            question = q_data.get('question', '')
            options = q_data.get('options', [])
            correct_answer = q_data.get('correct_answer', '')
            explanation = q_data.get('explanation', '')
            group_id = q_data.get('group_id', -1)
            group_description = q_data.get('group_description', '')
            region_ids = q_data.get('region_ids', [])
            verified_by_vision = q_data.get('verified_by_vision', False)
            generation_mode = q_data.get('generation_mode', 'grounding')

            # Prefer question-level image info, fallback to result-level
            image_path = q_data.get('image_path', base_image_path)
            image_name = q_data.get('image_name', '')
            image_size = q_data.get('image_size', result.get('image_size', {}))

            if not question or not options:
                continue

            # Extract image size
            img_width = image_size.get('width', 0) if isinstance(image_size, dict) else 0
            img_height = image_size.get('height', 0) if isinstance(image_size, dict) else 0
            if not img_width or not img_height:
                # Try to get from image if available
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        img_width, img_height = img.size
                    except (IOError, OSError, ValueError):
                        pass

            # Process options to extract element information
            options_info = []
            correct_option_bbox = None
            correct_option_region_id = None
            
            for opt in options:
                opt_label = opt.get('label', '')
                opt_region_id = opt.get('region_id', '')
                opt_bbox = opt.get('bbox', [])
                opt_metrics = opt.get('metrics', {})
                opt_region_type = opt.get('region_type', 'Unknown')

                # Normalize bbox to [x_min, y_min, x_max, y_max] format if needed
                if len(opt_bbox) == 4:
                    # Assume bbox is already in correct format
                    normalized_bbox = opt_bbox
                else:
                    normalized_bbox = []

                option_info = {
                    'label': opt_label,
                    'region_id': opt_region_id,
                    'bbox': normalized_bbox,
                    'metrics': opt_metrics,
                    'region_type': opt_region_type,
                }
                options_info.append(option_info)

                # Find correct answer's bbox
                if opt_label == correct_answer:
                    correct_option_bbox = normalized_bbox
                    correct_option_region_id = opt_region_id

            # Create dataset entry
            entry = {
                'dataset_name': dataset_name,
                'image_path': image_path,
                'image_name': image_name,
                'image_width': img_width,
                'image_height': img_height,
                'num_regions': num_regions,
                'question': question,
                'options': options_info,
                'correct_answer': correct_answer,
                'explanation': explanation,
                'group_id': group_id,
                'group_description': group_description,
                'region_ids': region_ids,
                'verified_by_vision': verified_by_vision,
                'generation_mode': generation_mode,
                'num_options': len(options_info),
                'correct_option_bbox': correct_option_bbox or [],
                'correct_option_region_id': correct_option_region_id or '',
            }

            dataset_entries.append(entry)
            cnt += 1

        if (idx + 1) % 10 == 0 or idx == len(result_files):
            debug_print(f"   Processed {idx}/{len(result_files)} files, created {cnt} entries so far", level="info")

    debug_print(f"✅ Created {len(dataset_entries)} dataset entries from {len(result_files)} files", level="success")

    # Shuffle entries
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
        'image_width': Value('int32'),
        'image_height': Value('int32'),
        'image_size': Sequence(Value('int32')),  # [width, height]
        'num_regions': Value('int32'),
        'question': Value('string'),
        'options': Value('string'),  # JSON string
        'correct_answer': Value('string'),
        'explanation': Value('string'),
        'group_id': Value('int32'),
        'group_description': Value('string'),
        'region_ids': Value('string'),  # JSON string
        'verified_by_vision': Value('bool'),
        'generation_mode': Value('string'),
        'num_options': Value('int32'),
        'correct_option_bbox': Sequence(Value('float32')),  # [x_min, y_min, x_max, y_max]
        'correct_option_region_id': Value('string'),
    })

    # Prepare data for dataset
    dataset_dict = {
        'image': [],
        'image_name': [],
        'dataset_name': [],
        'image_width': [],
        'image_height': [],
        'image_size': [],
        'num_regions': [],
        'question': [],
        'options': [],
        'correct_answer': [],
        'explanation': [],
        'group_id': [],
        'group_description': [],
        'region_ids': [],
        'verified_by_vision': [],
        'generation_mode': [],
        'num_options': [],
        'correct_option_bbox': [],
        'correct_option_region_id': [],
    }

    for entry in tqdm(entries, total=len(entries), desc="Preparing dataset"):
        # Validate required fields
        if not isinstance(entry.get('group_id'), int) or entry.get('group_id') < 0:
            continue
        if not entry.get('question'):
            continue
        if not entry.get('options') or len(entry.get('options', [])) == 0:
            continue

        image_path = entry['image_path']

        # Load image if requested
        try:
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path).convert('RGB')
                dataset_dict['image'].append(img)
                # Update image size from actual image
                img_width, img_height = img.size
            else:
                # Use provided dimensions or skip
                img_width = entry.get('image_width', 0)
                img_height = entry.get('image_height', 0)
                if not img_width or not img_height:
                    debug_print(f"⚠️  Skipping entry without valid image: {image_path}", level="warn")
                    continue
                # Create a placeholder image or skip
                debug_print(f"⚠️  Image not found, using placeholder: {image_path}", level="warn")
                img = Image.new('RGB', (img_width, img_height), color='white')
                dataset_dict['image'].append(img)
        except (IOError, OSError, ValueError) as e:
            debug_print(f"⚠️  Failed to load image {image_path}: {e}", level="warn")
            continue

        dataset_dict['dataset_name'].append(entry['dataset_name'])
        dataset_dict['image_name'].append(entry['image_name'])
        dataset_dict['image_width'].append(img_width)
        dataset_dict['image_height'].append(img_height)
        dataset_dict['image_size'].append([img_width, img_height])
        dataset_dict['num_regions'].append(entry.get('num_regions', 0))
        dataset_dict['question'].append(entry['question'])
        
        # Convert options and region_ids to JSON strings
        dataset_dict['options'].append(
            json.dumps(entry['options'], ensure_ascii=False)
        )
        dataset_dict['region_ids'].append(
            json.dumps(entry.get('region_ids', []), ensure_ascii=False)
        )
        
        dataset_dict['correct_answer'].append(entry['correct_answer'])
        dataset_dict['explanation'].append(entry.get('explanation', ''))
        dataset_dict['group_id'].append(entry['group_id'])
        dataset_dict['group_description'].append(entry.get('group_description', ''))
        dataset_dict['verified_by_vision'].append(entry.get('verified_by_vision', False))
        dataset_dict['generation_mode'].append(entry.get('generation_mode', 'grounding'))
        dataset_dict['num_options'].append(entry.get('num_options', 0))
        
        # Handle correct_option_bbox
        correct_bbox = entry.get('correct_option_bbox', [])
        if len(correct_bbox) == 4:
            dataset_dict['correct_option_bbox'].append(correct_bbox)
        else:
            dataset_dict['correct_option_bbox'].append([0.0, 0.0, 0.0, 0.0])
        
        dataset_dict['correct_option_region_id'].append(entry.get('correct_option_region_id', ''))

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

    except (ValueError, RuntimeError, ConnectionError) as e:
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
    except (IOError, OSError, ValueError) as e:
        debug_print(f"❌ Failed to save locally: {e}", level="error")
        raise


def main(args):
    """Main conversion and upload function"""
    
    debug_print("═" * 60, level="title")
    debug_print("🔄 Convert FuncElemQA to Hugging Face Dataset", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    debug_print(f"   Grounding Mode Dir: {Fore.CYAN}{args.grounding_dir}{Style.RESET_ALL}", level="info")
    
    debug_print("\n📤 OUTPUT CONFIGURATION", level="step")
    debug_print(f"   Push to Hub: {Fore.GREEN}YES{Style.RESET_ALL}", level="info")
    debug_print(f"   Repository: {Fore.CYAN}{args.repo_id}{Style.RESET_ALL}", level="info")
    debug_print(f"   Private: {Fore.YELLOW}{args.private}{Style.RESET_ALL}", level="info")
    debug_print(f"   Local Cache: {Fore.BLUE}YES{Style.RESET_ALL} (saved to grounding_dir)", level="info")
    
    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    debug_print(f"   Include Images: {Fore.YELLOW}{args.include_images}{Style.RESET_ALL}", level="info")
    
    debug_print("\n" + "═" * 60, level="title")
    
    # Convert to dataset format
    entries = convert_to_dataset_format(args.grounding_dir)

    if not entries:
        debug_print("❌ No valid entries found", level="error")
        return

    # Create HF dataset
    dataset = create_hf_dataset(entries, include_images=args.include_images)

    # Use grounding_dir as base for cache location
    cache_dir = os.path.join(args.grounding_dir, 'hf_dataset_cache', 'FuncElemQA')

    # Always save locally first as cache
    debug_print(f"\n💾 Saving dataset to cache: {cache_dir}", level="info")
    save_local(dataset, cache_dir)

    # Push to hub
    if not args.repo_id:
        debug_print("❌ --repo-id is required", level="error")
        return

    push_to_hub(dataset, args.repo_id, args.hf_token, args.private)
    
    # Print summary
    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Conversion Complete!", level="success")
    debug_print(f"📊 Total entries: {len(dataset['test'])}", level="info")
    
    # Print some statistics
    dataset_names = {}
    group_ids = {}
    num_options_dist = {}
    
    for entry in entries:
        dataset_name = entry.get('dataset_name', 'unknown')
        dataset_names[dataset_name] = dataset_names.get(dataset_name, 0) + 1
        
        group_id = entry.get('group_id', -1)
        group_ids[group_id] = group_ids.get(group_id, 0) + 1
        
        num_opts = entry.get('num_options', 0)
        num_options_dist[num_opts] = num_options_dist.get(num_opts, 0) + 1
    
    debug_print("\n📈 Dataset Distribution:", level="info")
    for dataset_name, count in sorted(dataset_names.items(), key=lambda x: x[1], reverse=True):
        debug_print(f"   {dataset_name}: {count}", level="info")
    
    debug_print("\n📈 Options Count Distribution:", level="info")
    for num_opts, count in sorted(num_options_dist.items(), key=lambda x: x[0]):
        debug_print(f"   {num_opts} options: {count}", level="info")
    
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert FuncElemQA grounding mode questions to Hugging Face dataset format and optionally upload"
    )
    
    # Input arguments
    parser.add_argument("--grounding-dir", 
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/FuncElemQA_eval_gen/result/grounding_mode",
                       help="Directory containing grounding_mode result JSON files")
    
    # Output arguments
    parser.add_argument("--repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncElemQA",
                       help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("LHX_HF_KEY"),
                       help="Hugging Face token (uses LHX_HF_KEY env var if not provided)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private on Hugging Face")
    
    # Processing arguments
    parser.add_argument("--include-images", action="store_true",
                       help="Include actual image data in dataset (otherwise just paths)")
    
    parsed_args, _ = parser.parse_known_args()

    main(parsed_args)

