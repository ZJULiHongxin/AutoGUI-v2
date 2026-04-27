"""
Convert FuncRegionCap captioning mode questions to Hugging Face dataset format and upload

This script reads the output from captioning_mode directory and converts
it to a Hugging Face dataset, including images, region IDs, questions,
options, and metadata.

Dataset Fields Explanation:
- annotated_image: PIL Image object (or path if --include-images is False), loaded from annotated_image_path
- annotated_image_name: Annotated image filename (extracted from annotated_image_path)
- dataset_name: Name of the dataset (default: osworld_g)
- image_size: [width, height] of the original image
- question: The question text
- correct_answer: Correct answer label (e.g., "A", "B", "C", "D")
- correct_option_idx: Index of correct answer in option arrays (e.g., 0, 1, 2, 3)
- target_region_id: Region ID of the correct option
- explanation: Explanation for the correct answer
- num_options: Number of options in the question
- option_labels: List of option labels ["A", "B", "C", "D"]
- option_region_ids: List of region IDs for each option
- option_region_types: List of region types (e.g., "button", "icon", "text")
- option_area_classes: List of area classifications (e.g., "small", "medium", "large")
- option_density_classes: List of density classifications (e.g., "low", "high")
- option_contexts: List of contextual descriptions for each option
- option_descriptions: List of textual descriptions for each option
- option_functionalities: List of functionality summaries for each option
- group_id: Group ID for the question
- group_description: Description of the group
- generation_mode: Mode of generation (e.g., "captioning_mode")
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict
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
        'success': Fore.GREEN,
        'warn': Fore.YELLOW,
        'error': Fore.RED,
        'title': Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def load_captioning_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all captioning mode result files
    
    Args:
        data_dir: Directory containing *_result.json files
    
    Returns:
        List of all questions from all files
    """
    debug_print("\n📂 Loading captioning mode data files...", level="step")
    
    # Find all result files
    result_files = glob.glob(os.path.join(data_dir, "*_result.json"))
    result_files = [f for f in result_files if not f.endswith("_processing_summary.json")]
    
    debug_print(f"   Found {len(result_files)} result files", level="info")
    
    all_questions = []
    
    for result_file in tqdm(result_files, desc="Loading result files"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = data.get('result', {})
            questions = result.get('questions', [])
            
            # Add metadata to each question
            for q in questions:
                # Ensure image_path and image_name are set
                if 'image_path' not in q or not q['image_path']:
                    q['image_path'] = result.get('image_path', '')
                if 'image_name' not in q or not q['image_name']:
                    # Extract image name from image_path
                    if q['image_path']:
                        q['image_name'] = os.path.basename(q['image_path'])
                    else:
                        # Try to get from image_key if available
                        image_key = data.get('image_key', '')
                        if image_key:
                            q['image_name'] = os.path.basename(image_key)
                        else:
                            q['image_name'] = os.path.basename(result_file).replace('_result.json', '.png')
                
                # Add image_size if available (from annotated_image_size or result)
                if 'image_size' not in q:
                    # Try annotated_image_size first (it's [H, W])
                    if 'annotated_image_size' in q and q['annotated_image_size']:
                        q['image_size'] = q['annotated_image_size']
                    else:
                        q['image_size'] = result.get('image_size', [0, 0])
                
                # generation_timestamp is intentionally not recorded in HF dataset
                q['generation_timestamp'] = data.get('metadata', {}).get('timestamp', '')
            
            all_questions.extend(questions)
            
        except Exception as e:
            debug_print(f"⚠️  Failed to load {result_file}: {e}", level="warn")
            continue
    
    debug_print(f"✅ Loaded {len(all_questions)} total questions", level="success")
    
    return all_questions


def convert_to_dataset_format(questions: List[Dict[str, Any]], base_image_dir: Optional[str] = None, dataset_name: str = 'osworld_g') -> List[Dict[str, Any]]:
    """Convert questions to dataset format
    
    Args:
        questions: List of question dictionaries
        base_image_dir: Base directory to resolve relative image paths (optional)
        dataset_name: Name of the dataset (default: 'osworld_g')
    
    Returns:
        List of dictionaries ready for HF dataset
    """
    debug_print("\n🔄 Converting to dataset format...", level="step")
    
    dataset_entries = []
    skipped_count = 0
    
    for q in tqdm(questions, desc="Processing questions"):
        try:
            # Get image path
            image_path = q.get('image_path', '')
            if not image_path:
                skipped_count += 1
                continue
            
            # Resolve image path if base_image_dir is provided
            if base_image_dir and not os.path.isabs(image_path):
                image_path = os.path.join(base_image_dir, image_path)
            
            # Check if image exists
            if not os.path.exists(image_path):
                debug_print(f"⚠️  Image not found: {image_path}", level="warn")
                skipped_count += 1
                continue
            
            # Extract basic fields
            question_text = q.get('question', '')
            correct_answer = q.get('correct_answer', '')
            explanation = q.get('explanation', '')
            
            if not question_text or not correct_answer:
                skipped_count += 1
                continue
            
            # Extract options
            options = q.get('options', [])
            if not options:
                skipped_count += 1
                continue
            
            # Process options into separate lists
            option_labels = []
            option_region_ids = []
            option_region_types = []
            option_area_classes = []
            option_density_classes = []
            option_contexts = []
            option_descriptions = []
            option_functionalities = []
            
            for opt in options:
                option_labels.append(opt.get('label', ''))
                option_region_ids.append(opt.get('region_id', ''))
                option_region_types.append(opt.get('region_type', ''))
                
                # Extract metrics
                metrics = opt.get('metrics', {})
                area_info = metrics.get('area', {})
                density_info = metrics.get('density', {})
                
                option_area_classes.append(area_info.get('area_class', ''))
                option_density_classes.append(density_info.get('density_class', ''))
                option_contexts.append(opt.get('option_context', ''))
                option_descriptions.append(opt.get('description', ''))
                option_functionalities.append(opt.get('functionality', ''))
            
            # Get correct option index
            correct_option_idx = -1
            for idx, label in enumerate(option_labels):
                if label == correct_answer:
                    correct_option_idx = idx
                    break
            
            if correct_option_idx == -1:
                skipped_count += 1
                continue
            
            # Get target region ID (instead of bbox)
            target_region_id = q.get('target_region_id', '')
            if not target_region_id:
                # Fallback to correct option's region_id
                target_region_id = option_region_ids[correct_option_idx] if correct_option_idx >= 0 else ''
            
            # Extract image size (accept dict or [H, W] list/tuple)
            image_size = q.get('image_size', [])
            img_width = 0
            img_height = 0
            if isinstance(image_size, dict):
                img_width = int(image_size.get('width', 0) or 0)
                img_height = int(image_size.get('height', 0) or 0)
            elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                # Incoming appears to be [H, W]; store as [W, H]
                img_height = int(image_size[0] or 0)
                img_width = int(image_size[1] or 0)
            
            # Get annotated image info (optional)
            annotated_image_path = q.get('annotated_image_path', '')
            # Extract annotated_image_name from path (last part after /)
            annotated_image_name = ''
            if annotated_image_path:
                annotated_image_name = os.path.basename(annotated_image_path)
            
            # Create dataset entry
            entry = {
                'annotated_image_path': annotated_image_path,  # Keep temporarily for loading image
                'annotated_image_name': annotated_image_name,
                # Prefer dataset_name in question if provided; otherwise fallback to arg
                'dataset_name': q.get('dataset_name', dataset_name),
                'image_size': [img_width, img_height],  # [width, height]
                'question': question_text,
                'correct_answer': correct_answer,
                'correct_option_idx': correct_option_idx,
                'target_region_id': target_region_id,
                'explanation': explanation,
                'num_options': len(options),
                'option_labels': option_labels,
                'option_region_ids': option_region_ids,
                'option_region_types': option_region_types,
                'option_area_classes': option_area_classes,
                'option_density_classes': option_density_classes,
                'option_contexts': option_contexts,
                'option_descriptions': option_descriptions,
                'option_functionalities': option_functionalities,
                'group_id': q.get('group_id', -1),
                'group_description': q.get('group_description', ''),
                'generation_mode': q.get('generation_mode', 'captioning_mode'),
            }
            
            dataset_entries.append(entry)
            
        except Exception as e:
            debug_print(f"⚠️  Failed to process question: {e}", level="warn")
            skipped_count += 1
            continue
    
    debug_print(f"✅ Processed {len(dataset_entries)} valid entries", level="success")
    if skipped_count > 0:
        debug_print(f"⚠️  Skipped {skipped_count} invalid entries", level="warn")
    
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
        'annotated_image': HFImage() if include_images else Value('string'),
        'annotated_image_name': Value('string'),
        'dataset_name': Value('string'),
        'image_size': Sequence(Value('int32')),  # [width, height]
        'question': Value('string'),
        'correct_answer': Value('string'),
        'correct_option_idx': Value('int32'),
        'target_region_id': Value('string'),
        'explanation': Value('string'),
        'num_options': Value('int32'),
        'option_labels': Sequence(Value('string')),
        'option_region_ids': Sequence(Value('string')),
        'option_region_types': Sequence(Value('string')),
        'option_area_classes': Sequence(Value('string')),
        'option_density_classes': Sequence(Value('string')),
        'option_contexts': Sequence(Value('string')),
        'option_descriptions': Sequence(Value('string')),
        'option_functionalities': Sequence(Value('string')),
        'group_id': Value('int32'),
        'group_description': Value('string'),
        'generation_mode': Value('string'),
    })
    
    # Prepare data for dataset
    dataset_dict = {key: [] for key in features.keys()}
    
    for entry in tqdm(entries, desc="Preparing dataset"):
        try:
            # Validate required fields
            if entry['correct_option_idx'] < 0 or entry['num_options'] == 0:
                continue
            if not entry['target_region_id']:
                continue
            
            annotated_image_path = entry['annotated_image_path']
            
            # Load annotated image if requested
            if include_images:
                if not annotated_image_path:
                    debug_print("⚠️  Missing annotated_image_path, skipping entry", level="warn")
                    continue
                try:
                    img = Image.open(annotated_image_path).convert('RGB')
                    dataset_dict['annotated_image'].append(img)
                except Exception as e:
                    debug_print(f"⚠️  Failed to load annotated image {annotated_image_path}: {e}", level="warn")
                    continue
            else:
                dataset_dict['annotated_image'].append(annotated_image_path)
            
            # Add all other fields
            dataset_dict['annotated_image_name'].append(entry['annotated_image_name'])
            dataset_dict['dataset_name'].append(entry['dataset_name'])
            dataset_dict['image_size'].append(entry['image_size'])
            dataset_dict['question'].append(entry['question'])
            dataset_dict['correct_answer'].append(entry['correct_answer'])
            dataset_dict['correct_option_idx'].append(entry['correct_option_idx'])
            dataset_dict['target_region_id'].append(entry['target_region_id'])
            dataset_dict['explanation'].append(entry['explanation'])
            dataset_dict['num_options'].append(entry['num_options'])
            dataset_dict['option_labels'].append(entry['option_labels'])
            dataset_dict['option_region_ids'].append(entry['option_region_ids'])
            dataset_dict['option_region_types'].append(entry['option_region_types'])
            dataset_dict['option_area_classes'].append(entry['option_area_classes'])
            dataset_dict['option_density_classes'].append(entry['option_density_classes'])
            dataset_dict['option_contexts'].append(entry['option_contexts'])
            dataset_dict['option_descriptions'].append(entry['option_descriptions'])
            dataset_dict['option_functionalities'].append(entry['option_functionalities'])
            dataset_dict['group_id'].append(entry['group_id'])
            dataset_dict['group_description'].append(entry['group_description'])
            dataset_dict['generation_mode'].append(entry['generation_mode'])
            
        except Exception as e:
            debug_print(f"⚠️  Failed to add entry to dataset: {e}", level="warn")
            continue
    
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
    debug_print("🔄 Convert FuncRegionCap Captioning to HF Dataset", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    if getattr(args, "data_dirs", None):
        debug_print(f"   Data Directories: {Fore.CYAN}{', '.join(args.data_dirs)}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Data Directory: {Fore.CYAN}{args.data_dir}{Style.RESET_ALL}", level="info")
    debug_print(f"   Dataset Name: {Fore.YELLOW}{args.dataset_name}{Style.RESET_ALL}", level="info")
    if args.base_image_dir:
        debug_print(f"   Base Image Dir: {Fore.YELLOW}{args.base_image_dir}{Style.RESET_ALL}", level="info")
    
    debug_print("\n📤 OUTPUT CONFIGURATION", level="step")
    upload_status = f"{Fore.GREEN}YES{Style.RESET_ALL}" if args.upload else f"{Fore.YELLOW}NO (use --upload to enable){Style.RESET_ALL}"
    debug_print(f"   Push to Hub: {upload_status}", level="info")
    if args.upload:
        debug_print(f"   Repository: {Fore.CYAN}{args.repo_id}{Style.RESET_ALL}", level="info")
        debug_print(f"   Private: {Fore.YELLOW}{args.private}{Style.RESET_ALL}", level="info")
    debug_print(f"   Local Cache: {Fore.BLUE}{args.output_dir}{Style.RESET_ALL}", level="info")
    
    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    debug_print(f"   Include Images: {Fore.YELLOW}{args.include_images}{Style.RESET_ALL}", level="info")
    
    debug_print("\n" + "═" * 60, level="title")
    
    # Load captioning data
    questions = []
    if getattr(args, "data_dirs", None):
        for data_dir in args.data_dirs:
            questions.extend(load_captioning_data(data_dir))
    else:
        questions = load_captioning_data(args.data_dir)
    
    if not questions:
        debug_print("❌ No valid questions found", level="error")
        return
    
    # Convert to dataset format
    entries = convert_to_dataset_format(questions, args.base_image_dir, args.dataset_name)
    
    if not entries:
        debug_print("❌ No valid entries after conversion", level="error")
        return
    
    # Create HF dataset
    dataset = create_hf_dataset(entries, include_images=args.include_images)
    
    # Save locally first as cache
    debug_print(f"\n💾 Saving dataset to cache: {args.output_dir}", level="info")
    save_local(dataset, args.output_dir)
    
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
    debug_print("\n📈 Dataset Statistics:", level="info")
    
    num_options_dist = defaultdict(int)
    for entry in entries:
        num_options_dist[entry['num_options']] += 1
    
    debug_print(f"\n   Number of options distribution:", level="info")
    for num_opts, count in sorted(num_options_dist.items()):
        debug_print(f"      {num_opts} options: {count}", level="info")
    
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert FuncRegionCap captioning mode questions to Hugging Face dataset format and optionally upload"
    )
    
    # Input arguments
    parser.add_argument("--data-dir", 
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
                       help="Directory containing *_result.json files")
    parser.add_argument("--data-dirs", nargs="+", type=str, default=None,
                       help="Multiple directories to merge; each containing *_result.json files. "
                            "If not provided, will use default 4 directories: "
                            "osworld_g, screenspot_pro, agentnet, amex")
    parser.add_argument("--base-image-dir", type=str, default=None,
                       help="Base directory to resolve relative image paths (optional)")
    parser.add_argument("--dataset-name", type=str, default="osworld_g",
                       help="Name of the dataset (default: osworld_g)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str,
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionCap",
                       help="Local directory to save the dataset")
    parser.add_argument("--upload", action="store_true",
                       help="Upload dataset to Hugging Face Hub (default: False, only save locally)")
    parser.add_argument("--repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncRegionCap",
                       help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("LHX_HF_KEY"),
                       help="Hugging Face token (uses LHX_HF_KEY env var if not provided)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private on Hugging Face")
    
    # Processing arguments
    parser.add_argument("--include-images", action="store_true", default=True,
                       help="Include actual image data in dataset (default: True; set to False by overriding in code or changing flag logic)")
    
    args = parser.parse_args()
    
    # If data_dirs not provided, use default 4 directories
    if not args.data_dirs and not args.data_dir:
        args.data_dirs = [
            "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
            "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode",
            "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode",
            "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode",
        ]
    
    main(args)

