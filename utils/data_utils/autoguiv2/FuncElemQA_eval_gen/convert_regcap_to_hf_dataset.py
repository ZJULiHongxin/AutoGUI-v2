"""
Convert FuncRegionCap captioning mode questions to Hugging Face dataset format and upload

This script supports two input modes:
  1) Legacy mode: scan one or more directories of ``*_result.json`` files
     produced by the captioning_mode pipeline (use ``--data-dir`` or
     ``--data-dirs``).
  2) Consolidated mode: read a single curated JSON file such as
     ``func_region_cap_hard.json`` (use ``--consolidated-json``). The
     consolidated file is expected to follow the schema:
         {
           "metadata": {...},
           "questions": [
             {
               "entry_id": str,
               "source_dataset": str,
               "annotated_image_path": str,
               "question": str,
               "correct_answer": "A"|"B"|...,
               "group_id": int,
               "options": [
                 {"label": "A", "option_context": str, "is_correct": bool, ...},
                 ...
               ],
               ...
             },
             ...
           ]
         }
     Image size will be derived from the annotated image via PIL since the
     consolidated file does not carry it.

Dataset Fields Explanation:
- annotated_image: PIL Image object (or path if --include-images is False), loaded from annotated_image_path
- annotated_image_name: Annotated image filename (extracted from annotated_image_path)
- dataset_name: Name of the dataset (default: osworld_g)
- image_size: [width, height] of the original image
- question: The question text
- correct_answer: Correct answer label (e.g., "A", "B", "C", "D")
- correct_option_idx: Index of correct answer in option arrays (e.g., 0, 1, 2, 3)
- target_region_id: Region ID of the correct option
- explanation: Explanation for the correct answer (legacy mode only; dropped in consolidated mode)
- num_options: Number of options in the question
- option_labels: List of option labels ["A", "B", "C", "D"]
- option_region_ids: List of region IDs for each option
- option_region_types: List of region types (e.g., "button", "icon", "text")
- option_area_classes: List of area classifications (e.g., "small", "medium", "large")
- option_density_classes: List of density classifications (e.g., "low", "high")
- option_contexts: List of contextual descriptions for each option
- option_descriptions: List of textual descriptions for each option (legacy mode only; dropped in consolidated mode)
- option_functionalities: List of functionality summaries for each option
- group_id: Group ID for the question
- group_description: Description of the group (legacy mode only; dropped in consolidated mode)
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


def load_consolidated_captioning_data(json_path: str) -> List[Dict[str, Any]]:
    """Load questions from a single consolidated JSON file.

    Expected to handle files like ``func_region_cap_hard.json`` whose root
    object is ``{"metadata": {...}, "questions": [...]}``.

    The returned questions are normalized so that downstream
    :func:`convert_to_dataset_format` can consume them without further
    branching:

    - ``image_path`` is filled from ``annotated_image_path`` so the existence
      check passes (the consolidated format does not carry the original
      screenshot path).
    - ``dataset_name`` is filled from ``source_dataset`` (per question).
    - ``image_size`` is derived from the annotated image via PIL and stored as
      ``{"width": W, "height": H}``.
    """
    debug_print(f"\n📂 Loading consolidated captioning JSON: {json_path}", level="step")

    if not os.path.exists(json_path):
        debug_print(f"❌ File not found: {json_path}", level="error")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_questions = data.get('questions', [])
    debug_print(f"   Found {len(raw_questions)} questions in consolidated JSON", level="info")

    metadata = data.get('metadata', {})
    if metadata:
        debug_print(
            f"   metadata.description: {metadata.get('description', '')}",
            level="info",
        )

    normalized: List[Dict[str, Any]] = []
    missing_image_count = 0
    pil_size_cache: Dict[str, Dict[str, int]] = {}

    for q in tqdm(raw_questions, desc="Normalizing consolidated questions"):
        annotated_path = q.get('annotated_image_path', '')

        # Mirror annotated_image_path into image_path so the downstream
        # existence check in convert_to_dataset_format works unchanged.
        if not q.get('image_path'):
            q['image_path'] = annotated_path

        # Map source_dataset -> dataset_name (per question).
        if not q.get('dataset_name') and q.get('source_dataset'):
            q['dataset_name'] = q['source_dataset']

        # Derive image size from the annotated image (PIL is lazy on open()).
        if not q.get('image_size') and annotated_path:
            if annotated_path in pil_size_cache:
                q['image_size'] = pil_size_cache[annotated_path]
            elif os.path.exists(annotated_path):
                try:
                    with Image.open(annotated_path) as img:
                        size_dict = {'width': img.width, 'height': img.height}
                    q['image_size'] = size_dict
                    pil_size_cache[annotated_path] = size_dict
                except Exception as e:
                    debug_print(
                        f"⚠️  Failed to read image size for {annotated_path}: {e}",
                        level="warn",
                    )
                    q['image_size'] = {'width': 0, 'height': 0}
            else:
                missing_image_count += 1
                q['image_size'] = {'width': 0, 'height': 0}

        # Default generation_mode for hard captioning entries.
        if not q.get('generation_mode'):
            q['generation_mode'] = 'captioning_mode_hard'

        normalized.append(q)

    if missing_image_count:
        debug_print(
            f"⚠️  {missing_image_count} entries reference annotated images that do not exist on disk",
            level="warn",
        )

    debug_print(
        f"✅ Normalized {len(normalized)} questions from consolidated JSON",
        level="success",
    )
    return normalized


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
            correct_answers = q.get('correct_answers') or []
            if isinstance(correct_answers, str):
                correct_answers = [x.strip() for x in correct_answers.split(',') if x.strip()]
            if not correct_answer and correct_answers:
                correct_answer = correct_answers[0]
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
            
            # Get correct option index/indices. Single-answer datasets expose
            # correct_answer; multi-answer variants expose correct_answers.
            correct_answer_labels = correct_answers or [correct_answer]
            correct_option_indices = [
                idx for idx, label in enumerate(option_labels)
                if label in correct_answer_labels
            ]

            if not correct_option_indices:
                skipped_count += 1
                continue

            correct_option_idx = correct_option_indices[0]
            
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
                'correct_answers': correct_answer_labels,
                'correct_option_indices': correct_option_indices,
                'num_correct': len(correct_option_indices),
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


# Fields that do not exist in the consolidated JSON (e.g. ``func_region_cap_hard.json``)
# and therefore should be omitted from the v2 HF schema entirely.
_CONSOLIDATED_DROP_FIELDS = (
    'explanation',           # not present in consolidated JSON
    'option_descriptions',   # consolidated options carry only ``option_context``
    'group_description',     # only ``group_id`` is present in consolidated JSON
)


def create_hf_dataset(
    entries: List[Dict[str, Any]],
    include_images: bool = True,
    is_consolidated: bool = False,
    is_multi_answer: bool = False,
) -> DatasetDict:
    """Create Hugging Face Dataset from entries
    
    Args:
        entries: List of dataset entry dictionaries
        include_images: Whether to include actual image data (vs just paths)
        is_consolidated: When True, drop columns that are always empty in the
            consolidated-JSON input mode (see ``_CONSOLIDATED_DROP_FIELDS``).
        is_multi_answer: When True, include multi-answer label/index columns.
    """
    debug_print("\n📦 Creating Hugging Face dataset...", level="step")
    
    # Define full features schema (legacy ``*_result.json`` mode keeps every field).
    full_features: Dict[str, Any] = {
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
    }

    if is_multi_answer:
        full_features.update({
            'correct_answers': Sequence(Value('string')),
            'correct_option_indices': Sequence(Value('int32')),
            'num_correct': Value('int32'),
        })

    if is_consolidated:
        for k in _CONSOLIDATED_DROP_FIELDS:
            full_features.pop(k, None)
        debug_print(
            f"   Consolidated mode: dropping always-empty columns "
            f"{list(_CONSOLIDATED_DROP_FIELDS)}",
            level="info",
        )

    features = Features(full_features)

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

            # Append every remaining column that is part of the active schema.
            for key in dataset_dict:
                if key == 'annotated_image':
                    continue
                dataset_dict[key].append(entry[key])

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
    if getattr(args, "consolidated_json", None):
        debug_print(f"   Consolidated JSON: {Fore.CYAN}{args.consolidated_json}{Style.RESET_ALL}", level="info")
    elif getattr(args, "data_dirs", None):
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
    if getattr(args, "consolidated_json", None):
        questions = load_consolidated_captioning_data(args.consolidated_json)
    elif getattr(args, "data_dirs", None):
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
    is_consolidated = bool(getattr(args, "consolidated_json", None))
    is_multi_answer = any(len(entry.get('correct_option_indices', [])) > 1 for entry in entries)
    dataset = create_hf_dataset(
        entries,
        include_images=args.include_images,
        is_consolidated=is_consolidated,
        is_multi_answer=is_multi_answer,
    )
    
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
    parser.add_argument("--data-dir", default=None,
                       help="Directory containing *_result.json files (legacy mode)")
    parser.add_argument("--data-dirs", nargs="+", type=str, default=None,
                       help="Multiple directories to merge; each containing *_result.json files (legacy mode).")
    parser.add_argument("--consolidated-json", type=str, default=None,
                       help="Path to a single consolidated questions JSON (e.g. func_region_cap_hard.json). "
                            "When provided, takes precedence over --data-dir / --data-dirs.")
    parser.add_argument("--base-image-dir", type=str, default=None,
                       help="Base directory to resolve relative image paths (optional)")
    parser.add_argument("--dataset-name", type=str, default="osworld_g",
                       help="Fallback dataset name when not present per-question (default: osworld_g)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str,
                       default="hf_dataset_cache/FuncRegionCap-v2",
                       help="Local directory to save the dataset")
    parser.add_argument("--upload", action="store_true",
                       help="Upload dataset to Hugging Face Hub (default: False, only save locally)")
    parser.add_argument("--repo-id", type=str, default="HongxinLi/AutoGUIv2-FuncRegionCap-v2",
                       help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("LHX_HF_KEY"),
                       help="Hugging Face token (uses LHX_HF_KEY env var if not provided)")
    parser.add_argument("--private", action="store_true",
                       help="Make the dataset private on Hugging Face")
    
    # Processing arguments
    parser.add_argument("--include-images", action="store_true", default=True,
                       help="Include actual image data in dataset (default: True; set to False by overriding in code or changing flag logic)")
    
    args = parser.parse_args()
    
    if not args.consolidated_json and not args.data_dirs and not args.data_dir:
        parser.error("Provide --consolidated-json, --data-dir, or --data-dirs")
    
    main(args)
