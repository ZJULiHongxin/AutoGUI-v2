#!/usr/bin/env python3
"""
Build a corrected HuggingFace dataset from annotation corrections.

This script:
1. Loads all corrections from grounding_questions_corrections.json files
2. Builds dataset entries from corrections (with modified_bbox, modified_questions_by_action)
3. Filters out abandoned entries
4. Creates a proper HuggingFace Dataset with images
5. Saves the dataset locally and optionally uploads to HuggingFace Hub

Usage:
    python build_corrected_hf_dataset.py \
        --datasets-root /path/to/AutoGUIv2 \
        --output-dir /path/to/output \
        --hf-repo "username/repo-name" \
        --upload
"""
import random
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.data_utils.autoguiv2.misc import ACTION_TYPE_ALIASES

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' package not installed. Install with: pip install datasets")

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


class EnhancedNIDAnalyzer:
    """Normalized Interference Density analyzer (mirrors 4_calc_task_attributes.py)."""

    def __init__(self, k_sigma: float = 1.5, alpha: float = 1.0):
        self.k_sigma = k_sigma
        self.alpha = alpha
        self.nid_scores = None
        self.thresholds = None

    def _gaussian_2d(self, x, y, mu_x, mu_y, sigma_x, sigma_y):
        return np.exp(-0.5 * (((x - mu_x) / sigma_x) ** 2 + ((y - mu_y) / sigma_y) ** 2))

    def _analysis_region(self, bbox, screen_width, screen_height):
        x1, y1, x2, y2 = bbox
        ex = self.alpha * (x2 - x1)
        ey = self.alpha * (y2 - y1)
        return [
            max(0, x1 - 1.5 * ex),
            max(0, y1 - 1.5 * ey),
            min(screen_width, x2 + 1.5 * ex),
            min(screen_height, y2 + 1.5 * ey),
        ]

    def _in_bbox(self, point, bbox):
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    def _center(self, bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def calculate_nid_score(self, target_bbox, surr_norm_bboxes, screen_width, screen_height):
        """target_bbox is in pixel coords; surr_norm_bboxes are 0-1 normalised."""
        cx, cy = self._center(target_bbox)
        region = self._analysis_region(target_bbox, screen_width, screen_height)
        sigma_x = self.k_sigma * (target_bbox[2] - target_bbox[0])
        sigma_y = self.k_sigma * (target_bbox[3] - target_bbox[1])
        # avoid division by zero for degenerate bboxes
        if sigma_x == 0:
            sigma_x = 1.0
        if sigma_y == 0:
            sigma_y = 1.0
        total = 0.0
        for nb in surr_norm_bboxes:
            ub = [nb[0] * screen_width, nb[1] * screen_height, nb[2] * screen_width, nb[3] * screen_height]
            ec = self._center(ub)
            if self._in_bbox(ec, region) and not self._in_bbox(ec, target_bbox):
                total += self._gaussian_2d(ec[0], ec[1], cx, cy, sigma_x, sigma_y)
        return total

    def calculate_all_nid_scores(self, bboxes, surr_list, widths, heights):
        scores = [
            self.calculate_nid_score(b, s, w, h)
            for b, s, w, h in zip(bboxes, surr_list, widths, heights)
        ]
        self.nid_scores = np.array(scores)
        return self.nid_scores

    def classify_by_percentiles(self, percentiles=(33, 67)):
        self.thresholds = np.percentile(self.nid_scores, percentiles)

    def classify_element(self, score):
        if self.thresholds is None:
            raise ValueError("Call classify_by_percentiles first")
        if score <= self.thresholds[0]:
            return "sparse"
        if score <= self.thresholds[1]:
            return "medium"
        return "dense"

    def classify_all_elements(self, scores):
        return [self.classify_element(s) for s in scores]


def compute_density_lookup(
    all_corrections: Dict[str, Dict[str, Any]],
    datasets_root: str,
    gq_cache: Optional[Dict[str, Dict]] = None,
) -> Dict[str, str]:
    """
    Compute NID-based density_class for corrected elements only.

    For each non-abandoned correction the target bbox (modified if present,
    else original) and the full OmniParser surroundings for that image are used
    to compute a Gaussian-weighted NID score.  Percentile thresholds [33, 67]
    are then applied across this population to assign sparse/medium/dense.

    Returns a lookup dict keyed by
        "{dataset_name}__{image_key}__{group_index}__{elem_id}"
    """
    if gq_cache is None:
        gq_cache = {}

    keys: List[str] = []
    bbox_data: List[List[float]] = []
    surr_bboxes_all: List[List] = []
    screen_widths: List[int] = []
    screen_heights: List[int] = []

    omniparser_cache: Dict[str, List] = {}

    for full_key, correction in all_corrections.items():
        if correction.get("abandoned", False):
            continue

        parts = full_key.split("__")
        if len(parts) != 4:
            continue
        dataset_name, image_key = parts[0], parts[1]
        try:
            generated_list_idx, q_idx = int(parts[2]), int(parts[3])
        except ValueError:
            continue

        # Load grounding questions (cached)
        if dataset_name not in gq_cache:
            qpath = os.path.join(datasets_root, dataset_name, "FuncElemGnd", "grounding_questions.json")
            gq_cache[dataset_name] = load_json(qpath)

        image_data = gq_cache[dataset_name].get("results", {}).get(image_key)
        if not image_data:
            continue

        generated = image_data.get("generated", [])
        if generated_list_idx >= len(generated):
            continue
        group = generated[generated_list_idx]
        questions = group.get("questions", [])
        if q_idx >= len(questions):
            continue

        question_obj = questions[q_idx]
        group_index = group.get("group_index", generated_list_idx)
        target_elem_id = question_obj.get("target_element_id") or question_obj.get("target_element_index")

        # Prefer modified_bbox from correction, fall back to original element bbox
        bbox = correction.get("modified_bbox")
        if not bbox or len(bbox) != 4:
            for el in group.get("elements", []):
                if el.get("id") == target_elem_id:
                    bbox = el.get("revised bbox") or el.get("revised_bbox") or el.get("bbox") or el.get("bbox_global")
                    break
        if not bbox:
            continue
        if isinstance(bbox, str):
            bbox = eval(bbox)  # noqa: S307

        # Image dimensions
        raw_path = image_data.get("image_path", "")
        image_path = os.path.join(datasets_root, raw_path.split("AutoGUIv2/")[-1]) if "AutoGUIv2" in raw_path else raw_path
        try:
            with Image.open(image_path) as img:
                W, H = img.size
        except Exception:
            continue

        # OmniParser surrounding bboxes (cached per image)
        omni_key = f"{dataset_name}__{image_key}"
        if omni_key not in omniparser_cache:
            omni_file = os.path.join(datasets_root, dataset_name, "omniparser", f"{image_key.rsplit('.', 1)[0]}.json")
            if not os.path.exists(omni_file):
                omniparser_cache[omni_key] = []
            else:
                with open(omni_file) as f:
                    omniparser_cache[omni_key] = [x["bbox"] for x in json.load(f)]
        surr = omniparser_cache[omni_key]
        if not surr:
            continue

        key = f"{dataset_name}__{image_key}__{str(group_index)}__{str(target_elem_id)}"
        keys.append(key)

        # unnorm
        unnorm = [bbox[0] / 1000 * W, bbox[1] / 1000 * H, bbox[2] / 1000 * W, bbox[3] / 1000 * H]
        bbox_data.append(list(map(float, unnorm)))
        surr_bboxes_all.append(surr)
        screen_widths.append(W)
        screen_heights.append(H)

    if not keys:
        debug_print("Warning: no corrected elements found for NID computation", level="warn")
        return {}

    debug_print(f"  Computing NID scores for {len(keys)} corrected elements …", level="info")
    analyzer = EnhancedNIDAnalyzer(k_sigma=1.5, alpha=1.0)
    nid_scores = analyzer.calculate_all_nid_scores(bbox_data, surr_bboxes_all, screen_widths, screen_heights)
    analyzer.classify_by_percentiles(percentiles=(33, 67))
    classifications = analyzer.classify_all_elements(nid_scores)

    counts = {c: classifications.count(c) for c in ("sparse", "medium", "dense")}
    debug_print(f"  Density distribution: {counts}", level="info")

    return dict(zip(keys, classifications))


def normalize_action_type(action_type: str) -> str:
    """Map any action_type surface form to its canonical base verb.

    Lookup is case-insensitive. Unknown values are returned lowercased
    with spaces/hyphens replaced by underscores as a best-effort fallback.
    """
    key = action_type.strip().lower()
    if key in ACTION_TYPE_ALIASES:
        return ACTION_TYPE_ALIASES[key]
    # best-effort fallback for unseen variants
    return key.replace("-", "_").replace(" ", "_")


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any], indent: int = 2):
    """Save a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_all_corrections(corrections_root: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all corrections from all datasets.

    Returns a dict keyed by (dataset_name, image_key, generated_list_idx, q_idx)
    """
    all_corrections: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(corrections_root):
        print(f"Warning: corrections_root does not exist: {corrections_root}")
        return all_corrections

    for dataset_name in os.listdir(corrections_root):
        corrections_path = os.path.join(
            corrections_root, dataset_name, "FuncElemGnd", "grounding_questions_corrections.json"
        )
        if os.path.exists(corrections_path):
            corrections = load_json(corrections_path)
            for c_key, c_value in corrections.items():
                # c_key format: "image_key__generated_list_idx__q_idx"
                full_key = f"{dataset_name}__{c_key}"
                all_corrections[full_key] = c_value
            print(f"Loaded {len(corrections)} corrections from {dataset_name}")

    return all_corrections


def load_grounding_questions(datasets_root: str, dataset_name: str) -> Dict[str, Any]:
    """Load the grounding_questions.json for a dataset."""
    qpath = os.path.join(datasets_root, dataset_name, "FuncElemGnd", "grounding_questions.json")
    if os.path.exists(qpath):
        return load_json(qpath)
    return {}


def build_entries_from_corrections(
    all_corrections: Dict[str, Dict[str, Any]],
    used_keys: Set[str],
    datasets_root: str,
    image_cache_dir: Optional[str] = None,
    density_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build dataset entries from corrections.

    Args:
        all_corrections: All loaded corrections
        used_keys: Keys to skip (already processed)
        datasets_root: Root directory containing grounding_questions.json files
        image_cache_dir: Optional directory where images are cached (for image_path)

    Returns:
        - List of entries
        - Statistics dict
    """
    entries: List[Dict[str, Any]] = []
    stats = {
        "extra_total": 0,
        "extra_abandoned": 0,
        "extra_added": 0,
    }

    grounding_questions_cache: Dict[str, Dict[str, Any]] = {}

    for full_key, correction in all_corrections.items():
        if full_key in used_keys:
            continue  # Already processed

        stats["extra_total"] += 1

        # Check if abandoned
        if correction.get("abandoned", False):
            stats["extra_abandoned"] += 1
            continue

        # Parse the full_key: "dataset_name__image_key__generated_list_idx__q_idx"
        parts = full_key.split("__")
        if len(parts) != 4:
            continue

        dataset_name = parts[0]
        image_key = parts[1]
        try:
            generated_list_idx = int(parts[2])
            q_idx = int(parts[3])
        except ValueError:
            continue

        # Load grounding questions for this dataset
        if dataset_name not in grounding_questions_cache:
            qpath = os.path.join(datasets_root, dataset_name, "FuncElemGnd", "grounding_questions.json")
            if os.path.exists(qpath):
                grounding_questions_cache[dataset_name] = load_json(qpath)
            else:
                grounding_questions_cache[dataset_name] = {}

        gq_data = grounding_questions_cache.get(dataset_name, {})
        results = gq_data.get("results", {})
        image_data = results.get(image_key)

        if not image_data:
            continue

        generated = image_data.get("generated", [])
        if generated_list_idx < 0 or generated_list_idx >= len(generated):
            continue

        group = generated[generated_list_idx]
        questions = group.get("questions", [])
        if q_idx < 0 or q_idx >= len(questions):
            continue

        question_obj = questions[q_idx]
        group_index = group.get("group_index", generated_list_idx)
        target_elem_id = question_obj.get("target_element_id") or question_obj.get("target_element_index")

        # Get elements to find the bbox and description
        elements = group.get("elements", [])
        original_bbox = [0, 0, 0, 0]
        description = ""
        functionality = ""
        target_element = None
        for el in elements:
            if el.get("id") == target_elem_id:
                target_element = el
                original_bbox = el.get("revised bbox") or el.get("bbox") or el.get("bbox_global") or [0, 0, 0, 0]
                # Get description and functionality from target element
                description = el.get("detailed desctiption", "")  # Note: typo in original data
                functionality = el.get("unique functionality", "")
                break

        # Apply corrections
        modified_bbox = correction.get("modified_bbox", original_bbox)
        if not modified_bbox or len(modified_bbox) != 4:
            modified_bbox = original_bbox

        # Get question and action_intent - prefer modified, fallback to original
        raw_modified_questions = correction.get("modified_questions_by_action", {})
        modified_questions = {normalize_action_type(k): v for k, v in raw_modified_questions.items()}
        raw_ref_expr = question_obj.get("referring_expressions") or question_obj.get("referring expressions") or {}
        ref_expr = {normalize_action_type(k): v for k, v in raw_ref_expr.items()}
        action_intent = ""

        if modified_questions:
            # Randomly choose an action type from available modified questions
            action_type = random.choice(list(modified_questions.keys()))
            question_text = modified_questions[action_type]
            # Get action_intent from original ref_expr for this action_type
            if ref_expr and action_type in ref_expr and isinstance(ref_expr[action_type], dict):
                action_intent = ref_expr[action_type].get("action_intent", "")
        else:
            # Use original question from referring_expressions
            # Pick any available action
            if ref_expr:
                action_type = random.choice(list(ref_expr.keys()))
                if isinstance(ref_expr[action_type], dict):
                    question_text = ref_expr[action_type].get("question", "")
                    action_intent = ref_expr[action_type].get("action_intent", "")
                else:
                    question_text = ""
            else:
                action_type = "clicking"
                question_text = question_obj.get("question", "")

        # Build image path
        original_image_path = image_data.get("image_path", "")
        # Try to construct a reasonable image path
        if image_cache_dir:
            image_path = os.path.join(image_cache_dir, dataset_name, image_key)
        else:
            # Use the path from grounding_questions, adjusted for datasets_root
            if "AutoGUIv2" in original_image_path:
                image_path = os.path.join(datasets_root, original_image_path.split("AutoGUIv2/")[-1])
            else:
                image_path = original_image_path

        # Count similar elements
        num_similar_elements = len(elements)

        # Look up NID-based density class from pre-computed lookup
        density_key = f"{dataset_name}__{image_key}__{str(group_index)}__{str(target_elem_id)}"
        if density_lookup:
            density_class = density_lookup.get(density_key, "unknown")
        else:
            density_class = "dense" if num_similar_elements > 2 else "sparse"

        # Build entry
        entry = {
            "entry_id": f"{image_key}_{group_index}_{target_elem_id}_{action_type}_{len(entries)}",
            "image_path": image_path,
            "image_name": image_key,
            "dataset_name": dataset_name,
            "question": question_text,
            "action_type": action_type,
            "action_intent": action_intent,
            "description": description,
            "functionality": functionality,
            "gt_bbox": [float(x) for x in modified_bbox],
            "group_index": group_index,
            "target_elem_id": target_elem_id,
            "density_class": density_class,
            "num_similar_elements": num_similar_elements,
        }

        entries.append(entry)
        stats["extra_added"] += 1

    return entries, stats


def create_hf_dataset(entries: List[Dict[str, Any]], include_images: bool = True) -> DatasetDict:
    """Create Hugging Face Dataset from entries

    Args:
        entries: List of dataset entry dictionaries
        include_images: Whether to include actual image data (vs just paths)
    """
    if not HAS_DATASETS:
        raise ImportError("'datasets' package is required. Install with: pip install datasets")

    debug_print("\n📦 Creating Hugging Face dataset...", level="step")

    # Define features schema
    features = Features({
        'image': HFImage() if include_images else Value('string'),
        'image_name': Value('string'),
        'dataset_name': Value('string'),
        'image_size': Sequence(Value('int32')),
        'question': Value('string'),
        'action_intent': Value('string'),
        'description': Value('string'),
        'functionality': Value('string'),
        'action_type': Value('string'),
        'group_index': Value('string'),  # Can be int or string like 'newly_added_group_1'
        'target_elem_id': Value('string'),  # Can be int or string
        'bbox': Sequence(Value('float32')),  # [x_min, y_min, x_max, y_max]
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
        'description': [],
        'functionality': [],
        'action_type': [],
        'group_index': [],
        'target_elem_id': [],
        'bbox': [],
        'num_similar_elements': [],
        'density_class': [],
    }

    skipped = 0
    for entry in tqdm(entries, total=len(entries), desc="Preparing dataset"):
        image_path = entry.get('image_path', '')

        # Load image if requested
        if include_images:
            try:
                img = Image.open(image_path).convert('RGB')
                dataset_dict['image'].append(img)
                dataset_dict['image_size'].append(list(img.size))
            except Exception as e:
                debug_print(f"⚠️  Failed to load image {image_path}: {e}", level="warn")
                skipped += 1
                continue
        else:
            dataset_dict['image'].append(image_path)
            dataset_dict['image_size'].append([0, 0])

        dataset_dict['dataset_name'].append(entry.get('dataset_name', ''))
        dataset_dict['image_name'].append(entry.get('image_name', ''))
        dataset_dict['question'].append(entry.get('question', ''))
        dataset_dict['action_intent'].append(entry.get('action_intent', ''))
        dataset_dict['description'].append(entry.get('description', ''))
        dataset_dict['functionality'].append(entry.get('functionality', ''))
        dataset_dict['action_type'].append(entry.get('action_type', ''))
        dataset_dict['group_index'].append(str(entry.get('group_index', '')))
        dataset_dict['target_elem_id'].append(str(entry.get('target_elem_id', '')))
        dataset_dict['bbox'].append(entry.get('gt_bbox', [0, 0, 0, 0]))
        dataset_dict['num_similar_elements'].append(entry.get('num_similar_elements', 0))
        dataset_dict['density_class'].append(entry.get('density_class', 'unknown'))

    # Create dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Create DatasetDict with "test" split
    dataset_dict_obj = DatasetDict({"test": dataset})

    debug_print(f"✅ Created dataset with {len(dataset)} entries (skipped {skipped} due to missing images)", level="success")
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
    """Save dataset locally using HuggingFace format

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


def main():
    parser = argparse.ArgumentParser(
        description="Build a corrected HuggingFace dataset from annotation corrections."
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="/volume/pt-coder/users/gji/data/gui_data/AutoGUIv2",
        help="Root dir containing datasets (*/FuncElemGnd/grounding_questions.json and */FuncElemGnd/grounding_questions_corrections.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/volume/pt-coder/users/gji/projects/highres_autogui/utils/data_utils/autoguiv2/corrected_datasets",
        help="Output directory for the corrected dataset",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repository to upload to (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (if None, uses HF_TOKEN env var or huggingface-cli login)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub after saving locally",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HuggingFace dataset private",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        default=True,
        help="Include actual image data in the dataset (default: True)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not include actual image data, only paths",
    )
    args, _ = parser.parse_known_args()

    include_images = not args.no_images

    debug_print("=" * 60, level="title")
    debug_print("🔄 Building Corrected HuggingFace Dataset", level="title")
    debug_print("=" * 60, level="title")

    debug_print("\n📁 INPUT CONFIGURATION", level="step")
    debug_print(f"   Datasets Root: {Fore.CYAN}{args.datasets_root}{Style.RESET_ALL}", level="info")

    debug_print("\n📤 OUTPUT CONFIGURATION", level="step")
    upload_status = f"{Fore.GREEN}YES{Style.RESET_ALL}" if args.upload else f"{Fore.YELLOW}NO (use --upload to enable){Style.RESET_ALL}"
    debug_print(f"   Push to Hub: {upload_status}", level="info")
    if args.upload:
        debug_print(f"   Repository: {Fore.CYAN}{args.hf_repo}{Style.RESET_ALL}", level="info")
        debug_print(f"   Private: {Fore.YELLOW}{args.private}{Style.RESET_ALL}", level="info")
    debug_print(f"   Local Cache: {Fore.CYAN}{args.output_dir}{Style.RESET_ALL}", level="info")

    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    debug_print(f"   Include Images: {Fore.YELLOW}{include_images}{Style.RESET_ALL}", level="info")

    debug_print("\n" + "=" * 60, level="title")

    # Load all corrections
    debug_print(f"\n📂 Loading corrections from: {args.datasets_root}", level="step")
    all_corrections = load_all_corrections(args.datasets_root)
    debug_print(f"✅ Total corrections loaded: {len(all_corrections)}", level="success")

    # Pre-compute NID-based density classes for corrected elements
    debug_print("\n📐 Computing NID density classes for corrected elements …", level="step")
    density_lookup = compute_density_lookup(all_corrections, args.datasets_root)
    debug_print(f"✅ Density lookup built: {len(density_lookup)} entries", level="success")

    # Build entries from all corrections
    debug_print("\n🔧 Building entries from corrections...", level="step")
    corrected_entries, stats = build_entries_from_corrections(
        all_corrections,
        used_keys=set(),  # No used keys - include all corrections
        datasets_root=args.datasets_root,
        image_cache_dir=None,  # Will use paths from grounding_questions.json
        density_lookup=density_lookup,
    )

    # Print statistics
    debug_print("\n" + "-" * 40, level="info")
    debug_print("Dataset Statistics:", level="step")
    debug_print("-" * 40, level="info")
    debug_print(f"  Total corrections:    {stats['extra_total']}", level="info")
    debug_print(f"  Abandoned (removed):  {stats['extra_abandoned']}", level="info")
    debug_print(f"  Added:                {stats['extra_added']}", level="info")
    debug_print("-" * 40, level="info")
    debug_print(f"  Final entries:        {len(corrected_entries)}", level="success")
    debug_print("-" * 40, level="info")

    if not corrected_entries:
        debug_print("❌ No valid entries found", level="error")
        return 1

    # Create HuggingFace dataset
    dataset = create_hf_dataset(corrected_entries, include_images=include_images)

    # Save locally
    debug_print(f"\n💾 Saving dataset to cache: {args.output_dir}", level="info")
    save_local(dataset, args.output_dir)

    # Upload to HuggingFace if requested
    if args.upload:
        if not args.hf_repo:
            debug_print("\n❌ Error: --hf-repo is required for uploading", level="error")
            return 1
        push_to_hub(dataset, args.hf_repo, args.hf_token, args.private)
    else:
        debug_print("\n💡 Tip: Use --upload flag to push dataset to Hugging Face Hub", level="info")

    # Print summary
    debug_print("\n" + "=" * 60, level="title")
    debug_print("🎉 Conversion Complete!", level="success")
    debug_print(f"📊 Total entries: {len(dataset['test'])}", level="info")
    debug_print("=" * 60, level="title")

    return 0


if __name__ == "__main__":
    exit(main())
