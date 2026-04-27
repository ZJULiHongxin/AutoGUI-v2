"""
Detect visually similar elements with different functionality

This script identifies visually similar elements with different functionality
in GUI screenshots. Its JSON output serves as input to the question generation script.
"""

import os
import glob
import json
import cv2
import re
import base64
import time
import argparse
import traceback
import multiprocessing
import numpy as np
import random
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, Manager
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO

from utils.data_utils.misc import resize_image, remove_trailing_commas
from colorama import Fore, Style, init as colorama_init

# Set random seed for reproducibility
random.seed(999)

try:
    def print_list(l: List):
        pprint(l, indent=4)
        colorama_init(autoreset=True)
except Exception:
    class _Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = ""
    class _Style:
        RESET_ALL = ""
    Fore = _Fore()
    Style = _Style()

# Import utilities
import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.data_utils.misc import resize_image
from utils.openai_utils.openai import OpenAIModel



# Track parallel mode
parallel_mode = False


def debug_print(message: str, level: str = "info") -> None:
    """Colorized debug print using colorama.
    
    Levels: info (cyan), step (blue), success (green), warn (yellow), error (red), title (magenta)
    """
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


def on_off(value: bool) -> str:
    """Return colorized ON/OFF string"""
    return f"{Fore.GREEN}ON{Style.RESET_ALL}" if value else f"{Fore.YELLOW}OFF{Style.RESET_ALL}"


def image_to_base64(image_or_path):
    """Convert an image to a base64 data URL.
    
    Accepts:
      - str: filesystem path to the image
      - np.ndarray: OpenCV image (BGR or grayscale)
      - PIL.Image.Image: PIL image instance
    """
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.svg': 'image/svg+xml',
    }

    # Case 1: path string
    if isinstance(image_or_path, str):
        ext = Path(image_or_path).suffix.lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    # Case 2: OpenCV image (numpy array)
    if isinstance(image_or_path, np.ndarray):
        success, buf = cv2.imencode('.png', image_or_path)
        if not success:
            raise ValueError("Failed to encode numpy image to PNG")
        binary_data = buf.tobytes()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    # Case 3: PIL Image
    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        fmt = image_or_path.format if image_or_path.format else 'PNG'
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else 'image/png'
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str), numpy array, or PIL Image")


def save_checkpoint(results: Dict, output_file: str, metadata: Dict = None):
    """Save results to checkpoint file"""
    if metadata is None:
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    output = {
        "metadata": metadata,
        "results": dict(results),
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_checkpoint(output_file: str) -> Tuple[Dict, Dict]:
    """Load results from checkpoint file"""
    if not os.path.exists(output_file):
        return {}, {}

    try:
        with open(output_file, 'r') as f:
            checkpoint = json.load(f)

        results = checkpoint.get("results", {})
        new_results = {}

        for k, v in results.items():
            if 'error' not in v:
                new_results[k] = v

        print(f"Loaded existing results from {output_file} with {len(results)} processed images, skipped {len(results) - len(new_results)} errors")
        return new_results, checkpoint.get('metadata', {})
    except Exception as e:
        print(f"Error loading results from {output_file}: {e}")
        return {}, {}


def draw_element_boxes(image: np.ndarray, elements: List[Dict], output_path: str = None):
    """Draw bounding boxes for elements on the image"""
    image_copy = image.copy()
    
    # Color palette
    colors = [
        (255, 100, 100),   # Red
        (100, 255, 100),   # Green
        (100, 100, 255),   # Blue
        (255, 255, 100),   # Yellow
        (255, 100, 255),   # Magenta
        (100, 255, 255),   # Cyan
        (255, 150, 100),   # Orange
        (150, 100, 255),   # Purple
    ]
    
    for i, elem in enumerate(elements):
        bbox = elem['bbox_global']
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Draw label
        label = f"{i}: {elem.get('functionality', {}).get('with_context', '')[:25]}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Position label
        label_y = bbox[1] - 5 if bbox[1] - label_height - 5 > 0 else bbox[1] + label_height + 5
        
        cv2.rectangle(image_copy, (bbox[0], label_y - label_height - 2),
                     (bbox[0] + label_width, label_y + 2), color, -1)
        cv2.putText(image_copy, label, (bbox[0], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, image_copy)
    
    return image_copy


# Simple heuristic to auto-detect candidate UI elements when no annotations are available
def auto_detect_elements(image: np.ndarray) -> List[Dict]:
    """Detect candidate UI elements using contour-based heuristics.

    Returns a list of dicts with keys: node_id, bbox_global, functionality, description, type
    """
    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = image.shape[:2]
    min_size = 20
    min_area = 400
    max_area = int(0.25 * W * H)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w < min_size or h < min_size:
            continue
        if area < min_area or area > max_area:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 12.0:
            continue
        candidates.append((area, [x, y, x + w, y + h]))

    # Limit to top-N by area to reduce noise
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = candidates[:60]

    elements = []
    for i, (_, bbox) in enumerate(top):
        elements.append({
            'node_id': f'auto-{i}',
            'bbox_global': bbox,
            'functionality': {},
            'description': {},
            'type': ''
        })

    return elements


# Prompt for identifying visually similar elements
SIMILAR_ELEMENTS_PROMPT = """You are a GUI understanding expert. Your task is to identify visually similar elements that have different functionality in this screenshot.

**Instructions:**
1. You will be provided with initially found similar elements groups.
1. Examine the screenshot carefully to identify the authentic groups of elements that **look similar visually** (same icon, same color scheme, same shape, etc.) but serve **different purposes** or have **different functionality** in their own contexts.
2. These elements should be challenging for an AI agent to distinguish because they appear similar but behave differently when interacted with. For example, two visually similar magnifier icons that may represent distinct functionalities like searching and zooming. Moreover, two search bars, one inside the commentation region and the other at the top in an E-commerce website, may represent distinct functionalities like searching for comments and commodity, respectively. Besides, multiple identical 'like', 'favorite', or 'save' icons are often associated with different items in booking, rental, E-commerce, music, and video apps. For another example, four "X" icons in different regions may be used to close or dismiss different modal dialogs, respectively.
3. For each group of similar elements you find, provide:
   - The index of this group in the initially found similar elements groups
   - A description of what makes them visually similar
   - The revised bbox of each element in the group
   - The different descriptions, functionalities, and interaction outcomes of these elements in their own contexts
4. **Determine Functionality:** For each identified element, deduce its primary functionality in detail.
    4.1. Provide a high-level description of the element's function. Avoid detailing every specific functionality. Instead, focus on its broader impact on the webpage experience. For example, if interacting with a "Products" button reveals a dropdown menu, do not catalog the subsequent webpage changes in exhaustive detail.
    4.2. To ensure uniqueness, your functionality description should reflect the instance-specific context of the element whenever possible. For example, instead of predicting 'This element is used to search,' you should predict 'This element allows users to search for electronic products on Amazon,' where 'electronic products on Amazon' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m on the clock picker.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values.
    4.3 BAD cases: a) "This element closes the sidebar or a specific section within the sidebar when clicked." is BAD as the detailed description of the sidebar is not presented to demonstrate uniqueness. b) "This element closes the entire application or document window." is BAD as the functionality description is not deterministic enough. c) "This element displays a specific date, likely part of a calendar feature." is BAD as the specific date (e.g., '3, September, 2024') is not mentioned in the functionality description.
5. **Determine Interaction Outcomes:** For each element, deduce its interaction outcomes in detail. You should employ your knowledge about the GUI and the context to deduce the interaction outcomes in great details without any uncertain hallucinations. For example, if the element is a search bar, you should deduce the outcome of typing as "Typing product keywords (e.g., "wireless headphones") into this element and pressing Enter or clicking an associated search icon will navigate the user to a search results page. This page will display a list of electronic products matching the entered keywords." You should also deduce the interaction outcomes of clicking, hovering, dragging, selecting, unselecting, etc, if these actions are feasible.
6. **Generate Normalized Bounding Boxes:** For each element, provide the precise bounding box coordinates in a normalized format `[x_min, y_min, x_max, y_max]`. The coordinates must be between 0 and 1000, where `(x_min, y_min)` is the top-left corner and `(x_max, y_max)` is the bottom-right corner of the element. All boxes should tightly bound the element.

Here are the initially found similar elements groups (use these as hints only; the alt-texts may be incorrect so you should revise them before using them):
{initial_similar_elements_groups}

Return a JSON array where each object represents a group with the following structure:

```json
[
  {{
    "group index": 1 (the index of this group in the initially found groups),
    "visual_similarity": "Brief description of what makes these elements look similar (e.g., 'Same icon type', 'Same color buttons', 'Similar text fields')",
    "elements": [
      {{
        "id": (the index of this element in the group),
        "revised bbox": [x_min, y_min, x_max, y_max] (The correct bbox of this element after the initial bbox is revised),
        "detailed desctiption": "Detailed description of this element, including its context, position, appearance, and content",
        "unique functionality": "This element does XXX in its own context" (should be unique acocording to the element's context and the above description field)
        "interaction outcomes": {{
               "clicking": (What will happen to the GUI when the user clicks this element),
               "typing": (What will happen to the GUI when the user types into this element),
               "hovering": (What will happen to the GUI when the user hovers over this element),
               "dragging": ... (if not applicable, just leave it out),
               "selecting": ...,
               "unselecting": ...
               ...
            }} (clearly delineating the possible interactions (click, type, hover, etc.) and precisely describing the resulting changes.)
      }},
      {{
        "id": ...,
        "revised bbox": [x_min, y_min, x_max, y_max],
        "detailed desctiption": "Detailed description of this element",
        "unique functionality": "This element ... " (this functionality should be different from the others in the group),
        "interaction outcomes": ...
      }}
    ],
  }},
  {{
    "group index": 2,
    "elements": [] (If this group is not qualified as visually similar elements with different functionality, leave it as an empty array)
  }},
  ...
  {{
    "group index": N (N is the total number of initial groups),
    "elements": ...
  }}
]
```

Now analyze the screenshot and identify visually similar elements with different functionality:"""


# **Important Guidelines!!**
# - Elements should be genuinely visually similar (not just in the same category). You should focus more on iconic elements because they are more likely to be visually similar.
# - Elements with different texts can never be seen as similar. For example, a 'Cancel' button and 'Save' button can never be seen as similar.
# - Focus on cases where confusion would lead to incorrect actions
# - Prioritize elements that require understanding the contexts
# - If no visually similar elements with different functionality exist, return an empty array: []

# 
# **Element List:**
# Below is the list of elements in this screenshot with their descriptions and positions (may be inaccurate):
# {element_list}

## Question generation moved to generate_func_elemgnd_questions.py


class FunctionalElementSimilarityDetector:
    """Detector for visually similar elements (different functionality)"""
    
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o", 
                 temperature: float = 0.1, max_tokens: int = 8192, max_retries: int = 3, repeats: int = 3):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.max_retries = max_retries
        self.repeats = repeats
        self.crop_params = [
            [0, 0, 0.5, 0.5],
            [0.5, 0, 1, 0.5],
            [0, 0.5, 0.5, 1],
            [0.5, 0.5, 1, 1],
            [0.25, 0.25, 0.75, 0.75],
        ]
    
    # Class-level counters
    similarity_query_count = 0
    question_query_count = 0
    

    def _compute_element_descriptor(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Compute a compact visual descriptor for an element crop (RGB image).

        Uses an HSV color histogram (8x8x8 bins) normalized with L2 norm.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        # Convert to HSV for color robustness
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        if hist is None:
            return None
        hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
        return hist.flatten().astype(np.float32)

    def _derive_base_and_stem(self, image_path: str) -> Tuple[Path, str]:
        base_dir, stem = image_path.rsplit('images/', 1)
        return Path(base_dir), stem.replace('.png', '').replace('.jpg', '')

    def _load_initial_similar_groups(self, image_path: str) -> Dict[int, np.ndarray]:
        """Load precomputed DINO-v3 embeddings saved by embed_all_elems.py.

        Returns a mapping from element index -> L2-normalized embedding vector.
        """
        try:
            base_dir, stem = self._derive_base_and_stem(image_path)
            npz_path = base_dir / 'omniparser_embeddings' / f'{stem}.npz'
            if not npz_path.exists():
                return {}
            data = np.load(str(npz_path), allow_pickle=True)
            sim_groups = data.get('similar_groups')

            # If the omniparser result file is not found, search recursively in the base directory
            omniparser_result_file = base_dir / 'omniparser' / f'{stem}.json'
            if not os.path.exists(omniparser_result_file):
                search = glob.glob(os.path.join(base_dir, 'omniparser', f'**/{stem}.json'), recursive=True)
                if len(search) > 0:
                    omniparser_result_file = search[0]

            with open(omniparser_result_file, 'r') as f:
                omniparser_result = json.load(f) # example: "{'type': 'text', 'bbox': [0.008333333767950535, 0.003703703638166189, 0.04270833358168602, 0.024074073880910873], 'interactivity': False, 'content': 'Activities', 'source': 'box_ocr_content_ocr'}"
            return sim_groups.tolist(), omniparser_result
        except Exception:
            return [], []

    def _compute_top_visual_pairs(self, image_rgb: np.ndarray, elements: List[Dict], top_k: int = 5, embeddings_map: Dict[int, np.ndarray] = None) -> List[Tuple[int, int, float]]:
        """Compute pairwise visual similarity across elements and return top-K pairs.

        Returns a list of tuples: (idx_i, idx_j, similarity) with similarity in [0,1].
        """
        if (image_rgb is None or image_rgb.size == 0) and not embeddings_map:
            return []

        # If embeddings are provided, prefer them
        descriptors: Dict[int, np.ndarray] = {}
        if embeddings_map and len(embeddings_map) >= 2:
            descriptors = embeddings_map
        else:
            if image_rgb is None or image_rgb.size == 0 or not elements:
                return []
            H, W = image_rgb.shape[:2]
            for idx, elem in enumerate(elements):
                bbox = elem.get('bbox') or elem.get('bbox_global')
                if not bbox or len(bbox) != 4:
                    continue
                x1 = max(0, min(W - 1, int(bbox[0] * W)))
                y1 = max(0, min(H - 1, int(bbox[1] * H)))
                x2 = max(0, min(W, int(np.ceil(bbox[2] * W))))
                y2 = max(0, min(H, int(np.ceil(bbox[3] * H))))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 6 or (y2 - y1) < 6:
                    continue
                crop = image_rgb[y1:y2, x1:x2]
                desc = self._compute_element_descriptor(crop)
                if desc is not None:
                    # Normalize HSV descriptor to unit length for cosine
                    norm = np.linalg.norm(desc) + 1e-12
                    descriptors[idx] = (desc / norm).astype(np.float32)

        if len(descriptors) < 2:
            return []

        # Compute pairwise cosine similarity (descriptors are L2-normalized)
        pairs: List[Tuple[int, int, float]] = []
        indices = sorted(descriptors.keys())
        for i_idx, i in enumerate(indices):
            di = descriptors[i]
            for j in indices[i_idx + 1:]:
                dj = descriptors[j]
                sim = float(np.dot(di, dj))  # cosine since L2-normalized
                pairs.append((i, j, sim))

        # Sort by similarity descending and take top_k
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:top_k]

    def identify_similar_elements(self, image_path: str, 
                                   debug: bool = False) -> List[Dict]:
        """Identify visually similar elements with different functionality"""

        # Load and prepare image
        # Maximum image size for processing

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        MAX_SIZE = 1280 if W < H else 2560

        if max(image.shape) > MAX_SIZE:
            image, _ = resize_image(image, MAX_SIZE)

        image_base64 = image_to_base64(image)


        # Load offline DINO embeddings if available, and compute top pairs
        sim_groups, parsing = self._load_initial_similar_groups(image_path)

        all_similar_groups = {}
        
        if len(sim_groups):
            pairs_lines = []
            for i, sim_group in enumerate(sim_groups, start=1):
                elem_strs = []
                for j, elem_idx in enumerate(sim_group):

                    elem_strs.append(f"Elem {j+1} - bbox: {[round(p*1000) for p in parsing[elem_idx]['bbox']]}, alt-text: {parsing[elem_idx]['content']}")
                pairs_lines.append(
                    f"{i}. {'; '.join(elem_strs)}"
                )
            pairs_text = ("\n".join(pairs_lines) + "\nRemember that not every group above is qualified as visually similar elements with different functionality. Moreover, one group may contain a small number of redundant elements that cannot fit in a similar group. In this case, just discard the redundant elements.") if pairs_lines else "None"

            prompt = SIMILAR_ELEMENTS_PROMPT.format(initial_similar_elements_groups=pairs_text)

            # Prepare messages
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': image_base64}},
                    {'type': 'text', 'text': prompt}
                ]
            }]

            

            # Repeat several times
            for repeat in range(self.repeats):
                # TODO: Crop the image five times to enable the annotating model to spot visually similar elements easily.
                # crop_param = self.crop_params[repeat % len(self.crop_params)]
                # image_crop = image[crop_param[1]:crop_param[3], crop_param[0]:crop_param[2]]
                # image_base64 = image_to_base64(image)

                # Query LLM with retries
                for attempt in range(self.max_retries):
                    try:
                        if debug:
                            debug_print(f"🔍 Identifying similar elements (Model: {self.model.model}, repeat {repeat + 1}/{self.repeats} - attempt {attempt + 1}/{self.max_retries})", level="step")

                        start_time = time.time()
                        FunctionalElementSimilarityDetector.similarity_query_count += 1

                        success, response, _ = self.model.get_model_response_with_prepared_messages(
                            messages, temperature=0.3 if attempt == 0 else 0.6, timeout=240, max_new_tokens=8192
                        )

                        query_time = time.time() - start_time

                        if debug:
                            debug_print(f"⏱️  Query time: {query_time:.3f}s", level="info")

                        if not success:
                            if debug:
                                debug_print(f"❌ API call failed (attempt {attempt + 1}), retrying...", level="warn")
                            continue

                        # Parse response
                        # NOTE: The annotating model (e.g., Gemini) may expand the initially found groups so the number of elements may vary.
                        # NOTE: The group indices in the response are 1-indexed.
                        # Example
                        """
                        [
                        {
                            "group index": 1,
                            "visual_similarity": "These elements are identical tags, each composed of a blue circular icon with a white symbol and the Chinese text '圆桌精选' (Roundtable Selection). They share the same design, color, and font.",
                            "elements": [
                            {
                                "id": 1,
                                "revised bbox": [787, 712, 968, 735],
                                "detailed desctiption": "A 'Roundtable Selection' tag located to the right of the username '啓月昇' in the '正在热议' (Currently in discussion) section. It serves as a badge for the user's comment, indicating it has been featured.",
                                "unique functionality": "This element functions as a label to identify the comment by user '啓月昇' as a featured 'Roundtable Selection' within the '走进峥嵘岁月' discussion.",
                                "interaction outcomes": {
                                "clicking": "Clicking this tag might filter the discussion feed to show only comments that are 'Roundtable Selections', or it could open a pop-up or navigate to a new page explaining the criteria for being a 'Roundtable Selection'. It is also possible that it is a non-interactive static label.",
                                "hovering": "Hovering over the tag might display a tooltip with a brief explanation, such as 'Featured comment selected by the roundtable organizers'."
                                }
                            },
                            {
                                "id": 2,
                                "revised bbox": [786, 902, 967, 925],
                                "detailed desctiption": "A 'Roundtable Selection' tag located to the right of the username '阿飞' in the '正在热议' (Currently in discussion) section. It serves as a badge for the user's comment, indicating it has been featured.",
                                "unique functionality": "This element functions as a label to identify the comment by user '阿飞' as a featured 'Roundtable Selection' within the '走进峥嵘岁月' discussion.",
                                "interaction outcomes": {
                                "clicking": "Clicking this tag might filter the discussion feed to show only comments that are 'Roundtable Selections', or it could open a pop-up or navigate to a new page explaining the criteria for being a 'Roundtable Selection'. It is also possible that it is a non-interactive static label.",
                                "hovering": "Hovering over the tag might display a tooltip with a brief explanation, such as 'Featured comment selected by the roundtable organizers'."
                                }
                            }
                            ]
                        },
                        {
                            "group index": 2,
                            "elements": []
                        }
                        ]
                        """
                        similar_groups = self._parse_similarity_response(response)

                        if similar_groups is not None:
                            if debug:
                                debug_print(f"✅ Found {len(similar_groups)} groups of similar elements", level="success")

                            # Add new groups
                            for group in similar_groups:
                                if group['group index'] in all_similar_groups:
                                    continue
                                all_similar_groups[group['group index']] = group
                            break
                        else:
                            if debug:
                                debug_print(f"⚠️  Failed to parse response (attempt {attempt + 1}), retrying...", level="warn")
                            continue

                    except Exception as e:
                        if debug:
                            debug_print(f"❌ Error in similarity detection: {str(e)}", level="error")
                        continue

        return all_similar_groups
    
    def _parse_similarity_response(self, response: str) -> List[Dict]:
        """Parse LLM response for similar element groups"""
        try:
            if '```json' not in response:
                return None
            # Extract JSON from response
            json_str = response[response.rfind('```json')+7:response.rfind('```')]
            similar_groups = json.loads(remove_trailing_commas(json_str.strip()))
            
            # Validate structure
            if not isinstance(similar_groups, list):
                return None
            
            # Filter and validate groups
            valid_groups = []
            for group in similar_groups:
                if not isinstance(group, dict):
                    continue
                
                required_keys = ['visual_similarity', 'elements']
                if not all(key in group for key in required_keys):
                    continue
                
                elements = group.get('elements', [])
                if len(elements) < 2:
                    continue
                
                # Validate element structure
                valid_elements = []
                for elem in elements:
                    if isinstance(elem, dict) and 'id' in elem:
                        valid_elements.append(elem)

                if len(valid_elements) >= 2:
                    group['elements'] = valid_elements
                    valid_groups.append(group)
            
            return valid_groups
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _parse_questions_response(self, response: str) -> List[Dict]:
        """Parse LLM response for grounding questions"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                return None
            
            json_str = json_match.group(0)
            questions = json.loads(json_str)
            
            # Validate structure
            if not isinstance(questions, list):
                return None
            
            # Filter and validate questions
            valid_questions = []
            for q in questions:
                if not isinstance(q, dict):
                    continue
                
                required_keys = ['question', 'target_element_index']
                if not all(key in q for key in required_keys):
                    continue
                
                valid_questions.append(q)
            
            return valid_questions if valid_questions else None
            
        except json.JSONDecodeError:
            return None
        except Exception:
            return None
    
    def process_image(self, image_path: str, annotation_data: Dict, 
                     questions_per_group: int = 0, debug: bool = False) -> Dict:
        """Process a single image to generate grounding evaluation samples"""

        # Reset counters
        FunctionalElementSimilarityDetector.similarity_query_count = 0
        FunctionalElementSimilarityDetector.question_query_count = 0

        if debug:
            debug_print("=" * 60, level="title")
            debug_print(f"Processing: {os.path.basename(image_path)}", level="title")
            debug_print("=" * 60, level="title")


        # Identify similar elements
        similar_groups = self.identify_similar_elements(image_path, debug)
        
        if debug:
            debug_print(f"🔍 Found {len(similar_groups)} groups of similar elements", level="success")
        

        return {
            'image_path': image_path,
            'similar_groups': similar_groups,
            'similarity_queries': FunctionalElementSimilarityDetector.similarity_query_count,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def init_worker(base_url: str, api_key: str, model: str, max_retries: int, 
               debug: bool):
    """Initialize worker with generator instance"""
    global detector_instance, debug_flag, parallel_mode
    
    detector_instance = FunctionalElementSimilarityDetector(
        base_url, api_key, model, max_retries=max_retries
    )
    
    debug_flag = debug
    
    parallel_mode = True


def process_image_worker(args) -> Dict:
    """Process a single image with error handling"""
    image_key, worker_id = args
    start_time = time.time()
    
    try:
        global detector_instance, debug_flag, parallel_mode
        
        # Only worker 0 should emit debug output in parallel mode
        effective_debug = bool(debug_flag) and (not parallel_mode or worker_id == 0)
        
        # No preloaded annotations; process image directly
        annotation_data = {}
        image_path = image_key
        
        print(f"[Worker {worker_id}] Processing: {os.path.basename(image_path)}")
        
        result = detector_instance.process_image(
            image_path, annotation_data, debug=effective_debug
        )
        
        result['processing_time'] = time.time() - start_time
        
        print(f"[Worker {worker_id}] Completed {os.path.basename(image_path)} "
              f"with {len(result.get('similar_groups', []))} groups in {result['processing_time']:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"[Worker {worker_id}] Error processing {image_key}: {e}")
        traceback.print_exc()
        return {
            'image_path': image_key,
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def main(args):
    """Main processing function"""
    
    # Print configuration
    debug_print("═" * 60, level="title")
    debug_print("🎯 Functional Element Grounding - Configuration", level="title")
    debug_print("═" * 60, level="title")
    
    debug_print("\n📁 INPUT & OUTPUT", level="step")
    debug_print(f"   Images Root: {Fore.CYAN}{args.image_src_dir}{Style.RESET_ALL}", level="info")
    debug_print(f"   Output File: {Fore.CYAN}{args.output_file or 'Auto'}{Style.RESET_ALL}", level="info")
    debug_print(f"   Random Samples: {Fore.CYAN}{args.random_samples}{Style.RESET_ALL}", level="info")
    
    debug_print("\n🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    
    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    mode_text = "SEQUENTIAL" if args.workers == 1 else f"PARALLEL ({args.workers} workers)"
    mode_color = Fore.RED if args.workers == 1 else Fore.GREEN
    debug_print(f"   Mode: {mode_color}{mode_text}{Style.RESET_ALL}", level="info")
    # Question generation moved to a separate script
    debug_print(f"   Repeats per image: {Fore.YELLOW}{args.repeats}{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")
    
    debug_print("\n🔍 DEBUG CONFIGURATION", level="step")
    debug_print(f"   Debug Mode: {on_off(args.debug)}", level="info")
    debug_print(f"   Force Reprocess: {on_off(args.force)}", level="info")
    
    debug_print("\n" + "═" * 60, level="title")
    
    # Resolve images
    if not os.path.exists(args.image_src_dir) or not os.path.isdir(args.image_src_dir):
        debug_print(f"❌ Images root directory not found: {args.image_src_dir}", level="error")
        return
    
    images_root = Path(args.image_src_dir)
    image_keys = sorted([str(p) for p in images_root.glob('**/*.png')])
    if args.random_samples > 0:
        image_keys = random.sample(image_keys, args.random_samples)
    debug_print(f"📦 Found {len(image_keys)} images", level="success")
    
    # Determine default output path if not provided
    output_file = args.output_file
    if output_file is None or len(str(output_file).strip()) == 0:
        # Try to derive '/.../FuncElemGnd/similar_elements_anno.json' next to the dataset root
        # If the images root ends with 'images', use its parent; otherwise, use provided root
        parts = list(images_root.resolve().parts)
        base_dir = images_root.resolve()
        if 'images' in parts:
            idx = parts.index('images')
            base_dir = Path(*parts[:idx])
        output_dir_default = base_dir / 'FuncElemGnd'
        os.makedirs(str(output_dir_default), exist_ok=True)
        output_file = str(output_dir_default / 'similar_elements_anno.json')
    
    # Load existing results
    existing_results, metadata = load_checkpoint(output_file)
    
    # Filter images to process
    if not args.force:
        processed_keys = set(existing_results.keys())
        image_keys = [k for k in image_keys if k.split('images/')[-1] not in processed_keys]
    
    debug_print(f"📋 Images to process: {len(image_keys)}", level="info")
    debug_print(f"✅ Already processed: {len(existing_results)}", level="info")
    
    if not image_keys:
        debug_print("✨ No new images to process", level="success")
        return
    
    # Setup output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Initialize results
    if args.workers == 1:
        results = existing_results.copy()
        processed_count = len(existing_results)
        total_processing_time = 0.0
    else:
        manager = Manager()
        results = manager.dict()
        for k, v in existing_results.items():
            results[k] = v
        processed_count = manager.Value('i', len(existing_results))
        total_processing_time = manager.Value('d', 0.0)

    start_time = time.time()

    if args.workers == 1:
        # Sequential processing
        debug_print("\n🔧 Running in sequential mode", level="step")

        global detector_instance, debug_flag
        detector_instance = FunctionalElementSimilarityDetector(
            args.base_url, args.api_key, args.model, max_retries=args.max_retries, repeats=args.repeats
        )
        debug_flag = args.debug

        with tqdm(total=len(image_keys), desc="Processing images") as pbar:
            for i, image_key in enumerate(image_keys):
                try:
                    result = process_image_worker((image_key, i))

                    results[image_key.split('images/')[-1]] = result
                    processed_count += 1
                    total_processing_time += result.get('processing_time', 0)

                    pbar.update(1)

                    # Save checkpoint
                    if processed_count % 1 == 0:
                        checkpoint_metadata = {
                            "model": args.model,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_images_processed": processed_count,
                            "total_images": len(image_keys) + len(existing_results),
                            "total_groups": sum(len(v.get('similar_groups', [])) for v in results.values() if isinstance(v, dict)),
                            "avg_processing_time": total_processing_time / processed_count if processed_count > 0 else 0,
                        }
                        save_checkpoint(results, output_file, checkpoint_metadata)
                        print(f"\n💾 Saved checkpoint to {output_file} with {processed_count} processed images")

                except Exception as e:
                    print(f"❌ Error processing {image_key}: {e}")
                    processed_count += 1

    else:
        # Parallel processing
        debug_print(f"\n⚡ Running in parallel mode with {args.workers} workers", level="step")

        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model, args.max_retries,
                     args.debug)
        ) as pool:
            try:
                with tqdm(total=len(image_keys), desc="Processing images") as pbar:
                    # Prepare tasks
                    tasks = [(key, i % args.workers) for i, key in enumerate(image_keys)]

                    # Process in batches
                    batch_size = args.workers * 2
                    for batch_start in range(0, len(tasks), batch_size):
                        batch = tasks[batch_start:batch_start + batch_size]

                        # Submit batch
                        async_results = [pool.apply_async(process_image_worker, (task,)) for task in batch]

                        # Collect results
                        for async_result in async_results:
                            try:
                                result = async_result.get(timeout=3600)  # 1 hour timeout per image

                                image_key = result['image_path']
                                results[image_key] = result
                                processed_count.value += 1
                                total_processing_time.value += result.get('processing_time', 0)

                                pbar.update(1)

                                # Save checkpoint
                                if processed_count.value % 1 == 0:
                                    checkpoint_metadata = {
                                        "model": args.model,
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "num_images_processed": processed_count.value,
                                        "total_images": len(image_keys) + len(existing_results),
                                        "total_groups": sum(len(v.get('similar_groups', [])) for v in results.values() if isinstance(v, dict)),
                                        "avg_processing_time": total_processing_time.value / processed_count.value if processed_count.value > 0 else 0,
                                    }
                                    save_checkpoint(results, output_file, checkpoint_metadata)
                                    print(f"\n💾 Saved checkpoint to {output_file}")

                            except Exception as e:
                                print(f"❌ Task failed: {e}")
                                traceback.print_exc()

            except KeyboardInterrupt:
                debug_print("\n⚠️  Keyboard interrupt - terminating workers...", level="warn")
                pool.terminate()
                pool.join()
                raise
    
    # Save final results
    final_count = processed_count if args.workers == 1 else processed_count.value
    final_time = total_processing_time if args.workers == 1 else total_processing_time.value
    
    total_groups = sum(len(v.get('similar_groups', [])) for v in results.values() if isinstance(v, dict))
    
    final_metadata = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_src_dir": args.image_src_dir,
        "num_images_processed": final_count,
        "total_images": len(image_keys) + len(existing_results),
        "total_groups_detected": total_groups,
        "avg_processing_time": final_time / final_count if final_count > 0 else 0,
        "total_wall_time": time.time() - start_time,
    }
    save_checkpoint(results, output_file, final_metadata)
    
    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Similarity Detection Complete!", level="success")
    debug_print(f"📊 Total groups detected: {total_groups}", level="info")
    debug_print(f"💾 Results saved to: {args.output_file}", level="info")
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect visually similar GUI elements with different functionality"
    )

    parser.add_argument("--image-src-dir", default=[
        "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/images/",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/androidcontrol/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/images/",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/magicui/images/"
        ][-2],
                       help="Root directory to glob images from (e.g., /path/to/images)")
    parser.add_argument("--output-file", default=None,
                       help="Output JSON file for detected similar element groups (default: dataset_root/FuncElemGnd/similar_elements_anno.json)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
                       help="OpenAI API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
                       help="OpenAI API base URL")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking",
                       help="Model to use for similarity detection")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--repeats", type=int, default=3,
                       help="Number of times to repeat the similarity detection")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls")
    parser.add_argument("--random-samples", type=int, default=130,
                       help="Number of random samples to process")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of already processed images")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging")

    args, _ = parser.parse_known_args()

    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)

