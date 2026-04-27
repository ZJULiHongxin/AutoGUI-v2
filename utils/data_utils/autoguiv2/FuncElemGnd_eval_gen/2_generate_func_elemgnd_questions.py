"""
Generate grounding questions for visually similar GUI elements

This script reads the output of make_func_elemgnd_samples.py (similarity detection)
and generates concise but uniquely identifiable grounding questions using a VLM.
"""

import os
import json
import re
import time
import argparse
import multiprocessing
import cv2
import base64
import numpy as np
from multiprocessing import Pool, Manager
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO

from typing import List, Dict, Any

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

import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel
from utils.data_utils.misc import resize_image

# Maximum image size for processing
MAX_SIZE = 1920

parallel_mode = False


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


def on_off(value: bool) -> str:
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


def draw_element_boxes(image: np.ndarray, elements: List[Dict], output_path: str = None):
    """Draw bounding boxes for elements on the image
    
    Args:
        image: RGB image (numpy array)
        elements: List of element dicts with 'revised bbox' key (normalized 0-1000)
        output_path: Optional path to save the image
    """
    image_copy = image.copy()
    H, W = image.shape[:2]
    
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
        # Get bbox in normalized coordinates (0-1000)
        bbox_norm = elem.get('revised bbox', [])
        if len(bbox_norm) != 4:
            continue
        
        # Convert to pixel coordinates
        x1 = int(bbox_norm[0] * W / 1000)
        y1 = int(bbox_norm[1] * H / 1000)
        x2 = int(bbox_norm[2] * W / 1000)
        y2 = int(bbox_norm[3] * H / 1000)
        
        color = colors[i % len(colors)]
        
        # Draw bounding box
        #cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)
        
        # Draw label with element ID
        elem_id = elem.get('id', i)
        label = f"[{elem_id}]"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Position label
        label_y = y1 - 5 if y1 - label_height - 5 > 0 else y1 + label_height + 5
        
        cv2.rectangle(image_copy, (x1, label_y - label_height - 2),
                     (x1 + label_width, label_y + 2), color, -1)
        cv2.putText(image_copy, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
    
    return image_copy


def save_checkpoint(results: Dict, output_file: str, metadata: Dict = None):
    if metadata is None:
        metadata = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    output = {"metadata": metadata, "results": dict(results)}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_checkpoint(output_file: str):
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


# Prompt for generating grounding questions
GROUNDING_QUESTION_PROMPT = """You are a GUI expert in designing questions about GUI World Model Knowledge. Your task is to generate natural, realistic questions that average GUI users would ask when trying to locate specific UI elements to complete their tasks and learning about interaction outcomes.

**Context:**
You are shown a screenshot with visually similar elements marked with colored bounding boxes and IDs (e.g., [0], [1], [2]). These elements look similar but have different functionality.

**Your Task:**
Generate diverse, natural questions that a real user might ask to locate a specific element from this similar group. Do this for each element in the group.

**Question Requirements:**
1. **Natural & Realistic**: Sound like how an average user would actually ask (casual, sometimes brief, but clear)
2. **Task-Oriented**: Frame questions around what the user wants to accomplish, not just "find element X"
3. **Instance-Specific**: Include at least ONE concrete detail visible in the screenshot (e.g., specific text, numbers, item names, section labels, brand names) to uniquely identify the target
4. **Functionally Distinguishing**: The question should require understanding what functionality the element provides and what changes to the GUI will occur after interacting with the element, not just how it looks
5. **Diverse Phrasing**: Vary question types and structures
6. **One-to-One Mapping**: The question should uniquely map to an element without ambiguity. For example, "I want to know what this collapsed section in the bottom panel is for. Is there a way to see its name or purpose?" is a bad case as this question ambiguously mentions "this collapsed section in the bottom panel" without essential description useful to uniquely locate an element.
7. Do NOT mention the interaction action in the question as the question should focus on high-level element functionality and interaction outcome.
8. Avoid mentioning the displayed text or alt-text of the target element in the generated questions as an average user would not mention this text directly. For example, "How can I see all the options available under the 'Text Editor' category?" is bad as it mentions 'Text Editor', which is displayed on the element. The prederred question is "Where do I find the settings for changing the font or making text bold for the VS Code text editor?"
9. **Associated Action Intent**: Rephrase the question as an action intent that directly mentions the element targeted by the question. The displayed text can be mentioned in the action intent.

**Output Format:**
Return ONLY a JSON array. Each object must include:

```json
[
  {{
    "target_element_id": 0,
    "referring_expressions": {{
            "clicking": {{
                "question": (A question that expresses how the user will fulfill his/her high-level goal by clicking the element),
                "action_intent": (An action intent that can be used to accurately locate the element specified by the quesiton)
            }},
            "hovering": {{
                "question": (How the user will fulfill a goal by hovering over the element),
                "action_intent": ...
            }},
            ... (other actions)
        }}
  }}
  ... (The other elements in the same group)
]
```

**IMPORTANT:**
- Use the exact element IDs shown in the screenshot (visible in the bounding box labels)
- The `target_element_id` should match the ID from the elements list below
- Do NOT include explanations or any text outside the JSON array
- Generate questions that require understanding functional differences instead of mere alt-text or appearances

**Element Group Information (The elements are marked on the image):**
{element_group}

**Examples for Style Reference Only (DO NOT copy these):**

Example A — Two "X" (close) icons with different scopes:
```json
[
    {{
        "target_element_id": 1
        "referring_expressions": {{
            "clicking": {{
                "question": "This 'Join our newsletter' pop-up is blocking the article. How do I get rid of it?",
                "action_intent": "Click the 'X' icon in the top-right corner of the 'Join our newsletter' modal, which is overlaying the main page content."
            }},
            "hovering": {{
                "question": "I'm looking for the 'Close' tooltip to make sure this 'X' won't do something else.",
                "action_intent": "Hover over the 'X' icon in the corner of the 'Join our newsletter' pop-up. A tooltip saying 'Close' should appear."
            }}
        }}
    }},
    {{
        "target_element_id": ...,
        "referring_expressions": {{
            "clicking": {{
                "question": "I'm done with the 'Annual Report' spreadsheet, but I need to keep Excel open. How do I close just this file?",
                "action_intent": "Click the 'X' icon in the top-right corner of the 'Annual Report' document pane, positioned below the main application ribbon."
            }},
            "hovering": {{
                "question": "Where can I hover to see a tooltip confirming this 'X' will only 'Close' the current file, not the whole app?",
                "action_intent": "Hover over the 'X' icon on the document pane, *below* the main window's 'X'. A 'Close' tooltip should appear."
            }}
        }}
    }}
]
```

Example B — Multiple search bars for different items:
```json
[
    {{
        "target_element_id": 2,
        "referring_expressions": {{
            "clicking": {{
                "question": "I need to change my password and manage app notifications. Where are the main 'Account Settings'?",
                "action_intent": "Click the 'gear' icon located in the main navigation sidebar, positioned at the bottom of the menu list."
            }},
            "hovering": {{
                "question": "Where can I hover to see the label for the main 'Settings' option in the sidebar?",
                "action_intent": "Hover over the 'gear' icon in the main navigation sidebar. A tooltip or text label 'Settings' should appear."
            }}
        }}
    }},
    {{
        "target_element_id": ...,
        "referring_expressions": {{
            "clicking": {{
                "question": "I want to change who can see *this specific post* about my 'New Project'. Where are the options for just this post?",
                "action_intent": "Click the 'gear' icon (or three dots) in the top-right corner of the 'New Project' post card, within the main feed."
            }},
            "hovering": {{
                "question": "Where can I hover to see the 'Post Options' for the 'New Project' post?",
                "action_intent": "Hover over the 'gear' icon (or three dots) on the 'New Project' post card to reveal a 'More options' tooltip."
            }}
        }}
    }}
]
```

Example C - Visually Similar 'List Items' in an email box
```json
[
    {{
        "target_element_id": ...,
        "referring_expressions": {{
            "swiping": {{
                "question": "I'm done with this 'Support' email. How do I 'Archive' it to get it out of my inbox?",
                "action_intent": "Swipe right on the 'Support' email (subject 'Your ticket') to reveal the Archive action."
            }},
            "long pressing": {{
                "question": "I need to 'Mark as Unread' this 'Support' email to read later. How do I get those options?",
                "action_intent": "Long press the 'Support' email to open a context menu and select 'Mark as Unread.'"
            }}
        }}
    }},
    {{
        "target_element_id": ...,
        "referring_expressions": {{
            "swiping": {{
                "question": "This 'Weekly Newsletter' email is junk. How do I 'Delete' it?",
                "action_intent": "Swipe left on the 'Weekly Newsletter' email to reveal the Delete action."
            }},
            "long pressing": {{
                "question": "How do I block the sender for this 'Weekly Newsletter'? I need to see the options.",
                "action_intent": "Long press the 'Weekly Newsletter' email and select 'Block Sender' from the context menu."
            }}
        }}
    }}
]
```

Now generate questions for the provided element group:"""


class QuestionGenerator:
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=8192,
        )
        self.max_retries = max_retries

    def _parse_questions_response(self, response: str):
        """Parse LLM response for grounding questions"""
        try:
            m = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if not m:
                return None
            arr = json.loads(m.group(1))
            if not isinstance(arr, list):
                return None
            valid = []
            for q in arr:
                if not isinstance(q, dict):
                    continue
                # Check for required fields (support both old and new naming)
                if 'referring_expressions' not in q and 'referring expressions' not in q:
                    continue
                if 'target_element_id' not in q and 'target_element_index' not in q:
                    continue
                
                # Normalize field names
                if 'referring expressions' in q:
                    q['referring_expressions'] = q['referring expressions']
                if 'target_element_index' in q and 'target_element_id' not in q:
                    q['target_element_id'] = q['target_element_index']

                valid.append(q)
            return valid if valid else None
        except Exception:
            return None

    def generate_for_group(self, image_path: str, group_obj: Dict, debug: bool = False):
        """Generate questions for a group of similar elements

        Args:
            image_path: Path to the screenshot image
            group_obj: Group object with 'visual_similarity' and 'elements' fields
            debug: Enable debug output
        """
        # Load and prepare image
        try:
            image = cv2.imread(image_path)
            if image is None:
                if debug:
                    debug_print(f"❌ Failed to load image: {image_path}", level="error")
                return []

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if max(image_rgb.shape) > MAX_SIZE:
                image_rgb, _ = resize_image(image_rgb, MAX_SIZE)

            # Draw bounding boxes on image
            elements = group_obj.get('elements', [])
            marked_image = draw_element_boxes(image_rgb, elements)
            image_base64 = image_to_base64(marked_image)
            
        except Exception as e:
            if debug:
                debug_print(f"❌ Error preparing image: {e}", level="error")
            return []
        
        # Prepare element group description
        group_desc_lines = [f"Visual Similarity: {group_obj.get('visual_similarity', 'N/A')}\n", "Elements in this group:"]
        
        for elem in elements:
            interactions = elem.get('interaction outcomes', {})
            if isinstance(interactions, str):
                continue # Invalid format

            elem_id = elem.get('id', '?')
            bbox = elem.get('revised bbox', [])
            func = elem.get('unique functionality', '')
            desc = elem.get('detailed desctiption', '')  # Note: typo in script 1
            
            # Format element info
            elem_info = [f"\n[Element {elem_id}]"]
            elem_info.append(f"  Bounding Box (normalized 0-1000): {bbox}")
            elem_info.append(f"  Description: {desc}")
            elem_info.append(f"  Functionality: {func}")
            
            # Add interaction outcomes if available
            if isinstance(interactions, list):
                if ':' not in interactions[0]:
                    continue

                new_interactions = {}
                for x in interactions:
                    action, outcome = x.split(':')
                    new_interactions[action] = outcome
                interactions = new_interactions

            if interactions:
                elem_info.append(f"  Interactions:")
                for action, outcome in interactions.items():
                    if outcome:
                        elem_info.append(f"    - {action.capitalize()}: {outcome}")

            group_desc_lines.append('\n'.join(elem_info))

        element_group_text = '\n'.join(group_desc_lines)

        prompt = GROUNDING_QUESTION_PROMPT.format(
            element_group=element_group_text
        )

        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]

        for attempt in range(self.max_retries):
            try:
                if debug:
                    debug_print(f"  📝 Generating questions (attempt {attempt + 1}/{self.max_retries})", level="step")

                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3 if attempt == 0 else 0.5, timeout=240, max_new_tokens=8192
                )

                if not success:
                    if debug:
                        debug_print(f"  ⚠️  API call failed", level="warn")
                    continue

                questions = self._parse_questions_response(response)
                if questions:
                    if debug:
                        debug_print(f"  ✅ Generated {len(questions)} questions", level="success")
                    return questions
                else:
                    if debug:
                        debug_print(f"  ⚠️  Failed to parse response", level="warn")

            except Exception as e:
                if debug:
                    debug_print(f"  ❌ Error: {e}", level="error")
                continue

        if debug:
            debug_print(f"  💥 All attempts failed", level="error")
        return []


def init_worker(base_url: str, api_key: str, model: str, max_retries: int, detection_file: str, 
                image_src_dir: str, debug: bool):
    global qgen_instance, detection_data, image_src_dir_global, debug_flag, parallel_mode
    qgen_instance = QuestionGenerator(base_url, api_key, model, max_retries=max_retries)
    with open(detection_file, 'r', encoding='utf-8') as f:
        detection_checkpoint = json.load(f)
    detection_data = detection_checkpoint.get('results', {})
    image_src_dir_global = image_src_dir
    debug_flag = debug
    parallel_mode = True


def process_image_worker(args):
    """Process a single image to generate questions for all its similar element groups"""
    image_key, worker_id = args
    start = time.time()
    try:
        global qgen_instance, detection_data, image_src_dir_global, debug_flag, parallel_mode
        effective_debug = bool(debug_flag) and (not parallel_mode or worker_id == 0)
        
        if image_key not in detection_data:
            return {"image_path": image_key, "error": "No detection data", "processing_time": time.time() - start}
        
        det = detection_data[image_key]
        
        # Get the full image path
        # detection_data stores relative paths from the image_src_dir
        if 'image_path' in det:
            image_path = det['image_path']
        else:
            # Fallback: construct from image_src_dir and image_key
            image_path = os.path.join(image_src_dir_global, image_key)
        
        # Check if image exists
        if not os.path.exists(image_path):
            return {"image_path": image_key, "error": f"Image not found: {image_path}", "processing_time": time.time() - start}
        
        # Get similar groups - note that in script 1, similar_groups is a dict with group_index as keys
        groups_dict = det.get('similar_groups', {})
        
        # Convert dict to list of groups
        if isinstance(groups_dict, dict):
            groups = [v for k, v in sorted(groups_dict.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)]
        else:
            groups = groups_dict if isinstance(groups_dict, list) else []
        
        if effective_debug:
            debug_print(f"\n{'='*60}", level="title")
            debug_print(f"Processing: {os.path.basename(image_path)}", level="title")
            debug_print(f"Groups to process: {len(groups)}", level="info")
        
        results = []
        for group_idx, g in enumerate(groups):
            if effective_debug:
                debug_print(f"\n📦 Group {group_idx + 1}/{len(groups)}", level="step")
            
            # Only process groups with valid elements
            elements = g.get('elements', [])
            if len(elements) < 2:
                if effective_debug:
                    debug_print(f"  ⚠️  Skipping group with < 2 elements", level="warn")
                continue
            
            questions = qgen_instance.generate_for_group(image_path, g, effective_debug)
            
            if questions:
                results.append({
                    "group_index": g.get('group index', group_idx),
                    "visual_similarity": g.get('visual_similarity', ''),
                    "elements": elements,
                    "questions": questions
                })

        out = {
            "image_path": image_path,
            "num_groups": len(groups),
            "num_groups_with_questions": len(results),
            "total_questions": sum(len(x['questions']) for x in results),
            "generated": results,
            "processing_time": time.time() - start,
        }

        print(f"[Worker {worker_id}] {os.path.basename(image_path)}: "
              f"{len(results)}/{len(groups)} groups → {out['total_questions']} questions "
              f"({out['processing_time']:.1f}s)")

        return out

    except Exception as e:
        import traceback
        print(f"[Worker {worker_id}] ❌ Error processing {image_key}: {e}")
        traceback.print_exc()
        return {"image_path": image_key, "error": str(e), "processing_time": time.time() - start}


def main(args):
    debug_print("═" * 60, level="title")
    debug_print("📝 Question Generation - Configuration", level="title")
    debug_print("═" * 60, level="title")

    debug_print("\n📁 INPUT & OUTPUT", level="step")
    debug_print(f"   Image Source Dir: {Fore.CYAN}{args.image_src_dir}{Style.RESET_ALL}", level="info")
    debug_print(f"   Detection File: {Fore.CYAN}{args.detection_file}{Style.RESET_ALL}", level="info")
    debug_print(f"   Output File: {Fore.CYAN}{args.output_file}{Style.RESET_ALL}", level="info")

    debug_print("\n🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")

    debug_print("\n⚙️  PROCESSING CONFIGURATION", level="step")
    mode_text = "SEQUENTIAL" if args.workers == 1 else f"PARALLEL ({args.workers} workers)"
    mode_color = Fore.RED if args.workers == 1 else Fore.GREEN
    debug_print(f"   Mode: {mode_color}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")

    debug_print("\n🔍 DEBUG CONFIGURATION", level="step")
    debug_print(f"   Debug Mode: {on_off(args.debug)}", level="info")
    debug_print(f"   Force Reprocess: {on_off(args.force)}", level="info")

    debug_print("\n" + "═" * 60, level="title")

    # Validate inputs
    if not os.path.exists(args.image_src_dir):
        debug_print(f"❌ Image source directory not found: {args.image_src_dir}", level="error")
        return

    if not os.path.exists(args.detection_file):
        debug_print(f"❌ Detection file not found: {args.detection_file}", level="error")
        return

    # Load detection file
    with open(args.detection_file, 'r', encoding='utf-8') as f:
        detection_checkpoint = json.load(f)
    det_results = detection_checkpoint.get('results', {})

    debug_print(f"📦 Loaded detection data for {len(det_results)} images", level="success")

    # Load existing results
    existing_results, _ = load_checkpoint(args.output_file)

    # Filter images to process
    if args.force:
        image_keys = list(det_results.keys())
    else:
        processed = set(existing_results.keys())
        image_keys = [k for k in det_results.keys() if k not in processed]

    debug_print(f"📋 Images to process: {len(image_keys)}", level="info")
    debug_print(f"✅ Already processed: {len(existing_results)}", level="info")

    if not image_keys:
        debug_print("✨ No new images to process", level="success")
        return

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

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

        global qgen_instance, detection_data, image_src_dir_global, debug_flag
        qgen_instance = QuestionGenerator(args.base_url, args.api_key, args.model, max_retries=args.max_retries)
        detection_data = det_results
        image_src_dir_global = args.image_src_dir
        debug_flag = args.debug

        for i, image_key in enumerate(image_keys):
            try:
                res = process_image_worker((image_key, i))
                results[image_key.split('images/')[-1]] = res
                processed_count += 1
                total_processing_time += res.get('processing_time', 0)
                
                # Save checkpoint periodically
                if processed_count % 1 == 0:
                    total_questions = sum(r.get('total_questions', 0) for r in results.values() if isinstance(r, dict))
                    save_checkpoint(results, args.output_file, {
                        "model": args.model,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "num_images_processed": processed_count,
                        "total_questions": total_questions,
                    })
                    print(f"💾 Checkpoint: {processed_count} images, {total_questions} questions")
            
            except Exception as e:
                print(f"❌ Error processing {image_key}: {e}")
                processed_count += 1
    
    else:
        # Parallel processing
        debug_print(f"\n⚡ Running in parallel mode with {args.workers} workers", level="step")
        
        with Pool(processes=args.workers, initializer=init_worker,
                  initargs=(args.base_url, args.api_key, args.model, args.max_retries, 
                           args.detection_file, args.image_src_dir, args.debug)) as pool:
            tasks = [(k, i % args.workers) for i, k in enumerate(image_keys)]
            
            for task in tasks:
                try:
                    res = pool.apply_async(process_image_worker, (task,)).get(timeout=3600)
                    image_key = task[0]
                    results[image_key.split('images/')[-1]] = res
                    processed_count.value += 1
                    total_processing_time.value += res.get('processing_time', 0)
                    
                    # Save checkpoint periodically
                    if processed_count.value % 1 == 0:
                        total_questions = sum(r.get('total_questions', 0) for r in results.values() if isinstance(r, dict))
                        save_checkpoint(results, args.output_file, {
                            "model": args.model,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_images_processed": processed_count.value,
                            "total_questions": total_questions,
                        })
                        print(f"💾 Checkpoint saved at {args.output_file}")
                
                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    continue

    # Calculate final statistics
    final_count = processed_count if args.workers == 1 else processed_count.value
    final_time = total_processing_time if args.workers == 1 else total_processing_time.value
    
    # Calculate total questions generated
    total_questions = sum(r.get('total_questions', 0) for r in results.values() if isinstance(r, dict))
    total_groups = sum(r.get('num_groups_with_questions', 0) for r in results.values() if isinstance(r, dict))

    # Save final results
    save_checkpoint(results, args.output_file, {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_src_dir": args.image_src_dir,
        "detection_file": args.detection_file,
        "num_images_processed": final_count,
        "total_groups_with_questions": total_groups,
        "total_questions": total_questions,
        "avg_processing_time": final_time / final_count if final_count else 0,
        "total_wall_time": time.time() - start_time,
    })

    debug_print("\n" + "═" * 60, level="title")
    debug_print("🎉 Question Generation Complete!", level="success")
    debug_print(f"📊 Total questions generated: {total_questions} (from {total_groups} groups)", level="info")
    debug_print(f"💾 Results saved to: {args.output_file}", level="info")
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grounding questions for visually similar GUI elements with different functionality"
    )

    # Input/Output arguments
    parser.add_argument("--image-src-dir", default="/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/images",
                       help="Root directory containing the images (must match the directory used in detection)")
    parser.add_argument("--detection-file", default=None,
                       help="Path to detection JSON from make_func_elemgnd_samples.py (e.g., dataset_root/FuncElemGnd/similar_elements_anno.json)")
    parser.add_argument("--output-file", default=None,
                       help="Output JSON with generated questions (default: dataset_root/FuncElemGnd/grounding_questions.json)")

    # API arguments
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
                       help="OpenAI API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
                       help="OpenAI API base URL")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking",
                       help="Model to use for question generation")

    # Processing arguments
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocessing of already processed images")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging")

    args, _ = parser.parse_known_args()

    # Auto-determine file paths if not provided
    if args.detection_file is None or len(str(args.detection_file).strip()) == 0:
        # Try to derive detection file from image_src_dir
        # Example: "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/similar_elements_anno.json"
        images_root = Path(args.image_src_dir)
        parts = list(images_root.resolve().parts)
        base_dir = images_root.resolve()
        if 'images' in parts:
            idx = parts.index('images')
            base_dir = Path(*parts[:idx])
        detection_dir = base_dir / 'FuncElemGnd'
        args.detection_file = str(detection_dir / 'similar_elements_anno.json')

    if args.output_file is None or len(str(args.output_file).strip()) == 0:
        # Place output next to detection file
        detection_path = Path(args.detection_file)
        args.output_file = str(detection_path.parent / 'grounding_questions.json')
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)
