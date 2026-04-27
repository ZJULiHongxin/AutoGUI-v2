"""
Generate captioning questions for functional element grounding evaluation.

This script reads the output of make_func_elemgnd_samples.py (similarity detection)
and generates multiple-choice questions that prompt a tested model to describe the
interaction outcome after interacting with a target element.

The questions focus on what happens after clicking, hovering, dragging, typing,
or long-pressing GUI elements, with hard negatives from similar elements and
easy negatives from dissimilar elements.
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

import random
random.seed(999)

import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel
from utils.data_utils.misc import resize_image

# Maximum image size for processing
MAX_SIZE = 2560

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

        # cv2.rectangle(image_copy, (x1, label_y - label_height - 2),
        #              (x1 + label_width, label_y + 2), color, -1)
        cv2.putText(image_copy, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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



# Prompt for generating captioning questions
CAPTIONING_QUESTION_PROMPT = """You are a GUI expert in designing multiple-choice questions for evaluating models' understanding of GUI element functionality. Your task is to generate realistic questions that test what happens when a user interacts with a *single target element*.

**Context:**
You are shown a screenshot with visually similar elements marked with colored bounding boxes and IDs (e.g., [0], [1], [2]). Exactly one element from each group is selected as the target. The other elements from the same group are provided as *hard negative candidates*, while elements from other groups/screens serve as *easy negative candidates*.

**Your Task:**
Generate multiple-choice questions that ask about the outcome of interacting with the target element using the interaction types listed. Focus on what happens after clicking, hovering, dragging, typing, or long-pressing (only include the interactions explicitly marked as supported).

**Question Requirements:**
1. **Realistic & Task-Oriented**: Sound like things an average user might genuinely ask while trying to accomplish a task
2. **Functionality-Focused**: Highlight resulting state changes, navigation, data updates, or feedback that follows the interaction
3. **Interaction-Specific**: Explicitly refer to the interaction type tied to the target element
4. **Diverse Phrasing**: Vary tone and structure across the generated questions to avoid repetition
5. **No information leakage**: Do not mention the target element in the question in any of the candidates. For example, If a candidate mentions "The files and folders will be rearranged into a detailed list view", a question mentioning "I want to change the view in the 'document' folder to see more details like date and size. What will happen if I click the button in the toolbar that depicts three horizontal lines?" is bad as the quesiton unexpectedly tells us that the functionality is about "seeing more details like date and size".
6. **Multi-Modal Understanding**: The question MUST be answered with the visual information of the GUI screenshot. Questions that can be answered with common knowledge alone, without reference to the GUI screenshot, are not allowed.

**Multiple-Choice Structure (per interaction type):**
- You should remember that the question is used to assess the outcome prediction capability of vision language models (e.g., Gemini, GPT-5, and Qwen-VL) in the realm of GUI understanding. Do NOT ask "How can I ..." in the question!!!
- Provide exactly five answer options labelled `A`, `B`, `C`, `D`, `E`, where `A` MUST be the correct answer while the other four are incorrect answers. Each candidate describes interaction outcome in detail according to the given GUI content.
- Supply a short explanation referencing why the correct option is right and why the distractors are incorrect
- Please mention the target element in the question using a detailed description instead of its ID.
- **IMPORTANT**: For each choice (B, C, D, E), you MUST specify which candidate it was drawn from by including the candidate reference ID (e.g., "hard_neg_0", "easy_neg_2") in the choice key.

**Choice Generation Strategy:**
- **Correct Choice**: Rephrase and modify the ground truth candidate, remove or paraphrase excessively obvious hints in it, and make it hard to spot as the correct answer directly.
- **Modified Hard Negatives (aim for 2 choices)**: Draw from the hard negative candidates list (elements in the *same* visual similarity group as the target) and modify them to be super-hard and confusing distractors by either modifying the description or coming up with brand-new challenging distracting content. If fewer than two are available, use as many as you can and supplement the remaining slots with easy negatives.
- **Easy Negatives (fill remaining choices)**: Draw from the easy negative candidates list (elements outside the target's group)
- Ensure every distractor sounds plausible in a GUI context

**Output Format:**
Return ONLY a JSON array:

```json
{{
    "target_element_id": 0,
    "questions": {{
      "clicking": {{
        "question": "...",
        "choices": {{
          "A (correct)": "...",
          "B (hard_neg_0)": "...",
          "C (hard_neg_1)": "...",
          "D (easy_neg_2)": "...",
          "E (easy_neg_5)": "..."
        }},
        "explanation": "..."
      }},
      "hovering": {{
        "question": "...",
        "choices": {{
          "A (correct)": "...",
          "B (hard_neg_0)": "...",
          "C (hard_neg_1)": "...",
          "D (easy_neg_2)": "...",
          "E (easy_neg_5)": "..."
        }},
        "explanation": "..."
      }},
      "long-pressing": {{
          ...
      }},
      "swiping": {{
          ...
      }},
      "typing": {{
          ...
      }},
      "dragging": {{
          ...
      }},
      "right-clicking": {{
          ...
      }},
      "double-clicking": {{
          ...
      }},
      "pressing-key": {{
          ...
      }}
      ... (only include the interaction types explicitly listed as supported for the target)
    }}
}}
```

**IMPORTANT:**
- Include interaction types *only* if the target element explicitly lists an outcome for that interaction
- Do NOT include explanations or any text outside the JSON array

**Target Element & Hard Negatives:**
{element_group}

**Easy Negative Candidates (outside this similarity group):**
{context_elements}

Now generate captioning questions for the provided target element:"""

# Prompt for generating captioning questions
CAPTIONING_QUESTION_PROMPT_V2 = """You are a GUI expert in designing multiple-choice questions for evaluating models' understanding of GUI element functionality. Your task is to generate realistic questions that test what happens when a user interacts with a *single target element*.

**Context:**
You are shown a screenshot with visually similar elements marked with colored bounding boxes and IDs (e.g., [0], [1], [2]). Exactly one element from each group is selected as the target. The other elements from the same group are provided as *hard negative candidates*.

**Your Task:**
Generate multiple-choice questions that ask about the outcome of interacting with the target element using the interaction types listed. Focus on what happens after clicking, hovering, dragging, typing, or long-pressing (only include the interactions explicitly marked as supported).

**Question Requirements:**
1. **Realistic & Task-Oriented**: Sound like things an average user might genuinely ask while trying to accomplish a task
2. **Functionality-Focused**: Highlight resulting state changes, navigation, data updates, or feedback that follows the interaction
3. **Interaction-Specific**: Explicitly refer to the interaction type tied to the target element
4. **Diverse Phrasing**: Vary tone and structure across the generated questions to avoid repetition
5. **No information leakage**: Do not mention the target element in the question in any of the candidates. For example, If a candidate mentions "The files and folders will be rearranged into a detailed list view", a question mentioning "I want to change the view in the 'document' folder to see more details like date and size. What will happen if I click the button in the toolbar that depicts three horizontal lines?" is bad as the quesiton unexpectedly tells us that the functionality is about "seeing more details like date and size".
6. **Multi-Modal Understanding**: The question MUST be answered with the visual information of the GUI screenshot. Questions that can be answered with common knowledge alone, without reference to the GUI screenshot, are not allowed.

**Multiple-Choice Structure (per interaction type):**
- Pose one natural-language question about the interaction outcome for the target element. You should remember that the question is used to assess the outcome prediction capability of top-level vision language models (e.g., Gemini, GPT-5, and Qwen-VL) in the realm of GUI understanding and interaction. Do NOT ask "How can I ..." in the question!!!
- Provide exactly five answer options labelled `A`, `B`, `C`, `D`, `E`, where `A` MUST be the correct answer while the other four are incorrect answers. Each candidate should faithfully describe the outcome of the interaction for the target element in detail according to the provided functionality metadata.
- Supply a short explanation referencing why the correct option is right and why the distractors are incorrect
- Please mention the target element in the question using a detailed description instead of its ID.
- **IMPORTANT**: For each choice (B, C, D, E), you MUST assign a reference ID. For a choice drawn from the hard candidates, assign IDs like "hard_neg_0"; for a choice generated by you, assign IDs like "gen_neg".

**Negative Generation Strategy:**
- **Hard Negatives (aim for 2 choices)**: Draw from the hard negative candidates list (elements in the *same* visual similarity group as the target). If fewer than two are available, use as many as you can and supplement the remaining slots with generated distractors.
- **Generated Distractors (fill remaining choices)**: Generate super-hard and confusing distractors by either modifying the correct choice or coming up with brand-new challenging distractors.
- Ensure every distractor sounds plausible in a GUI context

**Output Format:**
Return ONLY a JSON array:

```json
{{
    "target_element_id": 0,
    "questions": {{
      "clicking": {{
        "question": "...",
        "choices": {{
          "A (correct)": "...",
          "B (hard_neg_0)": "...",
          "C (hard_neg_1)": "...",
          "D (gen_neg)": "...",
          "E (gen_neg)": "..."
        }},
        "explanation": "..."
      }},
      "hovering": {{
        "question": "...",
        "choices": {{
          "A (correct)": "...",
          "B (hard_neg_0)": "...",
          "C (hard_neg_1)": "...",
          "D (gen_neg)": "...",
          "E (gen_neg)": "..."
        }},
        "explanation": "..."
      }},
      "long-pressing": {{
          ...
      }},
      "swiping": {{
          ...
      }},
      "typing": {{
          ...
      }},
      "dragging": {{
          ...
      }},
      "right-clicking": {{
          ...
      }},
      "double-clicking": {{
          ...
      }},
      "pressing-key": {{
          ...
      }}
      ... (only include the interaction types explicitly listed as supported for the target)
    }}
}}
```

**IMPORTANT:**
- Only generate questions for the specified target element in this group
- Include interaction types *only* if the target element explicitly lists an outcome for that interaction
- Do NOT include explanations or any text outside the JSON array

**Target Element & Hard Negatives:**
{element_group}

Now generate captioning questions for the provided target element:"""


class CaptioningQuestionGenerator:
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.3,
            max_tokens=8192,
        )
        self.max_retries = max_retries

    @staticmethod
    def _normalize_interactions(raw_interactions: Any) -> Dict[str, str]:
        if isinstance(raw_interactions, dict):
            return {k: v for k, v in raw_interactions.items() if isinstance(v, str) and v.strip()}
        if isinstance(raw_interactions, list):
            normalized = {}
            for entry in raw_interactions:
                if not isinstance(entry, str) or ':' not in entry:
                    continue
                action, outcome = entry.split(':', 1)
                outcome = outcome.strip()
                if outcome:
                    normalized[action.strip()] = outcome
            return normalized
        return {}

            
    @staticmethod
    def _count_supported_actions(interactions: Dict[str, str]) -> int:
        return sum(1 for _, outcome in interactions.items() if outcome and outcome.strip())

    def _select_target_element(self, elements: List[Dict]) -> Dict:
        if not elements:
            return {}

        scored_elements = []
        for idx, elem in enumerate(elements):
            interactions = self._normalize_interactions(elem.get('interaction outcomes', {}))
            score = self._count_supported_actions(interactions)
            # Prefer elements with more detailed functionality text as a secondary tie-breaker
            functionality = elem.get('unique functionality', '') or ''
            scored_elements.append((score, len(functionality.strip()), -idx, elem, interactions))

        scored_elements.sort(reverse=True)
        # Return element along with precomputed normalized interactions
        top = scored_elements[0]
        elem = top[3]
        elem = dict(elem)  # shallow copy to avoid mutating detection data
        elem['normalized_interactions'] = top[4]
        return elem

    def _prepare_negative_candidates(self, elements: List[Dict], target_id: Any, group_index: Any = None) -> List[Dict]:
        negatives = []
        for elem in elements:
            if elem.get('id') == target_id:
                continue
            normalized = self._normalize_interactions(elem.get('interaction outcomes', {}))
            negatives.append({
                'id': elem.get('id'),
                'group_index': group_index,
                'description': elem.get('detailed desctiption', ''),
                'functionality': elem.get('unique functionality', ''),
                'interactions': normalized,
            })
        return negatives

    def _parse_questions_response(self, response: str, hard_negatives: List[Dict], easy_negatives: List[Dict], 
                                   target_element: Dict, group_index: Any):
        """Parse LLM response for captioning questions and create candidate mappings"""
        try:
            if '</think>' in response:
                response = response.split('</think>')[-1]

            m = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if not m:
                temp = response.find('"target_element_id":')
                if temp != -1:
                    start_idx = response.rfind('{', 0, temp)
                    if start_idx != -1:
                        end_idx = response.rfind('}')
                        json_str = response[start_idx:end_idx+1]
                        q = json.loads(json_str)
                    else:
                        return None
                else:
                    return None
            else:
                q = json.loads(m.group(1))

            if isinstance(q, list):
                q = q[0]

            if not isinstance(q, dict):
                return None
            if 'questions' not in q:
                return None
            if 'target_element_id' not in q:
                return None

            # Create candidate mapping for this question set
            candidate_mapping = {
                'target': {
                    'element_id': target_element.get('id'),
                    'group_index': group_index
                },
                'hard_negatives': {},
                'easy_negatives': {}
            }
            
            # Map hard negatives
            for i, hard_neg in enumerate(hard_negatives):
                candidate_mapping['hard_negatives'][f'hard_neg_{i}'] = {
                    'element_id': hard_neg.get('id'),
                    'group_index': hard_neg.get('group_index')
                }

            # Map easy negatives
            for i, easy_neg in enumerate(easy_negatives):
                candidate_mapping['easy_negatives'][f'easy_neg_{i}'] = {
                    'element_id': easy_neg.get('id'),
                    'group_index': easy_neg.get('group_index')
                }

            # Parse choice mappings from each question
            questions_dict = q.get('questions', {})
            for interaction_type, question_data in questions_dict.items():
                if not isinstance(question_data, dict):
                    continue
                choices = question_data.get('choices', {})
                choice_mapping = {}

                for choice_key, choice_text in choices.items():
                    # Extract the candidate reference from the choice key
                    # e.g., "B (hard_neg_0)" -> "hard_neg_0"
                    match = re.search(r'\(([^)]+)\)', choice_key)
                    if match:
                        ref = match.group(1)
                        choice_letter = choice_key.split()[0]

                        if ref == 'correct':
                            choice_mapping[choice_letter] = {
                                'type': 'target',
                                'element_id': target_element.get('id'),
                                'group_index': group_index
                            }
                        elif ref.startswith('hard_neg_'):
                            if ref in candidate_mapping['hard_negatives']:
                                choice_mapping[choice_letter] = {
                                    'type': 'hard_negative',
                                    **candidate_mapping['hard_negatives'][ref]
                                }
                        elif ref.startswith('easy_neg_'):
                            if ref in candidate_mapping['easy_negatives']:
                                choice_mapping[choice_letter] = {
                                    'type': 'easy_negative',
                                    **candidate_mapping['easy_negatives'][ref]
                                }
                        elif ref.startswith('gen_neg'):
                            choice_mapping[choice_letter] = {
                                'type': 'generated_negative'
                            }
                
                # Add choice mapping to the question data
                question_data['choice_mapping'] = choice_mapping
            
            q['candidate_mapping'] = candidate_mapping
            return q
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def _get_context_elements(self, detection_data: Dict, current_image_key: str, current_group_identifier: Any) -> List[Dict]:
        """Get elements from other images/groups for easy negative choices"""
        context_elements = []

        # Helper to register candidate while trimming duplicates
        def add_candidate(elem):
            normalized_interactions = self._normalize_interactions(elem.get('interaction outcomes', {}))
            candidate = {
                'id': elem.get('id'),
                'group_index': group_key,
                'description': elem.get('detailed desctiption', ''),
                'functionality': elem.get('unique functionality', ''),
                'interactions': normalized_interactions,
            }
            if candidate not in context_elements:
                context_elements.append(candidate)

        for img_key, img_data in detection_data.items():
            groups = img_data.get('similar_groups', {})
            if isinstance(groups, dict):
                iterable = groups.items()
            else:
                iterable = enumerate(groups)

            for group_key, group in iterable:
                if img_key == current_image_key and str(group_key) == str(current_group_identifier):
                    continue

                for elem in group.get('elements', [])[:5]:
                    add_candidate(elem)

            if len(context_elements) >= 40:
                break

        return context_elements[:40]

    def generate_for_group(self, image_path: str, group_obj: Dict, detection_data: Dict, debug: bool = False,
                           image_key: str = None, group_identifier: Any = None, use_gen: bool = True):
        """Generate captioning questions for a group of similar elements"""

        elements = group_obj.get('elements', []) or []
        if len(elements) < 2:
            if debug:
                debug_print("  ⚠️  Skipping group with < 2 elements", level="warn")
            return []

        target_element = self._select_target_element(elements)
        if not target_element:
            if debug:
                debug_print("  ⚠️  Failed to select target element", level="warn")
            return []

        target_id = target_element.get('id', '?')
        target_interactions = target_element.pop('normalized_interactions', {})
        supported_actions = [action for action, outcome in target_interactions.items() if outcome and outcome.strip()]

        if not supported_actions:
            if debug:
                debug_print(f"  ⚠️  Target element [{target_id}] lacks explicit interaction outcomes; skipping", level="warn")
            return []

        hard_negatives = self._prepare_negative_candidates(elements, target_id, group_identifier)

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

            #marked_image = draw_element_boxes(image_rgb, elements)
            image_base64 = image_to_base64(image_rgb)

        except Exception as e:
            if debug:
                debug_print(f"❌ Error preparing image: {e}", level="error")
            return []

        # Get context elements for easy negatives
        current_image_key = image_key if image_key is not None else (image_path.split('images/')[-1] if 'images/' in image_path else image_path)

        if not use_gen:
            context_elements = self._get_context_elements(detection_data, current_image_key, group_identifier)
            random.shuffle(context_elements)
        else:
            context_elements = []

        group_desc_lines = [
            f"Visual Similarity: {group_obj.get('visual_similarity', 'N/A')}",
            "",
            f"Target Element [ID {target_id}]",
            f"  Bounding Box (normalized 0-1000): {target_element.get('revised bbox', [])}",
            f"  Description: {target_element.get('detailed desctiption', '')}",
            f"  Functionality: {target_element.get('unique functionality', '')}",
            "  Supported Interactions:",
        ]

        for action in supported_actions:
            outcome = target_interactions.get(action, '')
            group_desc_lines.append(f"    - {action}: {outcome}")

        if hard_negatives:
            group_desc_lines.append("\nHard Negative Candidates (other elements in this group):")
            for i, neg in enumerate(hard_negatives):
                neg_lines = [
                    f"  • hard_neg_{i} - Element [ID {neg.get('id', '?')}] from Group {neg.get('group_index', '?')}",
                    f"    Description: {neg.get('description', '')}",
                    f"    Functionality: {neg.get('functionality', '')}",
                ]
                interactions = neg.get('interactions', {})
                if interactions:
                    neg_lines.append("    Interactions:")
                    for action, outcome in interactions.items():
                        neg_lines.append(f"      - {action}: {outcome}")
                group_desc_lines.extend(neg_lines)

        context_desc_lines = ["Easy Negative Candidates (outside the target group):"]
        for i, ctx_elem in enumerate(context_elements[:5]):
            ctx_lines = [
                f"  • easy_neg_{i} - Element [ID {ctx_elem.get('id', '?')}] from Group {ctx_elem.get('group_index', '?')}",
                f"    Description: {ctx_elem.get('description', '')}",
                f"    Functionality: {ctx_elem.get('functionality', '')}",
            ]
            interactions = ctx_elem.get('interactions', {})
            if interactions:
                ctx_lines.append("    Interactions:")
                for action, outcome in interactions.items():
                    ctx_lines.append(f"      - {action}: {outcome}")
            context_desc_lines.extend(ctx_lines)

        element_group_text = '\n'.join(group_desc_lines)
        context_elements_text = '\n'.join(context_desc_lines)

        if use_gen:
            prompt = CAPTIONING_QUESTION_PROMPT_V2.format(element_group=element_group_text)
        else:
            prompt = CAPTIONING_QUESTION_PROMPT.format(
                element_group=element_group_text,
                context_elements=context_elements_text
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
                    debug_print(f"  📝 Generating captioning questions (attempt {attempt + 1}/{self.max_retries})", level="step")

                success, response, _, usage_info = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3 if attempt == 0 else 0.5, timeout=240, max_new_tokens=8192
                )

                if not success:
                    if debug:
                        debug_print("  ⚠️  API call failed", level="warn")
                    continue

                question = self._parse_questions_response(response, hard_negatives, context_elements[:6], 
                                                          target_element, group_identifier)
                if question:
                    if debug:
                        debug_print(f"  ✅ Generated a captioning question for Group {group_identifier} of Image {image_path}", level="success")
                    return {
                            'question_meta': question,
                            'image_path': image_path,
                            'prompt': prompt,
                            'response': response,
                            }
                else:
                    if debug:
                        debug_print("  ⚠️  Failed to parse response", level="warn")

            except Exception as e:
                if debug:
                    debug_print(f"  ❌ Error: {e}", level="error")
                continue

        if debug:
            debug_print(f"  💥 All attempts failed", level="error")
        return {}


def init_worker(base_url: str, api_key: str, model: str, max_retries: int, detection_file: str,
                image_src_dir: str, debug: bool):
    global qgen_instance, detection_data, image_src_dir_global, debug_flag, parallel_mode
    qgen_instance = CaptioningQuestionGenerator(base_url, api_key, model, max_retries=max_retries)
    with open(detection_file, 'r', encoding='utf-8') as f: # Example: /mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/FuncElemGnd/similar_elements_anno.json
        detection_checkpoint = json.load(f)
    detection_data = detection_checkpoint.get('results', {})
    image_src_dir_global = image_src_dir
    debug_flag = debug
    parallel_mode = True


def process_image_worker(args):
    """Process a single image to generate captioning questions for all its similar element groups"""
    image_key, worker_id, use_gen = args
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

        if isinstance(groups_dict, dict):
            group_items = sorted(groups_dict.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
        else:
            group_items = [(str(idx), grp) for idx, grp in enumerate(groups_dict if isinstance(groups_dict, list) else [])]

        if effective_debug:
            debug_print(f"\n{'='*60}", level="title")
            debug_print(f"Processing: {os.path.basename(image_path)}", level="title")
            debug_print(f"Groups to process: {len(group_items)}", level="info")

        results = []
        for group_idx, (group_key, g) in enumerate(group_items):
            #if group_idx > 2: break
            # Each group provides only one target element.
            if effective_debug:
                debug_print(f"\n📦 Group {group_idx + 1}/{len(group_items)}", level="step")

            # Only process groups with valid elements
            elements = g.get('elements', [])
            if len(elements) < 2:
                if effective_debug:
                    debug_print(f"  ⚠️  Skipping group with < 2 elements", level="warn")
                continue

            question_meta = qgen_instance.generate_for_group(
                image_path, g, detection_data, effective_debug,
                image_key=image_key, group_identifier=group_key, use_gen=use_gen
            )

            if question_meta:
                results.append({
                    "group_index": g.get('group index', group_key),
                    "elements_in_group": elements,
                    "visual_similarity": g.get('visual_similarity', ''),
                    **question_meta
                })

        out = {
            "image_path": image_path,
            "num_groups": len(group_items),
            "num_groups_with_questions": len(results),
            "generated": results,
            "processing_time": time.time() - start,
        }

        print(f"[Worker {worker_id}] {os.path.basename(image_path)}: "
              f"{len(results)}/{len(group_items)} groups → {len(results)} questions "
              f"({out['processing_time']:.1f}s)")

        return out

    except Exception as e:
        import traceback
        print(f"[Worker {worker_id}] ❌ Error processing {image_key}: {e}")
        traceback.print_exc()
        return {"image_path": image_key, "error": str(e), "processing_time": time.time() - start}


def main(args):
    debug_print("═" * 60, level="title")
    debug_print("📝 Captioning Question Generation - Configuration", level="title")
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
    debug_print(f"   Use Generated Distractors: {on_off(args.use_gen)}", level="info")

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
        qgen_instance = CaptioningQuestionGenerator(args.base_url, args.api_key, args.model, max_retries=args.max_retries)
        detection_data = det_results
        image_src_dir_global = args.image_src_dir
        debug_flag = args.debug

        for i, image_key in enumerate(image_keys):
            #if i > 2: break
            try:
                res = process_image_worker((image_key, i, args.use_gen))
                results[image_key.split('images/')[-1]] = res
                processed_count += 1
                total_processing_time += res.get('processing_time', 0)

                # Save checkpoint periodically
                if processed_count % 1 == 0:
                    total_groups = sum(r.get('num_groups_with_questions', 0) for r in results.values() if isinstance(r, dict))
                    save_checkpoint(results, args.output_file, {
                        "model": args.model,
                        "use_gen": args.use_gen,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "num_images_processed": processed_count,
                        "total_questions": total_groups,
                    })
                    print(f"💾 Checkpoint: {processed_count} images, {total_groups} questions -> {args.output_file}")

            except Exception as e:
                print(f"❌ Error processing {image_key}: {e}")
                processed_count += 1

    else:
        # Parallel processing
        debug_print(f"\n⚡ Running in parallel mode with {args.workers} workers", level="step")

        with Pool(processes=args.workers, initializer=init_worker,
                  initargs=(args.base_url, args.api_key, args.model, args.max_retries,
                           args.detection_file, args.image_src_dir, args.debug)) as pool:
            tasks = [(k, i % args.workers, args.use_gen) for i, k in enumerate(image_keys)]

            for task in tasks:
                try:
                    res = pool.apply_async(process_image_worker, (task,)).get(timeout=3600)
                    image_key = task[0]
                    results[image_key.split('images/')[-1]] = res
                    processed_count.value += 1
                    total_processing_time.value += res.get('processing_time', 0)

                    # Save checkpoint periodically
                    if processed_count.value % 1 == 0:
                        total_groups = sum(r.get('num_groups_with_questions', 0) for r in results.values() if isinstance(r, dict))
                        save_checkpoint(results, args.output_file, {
                            "model": args.model,
                            "use_gen": args.use_gen,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_images_processed": processed_count.value,
                            "total_questions": total_groups,
                        })
                        print(f"💾 Checkpoint saved at {args.output_file}")

                except Exception as e:
                    print(f"❌ Task failed: {e}")
                    continue

    # Calculate final statistics
    final_count = processed_count if args.workers == 1 else processed_count.value
    final_time = total_processing_time if args.workers == 1 else total_processing_time.value

    # Calculate total questions generated (one question per group)
    total_groups = sum(r.get('num_groups_with_questions', 0) for r in results.values() if isinstance(r, dict))
    total_questions = total_groups  # One question per group

    # Save final results
    save_checkpoint(results, args.output_file, {
        "model": args.model,
        "use_gen": args.use_gen,
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
    debug_print("🎉 Captioning Question Generation Complete!", level="success")
    debug_print(f"📊 Total questions generated: {total_questions} (from {total_groups} groups)", level="info")
    debug_print(f"💾 Results saved to: {args.output_file}", level="info")
    debug_print("═" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate captioning questions for functional element grounding evaluation"
    )

    # Input/Output arguments
    parser.add_argument("--image-src-dir", default="/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/images",
                       help="Root directory containing the images (must match the directory used in detection)")
    parser.add_argument("--detection-file", default=None,
                       help="Path to detection JSON from make_func_elemgnd_samples.py (e.g., dataset_root/FuncElemGnd/similar_elements_anno.json)")
    parser.add_argument("--output-file", default=None,
                       help="Output JSON with generated questions (default: dataset_root/FuncElemGnd/captioning_questions.json)")

    # API arguments
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
                       help="OpenAI API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
                       help="OpenAI API base URL")
    parser.add_argument("--model", type=str, default=["gemini-2.5-pro-thinking", "gemini-3-flash-preview-thinking"][-1],
                       help="Model to use for question generation")

    # Processing arguments
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum retries for API calls")
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument("--use-gen", type=str2bool, default=False,
                       help="Use generated distractors instead of easy negatives")
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
        suffix = "_gen" if args.use_gen else ""
        args.output_file = str(detection_path.parent / f'captioning_questions_{args.model}{suffix}.json')

    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)

    main(args)
