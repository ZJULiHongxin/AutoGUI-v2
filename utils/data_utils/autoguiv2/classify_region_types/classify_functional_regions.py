import json
from typing import List, Dict, Any, Optional

import sys
sys.path.append('/'.join(__file__.split('/')[:-3]))
from utils.data_utils.task_prompt_lib import ANNO_PROMPT_V2_EN
from utils.data_utils.misc import resize_pil_image
from utils.data_utils.autoguiv2.annotate_functional_regions import load_image

MAX_SIZE = 1600

def extract_taxonomy_types_from_prompt(prompt_text: str) -> List[str]:
    """Extract region type labels from the taxonomy embedded in ANNO_PROMPT_V2_EN.

    Strategy: locate the anchor sentence, then parse the following top-level JSON
    object with nested categories, then flatten its child keys.
    """
    anchor = "Classify the types of the functional regions according to the following dictionary:"
    idx = prompt_text.find(anchor)
    if idx < 0:
        return []
    start = prompt_text.find('{', idx)
    if start < 0:
        return []
    depth = 0
    end = -1
    for j in range(start, len(prompt_text)):
        ch = prompt_text[j]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = j
                break
    if end < 0:
        return []
    block = prompt_text[start:end+1]
    try:
        data = json.loads(block)
        labels: List[str] = []
        if isinstance(data, dict):
            for _, group in data.items():
                if isinstance(group, dict):
                    labels.extend(list(group.keys()))
        # Deduplicate while preserving order
        seen = set()
        ordered: List[str] = []
        for l in labels:
            if l not in seen:
                seen.add(l)
                ordered.append(l)
        return ordered
    except Exception:
        return []


def build_region_type_classification_prompt(types: List[str]) -> str:
    taxonomy_lines = "\n".join([f"- {t}" for t in types])
    return (
        "You are an expert UI/UX analyst. Your task is to classify a CROPPED screenshot of a GUI functional region into a region TYPE from the taxonomy used in the functional-region annotation task.\n\n"
        "Follow these rules:\n"
        "- Consider ONLY the provided crop; do not assume outside context.\n"
        "- Choose the SINGLE best-fitting type from the taxonomy list below.\n"
        "- If none fits well, use 'Other', and provide a concise subtype (You should come up with standard type names you think are most suitable for the region).\n"
        "- Be consistent with the types used by the annotator (ANNO_PROMPT_V2_EN).\n\n"
        "TAXONOMY (choose exactly one):\n"
        f"{taxonomy_lines}\n\n"
        "Output STRICTLY one JSON object on a single line with keys: type, subtype, confidence.\n"
        "- type: exactly one of the taxonomy labels above (case-sensitive).\n"
        "- subtype: short phrase refining the type (e.g., 'left sidebar', 'modal - confirmation').\n"
        "- confidence: number in [0.0, 1.0].\n\n"
        "Example: {\"type\": \"Sidebar / Side Navigation\", \"subtype\": \"left navigation\", \"confidence\": 0.86}"
    )


def build_region_type_messages(image_base64_url: str, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if types is None:
        extracted = extract_taxonomy_types_from_prompt(ANNO_PROMPT_V2_EN)
        # Ensure 'Other' exists as a safe fallback option
        types = extracted if extracted else [
            "Entire GUI", "Application Window", "Browser Window / Tab", "Split-Screen Pane",
            "Header / Top Bar", "Footer", "Sidebar / Side Navigation", "Tab Bar", "Toolbar / Action Bar", "Breadcrumbs",
            "Main Content Area", "Card / Item List", "Dashboard / Widget Area", "Data Table / Grid", "Image Gallery / Carousel", "Map View", "Media Player",
            "Search Region", "Form", "Filter / Sort Controls", "Login / Authentication Form", "Comment Section", "Pagination Controls",
            "Modal / Dialog Box", "Popover / Tooltip", "Dropdown Menu", "Context Menu", "Notification / Toast / Alert Banner", "Cookie Consent Banner",
            "Other (You should come up with standard type names you think are most suitable for the region)"
        ]
        if "Other" not in types:
            types.append("Other (You should come up with standard type names you think are most suitable for the region)")

    prompt = build_region_type_classification_prompt(types)
    return [{
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': image_base64_url}},
            {'type': 'text', 'text': prompt}
        ]
    }]

import os, json, time, argparse, traceback, re, glob
import multiprocessing
from multiprocessing import Pool, Manager
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# Optional color output (safe fallback)
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

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

import sys
sys.path.append('/'.join(__file__.split('/')[:-3]))
from utils.openai_utils.openai import OpenAIModel


# ---------------------------
# Pretty printing helpers
# ---------------------------
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


# ---------------------------
# Image helpers
# ---------------------------
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

    if isinstance(image_or_path, str):
        ext = os.path.splitext(image_or_path)[1].lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    if isinstance(image_or_path, np.ndarray):
        success, buf = cv2.imencode('.png', image_or_path)
        if not success:
            raise ValueError("Failed to encode numpy image to PNG")
        binary_data = buf.tobytes()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        fmt = image_or_path.format if image_or_path.format else 'PNG'
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else 'image/png'
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str), numpy array, or PIL Image")


# ---------------------------
# Region type persistence helpers
# ---------------------------
def save_region_type_result(node_info: Dict[str, Any], anno_dir: str, payload: Dict[str, Any]) -> None:
    node_image_path = node_info.get('node_image_path') if isinstance(node_info, dict) else None
    if not isinstance(node_image_path, str) or not node_image_path:
        return

    node_image_full = node_image_path if os.path.isabs(node_image_path) else os.path.join(anno_dir, node_image_path)
    node_dir = os.path.dirname(node_image_full)
    if not node_dir:
        return

    base_name, _ = os.path.splitext(os.path.basename(node_image_full))
    out_path = os.path.join(node_dir, f"{base_name}_region-type.json")

    try:
        os.makedirs(node_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"\n[RegionTypes] Failed to write region type file for {base_name}: {exc}")


# ---------------------------
# Correction helpers
# ---------------------------
def load_correction_bboxes(node_dir: str) -> Dict[str, List[int]]:
    """Load corrected bounding boxes for all nodes under a directory.

    Supports both *_meta_fix*.json and *_fix*.json naming schemes, returning a
    mapping from node_id to the latest corrected bbox (by mtime).
    """

    mapping: Dict[str, List[int]] = {}
    if not node_dir or not os.path.isdir(node_dir):
        return mapping

    patterns = ("*_meta_fix*.json", "*_fix*.json")
    candidates: set[str] = set()

    for pattern in patterns:
        candidates.update(glob.glob(os.path.join(node_dir, pattern)))

    if not candidates:
        for pattern in patterns:
            candidates.update(glob.glob(os.path.join(node_dir, "**", pattern), recursive=True))

    if not candidates:
        return mapping

    latest_mtimes: Dict[str, float] = {}

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        bbox = payload.get("new_bbox") or payload.get("bbox_global")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue

        try:
            parsed_bbox = [int(round(float(v))) for v in bbox]
        except Exception:
            continue

        basename = os.path.basename(path)
        if "_meta_fix" in basename:
            node_id = basename.split("_meta_fix")[0]
        elif "_fix" in basename:
            node_id = basename.split("_fix")[0]
        else:
            continue

        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0

        if node_id not in latest_mtimes or mtime >= latest_mtimes[node_id]:
            latest_mtimes[node_id] = mtime
            mapping[node_id] = parsed_bbox

    return mapping


# ---------------------------
# Taxonomy and prompt
# ---------------------------
DEFAULT_TYPES: List[str] = [
    "Navigation bar",
    "Sidebar",
    "Toolbar",
    "Header",
    "Footer",
    "Main content area",
    "Panel/Pane",
    "Dialog/Modal",
    "Popup/Tooltip",
    "Menu",
    "Dropdown",
    "List",
    "Table",
    "Form",
    "Input field group",
    "Search bar",
    "Chat area",
    "Canvas/Map",
    "Code editor",
    "Settings/Preferences",
    "Advertisement",
    "Pagination",
    "Status bar",
    "Notification/Toast",
    "Breadcrumb",
    "Tabs",
    "Card",
    "Grid/Gallery",
    "Media player",
    "Chart/Graph",
    "Calendar",
    "File explorer",
    "Terminal/Console",
    "Login/Signup",
    "Other"
]

TAXONOMY = """{
    "Primary Interface Containers": {
        "Application Window": "Main container for an application, including window controls and all internal UI.",
        "Browser Window / Tab": "Container for a single webpage within a browser, including the address bar and tab UI.",
        "Split-Screen Pane": "A divided section of a window for displaying multiple views or documents simultaneously."
    },
    "Global Navigation & Structure": {
        "Header / Top Bar": "Top-most region with the logo, main navigation, search, and account access.",
        "Footer": "Bottom-most region with secondary links, copyright, and contact info.",
        "Sidebar / Side Navigation": "Vertical panel for navigation, content hierarchy (e.g., a file tree), or filters.",
        "Tab Bar": "A set of tabs to switch between different views, sections, or documents.",
        "Toolbar / Action Bar": "A set of controls or icon buttons for performing common actions.",
        "Breadcrumbs": "Navigation trail showing the user's current location in the UI hierarchy.",
        "Status Bar": "A horizontal bar that displays system status and notifications."
    },
    "Content & Data Display": {
        "Main Content Area": "Primary region for displaying main content like an article, video, or document.",
        "Card / Item List": "A list or grid of repeating items (cards), such as products or social media posts.",
        "Dashboard / Widget Area": "A summary view of data, metrics, and visualizations presented as widgets.",
        "Data Table / Grid": "Displays data in a sortable and filterable table with rows and columns.",
        "Image Gallery / Carousel": "An interactive viewer for a collection of images or promotional banners.",
        "Map View": "An interactive map for displaying geographical data.",
        "Media Player": "Region for playing video or audio with playback controls."
    },
    "Interaction & Input": {
        "Search Region": "An input field and button for performing a search.",
        "Form": "A set of fields for user data submission (e.g., registration, contact).",
        "Filter / Sort Controls": "Controls for filtering, refining, and sorting content.",
        "Login / Authentication Form": "A specific form for user login with username and password fields.",
        "Comment Section": "Region for users to read and write comments.",
        "Pagination Controls": "Controls (e.g., page numbers, 'next'/'previous') to navigate paged content."
    },
    "Contextual & Temporary Regions": {
        "Modal / Dialog Box": "A pop-up overlay that requires user interaction to be dismissed.",
        "Popover / Tooltip": "A small overlay that shows extra information on hover or click.",
        "Dropdown Menu": "A list of options that appears when an element is clicked.",
        "Context Menu": "A menu of relevant actions that appears on right-click.",
        "Notification / Toast / Alert Banner": "A temporary message that provides feedback or status updates.",
        "Cookie Consent Banner": "A banner that informs users about cookies and asks for their consent."
    },
    "Decorative and Non-Functional": {
        "Background": "Aesthetic backgrounds, patterns, or textures with no function.",
        "Divider or Spacer": "Lines or empty space used only for visual separation.",
        "Element Shadow": "Drop-shadow effects around windows, modals, or other elements."
    },
    "Purely Static Content": {
        "Body Text": "Paragraphs or blocks of text that are not clickable.",
        "Static Title or Heading": "A non-interactive headline or title.",
        "Image Caption": "Non-clickable descriptive text accompanying an image."
    },
    "Individual Element": {
        "Button": "A clickable button or control that performs an action when clicked.",
        "Link": "A clickable hyperlink that navigates to another webpage when clicked.",
        "Image": "A static image.",
        "Color Block": "A solid color block."
    },
    "Meaningless": {
        "Without Meaningful Content": "A crop that does not contain any interactive elements or meaningful content."
    },
    "System and Browser Artifacts": {
        "Scrollbar": "The scrollbar from the operating system or browser window.",
        "Mouse Cursor": "The user's mouse pointer captured in the image."
    }{extra_leaf_types}
}"""

EXTRA_LEAF_TYPES = """,
    "Fragmented or Incomplete": {
        "Partial Element": "An incomplete crop of a larger element, like the corner of a button.",
        "Isolated Icon": "An icon cropped without its parent component, like a toolbar or button.",
        "Text Fragment": "An incomplete word or phrase from a larger interactive text element."
    }"""

def build_classification_prompt(types: List[str], is_leaf: bool = False) -> str:
    # taxonomy = "\n".join([f"- {t}" for t in types])
    
    taxonomy_str = TAXONOMY.replace("{extra_leaf_types}", EXTRA_LEAF_TYPES if is_leaf else "")
    return (
        "You are a precise GUI region classifier.\n"
        "Given ONE cropped screenshot of a GUI region, classify it into the closest region type from the taxonomy.\n\n"
        "TAXONOMY (choose the single best type):\n"
        f"{taxonomy_str}\n\n"
        "If none fits well, use 'Other' and provide a concise subtype.\n\n"

        "Example 1: Given a prominent rectangular window located in the center of the desktop, with a distinct dark theme and a menu bar at the top.\n"
        "Type: Application Window\n"
        "Example 2: Given a horizontal bar that spans the entire width of the screen at the very top, featuring a black background and white text.\n"
        "Type: Header / Top Bar\n"
        "Example 3: Given a vertical, semi-transparent bar located on the left side of the screen, containing a series of application icons.\n"
        "Type: Sidebar / Side Navigation\n"
        "Example 4: Given a list or grid of repeating items (cards), such as products or social media posts.\n"
        "Type: Card / Item List\n"
        "Example 5: A small, rectangular pop-up window centered on the main application interface, with a title bar labeled 'Move/Copy Sheet' at the top.\n"
        "Type: Modal / Dialog Box\n"
        "Example 6: Given a set of controls or icon buttons for performing common actions.\n"
        "Type: Toolbar / Action Bar\n"
        "Example 7: Given a set of controls or icon buttons for performing common actions.\n"
        "Type: Toolbar / Action Bar\n"
        "Example 8: Given an complete VS Code application window with a pop-up modal dialog shown in the middle.\n"
        "Type: Application Window (Note that you should focus on the whole GUI content instead of a single dialog box. The GUI displays a main container for the VS Code application, including window controls and a dialog box inside, so the type is Application Window, not Modal / Dialog Box)\n\n"

        "Now it's your turn. The region screenshot has been provided.\n"
        "Observe the whole GUI screenshot, and then output STRICTLY the region type for this screenshot on a single line.\n"
    )


def parse_type_response(raw: str) -> Dict[str, Any]:
    """Extract a JSON object with fields type, subtype, confidence from model output."""
    # Prefer fenced JSON
    fenced = re.findall(r"```json\s*([\s\S]*?)```", raw, re.IGNORECASE)
    candidates: List[str] = []
    if fenced:
        candidates.extend(fenced)
    # Fallback: first JSON object
    if not candidates:
        obj_match = re.search(r"\{[\s\S]*?\}", raw)
        if obj_match:
            candidates.append(obj_match.group(0))

    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                type_str = str(data.get('type', '')).strip()
                subtype = str(data.get('subtype', '')).strip()
                conf_raw = data.get('confidence', 0.0)
                try:
                    confidence = float(conf_raw)
                except Exception:
                    # Try to extract number from strings
                    m = re.search(r"\d+\.?\d*", str(conf_raw))
                    confidence = float(m.group(0)) if m else 0.0
                # Clamp
                confidence = max(0.0, min(1.0, confidence))
                if type_str:
                    return {
                        'type': type_str,
                        'subtype': subtype,
                        'confidence': confidence,
                        'raw_response': raw
                    }
        except Exception:
            continue

    # Very fallback: heuristic extraction
    line = raw.strip().splitlines()[0] if raw.strip().splitlines() else raw[:200]
    return {
        'type': 'Other',
        'subtype': line[:100],
        'confidence': 0.0,
        'raw_response': raw
    }


# ---------------------------
# Classifier
# ---------------------------
class RegionTypeClassifier:
    def __init__(self, base_url: str, api_key: str, model: str, types: List[str]):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=1024,
        )
        self.types = types

    def classify(self, region_image: Image.Image, retries: int = 3, debug: bool = False, is_leaf: bool = False) -> Dict[str, Any]:
        prompt = build_classification_prompt(self.types, is_leaf=is_leaf)
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': image_to_base64(region_image)}},
            ]
        }]

        parsed_type = ""
        sleep_time, time_out = 1, 120
        for i in range(retries):
            sleep_time = int(1.5 * sleep_time)
            time_out *= 2
            try:
                temperature = 0.0 if i == 0 else 0.4
                success, raw_resp, _ = self.model.get_model_response_with_prepared_messages(messages, temperature=temperature, timeout=time_out)
                if not success:
                    continue

                if '</think>' in raw_resp:
                    resp = raw_resp.split('</think>')[-1]
                else:
                    resp = raw_resp

                parsed_type = resp.replace("Type:", "").strip()
                if parsed_type:
                    break
            except Exception as e:
                if debug:
                    debug_print(f"Classification attempt {i+1}/{retries} failed: {e}", level="warn")

                # Sleep for `sleep_time` second
                time.sleep(sleep_time)
                continue

        return parsed_type, prompt, raw_resp


# ---------------------------
# Multiprocessing glue
# ---------------------------
classifier_instance: RegionTypeClassifier = None  # type: ignore
debug_flag = False


def init_worker(base_url: str, api_key: str, model: str, types: List[str], debug: bool):
    global classifier_instance, debug_flag
    classifier_instance = RegionTypeClassifier(base_url, api_key, model, types)
    debug_flag = debug


def process_sample(args) -> Dict[str, Any]:
    anno_dir, bmk_name, anno_model_name, version, sample_key, sample_obj, worker_id = args
    t0 = time.time()
    try:
        results: Dict[str, Any] = {}
        regions: Dict[str, Any] = sample_obj.get('result', {})
        total = len(regions)
        
        if total <= 1:
            return {
                'image_path': sample_obj.get('image_path'),
                'region_types': {},
                'num_regions': 0,  # Number of nodes actually classified with model
                'total_regions_seen': total,
                'nodes_scanned': 0,
                'nodes_with_fixed_bbox': 0,
                'processing_time': time.time() - t0,
            }

        # Build tree structure
        parent = {}
        children_lists = {}
        for nid, ninfo in regions.items():
            ch = ninfo.get('children', [])
            children_lists[nid] = ch
            for c in ch:
                parent[c] = nid
        root_img_path = os.path.join(anno_dir, bmk_name, 'images', regions['0-0']['root_image_path'].split('images/')[-1])
        root_img = load_image(root_img_path)
        
        bmk_dir, image_key = root_img_path.split('/images/')
        dataset_root_dir, bmk_name = bmk_dir.rsplit('/', 1)
        
        correction_cache: Dict[str, Dict[str, List[int]]] = {}
        bbox_cache: Dict[str, List[int]] = {}

        def get_corrected_bbox(node_id: str, node_info: Dict[str, Any]) -> Optional[List[int]]:
            search_dirs: List[str] = []

            node_image_path = node_info.get('node_image_path')
            if isinstance(node_image_path, str) and node_image_path:
                node_image_full = node_image_path if os.path.isabs(node_image_path) else os.path.join(anno_dir, node_image_path)
                search_dirs.append(os.path.dirname(node_image_full))


            for dir_path in search_dirs:
                if not dir_path:
                    continue
                norm_dir = os.path.normpath(dir_path)
                if norm_dir not in correction_cache:
                    correction_cache[norm_dir] = load_correction_bboxes(norm_dir)
                corrected = correction_cache[norm_dir].get(node_id)
                if corrected:
                    return corrected
            return None

        def get_bbox(node_id: str) -> Optional[List[int]]:
            if node_id in bbox_cache:
                return bbox_cache[node_id]
            node_info = regions.get(node_id)
            if not node_info:
                return None

            bbox_candidate = get_corrected_bbox(node_id, node_info)
            if not (isinstance(bbox_candidate, (list, tuple)) and len(bbox_candidate) == 4):
                return None

            try:
                parsed_bbox = [int(round(float(v))) for v in bbox_candidate]
            except Exception:
                return None

            bbox_cache[node_id] = parsed_bbox
            return parsed_bbox

        # Identify nodes to classify with model (those with siblings)
        to_classify, not_to_classify = [], []
        total_nodes_scanned = 0
        nodes_with_fixed_bbox = 0
        for nid in regions:
            if nid == '0-0':
                continue
            total_nodes_scanned += 1
            p = parent.get(nid)
            if p is None:
                not_to_classify.append(nid)
                continue

            bbox = get_bbox(nid)
            if bbox is None:
                not_to_classify.append(nid)
                continue
            nodes_with_fixed_bbox += 1

            x1, y1, x2, y2 = bbox
            node_w, node_h = x2 - x1, y2 - y1

            if len(children_lists[p]) > 1 and node_w >= 10 and node_h >= 10:
                to_classify.append(nid)
            else:
                not_to_classify.append(nid)

        # Classify selected nodes
        done = 0

        for node_id in to_classify:
            node_info = regions[node_id]

            bbox = get_bbox(node_id)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            node_img = root_img.crop((x1, y1, x2, y2))
            if MAX_SIZE > 0 and max(node_img.size) > MAX_SIZE:
                node_img, _ = resize_pil_image(node_img, MAX_SIZE)

            is_leaf = len(children_lists[node_id]) == 0
            out, prompt, raw_resp = classifier_instance.classify(node_img, retries=10, debug=debug_flag, is_leaf=is_leaf)
            results[node_id] = {
                'type': out,
                'prompt': prompt,
                'raw_resp': raw_resp,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_region_type_result(node_info, anno_dir, results[node_id])
            done += 1

        sample_label = sample_obj.get('image_path') or sample_key
        summary_msg = (
            f"[RegionTypes] {bmk_name} | {os.path.basename(sample_label)} | "
            f"fixed nodes classified: {done}/{nodes_with_fixed_bbox} | scanned: {total_nodes_scanned} | unclassified: {len(not_to_classify)}"
        )
        debug_print(summary_msg, level="info") if debug_flag else print(summary_msg)

        # Assign inherited types for the rest
        def get_inherited_type(nid):
            if nid in results:
                return results[nid]
            if nid == '0-0':
                return "Entire GUI"
            p = parent.get(nid)
            if p is None:
                return "Other"
            return get_inherited_type(p)
        for nid in regions:
            if nid == '0-0': continue
            if nid not in results:
                results[nid] = "Unclassified"
        elapsed = time.time() - t0
        return {
            'image_path': sample_obj.get('image_path'),
            'region_types': results,
            'num_regions': done,  # Number of nodes actually classified with model
            'total_regions_seen': total,
            'nodes_scanned': total_nodes_scanned,
            'nodes_with_fixed_bbox': nodes_with_fixed_bbox,
            'processing_time': elapsed,
        }
    except Exception as e:
        return {
            'image_path': sample_obj.get('image_path'),
            'error': str(e),
            'processing_time': time.time() - t0,
        }


# ---------------------------
# I/O helpers
# ---------------------------
def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(results: Dict[str, Any], output_file: str, metadata: Dict[str, Any] = None):
    if metadata is None:
        metadata = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    out = {"metadata": metadata, "results": dict(results)}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def load_types_from_file(types_file: str) -> List[str]:
    items: List[str] = []
    with open(types_file, 'r', encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            if t:
                items.append(t)
    # Ensure Other exists
    if 'Other' not in items:
        items.append('Other')
    return items


# ---------------------------
# Main
# ---------------------------
def main(args):
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("🧭 Functional Region Type Classification - Run Configuration", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    debug_print("", level="info")
    debug_print("📁 DATA & OUTPUT CONFIGURATION", level="step")
    debug_print(f"   Annotation Directory: {Fore.CYAN}{args.anno_dir}{Style.RESET_ALL}", level="info")
    debug_print(f"   Input File: {Fore.CYAN}{args.input_file}{Style.RESET_ALL}", level="info")
    debug_print(f"   Output File: {Fore.CYAN}{args.output_file}{Style.RESET_ALL}", level="info")

    debug_print("", level="info")
    debug_print("🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")

    debug_print("", level="info")
    debug_print("⚙️  PROCESSING CONFIGURATION", level="step")
    mode_text = "SEQUENTIAL" if args.sequential else f"PARALLEL ({args.workers} workers)"
    mode_color = Fore.RED if args.sequential else Fore.GREEN
    debug_print(f"   Execution Mode: {mode_color}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Task Timeout: {Fore.YELLOW}{args.task_timeout}s{Style.RESET_ALL}", level="info")
    debug_print(f"   Debug: {on_off(args.debug)}", level="info")

    # Types
    if args.types_file and os.path.exists(args.types_file):
        types = load_types_from_file(args.types_file)
    else:
        types = [k for v in eval(TAXONOMY.replace("{extra_leaf_types}", EXTRA_LEAF_TYPES)).values() for k in v.keys()]
    debug_print(f"   Types: {len(types)} entries", level="info")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Load input annotations
    input_payload = load_json(args.input_file)
    input_file_parts = args.input_file.split('/')
    root = '/'.join(input_file_parts[:-4])
    bmk_name, anno_model_name, version = input_file_parts[-4:-1]
    annotated_results: Dict[str, Any] = input_payload.get('results', {}) if isinstance(input_payload, dict) else {}
    if not annotated_results:
        print("No annotated regions found in input file.")
        return

    # Load previous classifications if any
    existing_cls_payload = load_json(args.output_file)
    existing_results: Dict[str, Any] = existing_cls_payload.get('results', {}) if isinstance(existing_cls_payload, dict) else {}

    # Prepare task list
    tasks: List[Tuple[str, Dict[str, Any], int]] = []
    for i, (sample_key, sample_obj) in enumerate(sorted(list(annotated_results.items()), key=lambda x: x[0])):
        if '0FOB4CLBT2' in sample_key:
            1+1
        if not args.force:
            remaining_nodes_to_classify = {}
            for node_id, node_info in sample_obj['result'].items():
                if node_id == '0-0': continue
                if node_info.get('type', '') != '':
                    continue
                node_region_anno_file = node_info['node_image_path'].replace('.png', '_region-type.json')
                if root not in node_region_anno_file:
                    node_region_anno_file = os.path.join(root, node_region_anno_file)
                if not os.path.exists(node_region_anno_file):
                    remaining_nodes_to_classify[node_id] = node_info

            if len(remaining_nodes_to_classify) == 0:
                continue

            sample_obj['result'] = remaining_nodes_to_classify
        tasks.append((args.anno_dir, bmk_name, anno_model_name, version, sample_key, sample_obj, i))

    debug_print(f"\nFound {len(annotated_results)} images to process.\nSkip {len(annotated_results) - len(tasks)} samples that are already classified.", level="info")

    if not tasks:
        print("Nothing to classify. All samples are up-to-date.")
        return

    # Prepare manager-backed results combining existing
    if args.sequential:
        results: Dict[str, Any] = dict(existing_results)
        processed_count = len([k for k, v in results.items() if 'region_types' in v])
        total_processing_time = 0.0
    else:
        manager = Manager()
        results = manager.dict({k: v for k, v in existing_results.items()})  # type: ignore
        processed_count = manager.Value('i', len([k for k, v in existing_results.items() if 'region_types' in v]))
        total_processing_time = manager.Value('d', 0.0)

    start_time = time.time()

    # Initialize parallel pool or run sequentially
    if args.sequential:
        # Single-process mode
        global classifier_instance, debug_flag
        classifier_instance = RegionTypeClassifier(args.base_url, args.api_key, args.model, types)
        debug_flag = args.debug
        with tqdm(total=len(tasks), desc=f"Classifying {len(tasks)} samples | Model: {args.model} | #Workers: {args.workers}", dynamic_ncols=True) as pbar:
            for (anno_dir, bmk_name_task, anno_model_name_task, version_task, sample_key, sample_obj, worker_id) in tasks:
                try:
                    sample_out = process_sample((anno_dir, bmk_name_task, anno_model_name_task, version_task, sample_key, sample_obj, worker_id))
                    
                    if len(sample_out['region_types']):
                        results[sample_key] = sample_out

                    processed_count += 1
                    total_processing_time += sample_out.get('processing_time', 0.0)
                    pbar.update(1)

                    # Periodic checkpoint
                    if (processed_count % 1) == 0:
                        meta = {
                            "model": args.model,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_samples_processed": processed_count,
                            "total_samples": len(annotated_results),
                            "total_processing_time": total_processing_time,
                            "processing_time_so_far": time.time() - start_time,
                        }
                        save_checkpoint(results, args.output_file, meta)
                        pbar.set_postfix_str(f"checkpoint @ {processed_count}")
                except Exception as e:
                    print(f"Error processing sample {sample_key}: {e}")
                    pbar.update(1)
    else:
        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model, types, args.debug)
        ) as pool:
            try:
                # Submit tasks
                async_results = [(pool.apply_async(process_sample, args=((anno_dir, bmk_name_task, anno_model_name_task, version_task, k, s, i),)), (anno_dir, bmk_name_task, anno_model_name_task, version_task, k, s, i)) for (anno_dir, bmk_name_task, anno_model_name_task, version_task, k, s, i) in tasks]

                # Collect with timeout per task and progress bar
                with tqdm(total=len(async_results), desc=f"Classifying {len(tasks)} samples (parallel)", dynamic_ncols=True) as pbar:
                    for async_res, (anno_dir, bmk_name_task, anno_model_name_task, version_task, k, s, i) in async_results:
                        try:
                            sample_out = async_res.get(timeout=args.task_timeout)
                            results[k] = sample_out  # type: ignore
                            processed_count.value += 1  # type: ignore
                            total_processing_time.value += sample_out.get('processing_time', 0.0)  # type: ignore
                            pbar.update(1)

                            if (processed_count.value % 1) == 0:  # type: ignore
                                meta = {
                                    "model": args.model,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "num_samples_processed": processed_count.value,  # type: ignore
                                    "total_samples": len(annotated_results),
                                    "total_processing_time": total_processing_time.value,  # type: ignore
                                    "processing_time_so_far": time.time() - start_time,
                                }
                                save_checkpoint(results, args.output_file, meta)  # type: ignore
                                pbar.set_postfix_str(f"checkpoint @ {processed_count.value}")  # type: ignore
                        except Exception as e:
                            traceback.print_exc()
                            print(f"Task failed for sample {os.path.basename(s.get('image_path', 'unknown'))}: {e}")
                            pbar.update(1)
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Terminating workers...")
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                print(f"Parallel processing error: {e}")
                pool.terminate()
                pool.join()

    # Final save
    final_count = processed_count if isinstance(processed_count, int) else processed_count.value  # type: ignore
    final_time = total_processing_time if isinstance(total_processing_time, float) else total_processing_time.value  # type: ignore
    meta = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples_processed": final_count,
        "total_samples": len(annotated_results),
        "total_processing_time": final_time,
        "total_processing_time_wall": time.time() - start_time,
    }
    save_checkpoint(results, args.output_file, meta)
    print("\nClassification complete.")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify region types for annotated GUI regions")
    parser.add_argument("--anno-dir", type=str, default="/mnt/vdb1/hongxin_li/AutoGUIv2", help="Path to the root directory of annotation results")
    parser.add_argument("--input-file", type=str, default=[
        "/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/gemini-2.5-pro-thinking/v2/functional_regions_gemini-2.5-pro-thinking.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/gemini-2.5-pro-thinking/v2/HongxinLi-ScreenSpot-Pro.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/gemini-2.5-pro-thinking/v2/functional_regions_gemini-2.5-pro-thinking.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/gemini-2.5-pro-thinking/v2/sujr-autogui-agentnet.json",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/gemini-2.5-pro-thinking/v2/functional_regions_gemini-2.5-pro-thinking.json"
        ][-1], help="Path to functional_regions_*.json from annotation stage")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON for region types; defaults next to input file")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"), help="API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"), help="API base URL")
    parser.add_argument("--model", type=str, default=["gpt-4o-mini", "gemini-2.5-pro-thinking"][-1], help="Model to use for classification")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (debug mode)")
    parser.add_argument("--debug", action="store_true", help="Verbose debug output")
    parser.add_argument("--force", action="store_true", help="Recompute even if already present in output file")
    parser.add_argument("--task-timeout", type=int, default=18000, help="Timeout per task in seconds (parallel mode)")
    parser.add_argument("--types-file", type=str, default=None, help="Optional file with taxonomy, one label per line")

    args, _ = parser.parse_known_args()
    if args.output_file is None:
        base_dir = os.path.dirname(os.path.abspath(args.input_file))
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = os.path.join(base_dir, f"{base_name}_region_types_{args.model}.json")

    multiprocessing.set_start_method('spawn', force=True)
    main(args)


