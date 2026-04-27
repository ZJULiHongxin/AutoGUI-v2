import os
import json
import glob
import time
import cv2
import random
import base64
import traceback
import argparse
import multiprocessing

from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

random.seed(999)

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

sys.path.append("/".join(__file__.split("/")[:-3]))

from utils.data_utils.misc import resize_image
from utils.openai_utils.openai import OpenAIModel


MAX_SIZE = 1920


REANNOTATION_PROMPT_TEMPLATE = """
You are an expert UI/UX analyst. Your task is to analyze a graphical user interface (GUI) screenshot and revise the functional description for a GUI region after a human corrected its bounding box.

Context:
  • Previously annotated functionality: {previous_functionality}
  • Previously annotated description: {previous_description}

Look carefully at the red rectangle on the full screenshot and the cropped image of that region. Revise and polish the functionality and description of the specified region.

Requirements:
1. The previously annotated functionality and description may be incorrect, either containing hallucinated details or missing important discernible details. Therefore, you MUST correct the functionality and description according to the bounding box marked on the screenshot and the problems encountered in the original annotation.
2. You should revise the functionality according to the format requirements shown below.
2.1. The revised functionality MUST provide a high-level description of the region/element's function. Avoid detailing every specific functionality. Instead, focus on its broader impact on the webpage experience. For example, if interacting with a "Products" region reveals a dropdown menu, do not catalog the subsequent webpage changes in exhaustive detail.
2.2. To ensure uniqueness, your functionality description should reflect the instance-specific context of the region whenever possible. For example, instead of predicting 'This region is used to search,' you should predict 'This region allows users to search for electronic products on Amazon,' where 'electronic products on Amazon' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m on the clock picker.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values.
3. You should also revise the description. A description should describe the region's layout and appearance in English and in detail.

Respond strictly in JSON with this schema:
```json
{{
  "revision rationale": "Carefully describe the whole GUI screenshot, the region marked with a red rectangle, and their relationship. Then, describe the rationale for the revision.",
  "revised description": "...",
  "revised functionality": "..."
}}
```
Do not include any extra keys or commentary outside of the JSON object.
"""


def debug_print(message: str, level: str = "info") -> None:
    level_to_color = {
        "info": Fore.CYAN,
        "step": Fore.BLUE,
        "success": Fore.GREEN,
        "warn": Fore.YELLOW,
        "error": Fore.RED,
        "title": Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def on_off(value: bool) -> str:
    return f"{Fore.GREEN}ON{Style.RESET_ALL}" if value else f"{Fore.YELLOW}OFF{Style.RESET_ALL}"


def image_to_base64(image_or_path: Any) -> str:
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".svg": "image/svg+xml",
    }

    if isinstance(image_or_path, str):
        ext = Path(image_or_path).suffix.lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    if isinstance(image_or_path, np.ndarray):
        success, buf = cv2.imencode(".png", image_or_path)
        if not success:
            raise ValueError("Failed to encode numpy image to PNG")
        binary_data = buf.tobytes()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        fmt = image_or_path.format if image_or_path.format else "PNG"
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else "image/png"
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str), numpy array, or PIL Image")


def clamp_bbox_to_image(bbox: List[int], width: int, height: int) -> Optional[List[int]]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def draw_bbox_on_image(image: np.ndarray, bbox: List[int], color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 8) -> np.ndarray:
    annotated = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    return annotated


def extract_reannotation_from_response(raw_response: str) -> Optional[Dict[str, Any]]:
    response = raw_response.split("</think>")[-1] if "</think>" in raw_response else raw_response
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return None


    return payload


def save_checkpoint(results: Dict[str, Any], output_file: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    metadata = metadata or {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "results": dict(results)}, f, indent=2, ensure_ascii=False)

@dataclass
class CorrectionTask:
    namespace: str
    model_name: str
    version: str
    image_id: str
    node_id: str
    root_image_path: str
    nodes_dir: str
    correction_file: str


def discover_corrections(cache_root: str, namespace: str, model_name: str, version: str) -> List[CorrectionTask]:
    """Discover corrections with support for wildcards ('*') in namespace, model_name, and version."""
    # If all are wildcards, delegate to discover_all_corrections
    if namespace == "*" and model_name == "*" and version == "*":
        return discover_all_corrections(cache_root)

    # Otherwise, handle wildcards by scanning relevant directories
    tasks: List[CorrectionTask] = []

    # Get namespaces to scan
    if namespace == "*":
        try:
            namespaces_to_scan = [d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))]
        except Exception:
            return []
    else:
        namespaces_to_scan = [namespace]

    for ns in namespaces_to_scan:
        ns_dir = os.path.join(cache_root, ns)

        # Get models to scan
        if model_name == "*":
            try:
                models_to_scan = [d for d in os.listdir(ns_dir) if os.path.isdir(os.path.join(ns_dir, d))]
            except Exception:
                continue
        else:
            models_to_scan = [model_name]

        for model in models_to_scan:
            model_dir = os.path.join(ns_dir, model)

            # Get versions to scan
            if version == "*":
                try:
                    versions_to_scan = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                except Exception:
                    continue
            else:
                versions_to_scan = [version]

            for ver in versions_to_scan:
                # Use the original logic for each specific namespace/model/version combination
                version_tasks = _discover_corrections_single(cache_root, ns, model, ver)
                tasks.extend(version_tasks)

    return tasks


def _discover_corrections_single(cache_root: str, namespace: str, model_name: str, version: str) -> List[CorrectionTask]:
    """Original discover_corrections logic for a single specific namespace/model/version combination."""
    version_dir = os.path.join(cache_root, namespace, model_name, version)
    if not os.path.isdir(version_dir):
        return []

    tasks: List[CorrectionTask] = []

    # Collect unique nodes directories to avoid processing the same folder multiple times
    node_image_paths = sorted(glob.glob(os.path.join(version_dir, "**", "nodes/*.png"), recursive=True))
    nodes_dirs = sorted({os.path.dirname(p) for p in node_image_paths})
    for nodes_dir in tqdm(nodes_dirs, total=len(nodes_dirs), desc=f"Discovering corrections for {namespace} {model_name} {version}"):
        # Derive image_id from the nodes directory path relative to version_dir
        rel_nodes_dir = os.path.relpath(nodes_dir, version_dir)
        image_id = os.path.dirname(rel_nodes_dir)
        root_image_path = os.path.join(os.path.dirname(nodes_dir), "root.png")
        if not os.path.isdir(nodes_dir) or not os.path.exists(root_image_path):
            continue

        TAG = "meta_fix"
        correction_files = sorted(glob.glob(os.path.join(nodes_dir, "*_meta_fix*.json")))
        if len(correction_files) == 0:
            correction_files = sorted(glob.glob(os.path.join(nodes_dir, "*_fix*.json")))
            TAG = "fix"

        if not correction_files:
            continue

        latest_for_node: Dict[str, str] = {}
        for file_path in correction_files:
            node_id = os.path.basename(file_path).split(f"_{TAG}")[0]
            prev = latest_for_node.get(node_id)
            if prev is None or os.path.getmtime(file_path) > os.path.getmtime(prev):
                latest_for_node[node_id] = file_path

        for node_id, corr_path in latest_for_node.items():
            try:
                with open(corr_path, "r", encoding="utf-8") as f:
                    correction = json.load(f)

                new_bbox = correction["new_bbox"] if "new_bbox" in correction else correction["bbox_global"]

                if not new_bbox or len(new_bbox) != 4:
                    continue

                tasks.append(
                    CorrectionTask(
                        namespace=namespace,
                        model_name=model_name,
                        version=version,
                        image_id=image_id,
                        node_id=node_id,
                        root_image_path=root_image_path,
                        nodes_dir=nodes_dir,
                        correction_file=corr_path,
                    )
                )
            except Exception:
                continue
    return tasks


def discover_all_corrections(cache_root: str) -> List[CorrectionTask]:
    """Discover all correction tasks across all namespaces, models, and versions."""
    if not os.path.isdir(cache_root):
        return []

    tasks: List[CorrectionTask] = []

    # Get all namespaces
    try:
        namespaces = [d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))]
    except Exception:
        return []

    for namespace in sorted(namespaces):
        ns_dir = os.path.join(cache_root, namespace)
        try:
            models = [d for d in os.listdir(ns_dir) if os.path.isdir(os.path.join(ns_dir, d))]
        except Exception:
            continue

        for model_name in sorted(models):
            model_dir = os.path.join(ns_dir, model_name)
            try:
                versions = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            except Exception:
                continue

            for version in sorted(versions):
                # Use existing discover_corrections function for each namespace/model/version combination
                version_tasks = discover_corrections(cache_root, namespace, model_name, version)
                tasks.extend(version_tasks)

    return tasks


class FunctionalRegionReannotator:
    annotation_query_count = 0

    def __init__(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        model: str,
        max_retries: int = 3,
        timeout: int = 300,
    ) -> None:
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.1,
            max_tokens=4096,
        )
        self.max_retries = max_retries
        self.timeout = timeout

    def _load_node_meta(self, nodes_dir: str, node_id: str) -> Tuple[Dict[str, Any], str]:
        candidates = [
            os.path.join(nodes_dir, f"{node_id}_meta.json"),
            os.path.join(nodes_dir, f"{node_id}.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f), path
        raise FileNotFoundError(f"Meta file not found for node {node_id}")

    def _prepare_message(
        self,
        namespace: str,
        model_name: str,
        version: str,
        image_id: str,
        node_id: str,
        previous_functionality: str,
        previous_description: str,
        root_highlight_img: np.ndarray,
        corrected_crop: np.ndarray,
    ) -> List[Dict[str, Any]]:
        prompt = REANNOTATION_PROMPT_TEMPLATE.format(
            previous_functionality=previous_functionality or "(none)",
            previous_description=previous_description or "(none)"
        )

        message_content = [
            {"type": "text", "text": prompt},
            {
                "type": "text",
                "text": "Corrected region (red rectangle) on complete GUI screenshot:",
            },
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64(root_highlight_img)},
            },
            {"type": "text", "text": "Cropped region after correction:"},
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64(corrected_crop)},
            },
        ]

        return [{"role": "user", "content": message_content}]

    def reannotate_node(
        self,
        task: CorrectionTask,
        debug: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        FunctionalRegionReannotator.annotation_query_count = 0

        root_image = cv2.imread(task.root_image_path)
        if root_image is None:
            raise FileNotFoundError(f"Root image not found: {task.root_image_path}")

        H, W = root_image.shape[:2]

        with open(task.correction_file, "r", encoding="utf-8") as f:
            correction_payload = json.load(f)
        corrected_bbox = clamp_bbox_to_image(correction_payload.get("new_bbox", correction_payload.get("bbox_global", [])), W, H)
        if corrected_bbox is None:
            raise ValueError("Invalid corrected bbox")

        node_meta, meta_path = self._load_node_meta(task.nodes_dir, task.node_id)
        prev_func = ""
        prev_desc = ""
        if isinstance(node_meta.get("functionality"), dict):
            prev_func = node_meta["functionality"].get("with_context") or node_meta["functionality"].get("wo_context", "")
        elif isinstance(node_meta.get("functionality"), str):
            prev_func = node_meta["functionality"]
        if isinstance(node_meta.get("description"), dict):
            prev_desc = node_meta["description"].get("with_context") or node_meta["description"].get("wo_context", "")
        elif isinstance(node_meta.get("description"), str):
            prev_desc = node_meta["description"]

        highlighted = draw_bbox_on_image(root_image, corrected_bbox)
        region_crop = root_image[corrected_bbox[1] : corrected_bbox[3], corrected_bbox[0] : corrected_bbox[2]]
        if region_crop.size == 0:
            raise ValueError("Empty crop for corrected bbox")

        if MAX_SIZE > 0 and max(highlighted.shape[:2]) > MAX_SIZE:
            highlighted, _ = resize_image(highlighted, MAX_SIZE)

        if MAX_SIZE > 0 and max(region_crop.shape[:2]) > MAX_SIZE:
            region_crop, _ = resize_image(region_crop, MAX_SIZE)

        messages = self._prepare_message(
            task.namespace,
            task.model_name,
            task.version,
            task.image_id,
            task.node_id,
            prev_func,
            prev_desc,
            highlighted,
            region_crop,
        )

        last_error: Optional[str] = None
        raw_response: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            if debug:
                debug_print(
                    f"  🔄 Re-annotating node {task.node_id} (attempt {attempt}/{self.max_retries})",
                    level="step",
                )
            try:
                FunctionalRegionReannotator.annotation_query_count += 1
                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages,
                    temperature=0.1 if attempt == 1 else 0.6,
                    timeout=self.timeout,
                    max_new_tokens=8192
                )
                if not success or not isinstance(response, str):
                    last_error = str(response)
                    continue
                parsed = extract_reannotation_from_response(response)
                if parsed is None:
                    last_error = "Failed to parse JSON response"
                    raw_response = response
                    continue
                raw_response = response
                processing_time = time.time() - start_time
                return {
                    "image_id": task.image_id,
                    "node_id": task.node_id,
                    "namespace": task.namespace,
                    "original_anno_model_name": task.model_name,
                    "version": task.version,
                    "corrected_bbox": corrected_bbox,
                    "previous_functionality": prev_func,
                    "previous_description": prev_desc,
                    "new_functionality": parsed,
                    "raw_response": raw_response,
                    "correction_file": task.correction_file.split('cache/')[-1],
                    "meta_path": meta_path.split('cache/')[-1],
                    "processing_time": processing_time,
                    "annotation_queries": FunctionalRegionReannotator.annotation_query_count,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as exc:
                last_error = str(exc)
                traceback.print_exc()

        raise RuntimeError(last_error or "Re-annotation failed")


def init_worker(base_url: Optional[str], api_key: Optional[str], model: str, max_retries: int, timeout: int, debug: bool) -> None:
    global reannotator_instance, debug_flag
    reannotator_instance = FunctionalRegionReannotator(base_url, api_key, model, max_retries=max_retries, timeout=timeout)
    debug_flag = debug


def process_task(task: CorrectionTask) -> Dict[str, Any]:
    global reannotator_instance, debug_flag
    return reannotator_instance.reannotate_node(task, debug=debug_flag)


def process_task_with_timeout(task: CorrectionTask, timeout_seconds: int) -> Dict[str, Any]:
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing timeout after {timeout_seconds} seconds")

    import signal

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = process_task(task)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise



def summarize_configuration(args: argparse.Namespace, output_file: str, tasks_count: int) -> None:
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("🛠️ Functional Region Re-Annotation", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("📁 INPUT", level="step")
    debug_print(f"   Cache Dir: {Fore.CYAN}{args.cache_dir}{Style.RESET_ALL}", level="info")
    debug_print(f"   Namespace: {Fore.CYAN}{args.namespace}{Style.RESET_ALL}", level="info")
    debug_print(f"   Model Run: {Fore.CYAN}{args.target_model}{Style.RESET_ALL}", level="info")
    debug_print(f"   Version: {Fore.CYAN}{args.version}{Style.RESET_ALL}", level="info")
    debug_print("", level="info")
    debug_print("🤖 LLM", level="step")
    debug_print(f"   Annotation Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    debug_print("", level="info")
    debug_print("⚙️  RUNTIME", level="step")
    mode_text = "SEQUENTIAL" if args.sequential else f"PARALLEL ({args.workers} workers)"
    debug_print(f"   Execution Mode: {Fore.YELLOW}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Task Timeout: {Fore.YELLOW}{args.task_timeout}s{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")
    debug_print(f"   Debug Mode: {on_off(args.debug)}", level="info")
    debug_print("", level="info")
    debug_print("💾 OUTPUT", level="step")
    debug_print(f"   Output File: {Fore.CYAN}{output_file}{Style.RESET_ALL}", level="info")
    debug_print(f"   Pending Corrections: {Fore.YELLOW}{tasks_count}{Style.RESET_ALL}", level="info")
    debug_print("════════════════════════════════════════════════════════════", level="title")


def save_experiment_config(args: argparse.Namespace, output_file: str) -> None:
    exp_cfg_dir = os.path.dirname(output_file)
    os.makedirs(exp_cfg_dir, exist_ok=True)
    exp_cfg_path = os.path.join(exp_cfg_dir, "reannotation_config.json")
    try:
        with open(exp_cfg_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)
        print(f"Saved experiment config to {exp_cfg_path}")
    except Exception as exc:
        print(f"Failed to save experiment config: {exc}")


def main(args: argparse.Namespace) -> None:
    # Discover corrections with wildcard support
    raw_tasks = discover_corrections(args.cache_dir, args.namespace, args.target_model, args.version)

    # Determine output mode based on whether we're scanning specific combinations or all
    is_specific_scan = args.namespace != "*" and args.target_model != "*" and args.version != "*"

    # Wildcard scan - use individual file output
    output_file = None
    existing_results = {}
    metadata = {}
    processed_keys = set()
    # For individual file mode, check which corrections already have output files
    for task in tqdm(raw_tasks, total=len(raw_tasks), desc="Checking existing results"):
        output_path = task.correction_file.replace('_meta_fix', '_meta_reannotated').replace('_fix', '_meta_reannotated').replace('.json', f"_{args.model}.json")
        if os.path.exists(output_path) and not args.force:
            with open(output_path, "r", encoding="utf-8") as f:
                sample = json.load(f)
                if "error" not in sample:
                    processed_keys.add(f"{task.image_id}/{task.node_id}")
                else:
                    1+1
    tasks = [t for t in raw_tasks if f"{t.image_id}/{t.node_id}" not in processed_keys]

    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("🛠️ Functional Region Re-Annotation", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("📁 INPUT", level="step")
    debug_print(f"   Cache Dir: {Fore.CYAN}{args.cache_dir}{Style.RESET_ALL}", level="info")
    scan_desc = []
    if args.namespace == "*": scan_desc.append("all namespaces")
    else: scan_desc.append(f"namespace '{args.namespace}'")
    if args.target_model == "*": scan_desc.append("all models")
    else: scan_desc.append(f"model '{args.target_model}'")
    if args.version == "*": scan_desc.append("all versions")
    else: scan_desc.append(f"version '{args.version}'")
    debug_print(f"   Scanning: {Fore.CYAN}{', '.join(scan_desc)}{Style.RESET_ALL}", level="info")
    debug_print("", level="info")
    debug_print("🤖 LLM", level="step")
    debug_print(f"   Re-annotation Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    debug_print("", level="info")
    debug_print("⚙️  RUNTIME", level="step")
    mode_text = "SEQUENTIAL" if args.sequential else f"PARALLEL ({args.workers} workers)"
    debug_print(f"   Execution Mode: {Fore.YELLOW}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Task Timeout: {Fore.YELLOW}{args.task_timeout}s{Style.RESET_ALL}", level="info")
    debug_print(f"   Max Retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")
    debug_print(f"   Debug Mode: {on_off(args.debug)}", level="info")
    debug_print("", level="info")
    debug_print("💾 OUTPUT", level="step")
    debug_print(f"   Output: {Fore.CYAN}Individual JSON files per correction{Style.RESET_ALL}", level="info")
    debug_print(f"   Pending Corrections: {Fore.YELLOW}{len(tasks)}{Style.RESET_ALL}", level="info")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    if not tasks:
        print("No corrected nodes to re-annotate.")
        return

    if args.sequential:
        with tqdm(total=len(tasks), desc="Re-annotating nodes", leave=True) as pbar:
            init_worker(args.base_url, args.api_key, args.model, args.max_retries, args.task_timeout, args.debug)
            total_processing_time = 0.0
            success_count = 0
            for task in tasks:
                key = f"{task.image_id}/{task.node_id}"
                try:
                    sample = process_task(task)
                    # Save individual result file
                    output_path = task.correction_file.replace('_meta_fix', '_meta_reannotated').replace('_fix', '_meta_reannotated').replace('.json', f"_{args.model}.json")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(sample, f, indent=2, ensure_ascii=False)
                    total_processing_time += sample.get("processing_time", 0)
                    success_count += 1
                except Exception as exc:
                    traceback.print_exc()
                    # Save error result
                    error_sample = {
                        "error": str(exc),
                        "image_id": task.image_id,
                        "node_id": task.node_id,
                        "correction_file": task.correction_file,
                    }
                    output_path = task.correction_file.replace('_meta_fix', '_meta_reannotated').replace('_fix', '_meta_reannotated').replace('.json', f"_{args.model}.json")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(error_sample, f, indent=2, ensure_ascii=False)
                pbar.update(1)

        print(f"Re-annotation complete. Processed {success_count}/{len(tasks)} corrections successfully.")
        return


    if not is_specific_scan:
        with multiprocessing.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model, args.max_retries, args.task_timeout, args.debug),
        ) as pool:
            try:
                with tqdm(total=len(tasks), desc="Re-annotating nodes", leave=True) as pbar:
                    async_results = []
                    for task in tasks:
                        async_results.append(
                            (
                                task,
                                pool.apply_async(
                                    process_task_with_timeout,
                                    args=(task, args.task_timeout),
                                ),
                            )
                        )

                    success_count = 0
                    for task, async_result in async_results:
                        try:
                            sample = async_result.get(timeout=args.task_timeout + 60)
                            # Save individual result file
                            output_path = task.correction_file.replace('_meta_fix', '_meta_reannotated').replace('_fix', '_meta_reannotated').replace('.json', f"_{args.model}.json")
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(sample, f, indent=2, ensure_ascii=False)
                            success_count += 1
                        except Exception as exc:
                            traceback.print_exc()
                            # Save error result
                            error_sample = {
                                "error": str(exc),
                                "image_id": task.image_id,
                                "node_id": task.node_id,
                                "correction_file": task.correction_file,
                            }
                            output_path = task.correction_file.replace('_meta_fix', '_meta_reannotated').replace('_fix', '_meta_reannotated').replace('.json', f"_{args.model}.json")
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(error_sample, f, indent=2, ensure_ascii=False)
                        pbar.update(1)

            finally:
                pool.close()
                pool.join()

        print(f"Re-annotation complete. Processed {success_count}/{len(tasks)} corrections successfully.")
    else:
        manager = multiprocessing.Manager()
        results = manager.dict(existing_results)
        processed_counter = manager.Value("i", 0)
        total_processing_time = manager.Value("d", 0.0)

        with multiprocessing.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model, args.max_retries, args.task_timeout, args.debug),
        ) as pool:
            try:
                with tqdm(total=len(tasks), desc="Re-annotating nodes", leave=True) as pbar:
                    async_results = []
                    for task in tasks:
                        async_results.append(
                            (
                                task,
                                pool.apply_async(
                                    process_task_with_timeout,
                                    args=(task, args.task_timeout),
                                ),
                            )
                        )

                    for task, async_result in async_results:
                        key = f"{task.image_id}/{task.node_id}"
                        try:
                            sample = async_result.get(timeout=args.task_timeout + 60)
                            results[key] = sample
                            processed_counter.value += 1
                            total_processing_time.value += sample.get("processing_time", 0)
                        except Exception as exc:
                            traceback.print_exc()
                            results[key] = {
                                "error": str(exc),
                                "image_id": task.image_id,
                                "node_id": task.node_id,
                                "correction_file": task.correction_file,
                            }
                            processed_counter.value += 1
                        pbar.update(1)
                        if processed_counter.value % args.checkpoint_interval == 0:
                            checkpoint_metadata = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "total_processed": processed_counter.value,
                                "total_tasks": len(tasks),
                                "avg_processing_time": (
                                    total_processing_time.value / processed_counter.value
                                    if processed_counter.value
                                    else 0
                                ),
                                "annotation_queries": sum(
                                    int(v.get("annotation_queries", 0)) for v in results.values() if isinstance(v, dict)
                                ),
                            }
                            save_checkpoint(dict(results), output_file, checkpoint_metadata)
                            print(f"\nSaved checkpoint with {processed_counter.value} samples")

            finally:
                pool.close()
                pool.join()

        final_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_processed": processed_counter.value,
            "total_tasks": len(tasks),
            "avg_processing_time": (
                total_processing_time.value / processed_counter.value if processed_counter.value else 0
            ),
            "annotation_queries": sum(
                int(v.get("annotation_queries", 0)) for v in results.values() if isinstance(v, dict)
            ),
        }
        save_checkpoint(dict(results), output_file, final_metadata)
        print(f"Re-annotation complete. Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-annotate functional regions after bounding-box corrections")
    parser.add_argument("--cache-dir", type=str, default="/mnt/vdb1/hongxin_li/AutoGUIv2/cache", help="Cache directory where annotation assets are stored")
    parser.add_argument("--namespace", type=str, default="*", help="Namespace (benchmark) name. Use '*' to scan all namespaces.")
    parser.add_argument("--target-model", type=str, default="gemini-2.5-pro-thinking", help="Original annotation model name. Use '*' to scan all models.")
    parser.add_argument("--version", type=str, default="v2", help="Annotation version. Use '*' to scan all versions.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional explicit output JSON file")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking", help="LLM model to use for re-annotation")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY_XIAOAI"), help="API key for the LLM provider")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_API_BASE_XIAOAI"), help="Optional custom API base URL")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially for easier debugging")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per node re-annotation")
    parser.add_argument("--task-timeout", type=int, default=900, help="Timeout per node in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--force", action="store_true", help="Reprocess nodes even if already in checkpoint")
    args, _ = parser.parse_known_args()

    multiprocessing.set_start_method("spawn", force=True)
    main(args)

