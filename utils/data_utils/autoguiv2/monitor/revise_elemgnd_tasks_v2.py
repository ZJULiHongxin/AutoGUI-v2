import os
import json
import argparse
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from colorama import Fore, Style


def load_eval_set_filter(eval_set_path: str) -> Set[Tuple[str, str, int, int]]:
    """
    Load evaluation set file and return a set of (dataset_name, image_key, group_index, target_elem_id) tuples.

    Note: group_index here is the actual group_index field from the eval set (not the list index in 'generated').
    target_elem_id corresponds to target_element_id in questions.
    """
    if not eval_set_path or not os.path.exists(eval_set_path):
        return set()

    try:
        with open(eval_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Failed to load eval set file: {e}{Style.RESET_ALL}", flush=True)
        return set()

    entries = data.get("entries", [])
    filter_set: Set[Tuple[str, str, int, int]] = set()

    for entry in entries:
        # Parse from eval set entry fields
        dataset_name = entry.get("dataset_name", "")
        image_name = entry.get("image_name", "")  # This is the image_key in our system
        group_index = entry.get("group_index")
        target_elem_id = entry.get("target_elem_id")

        if dataset_name and image_name and group_index is not None and target_elem_id is not None:
            # Store (dataset_name, image_key, group_index, target_elem_id)
            # group_index is the actual group_index field, not list index
            filter_set.add((dataset_name, image_name, int(group_index), int(target_elem_id)))

    print(f"{Fore.GREEN}Loaded {len(filter_set)} entries from eval set: {eval_set_path}{Style.RESET_ALL}", flush=True)
    return filter_set


def get_corrected_sample_keys(datasets_root: str) -> Set[Tuple[str, str, int, int]]:
    """
    Get all sample keys that have corrections across all datasets.

    Returns a set of (dataset_name, image_key, group_index, target_element_id) tuples.
    Note: We need to look up the actual group_index and target_element_id from the questions file
    since corrections are stored by (image_key, generated_list_idx, q_idx).
    """
    corrected_keys: Set[Tuple[str, str, int, int]] = set()

    if not os.path.isdir(datasets_root):
        return corrected_keys

    for dataset_name in os.listdir(datasets_root):
        cpath = _corrections_path(datasets_root, dataset_name)
        qpath = _questions_path(datasets_root, dataset_name)
        if os.path.exists(cpath) and os.path.exists(qpath):
            try:
                corrections = _load_json(cpath)
                questions_data = _load_json(qpath)
                results = questions_data.get("results", {})

                for c_key in corrections.keys():
                    # Parse c_key: "image_key__generated_list_idx__q_idx"
                    parts = c_key.split("__")
                    if len(parts) == 3:
                        image_key = parts[0]
                        try:
                            generated_list_idx = int(parts[1])
                            q_idx = int(parts[2])

                            # Look up the actual group_index and target_element_id
                            if image_key in results:
                                image_data = results[image_key]
                                generated = image_data.get("generated", [])
                                if 0 <= generated_list_idx < len(generated):
                                    group = generated[generated_list_idx]
                                    group_index = group.get("group_index", generated_list_idx)
                                    questions = group.get("questions", [])
                                    if 0 <= q_idx < len(questions):
                                        q = questions[q_idx]
                                        target_elem_id = q.get("target_element_id") or q.get("target_element_index")
                                        if target_elem_id is not None:
                                            corrected_keys.add((dataset_name, image_key, int(group_index), int(target_elem_id)))
                        except ValueError:
                            continue
            except Exception:
                continue

    return corrected_keys

def _load_original_bbox_and_questions_by_action(
    datasets_root: str,
    dataset: str,
    image_key: str,
    group_idx: int,
    q_idx: int,
) -> Tuple[List[int], Dict[str, str]]:
    """Load original bbox and original questions per action for comparison."""
    qpath = _questions_path(datasets_root, dataset)
    data = _load_json(qpath)
    image_data = (data.get("results", {}) or {}).get(image_key) or {}
    generated = image_data.get("generated", []) or []
    if group_idx < 0 or group_idx >= len(generated):
        return [], {}
    group = generated[group_idx] or {}
    questions = group.get("questions", []) or []
    if q_idx < 0 or q_idx >= len(questions):
        return [], {}
    qobj = questions[q_idx] or {}

    orig_bbox = _pick_bbox_for_question(group, qobj)
    actions = _pick_action_types(qobj)
    orig_questions_by_action: Dict[str, str] = {}
    for act in actions:
        orig_questions_by_action[act] = _get_original_question_for_action(qobj, act)
    # Always include clicking as a common default
    if "clicking" not in orig_questions_by_action:
        orig_questions_by_action["clicking"] = _get_original_question_for_action(qobj, "clicking")
    return orig_bbox, orig_questions_by_action


def delete_correction(
    datasets_root: str,
    dataset: str,
    image_key: str,
    group_idx: int,
    q_idx: int,
) -> bool:
    """Delete a correction entry if exists. Returns True if deleted."""
    cpath = _corrections_path(datasets_root, dataset)
    corrections = _load_json(cpath)
    c_key = f"{image_key}__{group_idx}__{q_idx}"
    if not isinstance(corrections, dict) or c_key not in corrections:
        return False
    try:
        del corrections[c_key]
    except Exception:
        return False
    _save_json(cpath, corrections)
    try:
        total = len(corrections) if isinstance(corrections, dict) else 0
    except Exception:
        total = 0
    print(f"{Fore.YELLOW}Removed {c_key} ({total} corrections) from {cpath}{Style.RESET_ALL}", flush=True)
    return True

def detect_datasets_root(cli_root: Optional[str]) -> str:
    """Auto-detect dataset root containing */FuncElemGnd/grounding_questions.json."""
    candidates: List[str] = []
    if cli_root:
        candidates.append(cli_root)
    env_root = os.environ.get("AUTOGUI_DATASETS_ROOT") or os.environ.get("AUTOGUI_CACHE_ROOT")
    if env_root:
        candidates.append(env_root)
    candidates.append("/mnt/vdb1/hongxin_li/AutoGUIv2")
    local_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    candidates.append(local_guess)

    for c in candidates:
        if c and os.path.isdir(c):
            return c
    for c in candidates:
        if c:
            return c
    return "."


def _questions_path(datasets_root: str, dataset: str) -> str:
    return os.path.join(datasets_root, dataset, "FuncElemGnd", "grounding_questions.json")


def _corrections_path(datasets_root: str, dataset: str) -> str:
    return os.path.join(datasets_root, dataset, "FuncElemGnd", "grounding_questions_corrections.json")


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_datasets(datasets_root: str) -> List[str]:
    if not os.path.isdir(datasets_root):
        return []
    out: List[str] = []
    for name in os.listdir(datasets_root):
        qpath = _questions_path(datasets_root, name)
        if os.path.exists(qpath):
            out.append(name)
    return sorted(out)


def _encode_image_base64_and_size(image_path: str) -> Tuple[Optional[str], Optional[List[int]]]:
    if not image_path or not os.path.exists(image_path):
        return None, None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8"), [w, h]
    except Exception:
        return None, None


def _pick_action_types(question_obj: Dict[str, Any]) -> List[str]:
    ref = question_obj.get("referring_expressions") or question_obj.get("referring expressions") or {}
    if not isinstance(ref, dict):
        return []
    # preserve stable order (common actions first)
    priority = [
        "clicking",
        "hovering",
        "typing",
        "dragging",
        "scrolling",
        "selecting",
        "swiping",
        "pressing",
        "long pressing",
        "double-clicking",
        "right-clicking",
        "middle-clicking",
    ]
    existing = [k for k in priority if k in ref]
    for k in ref.keys():
        if k not in existing:
            existing.append(k)
    return existing


def _get_original_question_for_action(question_obj: Dict[str, Any], action_type: str) -> str:
    ref = question_obj.get("referring_expressions") or question_obj.get("referring expressions") or {}
    if isinstance(ref, dict) and action_type in ref and isinstance(ref[action_type], dict):
        return ref[action_type].get("question", "") or ""
    # fallback: any question
    if isinstance(ref, dict):
        for v in ref.values():
            if isinstance(v, dict) and v.get("question"):
                return v.get("question")
    return question_obj.get("question", "") or ""


def _pick_bbox_for_question(group_obj: Dict[str, Any], question_obj: Dict[str, Any]) -> List[int]:
    elements = group_obj.get("elements", []) if isinstance(group_obj, dict) else []
    target_id = question_obj.get("target_element_id") or question_obj.get("target_element_index")
    chosen = None
    if target_id is not None:
        for el in elements:
            if el.get("id") == target_id:
                chosen = el
                break
    if chosen is None and elements:
        chosen = elements[0]
    if not chosen:
        return []
    bbox = chosen.get("revised bbox") or chosen.get("bbox") or chosen.get("bbox_global") or []
    return bbox if isinstance(bbox, list) else []


def list_samples(datasets_root: str, dataset: str, eval_set_filter: Optional[Set[Tuple[str, str, int, int]]] = None, corrected_keys: Optional[Set[Tuple[str, str, int, int]]] = None) -> List[Dict[str, Any]]:
    """
    List samples for a dataset.

    If eval_set_filter is provided (non-empty), only samples matching the filter OR having corrections will be shown.
    Filter key format: (dataset_name, image_key, group_index, target_elem_id)
    corrected_keys: Set of (dataset_name, image_key, group_index, target_elem_id) that have corrections
    """
    qpath = _questions_path(datasets_root, dataset)
    data = _load_json(qpath)
    results = data.get("results", {})
    corrections = _load_json(_corrections_path(datasets_root, dataset))
    samples: List[Dict[str, Any]] = []
    for image_key, image_data in results.items():
        generated = image_data.get("generated", []) or []
        for generated_list_idx, group in enumerate(generated):
            # Get the actual group_index from the group object
            group_index = group.get("group_index", generated_list_idx)

            questions = group.get("questions", []) or []
            for q_idx, q in enumerate(questions):
                target_id = q.get("target_element_id") or q.get("target_element_index")

                # Check if this sample should be included based on eval_set_filter
                if eval_set_filter:
                    # Key for eval set filter: (dataset_name, image_key, group_index, target_elem_id)
                    # Handle non-integer group_index (e.g., 'newly_added_group_1') by skipping filter match
                    try:
                        group_index_int = int(group_index)
                        target_id_int = int(target_id) if target_id is not None else -1
                        filter_key = (dataset, image_key, group_index_int, target_id_int)
                        in_eval_set = filter_key in eval_set_filter

                        # Also check if this sample has corrections
                        has_correction = False
                        if corrected_keys:
                            has_correction = filter_key in corrected_keys
                    except (ValueError, TypeError):
                        # Non-integer group_index or target_id - cannot match eval set filter
                        # But still check if it has corrections by using the generated_list_idx
                        in_eval_set = False
                        has_correction = False
                        # Check corrections using the actual correction key format
                        c_key_check = f"{image_key}__{generated_list_idx}__{q_idx}"
                        has_correction = c_key_check in corrections

                    # Skip if not in eval set AND not corrected
                    if not in_eval_set and not has_correction:
                        continue

                # pick a label question (prefer clicking)
                label_q = _get_original_question_for_action(q, "clicking")
                # Check if sample is abandoned
                c_key = f"{image_key}__{generated_list_idx}__{q_idx}"
                corr_entry = corrections.get(c_key, {})
                abandoned = corr_entry.get("abandoned", False)

                # Determine sample status
                if abandoned:
                    status = "abandoned"
                elif corr_entry:  # Has correction entry with any modifications
                    status = "modified"
                else:
                    status = "untouched"

                samples.append(
                    {
                        "image_key": image_key,
                        "group_idx": generated_list_idx,  # This is the list index, used for API calls
                        "q_idx": q_idx,
                        "target_element_id": target_id,
                        "group_index": group_index,  # Actual group_index field for display
                        "label": f"{image_key} | g{group_index} | q{q_idx} | id {target_id}",
                        "question_preview": (label_q[:120] + "…") if len(label_q) > 120 else label_q,
                        "abandoned": abandoned,
                        "status": status,
                    }
                )
    return samples


def get_sample(
    datasets_root: str,
    dataset: str,
    image_key: str,
    group_idx: int,
    q_idx: int,
    action_type: str = "clicking",
) -> Dict[str, Any]:
    qpath = _questions_path(datasets_root, dataset)
    data = _load_json(qpath)
    image_data = data.get("results", {}).get(image_key)
    if not image_data:
        raise HTTPException(status_code=404, detail="image_key not found")

    generated = image_data.get("generated", []) or []
    if group_idx < 0 or group_idx >= len(generated):
        raise HTTPException(status_code=404, detail="group_idx out of range")
    group = generated[group_idx]

    questions = group.get("questions", []) or []
    if q_idx < 0 or q_idx >= len(questions):
        raise HTTPException(status_code=404, detail="q_idx out of range")
    qobj = questions[q_idx]

    available_actions = _pick_action_types(qobj)
    if action_type not in available_actions and available_actions:
        action_type = available_actions[0]

    orig_question = _get_original_question_for_action(qobj, action_type)
    orig_bbox = _pick_bbox_for_question(group, qobj)

    image_path = image_data.get("image_path", "")

    image_path = os.path.join(datasets_root, image_path.split("AutoGUIv2/")[-1])
    image_base64, image_size = _encode_image_base64_and_size(image_path)

    # Corrections are stored per (image_key, group_idx, q_idx)
    corrections = _load_json(_corrections_path(datasets_root, dataset))
    c_key = f"{image_key}__{group_idx}__{q_idx}"
    corr_entry = corrections.get(c_key, {})
    modified_bbox = corr_entry.get("modified_bbox", orig_bbox)
    modified_questions_by_action = corr_entry.get("modified_questions_by_action", {})
    modified_question = modified_questions_by_action.get(action_type, orig_question)
    abandoned = corr_entry.get("abandoned", False)

    corrections_file = _corrections_path(datasets_root, dataset)
    correction_key = f"{image_key}__{group_idx}__{q_idx}"

    return {
        "image_key": image_key,
        "group_idx": group_idx,
        "q_idx": q_idx,
        "target_element_id": qobj.get("target_element_id") or qobj.get("target_element_index"),
        "visual_similarity": group.get("visual_similarity", ""),
        "available_action_types": available_actions,
        "action_type": action_type,
        "original_question": orig_question,
        "original_bbox": orig_bbox,
        "modified_question": modified_question,
        "modified_bbox": modified_bbox,
        "modified_questions_by_action": modified_questions_by_action,
        "abandoned": abandoned,
        "image_path": image_path,
        "image_size": image_size,  # [W, H] in pixels of the original image
        "image_base64": image_base64,
        "corrections_file": corrections_file,
        "correction_key": correction_key,
    }


def save_correction(
    datasets_root: str,
    dataset: str,
    image_key: str,
    group_idx: int,
    q_idx: int,
    modified_bbox: List[int],
    modified_questions_by_action: Dict[str, str],
    abandoned: bool = False,
) -> None:
    if not isinstance(modified_bbox, list) or len(modified_bbox) != 4:
        raise HTTPException(status_code=400, detail="modified_bbox must be a list of 4 numbers")

    cpath = _corrections_path(datasets_root, dataset)
    corrections = _load_json(cpath)
    c_key = f"{image_key}__{group_idx}__{q_idx}"
    prev = corrections.get(c_key, {})
    merged = dict(prev) if isinstance(prev, dict) else {}
    merged["modified_bbox"] = modified_bbox
    merged["modified_questions_by_action"] = modified_questions_by_action or {}
    merged["abandoned"] = abandoned
    merged["updated_at"] = datetime.utcnow().isoformat()
    corrections[c_key] = merged
    _save_json(cpath, corrections)
    # Terminal-friendly message for manual monitoring
    try:
        total = len(corrections) if isinstance(corrections, dict) else 0
    except Exception:
        total = 0
    print(f"{Fore.GREEN}Saved {c_key} ({total} corrections) to {cpath}{Style.RESET_ALL}", flush=True)


def create_app(datasets_root: str, eval_set_filter: Optional[Set[Tuple[str, str, int, int]]] = None, corrected_keys: Optional[Set[Tuple[str, str, int, int]]] = None) -> FastAPI:
    app = FastAPI(title="FuncElemGnd Task Reviser", version="2.0")

    @app.get("/health")
    def health():
        filter_active = eval_set_filter is not None and len(eval_set_filter) > 0
        return {
            "status": "ok",
            "datasets_root": datasets_root,
            "eval_set_filter_active": filter_active,
            "eval_set_filter_count": len(eval_set_filter) if eval_set_filter else 0,
            "corrected_keys_count": len(corrected_keys) if corrected_keys else 0,
        }

    @app.get("/api/datasets")
    def api_datasets():
        return {"datasets": list_datasets(datasets_root)}

    @app.get("/api/images")
    def api_images(dataset: str):
        if dataset not in list_datasets(datasets_root):
            raise HTTPException(status_code=404, detail="dataset not found")
        return {"images": list_samples(datasets_root, dataset, eval_set_filter, corrected_keys)}

    @app.get("/api/sample")
    def api_sample(dataset: str, image_key: str, group_idx: int, q_idx: int, action_type: str = "clicking"):
        if dataset not in list_datasets(datasets_root):
            raise HTTPException(status_code=404, detail="dataset not found")
        return JSONResponse(get_sample(datasets_root, dataset, image_key, group_idx, q_idx, action_type))

    @app.post("/api/save_correction")
    def api_save_correction(payload: Dict[str, Any] = Body(...)):
        dataset = payload.get("dataset")
        image_key = payload.get("image_key")
        group_idx = payload.get("group_idx")
        q_idx = payload.get("q_idx")
        modified_bbox = payload.get("modified_bbox", [])
        modified_questions_by_action = payload.get("modified_questions_by_action", {})
        abandoned = payload.get("abandoned", False)

        if dataset is None or image_key is None or group_idx is None or q_idx is None:
            raise HTTPException(status_code=400, detail="dataset, image_key, group_idx, q_idx are required")

        # If nothing changed (bbox unchanged + all questions unchanged), do not save.
        # If an existing correction entry exists, remove it.
        orig_bbox, orig_questions_by_action = _load_original_bbox_and_questions_by_action(
            datasets_root, str(dataset), str(image_key), int(group_idx), int(q_idx)
        )

        bbox_changed = True
        if isinstance(orig_bbox, list) and isinstance(modified_bbox, list) and len(orig_bbox) == 4 and len(modified_bbox) == 4:
            bbox_changed = (orig_bbox != modified_bbox)

        # Prune unchanged questions to keep corrections file clean
        pruned_questions: Dict[str, str] = {}
        if isinstance(modified_questions_by_action, dict):
            for act, txt in modified_questions_by_action.items():
                if not isinstance(act, str):
                    continue
                if txt is None:
                    continue
                txt_s = str(txt)
                orig_txt = orig_questions_by_action.get(act, "")
                if txt_s != orig_txt:
                    pruned_questions[act] = txt_s

        # NOTE: "abandoned" must be persisted even when bbox/question are unchanged.
        # Only skip/delete when there is truly nothing to record AND abandoned is False.
        if (not bbox_changed) and (len(pruned_questions) == 0) and (not bool(abandoned)):
            deleted = delete_correction(datasets_root, str(dataset), str(image_key), int(group_idx), int(q_idx))
            if not deleted:
                cpath = _corrections_path(datasets_root, str(dataset))
                c_key = f"{image_key}__{group_idx}__{q_idx}"
                print(f"{Fore.CYAN}No changes for {c_key}; skipped saving to {cpath}{Style.RESET_ALL}", flush=True)
            return {"status": "skipped", "deleted": bool(deleted)}

        save_correction(
            datasets_root,
            str(dataset),
            str(image_key),
            int(group_idx),
            int(q_idx),
            modified_bbox,
            pruned_questions,
            bool(abandoned),
        )
        return {"status": "saved"}

    static_dir = os.path.join(os.path.dirname(__file__), "revise_elemgnd_tasks")
    if not os.path.isdir(static_dir):
        raise RuntimeError(f"Static directory not found: {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="revise_elemgnd_tasks")

    return app


def main():
    parser = argparse.ArgumentParser(description="FuncElemGnd task revising UI server")
    parser.add_argument("--datasets-root", type=str, default="/volume/pt-coder/users/gji/data/gui_data/AutoGUIv2", help="Root dir containing datasets (*/FuncElemGnd/grounding_questions.json)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19805)
    parser.add_argument(
        "--eval-set",
        type=str,
        default="/volume/pt-coder/users/gji/projects/highres_autogui/utils/eval_utils/autoguiv2/elemgnd_hf_dataset_cache/a136b59f5a5f5e2809a96d5798412c8a_func-w-bbox.json",
        help="Path to an evaluation set JSON file (e.g., *_func-w-bbox.json). "
             "When provided, only samples in the 'entries' field of this file will be shown, "
             "plus any samples that have existing corrections."
    )
    args, _ = parser.parse_known_args()

    datasets_root = detect_datasets_root(args.datasets_root)

    # Load eval set filter if provided
    eval_set_filter: Optional[Set[Tuple[str, str, int, int]]] = None
    corrected_keys: Optional[Set[Tuple[str, str, int, int]]] = None

    if args.eval_set:
        eval_set_filter = load_eval_set_filter(args.eval_set)
        if eval_set_filter:
            # Also load corrected sample keys to ensure they're always shown
            corrected_keys = get_corrected_sample_keys(datasets_root)
            print(f"{Fore.CYAN}Corrected samples found: {len(corrected_keys)}{Style.RESET_ALL}", flush=True)
        else:
            print(f"{Fore.YELLOW}Warning: Eval set file loaded but no entries found. Showing all samples.{Style.RESET_ALL}", flush=True)

    app = create_app(datasets_root, eval_set_filter, corrected_keys)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


