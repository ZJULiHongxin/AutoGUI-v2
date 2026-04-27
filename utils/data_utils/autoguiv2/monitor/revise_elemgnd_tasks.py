import os
import json
import argparse
import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image

from colorama import Fore, Style

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
        "long_pressing",
        "double_clicking",
        "right_clicking",
        "middle_clicking",
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


def list_samples(datasets_root: str, dataset: str) -> List[Dict[str, Any]]:
    qpath = _questions_path(datasets_root, dataset)
    data = _load_json(qpath)
    results = data.get("results", {})
    corrections = _load_json(_corrections_path(datasets_root, dataset))
    samples: List[Dict[str, Any]] = []
    for image_key, image_data in results.items():
        generated = image_data.get("generated", []) or []
        for group_idx, group in enumerate(generated):
            questions = group.get("questions", []) or []
            for q_idx, q in enumerate(questions):
                target_id = q.get("target_element_id") or q.get("target_element_index")
                # pick a label question (prefer clicking)
                label_q = _get_original_question_for_action(q, "clicking")
                # Check if sample is abandoned
                c_key = f"{image_key}__{group_idx}__{q_idx}"
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
                        "group_idx": group_idx,
                        "q_idx": q_idx,
                        "target_element_id": target_id,
                        "label": f"{image_key} | g{group_idx} | q{q_idx} | id {target_id}",
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


def create_app(datasets_root: str) -> FastAPI:
    app = FastAPI(title="FuncElemGnd Task Reviser", version="2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "datasets_root": datasets_root}

    @app.get("/api/datasets")
    def api_datasets():
        return {"datasets": list_datasets(datasets_root)}

    @app.get("/api/images")
    def api_images(dataset: str):
        if dataset not in list_datasets(datasets_root):
            raise HTTPException(status_code=404, detail="dataset not found")
        return {"images": list_samples(datasets_root, dataset)}

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
    parser.add_argument("--port", type=int, default=17805)
    args, _ = parser.parse_known_args()

    datasets_root = detect_datasets_root(args.datasets_root)
    app = create_app(datasets_root)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


