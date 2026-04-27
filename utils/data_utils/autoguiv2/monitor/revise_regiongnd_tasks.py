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


def detect_datasets_root(cli_root: Optional[str]) -> str:
    """Auto-detect dataset root containing */FuncRegion/grounding_mode/*.json."""
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


def _questions_dir(datasets_root: str, dataset: str) -> str:
    """Return the directory containing *_result.json files."""
    return os.path.join(datasets_root, dataset, "FuncRegion", "grounding_mode")


def _corrections_path(datasets_root: str, dataset: str) -> str:
    """Return path to corrections file."""
    return os.path.join(datasets_root, dataset, "FuncRegion", "grounding_questions_corrections.json")


def _list_question_files(datasets_root: str, dataset: str) -> List[str]:
    """List all *_result.json files in the grounding_mode directory."""
    dir_path = _questions_dir(datasets_root, dataset)
    if not os.path.isdir(dir_path):
        return []
    files = [
        f for f in os.listdir(dir_path)
        if f.endswith("_result.json") and not f.startswith("_")
    ]
    return sorted(files)


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
    """List all datasets that have FuncRegion/grounding_mode directory."""
    if not os.path.isdir(datasets_root):
        return []
    out: List[str] = []
    for name in os.listdir(datasets_root):
        qdir = _questions_dir(datasets_root, name)
        if os.path.isdir(qdir):
            # Check if there are any result files
            files = _list_question_files(datasets_root, name)
            if files:
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


def _pick_region_types(options: List[Dict[str, Any]]) -> List[str]:
    """Extract unique region types from options, preserving priority order."""
    priority = [
        "Toolbar / Action Bar",
        "Header / Top Bar",
        "Sidebar / Side Navigation",
        "Card / Item List",
        "Tab Bar",
        "Application Window",
        "Browser Window / Tab",
        "Notification / Toast / Alert Banner",
        "Filter / Sort Controls",
        "Form",
        "Dropdown Menu",
        "Footer",
        "Search Region",
        "Main Content Area",
        "Media Player",
        "Modal / Dialog Box",
        "Data Table / Grid",
        "Static Title or Heading",
        "Breadcrumbs",
        "Cookie Consent Banner",
        "Image Gallery / Carousel",
        "Dashboard / Widget Area",
        "Body Text",
        "Isolated Icon",
        "Unclassified",
    ]
    seen = set()
    result = []
    
    # First add priority types that exist in options
    for rtype in priority:
        for opt in options:
            if opt.get("region_type") == rtype and rtype not in seen:
                seen.add(rtype)
                result.append(rtype)
                break
    
    # Then add any remaining types not in priority list
    for opt in options:
        rtype = opt.get("region_type", "")
        if rtype and rtype not in seen:
            seen.add(rtype)
            result.append(rtype)
    
    return result


def _load_original_data(
    datasets_root: str,
    dataset: str,
    json_file: str,
    q_idx: int,
) -> Tuple[str, str, List[Dict[str, Any]], str]:
    """
    Load original question, correct_answer, options, and explanation.
    Returns: (question, correct_answer, options, explanation)
    """
    qdir = _questions_dir(datasets_root, dataset)
    qpath = os.path.join(qdir, json_file)
    
    if not os.path.exists(qpath):
        return "", "", [], ""
    
    data = _load_json(qpath)
    result = data.get("result", {})
    questions = result.get("questions", [])
    
    if q_idx < 0 or q_idx >= len(questions):
        return "", "", [], ""
    
    q = questions[q_idx]
    question = q.get("question", "")
    correct_answer = q.get("correct_answer", "")
    options = q.get("options", [])
    explanation = q.get("explanation", "")
    
    return question, correct_answer, options, explanation


def delete_correction(
    datasets_root: str,
    dataset: str,
    json_file: str,
    q_idx: int,
) -> bool:
    """Delete a correction entry if exists. Returns True if deleted."""
    cpath = _corrections_path(datasets_root, dataset)
    corrections = _load_json(cpath)
    c_key = f"{json_file}__{q_idx}"
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


def list_samples(datasets_root: str, dataset: str) -> List[Dict[str, Any]]:
    """
    List all samples (questions) from all *_result.json files in the dataset.
    Each question is a sample.
    """
    qdir = _questions_dir(datasets_root, dataset)
    corrections = _load_json(_corrections_path(datasets_root, dataset))
    samples: List[Dict[str, Any]] = []
    
    json_files = _list_question_files(datasets_root, dataset)
    
    for json_file in json_files:
        qpath = os.path.join(qdir, json_file)
        data = _load_json(qpath)
        result = data.get("result", {})
        questions = result.get("questions", [])
        image_key = result.get("image_key", json_file.replace("_result.json", ""))
        
        for q_idx, q in enumerate(questions):
            question_text = q.get("question", "")
            correct_answer = q.get("correct_answer", "")
            
            # Check correction status
            c_key = f"{json_file}__{q_idx}"
            corr_entry = corrections.get(c_key, {})
            abandoned = corr_entry.get("abandoned", False)
            
            # Determine sample status
            if abandoned:
                status = "abandoned"
            elif corr_entry:
                status = "modified"
            else:
                status = "untouched"
            
            # Create preview label
            preview = (question_text[:100] + "…") if len(question_text) > 100 else question_text
            
            samples.append({
                "json_file": json_file,
                "q_idx": q_idx,
                "image_key": image_key,
                "label": f"{json_file} | q{q_idx} | ans:{correct_answer}",
                "question_preview": preview,
                "correct_answer": correct_answer,
                "abandoned": abandoned,
                "status": status,
            })
    
    return samples


def get_sample(
    datasets_root: str,
    dataset: str,
    json_file: str,
    q_idx: int,
    region_type: str = "",
) -> Dict[str, Any]:
    """
    Get a single sample (question) with all its data.
    region_type parameter is used to filter/highlight specific region type in UI.
    """
    qdir = _questions_dir(datasets_root, dataset)
    qpath = os.path.join(qdir, json_file)
    
    if not os.path.exists(qpath):
        raise HTTPException(status_code=404, detail="json_file not found")
    
    data = _load_json(qpath)
    result = data.get("result", {})
    questions = result.get("questions", [])
    
    if q_idx < 0 or q_idx >= len(questions):
        raise HTTPException(status_code=404, detail="q_idx out of range")
    
    q = questions[q_idx]
    
    # Extract original data
    orig_question = q.get("question", "")
    orig_correct_answer = q.get("correct_answer", "")
    options = q.get("options", [])
    explanation = q.get("explanation", "")
    image_path = q.get("image_path", "")
    image_size = q.get("image_size", [])
    
    # Get available region types
    available_region_types = _pick_region_types(options)
    
    # If no region_type specified, use the first one
    if not region_type and available_region_types:
        region_type = available_region_types[0]
    
    # Get bbox for correct answer option
    orig_bbox = []
    for opt in options:
        if opt.get("label") == orig_correct_answer:
            orig_bbox = opt.get("bbox", [])
            break
    
    # Load corrections
    corrections = _load_json(_corrections_path(datasets_root, dataset))
    c_key = f"{json_file}__{q_idx}"
    corr_entry = corrections.get(c_key, {})
    
    modified_question = corr_entry.get("modified_question", orig_question)
    modified_correct_answer = corr_entry.get("modified_correct_answer", orig_correct_answer)
    modified_bbox = corr_entry.get("modified_bbox", orig_bbox)
    abandoned = corr_entry.get("abandoned", False)
    
    # Encode image
    image_base64, img_size = _encode_image_base64_and_size(image_path)
    if img_size:
        image_size = img_size
    
    corrections_file = _corrections_path(datasets_root, dataset)
    correction_key = c_key
    
    return {
        "json_file": json_file,
        "q_idx": q_idx,
        "image_key": result.get("image_key", json_file.replace("_result.json", "")),
        "available_region_types": available_region_types,
        "region_type": region_type,
        "original_question": orig_question,
        "original_correct_answer": orig_correct_answer,
        "original_bbox": orig_bbox,
        "modified_question": modified_question,
        "modified_correct_answer": modified_correct_answer,
        "modified_bbox": modified_bbox,
        "options": options,  # All options for display
        "explanation": explanation,
        "abandoned": abandoned,
        "image_path": image_path,
        "image_size": image_size,
        "image_base64": image_base64,
        "corrections_file": corrections_file,
        "correction_key": correction_key,
    }


def save_correction(
    datasets_root: str,
    dataset: str,
    json_file: str,
    q_idx: int,
    modified_question: str,
    modified_correct_answer: str,
    modified_bbox: List[int],
    abandoned: bool = False,
) -> None:
    """Save corrections for a question."""
    if not isinstance(modified_bbox, list) or len(modified_bbox) != 4:
        raise HTTPException(status_code=400, detail="modified_bbox must be a list of 4 numbers")
    
    cpath = _corrections_path(datasets_root, dataset)
    corrections = _load_json(cpath)
    c_key = f"{json_file}__{q_idx}"
    
    prev = corrections.get(c_key, {})
    merged = dict(prev) if isinstance(prev, dict) else {}
    merged["modified_question"] = modified_question
    merged["modified_correct_answer"] = modified_correct_answer
    merged["modified_bbox"] = modified_bbox
    merged["abandoned"] = abandoned
    merged["updated_at"] = datetime.utcnow().isoformat()
    
    corrections[c_key] = merged
    _save_json(cpath, corrections)
    
    try:
        total = len(corrections) if isinstance(corrections, dict) else 0
    except Exception:
        total = 0
    print(f"{Fore.GREEN}Saved {c_key} ({total} corrections) to {cpath}{Style.RESET_ALL}", flush=True)


def create_app(datasets_root: str) -> FastAPI:
    app = FastAPI(title="FuncRegionGnd Task Reviser", version="1.0")

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
    def api_sample(dataset: str, json_file: str, q_idx: int, region_type: str = ""):
        if dataset not in list_datasets(datasets_root):
            raise HTTPException(status_code=404, detail="dataset not found")
        return JSONResponse(get_sample(datasets_root, dataset, json_file, q_idx, region_type))

    @app.post("/api/save_correction")
    def api_save_correction(payload: Dict[str, Any] = Body(...)):
        dataset = payload.get("dataset")
        json_file = payload.get("json_file")
        q_idx = payload.get("q_idx")
        modified_question = payload.get("modified_question", "")
        modified_correct_answer = payload.get("modified_correct_answer", "")
        modified_bbox = payload.get("modified_bbox", [])
        abandoned = payload.get("abandoned", False)

        if dataset is None or json_file is None or q_idx is None:
            raise HTTPException(status_code=400, detail="dataset, json_file, q_idx are required")

        # Load original data for comparison
        orig_question, orig_correct_answer, options, _ = _load_original_data(
            datasets_root, str(dataset), str(json_file), int(q_idx)
        )
        
        # Get original bbox for correct answer
        orig_bbox = []
        for opt in options:
            if opt.get("label") == orig_correct_answer:
                orig_bbox = opt.get("bbox", [])
                break
        
        # Check if anything changed
        question_changed = (str(modified_question) != orig_question)
        answer_changed = (str(modified_correct_answer) != orig_correct_answer)
        bbox_changed = (modified_bbox != orig_bbox)
        
        # If nothing changed and not abandoned, delete correction entry
        if not question_changed and not answer_changed and not bbox_changed and not bool(abandoned):
            deleted = delete_correction(datasets_root, str(dataset), str(json_file), int(q_idx))
            if not deleted:
                cpath = _corrections_path(datasets_root, str(dataset))
                c_key = f"{json_file}__{q_idx}"
                print(f"{Fore.CYAN}No changes for {c_key}; skipped saving to {cpath}{Style.RESET_ALL}", flush=True)
            return {"status": "skipped", "deleted": bool(deleted)}
        
        # Save correction
        save_correction(
            datasets_root,
            str(dataset),
            str(json_file),
            int(q_idx),
            str(modified_question),
            str(modified_correct_answer),
            modified_bbox,
            bool(abandoned),
        )
        return {"status": "saved"}

    static_dir = os.path.join(os.path.dirname(__file__), "revise_regiongnd_tasks")
    if not os.path.isdir(static_dir):
        raise RuntimeError(f"Static directory not found: {static_dir}")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="revise_regiongnd_tasks")

    return app


def main():
    parser = argparse.ArgumentParser(description="FuncRegionGnd task revising UI server")
    parser.add_argument("--datasets-root", type=str, default=None, help="Root dir containing datasets (*/FuncRegion/grounding_mode/*.json)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=17806)
    args, _ = parser.parse_known_args()

    datasets_root = detect_datasets_root(args.datasets_root)
    app = create_app(datasets_root)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

