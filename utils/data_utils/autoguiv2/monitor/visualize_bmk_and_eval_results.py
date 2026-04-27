import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import base64
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CAPTION_TASK_TYPES = {
    'func',
    'func-w-bbox',
    'func-w-ques',
    'desc',
    'desc-w-bbox',
    'desc-w-ques',
}

def is_caption_task(metadata: Dict[str, Any], results: List[Dict[str, Any]]) -> bool:
    """Determine if the evaluation file corresponds to caption (multi-choice) tasks."""
    task_type = str((metadata or {}).get('task_type', '')).lower()
    if task_type in CAPTION_TASK_TYPES:
        return True
    return any('correct' in (r or {}) for r in results)


def extract_cache_hash_and_root(image_path: str) -> Tuple[Optional[str], Optional[Path]]:
    """Extract the cached hash directory and cache root from an image path."""
    if not image_path:
        return None, None
    path = Path(image_path).resolve()
    parts = path.parts
    if 'images' not in parts:
        return None, None
    images_idx = parts.index('images')
    if images_idx + 1 >= len(parts):
        return None, None
    cache_hash = parts[images_idx + 1]
    cache_root = Path(*parts[:images_idx])
    return cache_hash, cache_root


def resolve_caption_cache_file(
    results: List[Dict[str, Any]],
    task_type: str,
    cache_override: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the dataset cache JSON file that stores caption metadata."""
    candidate_files: List[Path] = []
    cache_dir_override: Optional[Path] = None

    if cache_override:
        cache_path = Path(cache_override).expanduser()
        if cache_path.is_file():
            candidate_files.append(cache_path)
        elif cache_path.is_dir():
            # Defer adding until we know the hash
            cache_dir_override = cache_path
        else:
            cache_dir_override = None

    first_with_image = next((r for r in results if r and r.get('image_path')), None)
    cache_hash = None
    cache_root = None
    if first_with_image:
        cache_hash, cache_root = extract_cache_hash_and_root(first_with_image.get('image_path', ''))

    if cache_hash and cache_root:
        candidate_roots = [cache_root]
        if cache_dir_override:
            candidate_roots.insert(0, cache_dir_override)
        normalized_task_types = list({
            task_type,
            task_type.replace('_', '-'),
            task_type.replace('-w-', '-'),
            task_type.replace('-wo-', '-w-'),
            task_type.split('-')[0] if '-' in task_type else task_type,
        })
        for root in candidate_roots:
            for t in normalized_task_types:
                candidate_files.append(root / f"{cache_hash}_{t}.json")
    elif cache_dir_override:
        # If we only have override directory, try to find any json there
        candidate_files.extend(sorted(cache_dir_override.glob("*.json")))

    for candidate in candidate_files:
        if candidate.exists():
            return candidate
    return None


def load_caption_entries_map(
    results: List[Dict[str, Any]],
    task_type: str,
    cache_override: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load caption dataset entries, returning a map from entry_id to metadata."""
    cache_file = resolve_caption_cache_file(results, task_type, cache_override)
    if not cache_file:
        logger.warning("Could not locate caption dataset cache file. Choices will be parsed from prompts.")
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        entries = cache_data.get('entries', [])
        if not isinstance(entries, list):
            logger.warning(f"Unexpected entries format in cache file: {cache_file}")
            return {}
        return {entry.get('entry_id'): entry for entry in entries if entry.get('entry_id')}
    except Exception as exc:
        logger.warning(f"Failed to load caption cache ({cache_file}): {exc}")
        return {}


def detect_eval_root(cli_eval_root: Optional[str]) -> str:
    candidates = []
    if cli_eval_root:
        candidates.append(cli_eval_root)
    env_dir = os.environ.get("AUTOGUI_EVAL_ROOT")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/eval_utils/autoguiv2/eval_results")
    local_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../eval_utils/autoguiv2/eval_results"))
    candidates.append(local_guess)
    logger.info(f"Eval root candidates: {candidates}")
    for c in candidates:
        if c and os.path.isdir(c):
            logger.info(f"Selected eval root: {c}")
            return c
    selected = candidates[0] if candidates else "./eval_results"
    logger.warning(f"No valid eval root found, using: {selected}")
    return selected

def list_tasks(eval_root: str) -> List[str]:
    if not os.path.isdir(eval_root):
        return []
    return sorted([d for d in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, d))])

def list_models(eval_root: str, task: str) -> List[str]:
    task_dir = os.path.join(eval_root, task)
    if not os.path.isdir(task_dir):
        return []
    return sorted([d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))])

def list_evaluations(eval_root: str, task: str, model: str) -> List[Dict[str, Any]]:
    model_dir = os.path.join(eval_root, task, model)
    if not os.path.isdir(model_dir):
        return []
    json_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
    evals = []
    for f in json_files:
        file_path = os.path.join(model_dir, f)
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        evals.append({
            "timestamp": f.replace('.json', ''),
            "updated_at": mtime
        })
    return sorted(evals, key=lambda x: x['updated_at'], reverse=True)

def read_metrics(eval_root: str, task: str, model: str, timestamp: str) -> Dict[str, Any]:
    file_path = os.path.join(eval_root, task, model, f"{timestamp}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('metrics', {})

def list_samples(eval_root: str, task: str, model: str, timestamp: str) -> List[str]:
    file_path = os.path.join(eval_root, task, model, f"{timestamp}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [r['entry_id'] for r in data.get('results', []) if 'entry_id' in r]

def get_sample_details(eval_root: str, task: str, model: str, timestamp: str, entry_id: str) -> Dict[str, Any]:
    file_path = os.path.join(eval_root, task, model, f"{timestamp}.json")
    if not os.path.exists(file_path):
        return {"error": f"Evaluation file not found: {file_path}"}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        results = data.get('results', [])
        caption_mode = is_caption_task(metadata, results)

        # Load caption entries if this is a caption task
        caption_entries_map = {}
        if caption_mode:
            task_type = metadata.get('task_type', '')
            caption_entries_map = load_caption_entries_map(results, task_type)

        for r in results:
            if r.get('entry_id') == entry_id:
                sample = r.copy()

                # For caption tasks, add target bbox from caption entry
                if caption_mode:
                    caption_entry = caption_entries_map.get(entry_id)
                    if caption_entry:
                        target_bbox = caption_entry.get('target_bbox') or caption_entry.get('target_element', {}).get('bbox', [])
                        if target_bbox and len(target_bbox) == 4:
                            # Store as gt_bbox for consistency with grounding tasks
                            sample['gt_bbox'] = target_bbox
                            # Also keep the original target_bbox for clarity
                            sample['target_bbox'] = target_bbox

                image_path = sample.get('image_path')
                if image_path and os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            # img.thumbnail((800, 800)) # Use original size
                            buffered = BytesIO()
                            img.save(buffered, format="JPEG")
                            sample['image_base64'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error loading image {image_path}: {e}")
                        sample['image_error'] = str(e)
                return sample
        return {"error": f"Sample not found: {entry_id}"}
    except Exception as e:
        logger.error(f"Error reading sample details: {e}")
        return {"error": str(e)}

def update_failure_reason(eval_root: str, task: str, model: str, timestamp: str, entry_id: str, failure_reason: str) -> Dict[str, str]:
    eval_file_path = os.path.join(eval_root, task, model, f"{timestamp}.json")
    error_file_path = os.path.join(eval_root, task, model, f"{timestamp}_error_type.json")
    
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file_path}")
    
    # Load existing errors if file exists, otherwise start with empty dict
    error_data = {}
    if os.path.exists(error_file_path):
        with open(error_file_path, 'r', encoding='utf-8') as f:
            try:
                error_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {error_file_path}, starting with empty errors")
                error_data = {}
    
    # Update or add the error for this entry
    error_data[entry_id] = failure_reason
    
    with open(error_file_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)
        
    return {"message": f"Failure reason updated in {error_file_path}"}

def get_failure_reasons(eval_root: str, task: str, model: str, timestamp: str) -> Dict[str, str]:
    # Try to load from the separate error file first
    error_file_path = os.path.join(eval_root, task, model, f"{timestamp}_error_type.json")
    if os.path.exists(error_file_path):
        with open(error_file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {error_file_path}")
                return {}

    # Fallback: try to read from original eval file if it has failure_reason fields (legacy support)
    file_path = os.path.join(eval_root, task, model, f"{timestamp}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    reasons = {}
    for r in data.get('results', []):
        entry_id = r.get('entry_id')
        if entry_id and 'failure_reason' in r:
            reasons[entry_id] = r['failure_reason']
    return reasons

class UpdateFailureRequest(BaseModel):
    task: str
    model: str
    evaluation: str
    entry_id: str
    failure_reason: str

def create_app(eval_root: str) -> FastAPI:
    app = FastAPI(title="AutoGUIv2 Evaluation Visualizer", version="1.0")

    @app.get("/api/eval-root")
    def get_eval_root():
        return {"eval_root": eval_root}

    @app.get("/api/tasks")
    def api_tasks():
        logger.info("Received request for /api/tasks")
        if not os.path.isdir(eval_root):
            logger.warning(f"Eval root not found: {eval_root}")
            return {"tasks": [], "message": "Evaluation root directory not found or empty"}
        tasks = list_tasks(eval_root)
        logger.info(f"Returning {len(tasks)} tasks")
        return {"tasks": tasks}

    @app.get("/api/models")
    def api_models(task: str):
        logger.info(f"Received request for /api/models?task={task}")
        models = list_models(eval_root, task)
        logger.info(f"Returning {len(models)} models for task {task}")
        return {"models": models}

    @app.get("/api/evaluations")
    def api_evaluations(task: str, model: str):
        logger.info(f"Received request for /api/evaluations?task={task}&model={model}")
        evals = list_evaluations(eval_root, task, model)
        logger.info(f"Returning {len(evals)} evaluations for task {task}, model {model}")
        return {"evaluations": evals}

    @app.get("/api/metrics")
    def api_metrics(task: str, model: str, evaluation: str):
        logger.info(f"Received request for /api/metrics?task={task}&model={model}&evaluation={evaluation}")
        try:
            metrics = read_metrics(eval_root, task, model, evaluation)
            logger.info(f"Returning metrics for task {task}, model {model}, evaluation {evaluation}")
            return JSONResponse({"metrics": metrics})
        except FileNotFoundError:
            logger.warning(f"Metrics not found for task {task}, model {model}, evaluation {evaluation}")
            raise HTTPException(status_code=404, detail="Metrics not found")

    @app.get("/api/samples")
    def api_samples(task: str, model: str, evaluation: str):
        logger.info(f"Received request for /api/samples?task={task}&model={model}&evaluation={evaluation}")
        try:
            samples = list_samples(eval_root, task, model, evaluation)
            logger.info(f"Returning {len(samples)} samples for task {task}, model {model}, evaluation {evaluation}")
            return {"samples": samples}
        except FileNotFoundError:
            logger.warning(f"Evaluation not found for task {task}, model {model}, evaluation {evaluation}")
            raise HTTPException(status_code=404, detail="Evaluation not found")

    @app.get("/api/sample")
    def api_sample(task: str, model: str, evaluation: str, entry_id: str):
        logger.info(f"Received request for /api/sample?task={task}&model={model}&evaluation={evaluation}&entry_id={entry_id}")
        try:
            sample = get_sample_details(eval_root, task, model, evaluation, entry_id)
            logger.info(f"Returning sample details for task {task}, model {model}, evaluation {evaluation}, entry_id {entry_id}")
            return JSONResponse(sample)
        except FileNotFoundError:
            logger.warning(f"Sample not found for task {task}, model {model}, evaluation {evaluation}, entry_id {entry_id}")
            raise HTTPException(status_code=404, detail="Sample not found")

    @app.post("/api/update-failure")
    def api_update_failure(req: UpdateFailureRequest):
        task, model, evaluation, entry_id, failure_reason = req.task, req.model, req.evaluation, req.entry_id, req.failure_reason
        logger.info(f"Received request for /api/update-failure?task={task}&model={model}&evaluation={evaluation}&entry_id={entry_id}&failure_reason={failure_reason}")
        
        # Construct the full path to the error file being updated
        error_file_path = os.path.abspath(os.path.join(eval_root, task, model, f"{evaluation}_error_type.json"))
        print(f"Saving failure reason to file: {error_file_path}")
        print(f"Target entry: {entry_id}")
        
        try:
            result = update_failure_reason(eval_root, task, model, evaluation, entry_id, failure_reason)
            logger.info(f"Failure reason updated for task {task}, model {model}, evaluation {evaluation}, entry_id {entry_id}")
            return result
        except HTTPException as e:
            logger.warning(f"HTTPException during update-failure for task {task}, model {model}, evaluation {evaluation}, entry_id {entry_id}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during update-failure for task {task}, model {model}, evaluation {evaluation}, entry_id {entry_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/failure-reasons")
    def api_failure_reasons(task: str, model: str, evaluation: str):
        logger.info(f"Received request for /api/failure-reasons?task={task}&model={model}&evaluation={evaluation}")
        try:
            reasons = get_failure_reasons(eval_root, task, model, evaluation)
            logger.info(f"Returning {len(reasons)} failure reasons for task {task}, model {model}, evaluation {evaluation}")
            return {"reasons": reasons}
        except FileNotFoundError:
            logger.warning(f"Evaluation not found for failure-reasons for task {task}, model {model}, evaluation {evaluation}")
            raise HTTPException(status_code=404, detail="Evaluation not found")

    @app.get("/README.md")
    def serve_readme():
      readme_path = os.path.join(os.path.dirname(__file__), '..', 'README.md')
      if os.path.exists(readme_path):
        return FileResponse(readme_path)
      raise HTTPException(status_code=404, detail="README not found")

    @app.get("/health")
    def health():
      return {"status": "ok", "eval_root": eval_root, "static_dir": static_dir}

    static_dir = os.path.join(os.path.dirname(__file__), "static_visualize")
    if not os.path.isdir(static_dir):
      logger.error(f"Static directory not found: {static_dir}")
      @app.get("/")
      def root():
        return {"message": "Static directory not found. Check installation."}
    else:
      app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_visualize")

    return app

def main():
    parser = argparse.ArgumentParser(description="AutoGUIv2 Evaluation Visualizer Server")
    parser.add_argument("--eval-root", type=str, default=None, help="Evaluation results root directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=17802, help="Port to bind")
    args, _ = parser.parse_known_args()

    eval_root = detect_eval_root(args.eval_root)
    if not os.path.isdir(eval_root):
        print(f"[WARN] Eval root not found: {eval_root}")
    
    app = create_app(eval_root)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
