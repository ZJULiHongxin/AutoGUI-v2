import os
import json
import argparse
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse


def detect_cache_dir(cli_cache_dir: Optional[str]) -> str:
    """Auto-detect cache directory with reasonable fallbacks."""
    candidates: List[str] = []

    # 1) CLI flag has highest priority
    if cli_cache_dir:
        candidates.append(cli_cache_dir)

    # 2) Environment variable
    env_dir = os.environ.get("AUTOGUI_CACHE_DIR")
    if env_dir:
        candidates.append(env_dir)

    # 3) Common default used by annotator when not specified
    candidates.append("/mnt/jfs/copilot/lhx/ui_data/AutoGUIv2/cache")

    # 4) Repo-local default: ../../../../ui_data/AutoGUIv2/cache relative to this file
    local_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..", "ui_data/AutoGUIv2/cache"))
    candidates.append(local_guess)

    for c in candidates:
        if c and os.path.isdir(c):
            return c

    # If none exist, return first non-empty candidate so the user sees the intended path in error
    for c in candidates:
        if c:
            return c
    return "./cache"


def list_namespaces(cache_root: str) -> List[str]:
    if not os.path.isdir(cache_root):
        return []
    return sorted([d for d in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, d))])


def list_models(cache_root: str, namespace: str) -> List[str]:
    ns_dir = os.path.join(cache_root, namespace)
    if not os.path.isdir(ns_dir):
        return []
    return sorted([d for d in os.listdir(ns_dir) if os.path.isdir(os.path.join(ns_dir, d))])


def list_versions(cache_root: str, namespace: str, model_name: str) -> List[str]:
    model_dir = os.path.join(cache_root, namespace, model_name)
    if not os.path.isdir(model_dir):
        return []
    return sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])


def _find_image_base_dir(cache_root: str, namespace: str, model_name: str, version: str, image_id: str) -> Optional[str]:
    """Recursively find the base directory containing tree.json or nodes/ directory.

    Args:
        image_id: Can be a simple name (e.g., 'com.adsk.sketchbook') or a full path
                  (e.g., 'com.adsk.sketchbook/1194/1') relative to version_dir. Some
                  datasets insert additional categorisation directories that are not
                  reflected in the stored image_id, so we search broadly when a direct
                  join fails.
    """
    version_dir = os.path.join(cache_root, namespace, model_name, version)
    if not os.path.isdir(version_dir):
        return None

    # Start from the image_id directory (handles both simple names and nested paths)
    image_id = (image_id or "").strip("/")
    image_dir = os.path.join(version_dir, image_id)
    if not os.path.isdir(image_dir):
        image_dir = None

    def _validate_dir(candidate: str) -> Optional[str]:
        if not candidate or not os.path.isdir(candidate):
            return None
        if os.path.exists(os.path.join(candidate, "tree.json")):
            return candidate
        for root, dirs, files in os.walk(candidate):
            if "tree.json" in files or "nodes" in dirs:
                return root
        return None

    candidate = _validate_dir(image_dir)
    if candidate:
        return candidate

    # Fallback: walk the entire version directory looking for a path whose tail matches image_id
    normalized_parts = [part for part in image_id.split("/") if part]
    best_match: Optional[str] = None

    for root, dirs, files in os.walk(version_dir):
        rel_path = os.path.relpath(root, version_dir)
        if rel_path == ".":
            continue
        rel_parts = [part for part in rel_path.split(os.sep) if part]
        if not rel_parts:
            continue

        if normalized_parts and len(rel_parts) >= len(normalized_parts):
            if rel_parts[-len(normalized_parts):] == normalized_parts:
                candidate = _validate_dir(root)
                if candidate:
                    best_match = candidate
                    break

        # Also accept exact match when image_id is shorter/empty
        if rel_path == image_id:
            candidate = _validate_dir(root)
            if candidate:
                best_match = candidate
                break

    return best_match


def _find_nodes_dir_recursive(base_dir: str) -> Optional[str]:
    """Recursively find the nodes directory starting from base_dir."""
    if not base_dir or not os.path.isdir(base_dir):
        return None
    
    # Check direct nodes directory
    nodes_dir = os.path.join(base_dir, "nodes")
    if os.path.isdir(nodes_dir):
        return nodes_dir
    
    # Recursively search for nodes directory
    for root, dirs, files in os.walk(base_dir):
        if "nodes" in dirs:
            return os.path.join(root, "nodes")
    
    return None


def _find_node_file_recursive(search_dir: str, node_id: str, patterns: List[str]) -> Optional[str]:
    """Recursively find a node file matching any of the given patterns."""
    if not search_dir or not os.path.isdir(search_dir):
        return None
    
    # Try direct patterns first
    for pattern in patterns:
        candidate = os.path.join(search_dir, pattern.format(node_id=node_id))
        if os.path.exists(candidate):
            return candidate
    
    # Recursively search
    for root, dirs, files in os.walk(search_dir):
        for pattern in patterns:
            candidate = os.path.join(root, pattern.format(node_id=node_id))
            if os.path.exists(candidate):
                return candidate
    
    return None


def count_correction_files(cache_root: str, namespace: str, model_name: str, version: str, image_id: str) -> int:
    """Count the number of correction files for a specific image, searching recursively.

    Supports both naming patterns:
    - {node_id}_meta_fix*.json (correction-only files)
    - {node_id}_fix*.json (full metadata files)
    """
    base_dir = _find_image_base_dir(cache_root, namespace, model_name, version, image_id)
    if not base_dir:
        return 0
    
    # Find nodes directory recursively
    nodes_dir = _find_nodes_dir_recursive(base_dir)
    if not nodes_dir:
        return 0
    
    # Search recursively for correction files (both patterns)
    correction_files: List[str] = []
    for pattern in ["*_meta_fix*.json", "*_fix*.json"]:
        correction_files.extend(glob.glob(os.path.join(nodes_dir, "**", pattern), recursive=True))

    # Extract unique node IDs from correction filenames
    used: List[str] = []
    for file in correction_files:
        basename = os.path.basename(file)
        if "_meta_fix" in basename:
            node_id = basename.split("_meta_fix")[0]
        elif "_fix" in basename:
            node_id = basename.split("_fix")[0]
        else:
            continue
        if node_id and node_id not in used:
            used.append(node_id)
    
    return len(used)


def _pick_default_version(cache_root: str, namespace: str, model_name: str) -> Optional[str]:
    versions = list_versions(cache_root, namespace, model_name)
    if not versions:
        return None
    # pick the last (e.g., v2 > v1) in lexicographic order as default
    return versions[-1]


def list_images(cache_root: str, namespace: str, model_name: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
    # Resolve version if not provided
    use_version = version or _pick_default_version(cache_root, namespace, model_name)
    if not use_version:
        return []
    version_dir = os.path.join(cache_root, namespace, model_name, use_version)
    if not os.path.isdir(version_dir):
        return []
    images: List[Dict[str, Any]] = []
    
    # Collect all potential image directories recursively
    # Use a dict to track the best base_dir for each unique path
    image_dirs_dict = {}
    for root, dirs, files in os.walk(version_dir):
        # Extract image_id from path (relative to version_dir) - use full path
        rel_path = os.path.relpath(root, version_dir)
        # Skip the root directory itself (returns ".")
        if rel_path == ".":
            continue
        # Use the full relative path as image_id to support nested structures
        image_id = rel_path
        
        # Check if this directory contains tree.json or nodes directory
        has_tree = "tree.json" in files
        has_nodes = "nodes" in dirs
        
        if has_tree or has_nodes:
            # Only include directories that actually have tree.json or nodes
            # If it's a new path, add it. If we already have this path, prefer the one with tree.json
            if image_id not in image_dirs_dict:
                image_dirs_dict[image_id] = root
            elif has_tree:
                # Update if current has tree.json but existing doesn't
                existing_dir = image_dirs_dict[image_id]
                if not os.path.exists(os.path.join(existing_dir, "tree.json")):
                    image_dirs_dict[image_id] = root
    
    # Convert dict to list
    image_dirs = [{"image_id": img_id, "base_dir": base_dir} for img_id, base_dir in image_dirs_dict.items()]
    
    for img_info in sorted(image_dirs, key=lambda x: x["image_id"]):
        image_id = img_info["image_id"]
        base_dir = img_info["base_dir"]
        
        # Skip the images with 8-digit hash id.
        has_hashid = len(os.path.basename(base_dir).split('-')[-1]) == 8
        if has_hashid:
            continue
        
        tree_path = os.path.join(base_dir, "tree.json")
        stack_path = os.path.join(base_dir, "stack.json")
        nodes_dir = _find_nodes_dir_recursive(base_dir)
        root_img_path = os.path.join(base_dir, "root.png")
        
        # Calculate relative path for root_image_url
        rel_path = os.path.relpath(base_dir, cache_root)
        root_image_url = f"/_cache/{rel_path}/root.png" if os.path.exists(root_img_path) else None
        
        summary = {
            "namespace": namespace,
            "model_name": model_name,
            "version": use_version,
            "image_id": image_id,
            "root_image_url": root_image_url,
            "nodes_count": 0,
            "stack_len": 0,
            "corrections_count": 0,
            "updated_at": None,
        }
        try:
            if os.path.exists(tree_path):
                with open(tree_path, "r", encoding="utf-8") as f:
                    tree = json.load(f)
                summary["nodes_count"] = len(tree) if isinstance(tree, dict) else 0
                mtime = datetime.fromtimestamp(os.path.getmtime(tree_path)).isoformat()
                summary["updated_at"] = mtime
            if os.path.exists(stack_path):
                with open(stack_path, "r", encoding="utf-8") as f:
                    stack = json.load(f)
                summary["stack_len"] = len(stack) if isinstance(stack, list) else 0
            if nodes_dir and summary["nodes_count"] == 0:
                # Count JSON files recursively
                json_files = glob.glob(os.path.join(nodes_dir, "**", "*.json"), recursive=True)
                # Filter out correction files and region-type files
                json_files = [f for f in json_files if "_meta_fix" not in os.path.basename(f) and "_region-type" not in os.path.basename(f)]
                summary["nodes_count"] = len(json_files)
            
            # Count correction files
            summary["corrections_count"] = count_correction_files(cache_root, namespace, model_name, use_version, image_id)
        except Exception:
            pass

        # Skip samples with 0 nodes
        if summary["nodes_count"] <= 9:
            continue

        images.append(summary)
    return images


def _resolve_version_for_image(cache_root: str, namespace: str, model_name: str, image_id: str) -> Optional[str]:
    for ver in list_versions(cache_root, namespace, model_name):
        base_dir = _find_image_base_dir(cache_root, namespace, model_name, ver, image_id)
        if base_dir and os.path.exists(os.path.join(base_dir, "tree.json")):
            return ver
    return None


def read_tree(cache_root: str, namespace: str, model_name: str, image_id: str, version: Optional[str] = None) -> Dict[str, Any]:
    ver = version or _resolve_version_for_image(cache_root, namespace, model_name, image_id)
    if not ver:
        raise FileNotFoundError("tree.json not found")
    
    base_dir = _find_image_base_dir(cache_root, namespace, model_name, ver, image_id)
    if not base_dir:
        raise FileNotFoundError("tree.json not found")
    
    tree_path = os.path.join(base_dir, "tree.json")
    stack_path = os.path.join(base_dir, "stack.json")
    if not os.path.exists(tree_path):
        raise FileNotFoundError("tree.json not found")
    
    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)
    stack = []
    if os.path.exists(stack_path):
        try:
            with open(stack_path, "r", encoding="utf-8") as f:
                stack = json.load(f)
        except Exception:
            stack = []
    mtime = datetime.fromtimestamp(os.path.getmtime(tree_path)).isoformat()
    root_img_path = os.path.join(base_dir, "root.png")
    
    # Calculate relative path for root_image_url
    rel_path = os.path.relpath(base_dir, cache_root)
    root_image_url = f"/_cache/{rel_path}/root.png" if os.path.exists(root_img_path) else None
    
    return {"tree": tree, "stack": stack, "updated_at": mtime, "root_image_url": root_image_url}

  
def find_node_image_url(cache_root: str, namespace: str, model_name: str, version: str, image_id: str, node_id: str) -> Optional[str]:
    base_dir = _find_image_base_dir(cache_root, namespace, model_name, version, image_id)
    if not base_dir:
        return None
    
    nodes_dir = _find_nodes_dir_recursive(base_dir)
    if not nodes_dir:
        return None
    
    # Try common extensions recursively
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        # Search recursively for files matching the pattern
        patterns = [
            f"{node_id}_crop{ext}",
            f"{node_id}{ext}"
        ]
        for pattern in patterns:
            found_files = glob.glob(os.path.join(nodes_dir, "**", pattern), recursive=True)
            if found_files:
                found_file = found_files[0]
                # Calculate relative path from cache_root
                rel_path = os.path.relpath(found_file, cache_root)
                return f"/_cache/{rel_path}"
    
    # Fallback: pick any file starting with node_id recursively
    for root, dirs, files in os.walk(nodes_dir):
        for f in files:
            if f.startswith(node_id + "."):
                found_file = os.path.join(root, f)
                rel_path = os.path.relpath(found_file, cache_root)
                return f"/_cache/{rel_path}"
    
    return None


def find_region_types_file(cache_root: str, namespace: str, model_name: str, version: Optional[str] = None) -> Optional[str]:
    """Find region types file for the given namespace and model."""
    # Look for region types file in various common locations
    # Based on the classify_functional_regions.py output patterns
    search_dirs = [
        os.path.join(cache_root, ".."),  # Parent of cache dir
        os.path.join(cache_root, "..", ".."),  # Grandparent of cache dir
        cache_root,  # Cache dir itself
    ]

    candidates = []

    # Generate candidate file patterns
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        # Common patterns based on classify_functional_regions.py
        patterns = [
            f"functional_regions_*_region_types_{model_name}.json",
            f"functional_regions_{model_name}_region_types_*.json",
            f"*region_types_{model_name}.json",
            f"functional_regions_*region_types*.json",
            f"*{model_name}*region_types*.json",
        ]

        for pattern in patterns:
            candidates.append(os.path.join(search_dir, pattern))

        # Also look for any JSON file containing "region_types" in the name
        candidates.append(os.path.join(search_dir, "*region_types*.json"))

    # Check all candidates
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        # Try glob pattern
        if "*" in candidate:
            matches = glob.glob(candidate)
            if matches:
                # Return the most recently modified file
                return max(matches, key=os.path.getmtime)

    return None


# Global cache for region types data
_region_types_cache: Dict[str, Dict[str, Any]] = {}


def load_region_types_file(cache_root: str, namespace: str, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load and cache the entire region types file."""
    # Create a cache key based on the file path
    cache_key = f"{cache_root}:{namespace}:{model_name}"

    # Check if already cached
    if cache_key in _region_types_cache:
        return _region_types_cache[cache_key]

    # Find the region types file
    region_types_file = find_region_types_file(cache_root, namespace, model_name, version)
    if not region_types_file:
        return None

    try:
        with open(region_types_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Cache the data
        _region_types_cache[cache_key] = data
        return data

    except Exception:
        return None


def get_region_types_for_image(cache_root: str, namespace: str, model_name: str, image_id: str, version: Optional[str] = None) -> Dict[str, str]:
    """Get region types for a specific image from the cached data."""
    data = load_region_types_file(cache_root, namespace, model_name, version)
    if not data:
        return {}

    # Extract region types for this specific image
    if isinstance(data, dict) and "results" in data:
        results = data["results"]

        # Try to find the image by various matching strategies
        if image_id in results:
            sample_data = results[image_id]
            if isinstance(sample_data, dict) and "region_types" in sample_data:
                region_types = sample_data["region_types"]
                # Handle both old and new formats
                if isinstance(region_types, dict):
                    # Check if values are strings or dicts
                    first_value = next(iter(region_types.values()), None)
                    if isinstance(first_value, str):
                        return region_types
                    elif isinstance(first_value, dict) and "type" in first_value:
                        # Convert dict format to string format
                        return {k: v.get("type", "") for k, v in region_types.items()}

        # If exact match not found, try partial matching
        for key, sample_data in results.items():
            if isinstance(sample_data, dict) and "region_types" in sample_data:
                region_types = sample_data["region_types"]
                if isinstance(region_types, dict):
                    # Check if any node_id in this sample matches our image_id pattern
                    first_value = next(iter(region_types.values()), None)
                    if isinstance(first_value, str):
                        return region_types
                    elif isinstance(first_value, dict) and "type" in first_value:
                        return {k: v.get("type", "") for k, v in region_types.items()}

    return {}


def read_node(cache_root: str, namespace: str, model_name: str, image_id: str, node_id: str, version: Optional[str] = None) -> Dict[str, Any]:
    # Resolve version if not provided
    ver = version or _resolve_version_for_image(cache_root, namespace, model_name, image_id)
    if not ver:
        raise FileNotFoundError("node meta not found")
    
    base_dir = _find_image_base_dir(cache_root, namespace, model_name, ver, image_id)
    if not base_dir:
        raise FileNotFoundError("node meta not found")
    
    nodes_dir = _find_nodes_dir_recursive(base_dir)
    if not nodes_dir:
        raise FileNotFoundError("node meta not found")
    
    # Find node meta file recursively
    node_meta_path = _find_node_file_recursive(nodes_dir, node_id, [
        "{node_id}_meta.json",
        "{node_id}.json"
    ])
    
    if not node_meta_path:
        raise FileNotFoundError("node meta not found")
    
    with open(node_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["node_id"] = node_id
    meta["image_url"] = find_node_image_url(cache_root, namespace, model_name, ver, image_id, node_id)

    # Normalize functionality/description fields for downstream usage and keep originals for tooltips
    functionality_field = meta.get("functionality")
    if isinstance(functionality_field, dict):
        original_functionality = functionality_field.get("with_context") or functionality_field.get("wo_context")
        functionality_field.setdefault("with_context", original_functionality)
    elif isinstance(functionality_field, str):
        original_functionality = functionality_field
        meta["functionality"] = {"with_context": original_functionality, "wo_context": None}
    else:
        original_functionality = None
        meta["functionality"] = {"with_context": None, "wo_context": None}

    description_field = meta.get("description")
    if isinstance(description_field, dict):
        original_description = description_field.get("with_context") or description_field.get("wo_context")
        description_field.setdefault("with_context", original_description)
    elif isinstance(description_field, str):
        original_description = description_field
        meta["description"] = {"with_context": original_description, "wo_context": None}
    else:
        original_description = None
        meta["description"] = {"with_context": None, "wo_context": None}

    meta["functionality_revised"] = False
    meta["description_revised"] = False
    
    # Add root image path for correction modal
    root_img_path = os.path.join(base_dir, "root.png")
    if os.path.exists(root_img_path):
        rel_path = os.path.relpath(root_img_path, cache_root)
        meta["root_image_path"] = f"/_cache/{rel_path}"

    # Find the latest correction file recursively (support both naming patterns)
    correction_candidates: List[str] = []
    for pattern in [f"{node_id}_meta_fix*.json", f"{node_id}_fix*.json"]:
        correction_candidates.extend(glob.glob(os.path.join(nodes_dir, "**", pattern), recursive=True))
    if correction_candidates:
        try:
            latest_correction_file = max(correction_candidates, key=os.path.getmtime)
            with open(latest_correction_file, "r", encoding="utf-8") as f:
                correction_data = json.load(f)
            # Handle both formats
            if isinstance(correction_data, dict):
                if "new_bbox" in correction_data:
                    meta["corrected_bbox"] = correction_data["new_bbox"]
                elif "bbox_global" in correction_data:
                    meta["corrected_bbox"] = correction_data["bbox_global"]
        except Exception:
            pass  # Ignore malformed files or read errors

    # Load and add region type from per-node file (search recursively)
    region_type_path = _find_node_file_recursive(nodes_dir, node_id, [
        "{node_id}_region-type.json"
    ])
    if region_type_path and os.path.exists(region_type_path):
        with open(region_type_path, "r", encoding="utf-8") as f:
            region_data = json.load(f)
        if isinstance(region_data, dict):
            meta["region_type"] = region_data.get('type', "")
        else:
            meta["region_type"] = str(region_data)
    else:
        meta["region_type"] = ""

    # Load reannotated data if available (prefer latest in same directory)
    node_dir = os.path.dirname(node_meta_path)
    reannotated_files = sorted(glob.glob(os.path.join(node_dir, f"{node_id}_meta_reannotated*.json")))
    if not reannotated_files:
        fallback_reannotated = _find_node_file_recursive(nodes_dir, node_id, [
            "{node_id}_meta_reannotated*.json"
        ])
        if fallback_reannotated and os.path.exists(fallback_reannotated):
            reannotated_files = [fallback_reannotated]

    if reannotated_files:
        reannotated_path = reannotated_files[-1]
        try:
            with open(reannotated_path, "r", encoding="utf-8") as f:
                reannotated_data = json.load(f)

            new_func_data = reannotated_data.get("new_functionality") if isinstance(reannotated_data, dict) else None
            if isinstance(new_func_data, dict):
                revised_functionality = new_func_data.get("revised functionality")
                revised_description = new_func_data.get("revised description")

                if revised_functionality:
                    meta["original_functionality"] = original_functionality
                    meta["functionality"]["with_context"] = revised_functionality
                    meta["functionality_revised"] = revised_functionality != (original_functionality or "")

                if revised_description:
                    meta["original_description"] = original_description
                    meta["description"]["with_context"] = revised_description
                    meta["description_revised"] = revised_description != (original_description or "")

                if "revision rationale" in new_func_data:
                    meta["revision_rationale"] = new_func_data.get("revision rationale")
        except Exception:
            pass  # Ignore if file is malformed or other errors occur

    if meta.get("functionality_revised") and not meta.get("original_functionality"):
        meta["original_functionality"] = original_functionality
    if meta.get("description_revised") and not meta.get("original_description"):
        meta["original_description"] = original_description
 
    return meta

def save_corrected_node(cache_root: str, namespace: str, model_name: str, image_id: str, node_id: str, version: str, new_bbox: List[int]) -> str:
    # 1. Find the base directory and nodes directory recursively
    base_dir = _find_image_base_dir(cache_root, namespace, model_name, version, image_id)
    if not base_dir:
        raise FileNotFoundError("node meta not found")
    
    nodes_dir = _find_nodes_dir_recursive(base_dir)
    if not nodes_dir:
        raise FileNotFoundError("node meta not found")
    
    # 2. Find and read the original node meta file recursively
    node_meta_path = _find_node_file_recursive(nodes_dir, node_id, [
        "{node_id}_meta.json",
        "{node_id}.json"
    ])
    
    if not node_meta_path:
        raise FileNotFoundError("node meta not found")
    
    with open(node_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3. Update the bounding box coordinates
    original_bbox = meta.get("bbox_global")
    meta["bbox_global"] = new_bbox
    
    # Also update normalized bbox
    if "root_size(wxh)" in meta:
        w, h = meta["root_size(wxh)"]
        if w > 0 and h > 0:
            meta["bbox_global_norm"] = [
                new_bbox[0] / w,
                new_bbox[1] / h,
                new_bbox[2] / w,
                new_bbox[3] / h
            ]

    # Add a note about the correction
    meta["correction_info"] = {
        "original_bbox_global": original_bbox,
        "corrected_at": datetime.now().isoformat(),
        "source": "manual_correction_ui"
    }

    # 4. Create a new filename with a timestamp (save in the same directory as the original)
    timestamp = datetime.now().strftime("%Y%m%d")
    base, ext = os.path.splitext(node_meta_path)
    new_file_path = f"{base}_fix{timestamp}{ext}"

    # 5. Save the updated data to the new file
    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return new_file_path


def create_app(cache_root: str) -> FastAPI:
    app = FastAPI(title="AutoGUIv2 Annotation Monitor", version="1.0")

    # Mount cache so UI can load images directly
    if os.path.isdir(cache_root):
        app.mount("/_cache", StaticFiles(directory=cache_root), name="cache")

    @app.get("/api/cache-root")
    def get_cache_root():
        return {"cache_root": cache_root}

    @app.get("/api/namespaces")
    def api_namespaces():
        return {"namespaces": list_namespaces(cache_root)}

    @app.get("/api/models")
    def api_models(namespace: str):
        return {"models": list_models(cache_root, namespace)}

    @app.get("/api/versions")
    def api_versions(namespace: str, model_name: str):
        return {"versions": list_versions(cache_root, namespace, model_name)}

    @app.get("/api/images")
    def api_images(namespace: str, model_name: str, version: Optional[str] = None):
        return {"images": list_images(cache_root, namespace, model_name, version)}

    @app.get("/api/image/{namespace}/{model_name}/tree")
    def api_tree(namespace: str, model_name: str, image_id: str):
        try:
            payload = read_tree(cache_root, namespace, model_name, image_id)
            return JSONResponse(payload)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="tree not found")

    @app.get("/api/image/{namespace}/{model_name}/node/{node_id}")
    def api_node(namespace: str, model_name: str, node_id: str, image_id: str):
        try:
            payload = read_node(cache_root, namespace, model_name, image_id, node_id)
            return JSONResponse(payload)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="node not found")

    @app.post("/api/image/{namespace}/{model_name}/{version}/node/{node_id}/correct")
    def correct_node_bbox(namespace: str, model_name: str, version: str, node_id: str, image_id: Optional[str] = None, body: Dict[str, Any] = Body(...)):
        new_bbox = body.get("new_bbox")
        if not new_bbox or len(new_bbox) != 4:
            raise HTTPException(status_code=400, detail="Invalid 'new_bbox' provided")

        target_image_id = image_id or body.get("image_id")
        if not target_image_id:
            raise HTTPException(status_code=400, detail="'image_id' is required")

        try:
            new_file_path = save_corrected_node(cache_root, namespace, model_name, target_image_id, node_id, version, new_bbox)
            return {"message": "Correction saved", "new_file": os.path.basename(new_file_path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save correction: {e}")

    # Versioned endpoints - image_id is passed as query parameter to handle paths with slashes
    @app.get("/api/image/{namespace}/{model_name}/{version}/tree")
    def api_tree_v2(namespace: str, model_name: str, version: str, image_id: str):
        try:
            payload = read_tree(cache_root, namespace, model_name, image_id, version)
            return JSONResponse(payload)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="tree not found")

    @app.get("/api/image/{namespace}/{model_name}/{version}/node/{node_id}")
    def api_node_v2(namespace: str, model_name: str, version: str, node_id: str, image_id: str):
        try:
            payload = read_node(cache_root, namespace, model_name, image_id, node_id, version)
            return JSONResponse(payload)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="node not found")

    @app.post("/api/image/{namespace}/{model_name}/{version}/node/{node_id}/correct")
    async def api_correct_node(namespace: str, model_name: str, version: str, node_id: str, request: Request):
        try:
            data = await request.json()
            new_bbox = data.get("new_bbox")
            image_id = data.get("image_id")
            if not new_bbox or len(new_bbox) != 4:
                raise HTTPException(status_code=400, detail="Invalid new_bbox format")
            if not image_id:
                raise HTTPException(status_code=400, detail="image_id is required")

            new_file = save_corrected_node(cache_root, namespace, model_name, image_id, node_id, version, new_bbox)
            return {"message": "Correction saved", "new_file": new_file}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Node meta file not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    # Mount frontend LAST so API routes take precedence
    static_dir = os.path.join(os.path.dirname(__file__), "static_bboxcorrection_v2")
    
    assert os.path.exists(static_dir), f"Static directory not found: {static_dir}"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_bboxcorrection_v2")

    return app


def main():
    parser = argparse.ArgumentParser(description="AutoGUIv2 Annotation Monitor Server")
    parser.add_argument("--cache-dir", type=str, default="/mnt/vdb1/hongxin_li/AutoGUIv2/cache/", help="Cache directory. Auto-detected if not provided.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=17800, help="Port to bind")
    args, _ = parser.parse_known_args()

    cache_root = detect_cache_dir(args.cache_dir)
    if not os.path.isdir(cache_root):
        print(f"[WARN] Cache dir not found: {cache_root}. The UI will load but show no data until cache is created.")

    app = create_app(cache_root)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()