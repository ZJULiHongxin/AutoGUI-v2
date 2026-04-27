#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为生成的问题数据添加 area_class 和 density_class 指标

复用 FuncRegionGnd_eval_gen/make_funcreg_gnd_samples.py 中的计算工具：
- GUISizeClassifier: 计算目标功能区面积占比
- EnhancedNIDAnalyzer: 计算目标区域周围密集度
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.append('/'.join(__file__.split('/')[:-4]))

# Import the calculation tools
sys.path.append('/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/data_utils/autoguiv2/FuncRegionGnd_eval_gen')
from make_funcreg_gnd_samples import GUISizeClassifier, EnhancedNIDAnalyzer


def load_androidcontrol_from_cache(cache_dir: str, debug: bool = False) -> Dict:
    """从 androidcontrol cache 目录直接加载数据（包括修订后的bbox和功能描述）
    
    复用 gen_region-func_multichoice-qa_aliyun_androidcontrol.py 中的加载逻辑
    """
    if not os.path.exists(cache_dir) or not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    if debug:
        print(f"📂 Loading androidcontrol data from cache directory: {cache_dir}")
    
    results = {}
    bbox_correction_count = 0
    reannotation_count = 0
    
    # Scan all image directories: {app_name}/{episode_id}/{step_id}/
    app_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
    
    for app_name in app_dirs:
        app_path = os.path.join(cache_dir, app_name)
        episode_dirs = [d for d in os.listdir(app_path) if os.path.isdir(os.path.join(app_path, d))]
        
        for episode_id in episode_dirs:
            episode_path = os.path.join(app_path, episode_id)
            step_dirs = [d for d in os.listdir(episode_path) if os.path.isdir(os.path.join(episode_path, d))]
            
            for step_id in step_dirs:
                image_dir = os.path.join(episode_path, step_id)
                tree_json_path = os.path.join(image_dir, "tree.json")
                root_png_path = os.path.join(image_dir, "root.png")
                nodes_dir = os.path.join(image_dir, "nodes")
                
                # Check if tree.json exists
                if not os.path.exists(tree_json_path):
                    if debug:
                        print(f"   Skipping {app_name}/{episode_id}/{step_id}: tree.json not found")
                    continue
                
                # Load tree.json
                try:
                    with open(tree_json_path, 'r', encoding='utf-8') as f:
                        tree_data = json.load(f)
                except Exception as e:
                    if debug:
                        print(f"   Failed to load tree.json for {app_name}/{episode_id}/{step_id}: {e}")
                    continue
                
                # Construct image_key (relative path from cache root)
                image_key = f"{app_name}/{episode_id}/{step_id}"
                
                # Use root.png path if exists, otherwise try to infer from tree_data
                if os.path.exists(root_png_path):
                    root_image_path = root_png_path
                else:
                    # Try to get from tree_data (0-0 node usually has root_image_path)
                    root_image_path = tree_data.get('0-0', {}).get('root_image_path', root_png_path)
                
                # Store image data
                results[image_key] = tree_data.copy()
                results[image_key]['root_image_path'] = root_image_path
                
                # Load reannotations from nodes directory
                if os.path.isdir(nodes_dir):
                    for node_id in tree_data.keys():
                        if not isinstance(tree_data[node_id], dict):
                            continue
                        
                        # Find reannotation files
                        reannotation_files = glob.glob(os.path.join(nodes_dir, f"{node_id}_meta_reannotated*.json"))
                        
                        if reannotation_files:
                            latest_reannotation = sorted(reannotation_files)[-1]
                            
                            try:
                                with open(latest_reannotation, 'r', encoding='utf-8') as rf:
                                    reannotation_data = json.load(rf)
                                
                                # Extract corrected bbox
                                corrected_bbox = reannotation_data.get('corrected_bbox')
                                if corrected_bbox and len(corrected_bbox) == 4:
                                    original_bbox = tree_data[node_id].get('bbox_global')
                                    
                                    # If corrected_bbox differs from original, update the bbox
                                    if original_bbox != corrected_bbox:
                                        results[image_key][node_id]['bbox_global'] = corrected_bbox
                                        
                                        if 'root_size(wxh)' in results[image_key][node_id]:
                                            w, h = results[image_key][node_id]['root_size(wxh)']
                                            if w > 0 and h > 0:
                                                results[image_key][node_id]['bbox_global_norm'] = [
                                                    corrected_bbox[0] / w,
                                                    corrected_bbox[1] / h,
                                                    corrected_bbox[2] / w,
                                                    corrected_bbox[3] / h
                                                ]
                                    
                                    # Mark as corrected
                                    results[image_key][node_id]['bbox_corrected'] = True
                                    bbox_correction_count += 1
                                
                                # Extract revised functionality and description
                                new_functionality = reannotation_data.get('new_functionality', {})
                                if isinstance(new_functionality, dict):
                                    revised_func = new_functionality.get('revised functionality')
                                    revised_desc = new_functionality.get('revised description')
                                    
                                    # Update functionality if available
                                    if revised_func:
                                        # Store old functionality for reference
                                        if 'functionality' in results[image_key][node_id]:
                                            old_func = results[image_key][node_id]['functionality']
                                            if isinstance(old_func, dict):
                                                results[image_key][node_id]['functionality_original'] = old_func.copy()
                                            else:
                                                results[image_key][node_id]['functionality_original'] = old_func
                                        
                                        # Update with revised functionality
                                        results[image_key][node_id]['functionality'] = {
                                            'wo_context': revised_func,
                                            'with_context': revised_func
                                        }
                                    
                                    # Update description if available
                                    if revised_desc:
                                        # Store old description for reference
                                        if 'description' in results[image_key][node_id]:
                                            old_desc = results[image_key][node_id]['description']
                                            if isinstance(old_desc, dict):
                                                results[image_key][node_id]['description_original'] = old_desc.copy()
                                            else:
                                                results[image_key][node_id]['description_original'] = old_desc
                                        
                                        # Update with revised description
                                        results[image_key][node_id]['description'] = {
                                            'wo_context': revised_desc,
                                            'with_context': revised_desc
                                        }
                                    
                                    # Mark as reannotated
                                    if revised_func or revised_desc:
                                        results[image_key][node_id]['reannotated'] = True
                                        results[image_key][node_id]['reannotation_file'] = os.path.basename(latest_reannotation)
                                        reannotation_count += 1
                            
                            except Exception as e:
                                if debug:
                                    print(f"   Warning: Failed to load reannotation file {latest_reannotation}: {e}")
                                continue
    
    if debug:
        print(f"✅ Successfully loaded {len(results)} images from cache directory")
        print(f"  - {bbox_correction_count} regions with corrected bbox")
        print(f"  - {reannotation_count} regions with revised functionality/description")
    
    return results


def load_annotation_data(annotation_file: str = None, cache_dir: str = None) -> Dict:
    """加载原始标注数据（包括修订后的bbox和功能描述）
    
    支持两种模式：
    1. 从 annotation_file 加载（传统模式）
    2. 直接从 androidcontrol cache_dir 加载（androidcontrol 模式）
    """
    # Check if we should load directly from cache (androidcontrol mode)
    if annotation_file is None or not os.path.exists(annotation_file):
        # Try to load from cache directory if it looks like androidcontrol structure
        if cache_dir and os.path.isdir(cache_dir):
            # Check if cache_dir structure matches androidcontrol: {dataset}/{model}/{version}/
            cache_parts = cache_dir.rstrip('/').split(os.sep)
            if len(cache_parts) >= 3:
                # Check if it's androidcontrol cache structure
                if 'androidcontrol' in cache_parts or os.path.basename(cache_dir) == 'v2':
                    # This looks like androidcontrol cache, load directly
                    return load_androidcontrol_from_cache(cache_dir, debug=True)
        
        # If annotation_file is required but not found, raise error
        if annotation_file and not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        # If annotation_file is None and we can't load from cache, raise error
        if annotation_file is None:
            raise ValueError("Either annotation_file or valid cache_dir must be provided")
    
    # Original loading logic (for non-androidcontrol datasets)
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    raw_results = data.get('results', {})
    
    # Handle nested structure
    results = {}
    for image_key, image_data in raw_results.items():
        if isinstance(image_data, dict):
            if 'result' in image_data and isinstance(image_data['result'], dict):
                results[image_key] = image_data['result']
                if 'image_path' in image_data:
                    results[image_key]['root_image_path'] = image_data['image_path']
            else:
                results[image_key] = image_data
    
    # Load reannotations (corrected bbox + revised functionality/description)
    if cache_dir and os.path.isdir(cache_dir):
        print(f"📂 Loading reannotations from cache directory...")
        bbox_correction_count = 0
        reannotation_count = 0
        
        # Try to infer dataset/model/version from annotation_file path
        annotation_path_parts = annotation_file.split(os.sep)
        inferred_dataset = None
        inferred_model = None
        inferred_version = None
        
        for i, part in enumerate(annotation_path_parts):
            if part.startswith('v') and len(part) <= 3 and part[1:].isdigit():
                inferred_version = part
                if i >= 2:
                    inferred_model = annotation_path_parts[i-1]
                    inferred_dataset = annotation_path_parts[i-2]
                break
        
        for image_key, image_data in results.items():
            if not isinstance(image_data, dict):
                continue
            
            image_id = os.path.splitext(os.path.basename(image_key))[0]
            
            # Find corresponding cache directory
            cache_patterns = []
            
            if inferred_dataset and inferred_model and inferred_version:
                precise_path = os.path.join(cache_dir, inferred_dataset, inferred_model, inferred_version, image_id, "nodes")
                cache_patterns.append(precise_path)
            
            cache_patterns.extend([
                os.path.join(cache_dir, "**", image_id, "nodes"),
                os.path.join(cache_dir, "*", "*", "*", image_id, "nodes"),
            ])
            
            nodes_dir = None
            for pattern in cache_patterns:
                if '**' in pattern or '*' in pattern:
                    matches = glob.glob(pattern, recursive=True)
                    if matches:
                        non_backup_matches = [m for m in matches if '_bak' not in m]
                        if non_backup_matches:
                            nodes_dir = non_backup_matches[-1]
                        else:
                            nodes_dir = matches[-1]
                        break
                else:
                    if os.path.isdir(pattern):
                        nodes_dir = pattern
                        break
            
            if not nodes_dir or not os.path.isdir(nodes_dir):
                continue
            
            # Iterate all nodes to find reannotation files
            for node_id in image_data.keys():
                if not isinstance(image_data[node_id], dict):
                    continue
                
                # Load reannotations (from *_meta_reannotated*.json)
                reannotation_files = glob.glob(os.path.join(nodes_dir, f"{node_id}_meta_reannotated*.json"))
                
                if reannotation_files:
                    latest_reannotation = sorted(reannotation_files)[-1]
                    
                    try:
                        with open(latest_reannotation, 'r', encoding='utf-8') as rf:
                            reannotation_data = json.load(rf)
                        
                        # Extract corrected bbox
                        corrected_bbox = reannotation_data.get('corrected_bbox')
                        if corrected_bbox and len(corrected_bbox) == 4:
                            original_bbox = image_data[node_id].get('bbox_global')
                            
                            if original_bbox != corrected_bbox:
                                image_data[node_id]['bbox_global'] = corrected_bbox
                                
                                if 'root_size(wxh)' in image_data[node_id]:
                                    w, h = image_data[node_id]['root_size(wxh)']
                                    if w > 0 and h > 0:
                                        image_data[node_id]['bbox_global_norm'] = [
                                            corrected_bbox[0] / w,
                                            corrected_bbox[1] / h,
                                            corrected_bbox[2] / w,
                                            corrected_bbox[3] / h
                                        ]
                                
                                image_data[node_id]['bbox_corrected'] = True
                                bbox_correction_count += 1
                        
                        # Extract revised functionality and description
                        new_functionality = reannotation_data.get('new_functionality', {})
                        if isinstance(new_functionality, dict):
                            revised_func = new_functionality.get('revised functionality')
                            revised_desc = new_functionality.get('revised description')
                            
                            # Update functionality if available
                            if revised_func:
                                image_data[node_id]['functionality'] = {
                                    'wo_context': revised_func,
                                    'with_context': revised_func
                                }
                            
                            # Update description if available
                            if revised_desc:
                                image_data[node_id]['description'] = {
                                    'wo_context': revised_desc,
                                    'with_context': revised_desc
                                }
                            
                            # Mark as reannotated
                            if revised_func or revised_desc:
                                image_data[node_id]['reannotated'] = True
                                reannotation_count += 1
                    
                    except Exception as e:
                        print(f"   Warning: Failed to load {latest_reannotation}: {e}")
                        continue
        
        print(f"✅ Loaded {reannotation_count} reannotations")
        if bbox_correction_count > 0:
            print(f"   - {bbox_correction_count} regions with corrected bbox")
        if reannotation_count > 0:
            print(f"   - {reannotation_count} regions with revised functionality/description")
    
    return results


def load_omniparser_data(omniparser_dir: str, image_name: str) -> List[Dict]:
    """加载 OmniParser 数据（支持递归搜索和 androidcontrol 格式）
    
    Args:
        omniparser_dir: OmniParser 数据目录
        image_name: 图片名称（可能是普通文件名或 androidcontrol 格式的 {app_name}/{episode_id}/{step_id}）
    """
    # For androidcontrol format: {app_name}/{episode_id}/{step_id}
    # Try to find using the full path structure
    if '/' in image_name and not os.path.isfile(image_name):
        # androidcontrol format: try to find in omniparser_dir with same structure
        androidcontrol_path = os.path.join(omniparser_dir, f'{image_name}.json')
        if os.path.exists(androidcontrol_path):
            try:
                with open(androidcontrol_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {androidcontrol_path}: {e}")
        
        # Also try recursive search with androidcontrol path
        search_pattern = os.path.join(omniparser_dir, '**', f'{image_name}.json')
        matches = glob.glob(search_pattern, recursive=True)
        if matches:
            non_backup_matches = [m for m in matches if '_bak' not in m]
            if non_backup_matches:
                omniparser_file = non_backup_matches[0]
            else:
                omniparser_file = matches[0]
            
            try:
                with open(omniparser_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {omniparser_file}: {e}")
        
        # Also try using just the step_id part (last component)
        step_id = image_name.split('/')[-1]
        step_id_file = os.path.join(omniparser_dir, f'{step_id}.json')
        if os.path.exists(step_id_file):
            try:
                with open(step_id_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {step_id_file}: {e}")
    
    # Try direct path first (for regular image names)
    omniparser_file = os.path.join(omniparser_dir, f'{image_name}.json')
    if os.path.exists(omniparser_file):
        try:
            with open(omniparser_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {omniparser_file}: {e}")
    
    # Try recursive search
    search_pattern = os.path.join(omniparser_dir, '**', f'{image_name}.json')
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        # Use the first match (prefer non-backup files)
        non_backup_matches = [m for m in matches if '_bak' not in m]
        if non_backup_matches:
            omniparser_file = non_backup_matches[0]
        else:
            omniparser_file = matches[0]
        
        try:
            with open(omniparser_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {omniparser_file}: {e}")
    
    return []


def extract_bbox_and_screen_info(annotation_data: Dict, image_key: str, region_id: str) -> Tuple:
    """从标注数据中提取 bbox 和屏幕尺寸
    
    支持 androidcontrol 格式的 image_key: {app_name}/{episode_id}/{step_id}
    
    Returns:
        (bbox, screen_width, screen_height) or (None, None, None) if not found
    """
    if image_key not in annotation_data:
        return None, None, None
    
    image_data = annotation_data[image_key]
    
    # Handle nested structure (for non-androidcontrol datasets)
    if 'result' in image_data and isinstance(image_data['result'], dict):
        image_data = image_data['result']
    
    if region_id not in image_data:
        return None, None, None
    
    region_data = image_data[region_id]
    
    # Get bbox (prefer non-normalized)
    bbox = region_data.get('bbox_global')
    if not bbox or len(bbox) < 4:
        return None, None, None
    
    # Get screen size
    screen_size = region_data.get('root_size(wxh)')
    if not screen_size or len(screen_size) < 2:
        return None, None, None
    
    screen_width, screen_height = screen_size
    
    return bbox, screen_width, screen_height


def collect_all_target_regions(result_dir: str, allowed_modes: List[str] = None) -> List[Dict]:
    """收集所有问题中的目标区域信息
    
    Args:
        result_dir: 结果目录路径
        allowed_modes: 允许处理的 mode 目录列表（白名单），如果为 None 则只处理 captioning_mode 和 grounding_mode
    
    Returns:
        List of dicts with keys: {
            'file_path': str,
            'question_idx': int,
            'image_key': str,
            'image_name': str,
            'target_region_id': str,
            'mode': 'qa' or 'grounding'
        }
    """
    target_regions = []
    
    # Default allowed modes: only captioning_mode and grounding_mode
    if allowed_modes is None:
        allowed_modes = ['captioning_mode', 'grounding_mode']
    
    # Check allowed mode directories only
    modes = []
    for mode_dir in allowed_modes:
        mode_path = os.path.join(result_dir, mode_dir)
        if os.path.isdir(mode_path):
            if mode_dir == 'captioning_mode' or mode_dir == 'qa_mode':
                modes.append((mode_dir, 'qa'))
            elif mode_dir == 'grounding_mode':
                modes.append((mode_dir, 'grounding'))
            else:
                # Unknown mode, treat as 'qa' by default
                modes.append((mode_dir, 'qa'))
        else:
            print(f"⚠️  Warning: Mode directory not found: {mode_path}")
    
    # If no subdirectories found, check result_dir directly (only if explicitly allowed)
    if not modes and allowed_modes:
        result_files = glob.glob(os.path.join(result_dir, '*_result.json'))
        if result_files:
            print(f"⚠️  Warning: No mode subdirectories found, checking result_dir directly")
            modes.append(('', 'unknown'))
    
    for mode_dir, mode_name in modes:
        if mode_dir:
            search_pattern = os.path.join(result_dir, mode_dir, '*_result.json')
        else:
            search_pattern = os.path.join(result_dir, '*_result.json')
        
        result_files = glob.glob(search_pattern)
        
        # Filter out backup directories
        result_files = [f for f in result_files if 'backup' not in f.lower()]
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_key = data.get('image_key', '')
                if not image_key:
                    continue
                
                # Extract image name (without extension)
                # For androidcontrol format: {app_name}/{episode_id}/{step_id}, use the full path
                # For regular format: use basename without extension
                if '/' in image_key and not os.path.isfile(image_key):
                    # androidcontrol format: use the full path as image_name
                    image_name = image_key
                else:
                    # Regular format: extract basename without extension
                    image_name = os.path.splitext(os.path.basename(image_key))[0]
                
                result = data.get('result', {})
                questions = result.get('questions', [])
                
                for q_idx, question in enumerate(questions):
                    # For qa mode (including captioning_mode): use all region_ids
                    if mode_name == 'qa' and 'region_ids' in question:
                        for region_id in question['region_ids']:
                            target_regions.append({
                                'file_path': file_path,
                                'question_idx': q_idx,
                                'image_key': image_key,
                                'image_name': image_name,
                                'target_region_id': region_id,
                                'mode': mode_name,
                                'is_multi_region': True  # Flag for QA mode
                            })
                    # For grounding mode: also use all region_ids (same structure as qa_mode)
                    # Note: actual grounding_mode files have region_ids and options, not target_region_id
                    elif mode_name == 'grounding' and 'region_ids' in question:
                        for region_id in question['region_ids']:
                            target_regions.append({
                                'file_path': file_path,
                                'question_idx': q_idx,
                                'image_key': image_key,
                                'image_name': image_name,
                                'target_region_id': region_id,
                                'mode': mode_name,
                                'is_multi_region': True  # Grounding mode also has multiple regions
                            })
                    # Legacy support: if grounding mode has target_region_id (old format)
                    elif mode_name == 'grounding' and 'target_region_id' in question:
                        target_regions.append({
                            'file_path': file_path,
                            'question_idx': q_idx,
                            'image_key': image_key,
                            'image_name': image_name,
                            'target_region_id': question['target_region_id'],
                            'mode': mode_name,
                            'is_multi_region': False  # Legacy single region format
                        })
            
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    return target_regions


def calculate_metrics_for_all_regions(
    target_regions: List[Dict],
    annotation_data: Dict,
    omniparser_dir: str
) -> Tuple[Dict, Dict]:
    """批量计算所有区域的指标
    
    Returns:
        (area_metrics, density_metrics)
        Both are dicts mapping (image_key, region_id) -> metrics
    """
    print("收集所有区域的 bbox 和周围元素...")
    
    # Collect unique regions (image_key, region_id)
    unique_regions = {}
    for item in target_regions:
        key = (item['image_key'], item['target_region_id'])
        if key not in unique_regions:
            unique_regions[key] = item
    
    print(f"找到 {len(unique_regions)} 个唯一区域")
    
    # Collect bbox data with OmniParser (both area and density need OmniParser)
    # Only collect data when OmniParser is available
    bbox_data = []
    screen_widths = []
    screen_heights = []
    surr_bboxes = []
    valid_keys = []
    
    missing_bbox_count = 0
    missing_omniparser_count = 0
    
    for key, item in tqdm(unique_regions.items(), desc="提取 bbox 和周围元素"):
        image_key = item['image_key']
        region_id = item['target_region_id']
        image_name = item['image_name']
        
        # Get bbox and screen info
        bbox, screen_width, screen_height = extract_bbox_and_screen_info(
            annotation_data, image_key, region_id
        )
        
        if bbox is None:
            missing_bbox_count += 1
            if missing_bbox_count <= 5:  # Only print first 5 warnings
                print(f"Warning: 无法找到 {image_key} / {region_id} 的 bbox")
            continue
        
        # Get surrounding elements from OmniParser (required for both metrics)
        omniparser_data = load_omniparser_data(omniparser_dir, image_name)
        if not omniparser_data:
            missing_omniparser_count += 1
            if missing_omniparser_count <= 5:  # Only print first 5 warnings
                print(f"Warning: 无法找到 {image_name} 的 OmniParser 数据，跳过该区域")
            continue
        
        # Extract normalized bboxes from omniparser
        surr_bbox_list = [elem['bbox'] for elem in omniparser_data if 'bbox' in elem]
        
        # Only add to calculation list if OmniParser data is available
        bbox_data.append(bbox)
        screen_widths.append(screen_width)
        screen_heights.append(screen_height)
        surr_bboxes.append(surr_bbox_list)
        valid_keys.append(key)
    
    # Print statistics
    print(f"\n数据收集统计:")
    print(f"  成功收集数据（有 OmniParser）: {len(bbox_data)} 个区域")
    if missing_bbox_count > 0:
        print(f"  缺少 bbox: {missing_bbox_count} 个区域")
    if missing_omniparser_count > 0:
        print(f"  缺少 OmniParser（已跳过）: {missing_omniparser_count} 个区域")
    
    if not bbox_data:
        print("错误: 没有找到有效的区域数据（需要同时有 bbox 和 OmniParser 数据）")
        return {}, {}
    
    # ============================================================
    # 计算面积分类（仅使用有 OmniParser 数据的区域）
    # ============================================================
    print("\n计算面积占比和分类...")
    
    classifier = GUISizeClassifier()
    relative_areas = classifier.calculate_relative_areas(
        bbox_data, screen_widths, screen_heights
    )
    area_analysis = classifier.analyze_distribution(relative_areas, percentiles=[10, 30, 70])
    area_classifications = classifier.classify_all_elements(relative_areas)
    
    # Print statistics
    print("\n面积分类统计:")
    print(f"  样本数量: {len(relative_areas)}")
    print(f"  平均面积: {area_analysis['statistics']['mean']:.2f}%")
    print(f"  阈值: P10={area_analysis['thresholds'][0]:.2f}%, "
          f"P30={area_analysis['thresholds'][1]:.2f}%, "
          f"P70={area_analysis['thresholds'][2]:.2f}%")
    
    area_dist = classifier.get_class_distribution(area_classifications)
    for class_name, info in area_dist.items():
        print(f"  {class_name:>6}: {info['count']:>4} ({info['percentage']:.1f}%)")
    
    # ============================================================
    # 计算密集度分类（使用相同的区域数据）
    # ============================================================
    print("\n计算周围密集度和分类...")
    
    analyzer = EnhancedNIDAnalyzer(k_sigma=1.5, alpha=1.0)
    nid_scores = analyzer.calculate_all_nid_scores(
        bbox_data, surr_bboxes, 
        screen_widths, screen_heights
    )
    density_analysis = analyzer.classify_by_percentiles(percentiles=[33, 67])
    density_classifications = analyzer.classify_all_elements(nid_scores)
    
    # Print statistics
    print("\n密集度分类统计:")
    print(f"  样本数量: {len(nid_scores)}")
    print(f"  平均 NID: {density_analysis['statistics']['mean']:.2f}")
    print(f"  阈值: P33={density_analysis['thresholds'][0]:.2f}, "
          f"P67={density_analysis['thresholds'][1]:.2f}")
    
    density_counts = {cls: density_classifications.count(cls) for cls in ['sparse', 'medium', 'dense']}
    for cls, count in density_counts.items():
        percentage = (count / len(density_classifications)) * 100
        print(f"  {cls:>6}: {count:>4} ({percentage:.1f}%)")
    
    # ============================================================
    # 构建结果字典
    # ============================================================
    area_metrics = {}
    density_metrics = {}
    
    for i, key in enumerate(valid_keys):
        area_metrics[key] = {
            'relative_area': float(relative_areas[i]),
            'area_class': area_classifications[i]
        }
        density_metrics[key] = {
            'nid_score': float(nid_scores[i]),
            'density_class': density_classifications[i]
        }
    
    return area_metrics, density_metrics


def update_questions_with_metrics(
    result_dir: str,
    target_regions: List[Dict],
    area_metrics: Dict,
    density_metrics: Dict,
    output_dir: str = None
) -> None:
    """更新问题文件，添加 metrics 字段
    
    Args:
        result_dir: 输入的结果目录
        target_regions: 目标区域列表
        area_metrics: 面积指标字典
        density_metrics: 密集度指标字典
        output_dir: 输出目录（如果为 None，则原地更新）
    """
    
    print("\n更新问题文件...")
    
    # 确定输出模式
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        print(f"   原始文件不会被修改")
    else:
        print(f"⚠️  原地更新模式（会修改原始文件）")
    
    # Group by file path
    files_to_update = {}
    for item in target_regions:
        file_path = item['file_path']
        if file_path not in files_to_update:
            files_to_update[file_path] = []
        files_to_update[file_path].append(item)
    
    updated_count = 0
    
    for file_path, items in tqdm(files_to_update.items(), desc="更新文件"):
        try:
            # Load file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = data.get('result', {})
            questions = result.get('questions', [])
            
            # Update questions
            for item in items:
                q_idx = item['question_idx']
                if q_idx >= len(questions):
                    continue
                
                question = questions[q_idx]
                key = (item['image_key'], item['target_region_id'])
                
                # Get metrics (both area and density are required - they are calculated together)
                area_metric = area_metrics.get(key)
                density_metric = density_metrics.get(key)
                
                # Only update if both metrics exist (they should always exist together)
                if not area_metric or not density_metric:
                    continue
                
                # For modes with multiple regions (qa_mode, captioning_mode, and current grounding_mode):
                # add metrics to each option
                if item.get('is_multi_region', False):
                    # Find the option with this region_id
                    for option in question.get('options', []):
                        if option.get('region_id') == item['target_region_id']:
                            if 'metrics' not in option:
                                option['metrics'] = {}
                            
                            option['metrics']['area'] = area_metric
                            option['metrics']['density'] = density_metric
                            updated_count += 1
                            break
                
                # Legacy grounding mode: add metrics to question level (old format with single target_region_id)
                elif item['mode'] == 'grounding' and not item.get('is_multi_region', False):
                    if 'metrics' not in question:
                        question['metrics'] = {}
                    
                    question['metrics']['area'] = area_metric
                    question['metrics']['density'] = density_metric
                    updated_count += 1
            
            # Determine output file path
            if output_dir:
                # 计算相对路径
                rel_path = os.path.relpath(file_path, result_dir)
                output_file_path = os.path.join(output_dir, rel_path)
                
                # 创建输出文件的父目录
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            else:
                output_file_path = file_path
            
            # Save updated file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            continue
    
    print(f"\n✅ 成功更新 {updated_count} 个问题/选项的指标")


def main(args):
    print("=" * 60)
    print("为生成的问题数据添加 area_class 和 density_class 指标")
    print("=" * 60)
    print()
    
    # Check paths
    # annotation_file is optional for androidcontrol mode
    if args.annotation_file and not os.path.exists(args.annotation_file):
        print(f"错误: 标注文件不存在: {args.annotation_file}")
        return
    
    if not os.path.isdir(args.result_dir):
        print(f"错误: 结果目录不存在: {args.result_dir}")
        return
    
    if not os.path.isdir(args.omniparser_dir):
        print(f"错误: OmniParser 目录不存在: {args.omniparser_dir}")
        return
    
    # For androidcontrol mode, cache_dir is required
    if not args.annotation_file and (not args.cache_dir or not os.path.isdir(args.cache_dir)):
        print(f"错误: androidcontrol 模式需要提供有效的 cache_dir")
        return
    
    print(f"输入配置:")
    if args.annotation_file:
        print(f"  标注文件: {args.annotation_file}")
    else:
        print(f"  标注文件: 无（androidcontrol 模式，直接从 cache 加载）")
    print(f"  结果目录: {args.result_dir}")
    print(f"  OmniParser: {args.omniparser_dir}")
    print(f"  Cache目录: {args.cache_dir}")
    if args.output_dir:
        print(f"  输出目录: {args.output_dir}")
    else:
        print(f"  更新模式: ⚠️  原地更新")
    print()
    
    # Step 1: Load annotation data (including reannotations)
    print("加载原始标注数据（包含修订后的 bbox 和功能描述）...")
    annotation_data = load_annotation_data(args.annotation_file, cache_dir=args.cache_dir)
    print(f"加载了 {len(annotation_data)} 张图片的标注数据")
    print()
    
    # Step 2: Collect all target regions from question files
    print("收集问题文件中的目标区域...")
    allowed_modes = args.allowed_modes if args.allowed_modes else None
    if allowed_modes:
        print(f"  只处理以下 mode 目录: {', '.join(allowed_modes)}")
    else:
        print(f"  默认只处理: captioning_mode, grounding_mode")
    target_regions = collect_all_target_regions(args.result_dir, allowed_modes=allowed_modes)
    print(f"找到 {len(target_regions)} 个目标区域（包括重复）")
    print()
    
    if not target_regions:
        print("错误: 没有找到任何问题数据")
        return
    
    # Step 3: Calculate metrics for all regions
    area_metrics, density_metrics = calculate_metrics_for_all_regions(
        target_regions,
        annotation_data,
        args.omniparser_dir
    )
    
    if not area_metrics or not density_metrics:
        print("错误: 指标计算失败")
        return
    
    # Step 4: Update question files with metrics
    update_questions_with_metrics(
        args.result_dir,
        target_regions,
        area_metrics,
        density_metrics,
        output_dir=args.output_dir
    )
    
    print()
    print("=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为生成的问题数据添加 area_class 和 density_class 指标"
    )
    
    parser.add_argument(
        "--annotation-file",
        default=None,
        help="原始标注文件路径（可选，androidcontrol 模式可以省略，直接从 cache_dir 加载）"
    )
    
    parser.add_argument(
        "--result-dir",
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion",
        help="生成的问题数据目录"
    )
    
    parser.add_argument(
        "--allowed-modes",
        nargs='+',
        default=None,
        help="允许处理的 mode 目录列表（白名单），例如: --allowed-modes captioning_mode grounding_mode。如果不指定，默认只处理 captioning_mode 和 grounding_mode"
    )
    
    parser.add_argument(
        "--omniparser-dir",
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/omniparser",
        help="OmniParser 数据目录"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="/mnt/vdb1/hongxin_li/AutoGUIv2/cache/androidcontrol/gemini-2.5-pro-thinking/v2",
        help="Cache 目录路径（包含修订后的 bbox 和功能描述）。对于 androidcontrol 模式，应指向完整的 cache 路径，例如: /mnt/vdb1/hongxin_li/AutoGUIv2/cache/androidcontrol/gemini-2.5-pro-thinking/v2"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（如果不指定，则原地更新原始文件）"
    )
    
    args = parser.parse_args()
    main(args)

