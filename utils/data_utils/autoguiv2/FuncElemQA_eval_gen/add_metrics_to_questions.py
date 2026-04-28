#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为生成的问题数据添加 area_class 和 density_class 指标，并可选择性地补充其他字段

内置两个计算工具：
- GUISizeClassifier: 计算目标功能区面积占比
- EnhancedNIDAnalyzer: 计算目标区域周围密集度

支持补充的字段：
- region_type: 从 *region-type.json 文件读取
- description: 从 reannotated 文件读取
- functionality: 从 reannotated 文件读取
"""

import os
import sys
import json
import glob
import argparse
import math
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.append('/'.join(__file__.split('/')[:-5]))


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class GUISizeClassifier:
    """Classify GUI region size by relative screen area percentiles."""

    def __init__(self):
        self.thresholds: List[float] = []

    def calculate_relative_areas(self, bboxes: List[List[float]], widths: List[int], heights: List[int]) -> List[float]:
        areas = []
        for bbox, width, height in zip(bboxes, widths, heights):
            screen_area = max(float(width * height), 1.0)
            box_area = max(float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])), 0.0)
            areas.append(box_area / screen_area * 100.0)
        return areas

    def analyze_distribution(self, relative_areas: List[float], percentiles: List[float]) -> Dict:
        self.thresholds = [_percentile(relative_areas, p) for p in percentiles]
        mean = sum(relative_areas) / len(relative_areas) if relative_areas else 0.0
        return {
            'thresholds': self.thresholds,
            'statistics': {
                'mean': mean,
                'min': min(relative_areas) if relative_areas else 0.0,
                'max': max(relative_areas) if relative_areas else 0.0,
            },
        }

    def classify_element(self, relative_area: float) -> str:
        if not self.thresholds:
            raise ValueError("Call analyze_distribution before classify_element")
        if relative_area <= self.thresholds[0]:
            return 'tiny'
        if relative_area <= self.thresholds[1]:
            return 'small'
        if relative_area <= self.thresholds[2]:
            return 'medium'
        return 'large'

    def classify_all_elements(self, relative_areas: List[float]) -> List[str]:
        return [self.classify_element(area) for area in relative_areas]

    def get_class_distribution(self, classes: List[str]) -> Dict[str, Dict[str, float]]:
        total = len(classes) or 1
        return {
            name: {
                'count': classes.count(name),
                'percentage': classes.count(name) / total * 100.0,
            }
            for name in ['tiny', 'small', 'medium', 'large']
        }


class EnhancedNIDAnalyzer:
    """Normalized interference density analyzer for surrounding GUI elements."""

    def __init__(self, k_sigma: float = 1.5, alpha: float = 1.0):
        self.k_sigma = k_sigma
        self.alpha = alpha
        self.thresholds: List[float] = []

    @staticmethod
    def _center(bbox: List[float]) -> Tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def _in_bbox(point: Tuple[float, float], bbox: List[float]) -> bool:
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def _analysis_region(self, bbox: List[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = bbox
        expand_x = self.alpha * (x2 - x1)
        expand_y = self.alpha * (y2 - y1)
        return [
            max(0.0, x1 - 1.5 * expand_x),
            max(0.0, y1 - 1.5 * expand_y),
            min(float(width), x2 + 1.5 * expand_x),
            min(float(height), y2 + 1.5 * expand_y),
        ]

    def calculate_nid_score(self, target_bbox: List[float], surr_norm_bboxes: List[List[float]], width: int, height: int) -> float:
        cx, cy = self._center(target_bbox)
        region = self._analysis_region(target_bbox, width, height)
        sigma_x = max(self.k_sigma * (target_bbox[2] - target_bbox[0]), 1.0)
        sigma_y = max(self.k_sigma * (target_bbox[3] - target_bbox[1]), 1.0)
        score = 0.0
        for norm_bbox in surr_norm_bboxes:
            bbox = [norm_bbox[0] * width, norm_bbox[1] * height, norm_bbox[2] * width, norm_bbox[3] * height]
            center = self._center(bbox)
            if self._in_bbox(center, region) and not self._in_bbox(center, target_bbox):
                score += math.exp(-0.5 * (((center[0] - cx) / sigma_x) ** 2 + ((center[1] - cy) / sigma_y) ** 2))
        return score

    def calculate_all_nid_scores(self, bboxes: List[List[float]], surr_bboxes: List[List[List[float]]],
                                 widths: List[int], heights: List[int]) -> List[float]:
        self._last_scores = [
            self.calculate_nid_score(bbox, surr, width, height)
            for bbox, surr, width, height in zip(bboxes, surr_bboxes, widths, heights)
        ]
        return self._last_scores

    def classify_by_percentiles(self, percentiles: List[float]) -> Dict:
        scores = getattr(self, '_last_scores', None)
        if scores is None:
            scores = []
        self.thresholds = [_percentile(scores, p) for p in percentiles]
        mean = sum(scores) / len(scores) if scores else 0.0
        return {
            'thresholds': self.thresholds,
            'statistics': {
                'mean': mean,
                'min': min(scores) if scores else 0.0,
                'max': max(scores) if scores else 0.0,
            },
        }

    def classify_all_elements(self, scores: List[float]) -> List[str]:
        self._last_scores = scores
        if not self.thresholds:
            self.thresholds = [_percentile(scores, 33), _percentile(scores, 67)]
        classes = []
        for score in scores:
            if score <= self.thresholds[0]:
                classes.append('sparse')
            elif score <= self.thresholds[1]:
                classes.append('medium')
            else:
                classes.append('dense')
        return classes

# Module-level caches. Public usage should pass --cache-dir or keep region-type
# files next to the generated data instead of relying on private server paths.
_DATASET_REGION_TYPE_FILES: Dict[str, str] = {}
_LOADED_REGION_TYPE_INDICES: Dict[str, Dict] = {}
_FAILED_REGION_TYPE_FILES: set = set()


def reorder_option_fields(option: Dict) -> None:
    """Ensure option dict has fields ordered as label, region_id, bbox, metrics, then others."""
    preferred_order = ['label', 'region_id', 'bbox', 'metrics']
    reordered = {}
    for key in preferred_order:
        if key in option:
            reordered[key] = option[key]
    for key, value in option.items():
        if key not in preferred_order:
            reordered[key] = value
    option.clear()
    option.update(reordered)


def load_annotation_data(annotation_file: str, cache_dir: str = None) -> Dict:
    """加载原始标注数据（包括修订后的bbox和功能描述）
    
    复用 gen_region-func_multichoice-qa_easy.py 中的加载逻辑
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
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
                            
                            # Update functionality if available (store as string directly)
                            if revised_func:
                                image_data[node_id]['functionality'] = revised_func
                            
                            # Update description if available (store as string directly)
                            if revised_desc:
                                image_data[node_id]['description'] = revised_desc
                            
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
    """加载 OmniParser 数据（支持递归搜索）"""
    # Try direct path first
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


def load_region_type(cache_dir: str, image_key: str, region_id: str) -> str:
    """加载区域类型（从 *region-type.json 文件）
    
    Args:
        cache_dir: Cache 目录路径
        image_key: 图片键（完整路径或文件名）
        region_id: 区域 ID（节点 ID）
    
    Returns:
        区域类型字符串，如果找不到则返回空字符串
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        cache_dir = None
    
    # 提取图片 ID（不含扩展名）
    image_id = os.path.splitext(os.path.basename(image_key))[0]
    
    # 尝试推断 dataset/model/version
    # 首先尝试从 cache_dir 中递归搜索
    search_patterns = []
    if cache_dir:
        search_patterns = [
            os.path.join(cache_dir, '**', image_id, 'nodes', f'{region_id}_region-type.json'),
            os.path.join(cache_dir, '*', '*', '*', image_id, 'nodes', f'{region_id}_region-type.json'),
        ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            non_backup_matches = [m for m in matches if '_bak' not in m]
            region_type_file = non_backup_matches[0] if non_backup_matches else matches[0]
            try:
                with open(region_type_file, 'r', encoding='utf-8') as f:
                    region_data = json.load(f)
                if isinstance(region_data, dict):
                    value = region_data.get('type', '')
                    if value:
                        return value
                else:
                    return str(region_data)
            except Exception as e:
                if len(matches) == 1:
                    print(f"Warning: Failed to load {region_type_file}: {e}")
                # Fall through to dataset-level fallback
    
    # =========================
    # 数据集级别的后备查找
    # =========================
    # 从 image_key 推断数据集名称（取第一个路径段），需去掉可能的前导斜杠
    normalized_image_key = image_key.lstrip('/') if isinstance(image_key, str) else image_key
    dataset_name = None
    if isinstance(normalized_image_key, str) and '/' in normalized_image_key:
        candidate = normalized_image_key.split('/')[0]
        dataset_name = candidate if candidate in _DATASET_REGION_TYPE_FILES else None
    # 若首段未命中，尝试在路径中匹配已知数据集名
    if not dataset_name and isinstance(normalized_image_key, str):
        for known_ds in _DATASET_REGION_TYPE_FILES.keys():
            if normalized_image_key.startswith(known_ds + '/'):
                dataset_name = known_ds
                break
    # 如果 image_key 只是文件名，尝试从常见前缀中猜测（保守处理：不猜测）
    region_type = _load_region_type_from_dataset_index(dataset_name, normalized_image_key, image_id, region_id)
    if region_type:
        return region_type
    
    return ""


def _load_region_type_from_dataset_index(
    dataset_name: str,
    image_key: str,
    image_id: str,
    region_id: str
) -> str:
    """从指定数据集的 region_type 索引 JSON 中查找"""
    if not dataset_name or dataset_name not in _DATASET_REGION_TYPE_FILES:
        return ""
    
    index_path = _DATASET_REGION_TYPE_FILES[dataset_name]
    # 失败过的无需重复尝试
    if index_path in _FAILED_REGION_TYPE_FILES:
        return ""
    
    # 加载或复用缓存
    if dataset_name not in _LOADED_REGION_TYPE_INDICES:
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            _LOADED_REGION_TYPE_INDICES[dataset_name] = index_data
        except Exception as e:
            print(f"Warning: Failed to load dataset region_type index {index_path}: {e}")
            _FAILED_REGION_TYPE_FILES.add(index_path)
            return ""
    
    index_data = _LOADED_REGION_TYPE_INDICES.get(dataset_name, {})
    if not isinstance(index_data, dict):
        return ""
    
    # 构造一组可能的键以最大化匹配成功率
    candidates = []
    # 原始 image_key 及其前导斜杠变体
    if image_key:
        candidates.append(image_key)
        if image_key.startswith('/'):
            candidates.append(image_key.lstrip('/'))
        else:
            candidates.append('/' + image_key)
    # 带数据集常见前缀的键
    if image_key and not image_key.startswith(dataset_name + '/'):
        ds_prefixed = f"{dataset_name}/images/{os.path.basename(image_key)}"
        candidates.append(ds_prefixed)
        candidates.append('/' + ds_prefixed)
    # 仅文件名（带后缀 / 不带后缀）
    base_with_ext = os.path.basename(image_key) if image_key else f"{image_id}.png"
    base_without_ext = os.path.splitext(base_with_ext)[0]
    candidates.extend([
        base_with_ext,
        base_without_ext,
        f"{dataset_name}/images/{base_with_ext}",
        f"/{dataset_name}/images/{base_with_ext}"
    ])
    
    # 有些索引顶层可能是 'results' 或类似结构
    possible_roots = [index_data]
    for root_key in ['results', 'data', 'images']:
        if isinstance(index_data.get(root_key, None), dict):
            possible_roots.append(index_data[root_key])
    
    for root in possible_roots:
        if not isinstance(root, dict):
            continue
        for img_key in candidates:
            if img_key not in root:
                continue
            entry = root[img_key]
            # 兼容不同结构：
            # 1) { image: { region_id: "type" } }
            # 2) { image: { region_id: { "type": "Toolbar" } } }
            # 3) { image: { "regions": { region_id: "type" | {type:...} } } }
            # 4) { image: [ {id, type}, ... ] } 罕见，尽量兼容
            if isinstance(entry, dict):
                # 直接以 region_id 为键
                if region_id in entry:
                    value = entry[region_id]
                    if isinstance(value, dict):
                        rt = value.get('type') or value.get('region_type') or ''
                        if rt:
                            return rt
                    elif isinstance(value, str):
                        return value
                # 常见嵌套字段：'region_types'、'regions'、'nodes'
                for nested_key in ('region_types', 'regions', 'nodes'):
                    nested = entry.get(nested_key)
                    if isinstance(nested, dict) and region_id in nested:
                        value = nested[region_id]
                        if isinstance(value, dict):
                            rt = value.get('type') or value.get('region_type') or ''
                            if rt:
                                return rt
                        elif isinstance(value, str):
                            return value
                # 某些结构可能把区域列表放在数组里
            elif isinstance(entry, list):
                # 列表项里寻找匹配的 region_id
                for item in entry:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get('region_id') or item.get('id')) == str(region_id):
                        rt = item.get('type') or item.get('region_type')
                        if rt:
                            return rt
    return ""


def extract_bbox_and_screen_info(annotation_data: Dict, image_key: str, region_id: str) -> Tuple:
    """从标注数据中提取 bbox 和屏幕尺寸
    
    Returns:
        (bbox, screen_width, screen_height) or (None, None, None) if not found
    """
    if image_key not in annotation_data:
        return None, None, None
    
    image_data = annotation_data[image_key]
    if 'result' in image_data:
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
    annotation_data: Dict = None,
    cache_dir: str = None,
    add_fields: List[str] = None,
    output_dir: str = None
) -> None:
    """更新问题文件，添加 metrics 字段和可选的补充字段
    
    Args:
        result_dir: 输入的结果目录
        target_regions: 目标区域列表
        area_metrics: 面积指标字典
        density_metrics: 密集度指标字典
        annotation_data: 标注数据（用于读取 description 和 functionality）
        cache_dir: Cache 目录（用于读取 region_type）
        add_fields: 要补充的字段列表，可选值: ['region_type', 'description', 'functionality']
        output_dir: 输出目录（如果为 None，则原地更新）
    """
    
    print("\n更新问题文件...")
    
    if add_fields is None:
        add_fields = []
    
    # 确定输出模式
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录: {output_dir}")
        print(f"   原始文件不会被修改")
    else:
        print(f"⚠️  原地更新模式（会修改原始文件）")
    
    if add_fields:
        print(f"   将补充以下字段: {', '.join(add_fields)}")
    
    # Group by file path
    files_to_update = {}
    for item in target_regions:
        file_path = item['file_path']
        if file_path not in files_to_update:
            files_to_update[file_path] = []
        files_to_update[file_path].append(item)
    
    updated_count = 0
    field_stats = {
        'region_type': 0,
        'description': 0,
        'functionality': 0
    }
    missing_field_sets = {
        'region_type': set(),
        'description': set(),
        'functionality': set()
    }
    missing_warning_limit = 5
    
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
                image_key = item['image_key']
                region_id = item['target_region_id']
                
                # Get metrics (both area and density are required - they are calculated together)
                area_metric = area_metrics.get(key) if area_metrics else None
                density_metric = density_metrics.get(key) if density_metrics else None
                
                # Skip if metrics are required but not available
                # If metrics are None (skip-metrics mode), we can still add fields
                skip_metrics = (area_metrics is None or density_metrics is None)
                if not skip_metrics and (not area_metric or not density_metric):
                    continue
                
                # For modes with multiple regions (qa_mode, captioning_mode, and current grounding_mode):
                # add metrics to each option
                if item.get('is_multi_region', False):
                    # Find the option with this region_id
                    for option in question.get('options', []):
                        if option.get('region_id') == region_id:
                            # 插入来自 reannotated 的 bbox（仅当有修订标记时）
                            if annotation_data and image_key in annotation_data:
                                ann_image_data = annotation_data[image_key]
                                if 'result' in ann_image_data:
                                    ann_image_data = ann_image_data['result']
                                if region_id in ann_image_data:
                                    ann_region_data = ann_image_data[region_id]
                                    if ann_region_data.get('bbox_corrected') and ann_region_data.get('bbox_global') and len(ann_region_data.get('bbox_global')) >= 4:
                                        option['bbox'] = ann_region_data.get('bbox_global')[:4]
                            # 只在非 skip-metrics 模式下添加 metrics
                            if not skip_metrics and area_metric and density_metric:
                                if 'metrics' not in option:
                                    option['metrics'] = {}
                                
                                option['metrics']['area'] = area_metric
                                option['metrics']['density'] = density_metric
                            
                            # 补充其他字段
                            if add_fields:
                                # 补充 region_type
                                if 'region_type' in add_fields:
                                    region_type = load_region_type(cache_dir, image_key, region_id)
                                    region_key = (image_key, region_id)
                                    if region_type:
                                        option['region_type'] = region_type
                                        field_stats['region_type'] += 1
                                    elif region_key not in missing_field_sets['region_type']:
                                        missing_field_sets['region_type'].add(region_key)
                                        if len(missing_field_sets['region_type']) <= missing_warning_limit:
                                            print(f"⚠️ region_type 缺失: {image_key} / {region_id}")
                                
                                # 补充 description 和 functionality（从 annotation_data 读取）
                                if annotation_data and image_key in annotation_data:
                                    image_data = annotation_data[image_key]
                                    if 'result' in image_data:
                                        image_data = image_data['result']
                                    
                                    if region_id in image_data:
                                        region_data = image_data[region_id]
                                        
                                        # 补充 description
                                        if 'description' in add_fields:
                                            description_added = False
                                            description = region_data.get('description')
                                            if description:
                                                if isinstance(description, dict):
                                                    option['description'] = description.get('revised description') or description.get('with_context') or description.get('wo_context') or str(description)
                                                else:
                                                    option['description'] = description
                                                field_stats['description'] += 1
                                                description_added = True
                                            if not description_added:
                                                region_key = (image_key, region_id)
                                                if region_key not in missing_field_sets['description']:
                                                    missing_field_sets['description'].add(region_key)
                                                    if len(missing_field_sets['description']) <= missing_warning_limit:
                                                        print(f"⚠️ description 缺失: {image_key} / {region_id}")
                                        
                                        # 补充 functionality
                                        if 'functionality' in add_fields:
                                            functionality_added = False
                                            functionality = region_data.get('functionality')
                                            if functionality:
                                                if isinstance(functionality, dict):
                                                    option['functionality'] = functionality.get('revised functionality') or functionality.get('with_context') or functionality.get('wo_context') or str(functionality)
                                                else:
                                                    option['functionality'] = functionality
                                                field_stats['functionality'] += 1
                                                functionality_added = True
                                            if not functionality_added:
                                                region_key = (image_key, region_id)
                                                if region_key not in missing_field_sets['functionality']:
                                                    missing_field_sets['functionality'].add(region_key)
                                                    if len(missing_field_sets['functionality']) <= missing_warning_limit:
                                                        print(f"⚠️ functionality 缺失: {image_key} / {region_id}")
                                else:
                                    if 'description' in add_fields:
                                        region_key = (image_key, region_id)
                                        if region_key not in missing_field_sets['description']:
                                            missing_field_sets['description'].add(region_key)
                                            if len(missing_field_sets['description']) <= missing_warning_limit:
                                                print(f"⚠️ description 缺失: {image_key} / {region_id}（未找到标注数据）")
                                    if 'functionality' in add_fields:
                                        region_key = (image_key, region_id)
                                        if region_key not in missing_field_sets['functionality']:
                                            missing_field_sets['functionality'].add(region_key)
                                            if len(missing_field_sets['functionality']) <= missing_warning_limit:
                                                print(f"⚠️ functionality 缺失: {image_key} / {region_id}（未找到标注数据）")
                            
                            reorder_option_fields(option)
                            
                            updated_count += 1
                            break
                
                # Legacy grounding mode: add metrics to question level (old format with single target_region_id)
                elif item['mode'] == 'grounding' and not item.get('is_multi_region', False):
                    # 只在非 skip-metrics 模式下添加 metrics
                    if not skip_metrics and area_metric and density_metric:
                        if 'metrics' not in question:
                            question['metrics'] = {}
                        
                        question['metrics']['area'] = area_metric
                        question['metrics']['density'] = density_metric
                    
                    # 补充其他字段（legacy 模式也支持）
                    if add_fields:
                        # 补充 region_type
                        if 'region_type' in add_fields:
                            region_type = load_region_type(cache_dir, image_key, region_id)
                            region_key = (image_key, region_id)
                            if region_type:
                                question['region_type'] = region_type
                                field_stats['region_type'] += 1
                            elif region_key not in missing_field_sets['region_type']:
                                missing_field_sets['region_type'].add(region_key)
                                if len(missing_field_sets['region_type']) <= missing_warning_limit:
                                    print(f"⚠️ region_type 缺失: {image_key} / {region_id}")
                        
                        # 补充 description 和 functionality
                        if annotation_data and image_key in annotation_data:
                            image_data = annotation_data[image_key]
                            if 'result' in image_data:
                                image_data = image_data['result']
                            
                            if region_id in image_data:
                                region_data = image_data[region_id]
                                
                                # 补充 description
                                if 'description' in add_fields:
                                    description_added = False
                                    description = region_data.get('description')
                                    if description:
                                        if isinstance(description, dict):
                                            question['description'] = description.get('revised description') or description.get('with_context') or description.get('wo_context') or str(description)
                                        else:
                                            question['description'] = description
                                        field_stats['description'] += 1
                                        description_added = True
                                    if not description_added:
                                        region_key = (image_key, region_id)
                                        if region_key not in missing_field_sets['description']:
                                            missing_field_sets['description'].add(region_key)
                                            if len(missing_field_sets['description']) <= missing_warning_limit:
                                                print(f"⚠️ description 缺失: {image_key} / {region_id}")
                                
                                # 补充 functionality
                                if 'functionality' in add_fields:
                                    functionality_added = False
                                    functionality = region_data.get('functionality')
                                    if functionality:
                                        if isinstance(functionality, dict):
                                            question['functionality'] = functionality.get('revised functionality') or functionality.get('with_context') or functionality.get('wo_context') or str(functionality)
                                        else:
                                            question['functionality'] = functionality
                                        field_stats['functionality'] += 1
                                        functionality_added = True
                                    if not functionality_added:
                                        region_key = (image_key, region_id)
                                        if region_key not in missing_field_sets['functionality']:
                                            missing_field_sets['functionality'].add(region_key)
                                            if len(missing_field_sets['functionality']) <= missing_warning_limit:
                                                print(f"⚠️ functionality 缺失: {image_key} / {region_id}")
                            else:
                                if 'description' in add_fields:
                                    region_key = (image_key, region_id)
                                    if region_key not in missing_field_sets['description']:
                                        missing_field_sets['description'].add(region_key)
                                        if len(missing_field_sets['description']) <= missing_warning_limit:
                                            print(f"⚠️ description 缺失: {image_key} / {region_id}（未找到标注数据）")
                                if 'functionality' in add_fields:
                                    region_key = (image_key, region_id)
                                    if region_key not in missing_field_sets['functionality']:
                                        missing_field_sets['functionality'].add(region_key)
                                        if len(missing_field_sets['functionality']) <= missing_warning_limit:
                                            print(f"⚠️ functionality 缺失: {image_key} / {region_id}（未找到标注数据）")
                    
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
    
    if skip_metrics:
        print(f"\n✅ 成功更新 {updated_count} 个问题/选项的字段")
    else:
        print(f"\n✅ 成功更新 {updated_count} 个问题/选项的指标")
    
    if add_fields:
        print(f"\n字段补充统计:")
        for field in add_fields:
            count = field_stats.get(field, 0)
            print(f"  {field}: {count} 个")
        missing_summary = {field: len(missing_field_sets[field]) for field in add_fields if field in missing_field_sets and missing_field_sets[field]}
        if missing_summary:
            print(f"\n⚠️ 字段缺失统计:")
            for field, count in missing_summary.items():
                print(f"  {field}: {count} 个区域缺失")
                entries = sorted(missing_field_sets[field])
                max_preview = 20
                preview_entries = entries[:max_preview]
                for image_key, region_id in preview_entries:
                    print(f"    - {image_key} / {region_id}")
                if count > max_preview:
                    print(f"    ... 等 {count - max_preview} 个")


def main(args):
    print("=" * 60)
    if args.skip_metrics:
        print("为生成的问题数据添加字段（跳过 metrics 计算）")
    else:
        print("为生成的问题数据添加 area_class 和 density_class 指标")
    print("=" * 60)
    print()
    
    # Check paths
    if not os.path.exists(args.annotation_file):
        print(f"错误: 标注文件不存在: {args.annotation_file}")
        return
    
    if not os.path.isdir(args.result_dir):
        print(f"错误: 结果目录不存在: {args.result_dir}")
        return
    
    if not os.path.isdir(args.omniparser_dir):
        print(f"错误: OmniParser 目录不存在: {args.omniparser_dir}")
        return
    
    print(f"输入配置:")
    print(f"  标注文件: {args.annotation_file}")
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
    
    # Step 3: Calculate metrics for all regions (skip if --skip-metrics is set)
    if args.skip_metrics:
        print("跳过 metrics 计算（--skip-metrics 模式）")
        if not args.add_fields:
            print("⚠️  警告: --skip-metrics 模式下未指定 --add-fields，将不会添加任何内容")
        area_metrics = None
        density_metrics = None
    else:
        area_metrics, density_metrics = calculate_metrics_for_all_regions(
            target_regions,
            annotation_data,
            args.omniparser_dir
        )
        
        if not area_metrics or not density_metrics:
            print("错误: 指标计算失败")
            return
    
    # Step 4: Update question files with metrics (or just add fields if skip-metrics)
    update_questions_with_metrics(
        args.result_dir,
        target_regions,
        area_metrics,
        density_metrics,
        annotation_data=annotation_data,
        cache_dir=args.cache_dir,
        add_fields=args.add_fields,
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
        required=True,
        help="原始标注文件路径"
    )
    
    parser.add_argument(
        "--result-dir",
        required=True,
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
        required=True,
        help="OmniParser 数据目录"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Cache 目录路径（包含修订后的 bbox 和功能描述）"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（如果不指定，则原地更新原始文件）"
    )
    
    parser.add_argument(
        "--add-fields",
        nargs='+',
        choices=['region_type', 'description', 'functionality'],
        default=None,
        help="要补充的字段列表，可选值: region_type, description, functionality。可以指定多个，例如: --add-fields region_type description functionality"
    )
    
    parser.add_argument(
        "--skip-metrics",
        action='store_true',
        help="跳过 metrics 计算和添加，只添加 --add-fields 指定的字段"
    )
    
    args = parser.parse_args()
    main(args)
