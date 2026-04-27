import os
import json
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any
from huggingface_hub import HfApi, Repository

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_utils.misc import bbox_iou_np, get_image_dimensions

class EnhancedNIDAnalyzer:
    def __init__(self, k_sigma: float = 1.0, alpha: float = 1.0):
        """
        增强版归一化干扰密度分析器
        
        Args:
            alpha: 扩展因子，默认1.0
            screen_width: 屏幕宽度，用于边界校正
            screen_height: 屏幕高度，用于边界校正
        """
        self.k_sigma = k_sigma
        self.alpha = alpha
        self.nid_scores = None
        self.percentiles = None
        self.thresholds = None

    def gaussian_2d(self, x: float, y: float, mu_x: float, mu_y: float, sigma_x: float, sigma_y: float) -> float:
        """
        Calculate 2D Gaussian value at point (x, y)
        
        Args:
            x, y: Point coordinates
            mu_x, mu_y: Gaussian center
            sigma_x, sigma_y: Standard deviations in x and y directions
            
        Returns:
            Gaussian weight value
        """
        return np.exp(-0.5 * (
            ((x - mu_x) / sigma_x) ** 2 + 
            ((y - mu_y) / sigma_y) ** 2
        ))

    def calculate_analysis_region(self, bbox: List, screen_width: int, screen_height: int, normalized: bool = False) -> List:
        """
        计算分析区域（根据目标区域按比例扩展）
        
        Args:
            bbox: 目标功能区边界框

        Returns:
            分析区域的边界框
        """
        # 计算扩展量
        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1

        expand_x = self.alpha * bbox_width
        expand_y = self.alpha * bbox_height

        # 计算分析区域
        analysis_x1 = max(0, x1 - 1.5 * expand_x)
        analysis_y1 = max(0, y1 - 1.5 * expand_y)
        analysis_x2 = min(screen_width, x2 + 1.5 * expand_x)
        analysis_y2 = min(screen_height, y2 + 1.5 * expand_y)

        return [analysis_x1 / screen_width, analysis_y1 / screen_height, analysis_x2 / screen_width, analysis_y2 / screen_height] if normalized else [analysis_x1, analysis_y1, analysis_x2, analysis_y2]
    
    def is_point_in_bbox(self, point: Tuple[float, float], bbox: List) -> bool:
        """判断点是否在边界框内"""
        x, y = point
        return (bbox[0] <= x <= bbox[2] and 
                bbox[1] <= y <= bbox[3])
    
    def calculate_element_center(self, bbox: List) -> Tuple[float, float]:
        """计算元素中心点"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    def calculate_nid_score(self, target_bbox: List, all_bboxes: List[List], screen_width: int, screen_height: int) -> int:
        """
        计算单个目标元素的NID分数
        
        Args:
            target_bbox: 目标功能区
            all_bboxes: 所有UI元素的边界框列表
            
        Returns:
            NID分数（分析区域内的干扰元素数量）
        """
        target_center_x, target_center_y = self.calculate_element_center(target_bbox)
        # 获取分析区域
        analysis_region = self.calculate_analysis_region(target_bbox, screen_width, screen_height)

        outside_bboxes, weights = [], []

        # Adaptive sigma based on element size
        sigma_x = self.k_sigma * (target_bbox[2] - target_bbox[0])
        sigma_y = self.k_sigma * (target_bbox[3] - target_bbox[1])

        for norm_bbox in all_bboxes:
            # 计算元素中心点
            unnorm_bbox = [norm_bbox[0] * screen_width, norm_bbox[1] * screen_height, norm_bbox[2] * screen_width, norm_bbox[3] * screen_height]
            center_x, center_y = self.calculate_element_center(unnorm_bbox)

            # 检查中心点是否在分析区域内
            if self.is_point_in_bbox((center_x, center_y), analysis_region) and not self.is_point_in_bbox((center_x, center_y), target_bbox):
                outside_bboxes.append(unnorm_bbox)

                # Calculate Gaussian weight
                weight = self.gaussian_2d(
                    center_x, center_y, target_center_x, target_center_y,
                    sigma_x, sigma_y
                )
                
                weights.append(weight)

        return sum(weights)

    def calculate_all_nid_scores(self, all_bboxes: List[List], surr_bboxes: List[List[List]], screen_widths: List[int], screen_heights: List[int]) -> np.ndarray:
        """
        为所有元素计算NID分数
        
        Args:
            all_bboxes: 所有UI元素的边界框列表
            
        Returns:
            NID分数数组
        """
        nid_scores = []
        
        for target_bbox, surr, w, h in zip(all_bboxes, surr_bboxes, screen_widths, screen_heights):
            score = self.calculate_nid_score(target_bbox, surr, w, h)
            nid_scores.append(score)
            
        self.nid_scores = np.array(nid_scores)
        return self.nid_scores
    
    def classify_by_percentiles(self, percentiles: List[float] = [33, 67]) -> Dict:
        """
        基于百分位数自动分类
        
        Args:
            percentiles: 百分位数阈值 [稀疏上限, 中等上限]
            
        Returns:
            分类结果字典
        """
        if self.nid_scores is None:
            raise ValueError("必须先计算NID分数")
            
        self.percentiles = percentiles
        self.thresholds = np.percentile(self.nid_scores, percentiles)
        
        # 定义分类规则
        classification_rules = {
            'sparse': (0, self.thresholds[0]),
            'medium': (self.thresholds[0], self.thresholds[1]),
            'dense': (self.thresholds[1], np.inf)
        }
        
        return {
            'percentiles': percentiles,
            'thresholds': self.thresholds.tolist(),
            'classification_rules': classification_rules,
            'statistics': {
                'mean': np.mean(self.nid_scores),
                'std': np.std(self.nid_scores),
                'min': np.min(self.nid_scores),
                'max': np.max(self.nid_scores)
            }
        }
    
    def classify_element(self, nid_score: float) -> str:
        """根据NID分数分类单个元素"""
        if self.thresholds is None:
            raise ValueError("必须先调用classify_by_percentiles")
            
        if nid_score <= self.thresholds[0]:
            return 'sparse'
        elif nid_score <= self.thresholds[1]:
            return 'medium'
        else:
            return 'dense'

    def classify_all_elements(self, nid_scores: List) -> List[str]:
        """Classify all elements in the dataset."""
        return [self.classify_element(nid_score) for nid_score in nid_scores]
    

    def visualize_analysis(self, all_bboxes: List[List], target_indices: List[int] = None):
        """
        可视化NID分析结果
        
        Args:
            all_bboxes: 所有边界框
            target_indices: 要特别显示的目标索引列表
        """
        if target_indices is None:
            target_indices = [0, len(all_bboxes)//2, -1]  # 显示首、中、尾三个样本

        fig, axes = plt.subplots(1, len(target_indices), figsize=(5*len(target_indices), 5))
        if len(target_indices) == 1:
            axes = [axes]

        colors = {'sparse': 'green', 'medium': 'orange', 'dense': 'red'}

        for idx, (ax, target_idx) in enumerate(zip(axes, target_indices)):
            target_bbox = all_bboxes[target_idx]
            analysis_region = self.calculate_analysis_region(target_bbox)
            nid_score = self.nid_scores[target_idx]
            density_class = self.classify_element(nid_score)

            # 绘制屏幕边界
            ax.add_patch(plt.Rectangle((0, 0), self.screen_width, self.screen_height, 
                                     fill=False, edgecolor='black', linewidth=2))

            # 绘制所有元素
            for bbox in all_bboxes:
                color = 'lightgray'
                if bbox.element_id == target_bbox.element_id:
                    color = 'blue'  # 目标元素
                elif self.is_point_in_bbox(self.calculate_element_center(bbox), analysis_region):
                    color = colors[density_class]  # 干扰元素

                ax.add_patch(plt.Rectangle((bbox.x, bbox.y), bbox.width, bbox.height,
                                         fill=True, alpha=0.3, color=color, edgecolor='black'))

            # 绘制分析区域
            ax.add_patch(plt.Rectangle((analysis_region.x, analysis_region.y), 
                                     analysis_region.width, analysis_region.height,
                                     fill=False, edgecolor='blue', linestyle='--', linewidth=2))

            # 绘制目标元素
            ax.add_patch(plt.Rectangle((target_bbox.x, target_bbox.y), 
                                     target_bbox.width, target_bbox.height,
                                     fill=False, edgecolor='blue', linewidth=3))

            ax.set_xlim(0, self.screen_width)
            ax.set_ylim(self.screen_height, 0)  # 反转Y轴以匹配图像坐标
            ax.set_title(f'样本 {target_idx}\nNID={nid_score} ({density_class})')
            ax.set_xlabel('X坐标')
            ax.set_ylabel('Y坐标')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def load_generated_tasks(annotation_file: str) -> Dict:
    """Load the annotated functional regions from a JSON file."""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

def generated_task_attributes(generated_tasks_per_image: Dict, omniparser_dir: str) -> List[Dict]:
    """Generate referring expression grounding samples from annotated regions."""
    samples, bbox_data, screen_widths, screen_heights, surr_bboxes = [], [], [], [], []

    bad = {}
    total = 0
    for image_name, task_meta in generated_tasks_per_image['results'].items():
        if "error" in task_meta:
            bad[image_name] = task_meta['error']
            continue
        
        W, H = get_image_dimensions(task_meta['image_path'])
        image_name_wo_ext = image_name.split('.')[0]
        # Read the surrounding bboxes annotated by OmniParser
        omniparser_file = os.path.join(omniparser_dir, f'{image_name_wo_ext}.json')
        with open(omniparser_file) as f:
            all_bboxes = json.load(f)

        for task_info in task_meta['generated']:
            elements = task_info.get('elements', task_info.get('elements_in_group', []))

            for element in elements:
                elem_id = element["id"]
                bbox = element['revised bbox']

                sample = {
                    "image_name": image_name,
                    "group_id": task_info['group_index'],
                    "idx_in_group": elem_id,
                    "bbox": bbox,
                    "num_elements_in_group": len(elements),
                }
                total += 1
                samples.append(sample)
                bbox_data.append(eval(bbox) if isinstance(bbox, str) else bbox)
                screen_widths.append(W)
                screen_heights.append(H)
                surr_bboxes.append([x['bbox'] for x in all_bboxes]) # All are 0-1 normalized bboxes

    """Type 1: Display the num_elements_in_group statistics."""
    # Extract num_elements_in_group values
    num_elements_list = [sample['num_elements_in_group'] for sample in samples]
    num_elements_array = np.array(num_elements_list)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("Type 1: num_elements_in_group Statistics")
    print("="*60)
    
    print("\n基本统计:")
    print(f"  总数: {len(num_elements_list)}")
    print(f"  平均值: {np.mean(num_elements_array):.2f}")
    print(f"  中位数: {np.median(num_elements_array):.2f}")
    print(f"  标准差: {np.std(num_elements_array):.2f}")
    print(f"  最小值: {np.min(num_elements_array)}")
    print(f"  最大值: {np.max(num_elements_array)}")
    
    # Calculate percentiles5
    percentiles = [25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(num_elements_array, percentiles)
    print("\n百分位数:")
    for p, val in zip(percentiles, percentile_values):
        print(f"  P{p}: {val:.2f}")
    
    # Distribution by value
    unique_values, counts = np.unique(num_elements_array, return_counts=True)
    
    if len(unique_values) <= 10:
        # Show individual values if there are few unique values
        print("\n分布统计 (按值):")
        for val, count in zip(unique_values, counts):
            percentage = (count / len(num_elements_list)) * 100
            print(f"  {int(val)} 个元素: {count} 个样本 ({percentage:.1f}%)")
    else:
        # Group into ranges if too many unique values
        print("\n分布统计 (按范围):")
        bins = [0, 2, 5, 10, 20, np.inf]
        bin_labels = ['1-2', '3-5', '6-10', '11-20', '20+']
        hist, _ = np.histogram(num_elements_array, bins=bins)
        for label, count in zip(bin_labels, hist):
            if count > 0:  # Only show non-zero bins
                percentage = (count / len(num_elements_list)) * 100
                print(f"  {label} 个元素: {count} 个样本 ({percentage:.1f}%)")
    
    print("="*60 + "\n")

    """Type 2: Classify the tasks according to the surrounding elements."""
    # Step 1: Initialize the analyzer.
    analyzer = EnhancedNIDAnalyzer(k_sigma=1.5, alpha=1.0)

    # Step 2: Calculate the NID score.
    nid_scores = analyzer.calculate_all_nid_scores(bbox_data, surr_bboxes, screen_widths, screen_heights)

    # Step 3: Classification by percentile.
    density_analysis_results = analyzer.classify_by_percentiles(percentiles=[33, 67])
    density_classifications = analyzer.classify_all_elements(nid_scores)

    print("\n分类结果:")
    print(f"百分位数: {density_analysis_results['percentiles']}")
    print(f"阈值: {density_analysis_results['thresholds']}")

    print("\n分类规则:")
    for class_name, (min_val, max_val) in density_analysis_results['classification_rules'].items():
        if np.isinf(max_val):
            print(f"  {class_name}: NID > {min_val:.2f}")
        else:
            print(f"  {class_name}: {min_val:.2f} < NID ≤ {max_val:.2f}")

    # Step 4: Statistics
    class_counts = {cls: density_classifications.count(cls) for cls in ['sparse', 'medium', 'dense']}

    print("\n分类分布:")
    for cls, count in class_counts.items():
        percentage = (count / len(density_classifications)) * 100
        print(f"  {cls}: {count} 个元素 ({percentage:.1f}%)")

    # Attach the tags
    for sample, density_class in zip(samples, density_classifications):
        sample['density_class'] = density_class
    ## Calculate CDF
    return samples, bad, total, \
            {
                'nid_scores': nid_scores,
                'density_classifications': density_classifications
            }

def save_samples_to_json(samples: Dict, output_file: str) -> None:
    """Save the grounding samples to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} grounding samples to {output_file}")

def upload_to_huggingface(output_file: str, repo_name: str, hf_token: str) -> None:
    """Upload the generated samples to HuggingFace."""
    api = HfApi()
    repo_url = api.create_repo(repo_name, token=hf_token, exist_ok=True)
    repo = Repository(local_dir=os.path.dirname(output_file), clone_from=repo_url, use_auth_token=hf_token)
    
    # Add the file to the repository
    repo.git_add(output_file)
    repo.git_commit(f"Add grounding samples: {os.path.basename(output_file)}")
    repo.git_push()
    print(f"Uploaded {output_file} to HuggingFace repository: {repo_name}")

def main(annotation_file: str, output_dir: str, repo_name: str = None, hf_token: str = None):
    """Main function to generate and save grounding samples."""
    
    if '*' in annotation_file:
        annotation_files = glob.glob(annotation_file)
    else: annotation_files = [annotation_file]

    for anno_file in annotation_files:
        bmk_name = anno_file.split('/')[-3]
        # Load the annotated regions
        generated_tasks_per_image = load_generated_tasks(anno_file)

        # Cache
        output_file = os.path.join(os.path.dirname(anno_file), os.path.basename(anno_file).split('.')[0] + '_attributes.json')

        # Generate grounding samples
        omniparser_dir = os.path.join(anno_file.split('FuncElemGnd/')[0], 'omniparser')
        samples, bad, total, density_cls_info = generated_task_attributes(generated_tasks_per_image, omniparser_dir)

        print(f"[{bmk_name}] {total} tasks of {len(generated_tasks_per_image['results']) - len(bad)} images processed, {len(bad)} images failed due to annotation errors.")
        
        # Reorganize
        samples_dict = {}
        for sample in samples:
            image_name = sample['image_name']
            
            if sample['group_id'] is None or isinstance(sample['group_id'], str) and not sample['group_id'].isdigit():
                continue
            
            group_id = str(sample['group_id'])
            idx_in_group = str(sample['idx_in_group'])
            
            if image_name not in samples_dict:
                samples_dict[image_name] = {}
            if group_id not in samples_dict[image_name]:
                samples_dict[image_name][group_id] = {}
            samples_dict[image_name][group_id][idx_in_group] = {'bbox': sample['bbox'], 'num_elements_in_group': sample['num_elements_in_group'], 'density_class': sample['density_class']}

        # Save the samples to a JSON file
        save_samples_to_json(samples_dict, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate referring expression grounding samples from annotated functional regions.")
    parser.add_argument("--annotation-file", default="/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncElemGnd/grounding_questions.json", help="Path to the JSON file containing annotated functional regions.")
    parser.add_argument("--output-dir", default=None, help="Directory to save the generated grounding samples.")
    parser.add_argument("--repo-name", default=None, help="HuggingFace repository name to upload the samples.")
    parser.add_argument("--hf-token", default=None, help="HuggingFace API token for uploading.")

    args, _ = parser.parse_known_args()
    main(args.annotation_file, args.output_dir, args.repo_name, args.hf_token)