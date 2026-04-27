#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate annotation file statistics and reannotation data loading.

This script checks:
1. Total number of images in annotation file
2. Number of images with functional regions
3. Number of regions per image
4. Reannotation data loading statistics
5. Cache directory matching statistics
"""

import os
import json
import argparse
import glob
from collections import defaultdict
from typing import Dict

# Colorized output support
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


def load_annotation_results(annotation_file: str, cache_dir: str = None) -> Dict:
    """Load functional region annotation results (copied from main script)"""
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
    
    # If cache_dir provided, load reannotations
    if cache_dir and os.path.isdir(cache_dir):
        print(f"{Fore.CYAN}Loading reannotations from cache directory...{Style.RESET_ALL}")
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
                            best_match = None
                            max_corrections = 0
                            for match in non_backup_matches:
                                correction_count_local = len(glob.glob(os.path.join(match, '*_meta_fix*.json')))
                                if correction_count_local > max_corrections:
                                    max_corrections = correction_count_local
                                    best_match = match
                            nodes_dir = best_match if best_match else non_backup_matches[-1]
                        else:
                            nodes_dir = matches[-1]
                        break
                else:
                    if os.path.isdir(pattern):
                        nodes_dir = pattern
                        break
            
            if not nodes_dir or not os.path.isdir(nodes_dir):
                continue
            
            # Iterate all nodes to find correction and reannotation files
            for node_id in image_data.keys():
                if not isinstance(image_data[node_id], dict):
                    continue
                
                # Load reannotations
                reannotation_files = glob.glob(os.path.join(nodes_dir, f"{node_id}_meta_reannotated*.json"))
                
                if reannotation_files:
                    latest_reannotation = sorted(reannotation_files)[-1]
                    
                    try:
                        with open(latest_reannotation, 'r', encoding='utf-8') as rf:
                            reannotation_data = json.load(rf)
                        
                        # Extract corrected bbox
                        corrected_bbox = reannotation_data.get('corrected_bbox')
                        if corrected_bbox and len(corrected_bbox) == 4:
                            image_data[node_id]['bbox_corrected'] = True
                            bbox_correction_count += 1
                        
                        # Extract revised functionality and description
                        new_functionality = reannotation_data.get('new_functionality', {})
                        if isinstance(new_functionality, dict):
                            revised_func = new_functionality.get('revised functionality')
                            revised_desc = new_functionality.get('revised description')
                            
                            if revised_func or revised_desc:
                                image_data[node_id]['reannotated'] = True
                                reannotation_count += 1
                    
                    except Exception as e:
                        print(f"{Fore.YELLOW}Warning: Failed to load reannotation file {latest_reannotation}: {e}{Style.RESET_ALL}")
                        continue
        
        print(f"{Fore.GREEN}Successfully loaded {reannotation_count} reannotations{Style.RESET_ALL}")
        if bbox_correction_count > 0:
            print(f"  - {bbox_correction_count} regions with corrected bbox")
        if reannotation_count > 0:
            print(f"  - {reannotation_count} regions with revised functionality/description")
    
    return results


def analyze_annotation_file(annotation_file: str, cache_dir: str = None):
    """Analyze annotation file and print detailed statistics"""
    
    print("=" * 80)
    print(f"{Fore.CYAN}Annotation File Statistics{Style.RESET_ALL}")
    print("=" * 80)
    print()
    
    # Step 1: Check raw file structure
    print(f"{Fore.YELLOW}Step 1: Reading raw annotation file...{Style.RESET_ALL}")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    raw_results = raw_data.get('results', {})
    print(f"  Raw file contains {Fore.GREEN}{len(raw_results)}{Style.RESET_ALL} image entries")
    print()
    
    # Step 2: Load processed annotation results
    print(f"{Fore.YELLOW}Step 2: Loading and processing annotation data...{Style.RESET_ALL}")
    annotation_results = load_annotation_results(annotation_file, cache_dir)
    print(f"  Processed annotation data contains {Fore.GREEN}{len(annotation_results)}{Style.RESET_ALL} images")
    print()
    
    # Step 3: Analyze images and regions
    print(f"{Fore.YELLOW}Step 3: Analyzing images and functional regions...{Style.RESET_ALL}")
    
    images_with_regions = 0
    images_without_regions = 0
    total_regions = 0
    region_count_distribution = defaultdict(int)
    
    images_with_corrected_regions = 0
    images_with_reannotated_regions = 0
    images_with_both = 0
    
    total_bbox_corrected = 0
    total_reannotated = 0
    total_both = 0
    
    for image_key, image_data in annotation_results.items():
        if not isinstance(image_data, dict):
            continue
        
        # Count functional regions (exclude root node '0-0')
        regions = {k: v for k, v in image_data.items() 
                  if k != '0-0' and isinstance(v, dict)}
        region_count = len(regions)
        
        if region_count > 0:
            images_with_regions += 1
            total_regions += region_count
            region_count_distribution[region_count] += 1
            
            # Check for corrected/reannotated regions
            has_bbox_corrected = False
            has_reannotated = False
            
            for region_id, region_data in regions.items():
                if region_data.get('bbox_corrected', False):
                    has_bbox_corrected = True
                    total_bbox_corrected += 1
                if region_data.get('reannotated', False):
                    has_reannotated = True
                    total_reannotated += 1
                if region_data.get('bbox_corrected', False) and region_data.get('reannotated', False):
                    total_both += 1
            
            if has_bbox_corrected:
                images_with_corrected_regions += 1
            if has_reannotated:
                images_with_reannotated_regions += 1
            if has_bbox_corrected and has_reannotated:
                images_with_both += 1
        else:
            images_without_regions += 1
    
    print(f"  Images with functional regions: {Fore.GREEN}{images_with_regions}{Style.RESET_ALL}")
    if images_without_regions > 0:
        print(f"  Images without functional regions: {Fore.YELLOW}{images_without_regions}{Style.RESET_ALL} (will be skipped)")
    print(f"  Total functional regions: {Fore.GREEN}{total_regions}{Style.RESET_ALL}")
    print()
    
    # Region count distribution
    if region_count_distribution:
        print(f"  Region count distribution:")
        sorted_counts = sorted(region_count_distribution.items())
        for count, num_images in sorted_counts[:10]:  # Show top 10
            print(f"    {count} regions: {num_images} images")
        if len(sorted_counts) > 10:
            print(f"    ... ({len(sorted_counts) - 10} more)")
    print()
    
    # Step 4: Reannotation statistics
    if cache_dir:
        print(f"{Fore.YELLOW}Step 4: Reannotation data statistics...{Style.RESET_ALL}")
        print(f"  Regions with bbox_corrected=True: {Fore.GREEN}{total_bbox_corrected}{Style.RESET_ALL}")
        print(f"  Regions with reannotated=True: {Fore.GREEN}{total_reannotated}{Style.RESET_ALL}")
        print(f"  Regions with BOTH flags: {Fore.GREEN}{total_both}{Style.RESET_ALL}")
        print()
        print(f"  Images with bbox_corrected regions: {Fore.GREEN}{images_with_corrected_regions}{Style.RESET_ALL}")
        print(f"  Images with reannotated regions: {Fore.GREEN}{images_with_reannotated_regions}{Style.RESET_ALL}")
        print(f"  Images with BOTH: {Fore.GREEN}{images_with_both}{Style.RESET_ALL}")
        print()
        
        # Calculate percentages
        if total_regions > 0:
            pct_bbox = (total_bbox_corrected / total_regions) * 100
            pct_reannotated = (total_reannotated / total_regions) * 100
            pct_both = (total_both / total_regions) * 100
            print(f"  Coverage:")
            print(f"    bbox_corrected: {pct_bbox:.1f}% ({total_bbox_corrected}/{total_regions})")
            print(f"    reannotated: {pct_reannotated:.1f}% ({total_reannotated}/{total_regions})")
            print(f"    both: {pct_both:.1f}% ({total_both}/{total_regions})")
            print()
    
    # Step 5: Summary
    print("=" * 80)
    print(f"{Fore.CYAN}Summary{Style.RESET_ALL}")
    print("=" * 80)
    print(f"Total images in file: {Fore.GREEN}{len(raw_results)}{Style.RESET_ALL}")
    print(f"Valid images (with regions): {Fore.GREEN}{images_with_regions}{Style.RESET_ALL}")
    print(f"Total functional regions: {Fore.GREEN}{total_regions}{Style.RESET_ALL}")
    if cache_dir:
        print(f"Regions ready for processing (both flags): {Fore.GREEN}{total_both}{Style.RESET_ALL}")
        if total_regions > 0:
            print(f"  Coverage: {Fore.GREEN}{(total_both/total_regions)*100:.1f}%{Style.RESET_ALL}")
    print("=" * 80)
    
    # Step 6: Sample first few images for detailed inspection
    print()
    print(f"{Fore.YELLOW}Step 6: Sample inspection (first 5 images)...{Style.RESET_ALL}")
    sample_count = 0
    for image_key, image_data in annotation_results.items():
        if sample_count >= 5:
            break
        if not isinstance(image_data, dict):
            continue
        
        regions = {k: v for k, v in image_data.items() 
                  if k != '0-0' and isinstance(v, dict)}
        if len(regions) == 0:
            continue
        
        sample_count += 1
        print(f"\n  Image {sample_count}: {Fore.CYAN}{os.path.basename(image_key)}{Style.RESET_ALL}")
        print(f"    Total regions: {len(regions)}")
        
        if cache_dir:
            bbox_count = sum(1 for r in regions.values() if r.get('bbox_corrected', False))
            reannotated_count = sum(1 for r in regions.values() if r.get('reannotated', False))
            both_count = sum(1 for r in regions.values() 
                           if r.get('bbox_corrected', False) and r.get('reannotated', False))
            print(f"    bbox_corrected: {bbox_count}/{len(regions)}")
            print(f"    reannotated: {reannotated_count}/{len(regions)}")
            print(f"    both: {both_count}/{len(regions)}")
            
            if both_count == 0:
                print(f"    {Fore.RED}⚠️  No regions with both flags - will be skipped!{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(
        description="Test script to validate annotation file statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input-file", required=True,
                       help="Input functional region annotation JSON file")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Cache directory path for reading reannotations (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"{Fore.RED}Error: Annotation file not found: {args.input_file}{Style.RESET_ALL}")
        return
    
    if args.cache_dir and not os.path.isdir(args.cache_dir):
        print(f"{Fore.YELLOW}Warning: Cache directory not found: {args.cache_dir}{Style.RESET_ALL}")
        print("  Continuing without cache directory...")
        args.cache_dir = None
    
    analyze_annotation_file(args.input_file, args.cache_dir)


if __name__ == "__main__":
    main()

