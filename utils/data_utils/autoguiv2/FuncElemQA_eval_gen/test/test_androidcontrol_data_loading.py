#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for androidcontrol data loading functionality.

This script tests:
1. load_androidcontrol_from_cache function
2. load_annotation_results function (androidcontrol mode)
3. Data structure validation
4. Reannotation data loading
"""

import os
import sys
import argparse

# Add parent directory to path to import the main script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the main script
# Note: We need to import the module using importlib because of the hyphen in filename
import importlib.util
main_script_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'gen_region-func_multichoice-qa_aliyun_androidcontrol.py'
)
spec = importlib.util.spec_from_file_location("main_script", main_script_path)
main_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_script)

# Import functions
load_androidcontrol_from_cache = main_script.load_androidcontrol_from_cache
load_annotation_results = main_script.load_annotation_results
filter_corrected_regions = main_script.filter_corrected_regions
load_parent_child_relationships = main_script.load_parent_child_relationships
debug_print = main_script.debug_print


def test_load_androidcontrol_from_cache(cache_dir: str, verbose: bool = True):
    """Test load_androidcontrol_from_cache function"""
    print("=" * 80)
    print("TEST 1: load_androidcontrol_from_cache")
    print("=" * 80)
    
    if not os.path.exists(cache_dir):
        print(f"❌ ERROR: Cache directory not found: {cache_dir}")
        return False
    
    try:
        results = load_androidcontrol_from_cache(cache_dir, debug=verbose)
        
        if not results:
            print("❌ ERROR: No data loaded from cache directory")
            return False
        
        print(f"\n✅ Successfully loaded {len(results)} images")
        
        # Validate data structure
        print("\nValidating data structure...")
        validation_errors = []
        
        for image_key, image_data in list(results.items())[:5]:  # Check first 5 images
            # Check image_key format
            if '/' not in image_key:
                validation_errors.append(f"Invalid image_key format: {image_key} (expected app/ep/step)")
            
            # Check root_image_path
            if 'root_image_path' not in image_data:
                validation_errors.append(f"Missing root_image_path for {image_key}")
            
            # Check tree.json structure (should have 0-0 node)
            if '0-0' not in image_data:
                validation_errors.append(f"Missing root node (0-0) for {image_key}")
            
            # Count regions (excluding root node)
            regions = {k: v for k, v in image_data.items() 
                      if k != '0-0' and isinstance(v, dict)}
            
            # Check for corrected bbox and reannotated flags
            corrected_count = 0
            reannotated_count = 0
            both_count = 0
            
            for node_data in regions.values():
                has_bbox_corrected = node_data.get('bbox_corrected', False)
                has_reannotated = node_data.get('reannotated', False)
                
                if has_bbox_corrected:
                    corrected_count += 1
                if has_reannotated:
                    reannotated_count += 1
                if has_bbox_corrected and has_reannotated:
                    both_count += 1
            
            if verbose:
                print(f"\n  Image: {image_key}")
                print(f"    Total regions: {len(regions)}")
                print(f"    Regions with bbox_corrected: {corrected_count}")
                print(f"    Regions with reannotated: {reannotated_count}")
                print(f"    Regions with both: {both_count}")
                root_path = image_data.get('root_image_path', 'N/A')
                print(f"    Root image path: {root_path}")
        
        if validation_errors:
            print("\n❌ Validation errors found:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ Data structure validation passed")
        
        # Test filtering
        print("\n" + "=" * 80)
        print("TEST 2: filter_corrected_regions")
        print("=" * 80)
        
        # Test on first image
        first_image_key = list(results.keys())[0]
        first_image_data = results[first_image_key]
        
        # Extract regions
        regions_data = {k: v for k, v in first_image_data.items() 
                        if k != '0-0' and isinstance(v, dict)}
        
        print(f"\nTesting on image: {first_image_key}")
        print(f"  Total regions: {len(regions_data)}")
        
        # Test with filtering enabled
        filtered_regions = filter_corrected_regions(regions_data, only_corrected=True, debug=verbose)
        print(f"  Filtered regions (bbox_corrected + reannotated): {len(filtered_regions)}")
        
        # Test without filtering
        all_regions = filter_corrected_regions(regions_data, only_corrected=False, debug=verbose)
        print(f"  All regions (no filtering): {len(all_regions)}")
        
        if len(filtered_regions) <= len(all_regions):
            print("✅ Filtering test passed")
        else:
            print("❌ Filtering test failed: filtered count > total count")
            return False
        
        # Test load_parent_child_relationships
        print("\n" + "=" * 80)
        print("TEST 3: load_parent_child_relationships (androidcontrol mode)")
        print("=" * 80)
        
        test_image_key = list(results.keys())[0]
        test_image_data = results[test_image_key]
        root_image_path = test_image_data.get('root_image_path')
        
        if root_image_path and os.path.exists(root_image_path):
            parent_child_map = load_parent_child_relationships(
                root_image_path, 
                cache_dir, 
                image_key=test_image_key
            )
            
            if parent_child_map:
                print(f"\n✅ Successfully loaded parent-child relationships for {test_image_key}")
                print(f"   Found {len(parent_child_map)} parent nodes with children")
                
                if verbose:
                    for parent_id, children in list(parent_child_map.items())[:3]:
                        print(f"     {parent_id} -> {len(children)} children")
            else:
                print(f"\n⚠️  No parent-child relationships found for {test_image_key}")
        else:
            print(f"\n⚠️  Skipping parent-child test: root image not found at {root_image_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Exception during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_annotation_results_androidcontrol_mode(cache_dir: str, verbose: bool = True):
    """Test load_annotation_results in androidcontrol mode (no input_file)"""
    print("\n" + "=" * 80)
    print("TEST 4: load_annotation_results (androidcontrol mode, no input_file)")
    print("=" * 80)
    
    try:
        # Test with None input_file (should trigger androidcontrol mode)
        results = load_annotation_results(annotation_file=None, cache_dir=cache_dir)
        
        if not results:
            print("❌ ERROR: No data loaded")
            return False
        
        print(f"\n✅ Successfully loaded {len(results)} images in androidcontrol mode")
        
        # Validate a few sample images
        sample_count = min(3, len(results))
        print(f"\nValidating {sample_count} sample images...")
        
        for i, (image_key, image_data) in enumerate(list(results.items())[:sample_count]):
            print(f"\n  Sample {i+1}: {image_key}")
            
            # Check required fields
            required_fields = ['root_image_path', '0-0']
            missing_fields = [f for f in required_fields if f not in image_data]
            
            if missing_fields:
                print(f"    ❌ Missing fields: {missing_fields}")
                return False
            else:
                print(f"    ✅ All required fields present")
            
            # Count regions
            regions = {k: v for k, v in image_data.items() 
                      if k != '0-0' and isinstance(v, dict)}
            print(f"    Total regions: {len(regions)}")
            
            # Count corrected/reannotated
            corrected = sum(1 for r in regions.values() if r.get('bbox_corrected', False))
            reannotated = sum(1 for r in regions.values() if r.get('reannotated', False))
            both = sum(1 for r in regions.values() 
                      if r.get('bbox_corrected', False) and r.get('reannotated', False))
            
            print(f"    bbox_corrected: {corrected}, reannotated: {reannotated}, both: {both}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Exception during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency(cache_dir: str, verbose: bool = False):
    """Test data consistency between different loading methods"""
    print("\n" + "=" * 80)
    print("TEST 5: Data consistency check")
    print("=" * 80)
    
    try:
        # Load using both methods
        results_direct = load_androidcontrol_from_cache(cache_dir, debug=False)
        results_via_wrapper = load_annotation_results(annotation_file=None, cache_dir=cache_dir)
        
        if len(results_direct) != len(results_via_wrapper):
            print(f"❌ ERROR: Different number of images loaded")
            print(f"   Direct method: {len(results_direct)}")
            print(f"   Wrapper method: {len(results_via_wrapper)}")
            return False
        
        print(f"✅ Both methods loaded same number of images: {len(results_direct)}")
        
        # Check if image_keys match
        keys_direct = set(results_direct.keys())
        keys_wrapper = set(results_via_wrapper.keys())
        
        if keys_direct != keys_wrapper:
            missing_in_wrapper = keys_direct - keys_wrapper
            extra_in_wrapper = keys_wrapper - keys_direct
            
            if missing_in_wrapper:
                print(f"❌ ERROR: {len(missing_in_wrapper)} images missing in wrapper method")
                if verbose:
                    for key in list(missing_in_wrapper)[:5]:
                        print(f"     - {key}")
            
            if extra_in_wrapper:
                print(f"❌ ERROR: {len(extra_in_wrapper)} extra images in wrapper method")
                if verbose:
                    for key in list(extra_in_wrapper)[:5]:
                        print(f"     - {key}")
            
            return False
        
        print("✅ Image keys match between both methods")
        
        # Check data structure consistency for a sample image
        if results_direct:
            sample_key = list(results_direct.keys())[0]
            data_direct = results_direct[sample_key]
            data_wrapper = results_via_wrapper[sample_key]
            
            # Check if both have same node IDs
            nodes_direct = set(k for k in data_direct.keys() if k != 'root_image_path')
            nodes_wrapper = set(k for k in data_wrapper.keys() if k != 'root_image_path')
            
            if nodes_direct != nodes_wrapper:
                print(f"❌ ERROR: Different node IDs for {sample_key}")
                print(f"   Direct: {len(nodes_direct)} nodes")
                print(f"   Wrapper: {len(nodes_wrapper)} nodes")
                return False
            
            if verbose:
                print(f"✅ Data structure consistent for sample image: {sample_key}")
            else:
                print("✅ Data structure consistent")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Exception during consistency check: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test androidcontrol data loading functionality"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Cache directory path (e.g., /mnt/vdb1/hongxin_li/AutoGUIv2/cache/androidcontrol/gemini-2.5-pro-thinking/v2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AndroidControl Data Loading Test Suite")
    print("=" * 80)
    print(f"\nCache directory: {args.cache_dir}")
    print(f"Verbose mode: {args.verbose}")
    print()
    
    # Run all tests
    tests_passed = 0
    # Note: Test 1 includes Test 2 and Test 3 internally
    # Test 1: load_androidcontrol_from_cache (includes Test 2: filter_corrected_regions and Test 3: load_parent_child_relationships)
    # Test 4: load_annotation_results (androidcontrol mode)
    # Test 5: Data consistency
    tests_total = 3
    
    # Test 1: load_androidcontrol_from_cache (includes Test 2 and Test 3)
    if test_load_androidcontrol_from_cache(args.cache_dir, args.verbose):
        tests_passed += 1
    
    # Test 4: load_annotation_results (androidcontrol mode)
    if test_load_annotation_results_androidcontrol_mode(args.cache_dir, args.verbose):
        tests_passed += 1
    
    # Test 5: Data consistency
    if test_data_consistency(args.cache_dir, args.verbose):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

