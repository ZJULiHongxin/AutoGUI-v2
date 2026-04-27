"""
Visualize generated questions and their targeted elements

This script loads the generated questions JSON file, maps element indices
back to the OmniParser parsing result, and visualizes the targeted elements
with their questions overlaid on the screenshot.

Usage:
    # List all available images with questions
    python visualize_questions.py --list
    
    # Visualize ALL questions from ALL groups across ALL samples (batch mode)
    python visualize_questions.py --all
    
    # Visualize all and save to custom directory
    python visualize_questions.py --all --output-dir my_visualizations
    
    # Visualize the first image with questions (saves to test.png)
    python visualize_questions.py
    
    # Visualize a specific image
    python visualize_questions.py --image /path/to/image.png
    
    # Save to a different output path
    python visualize_questions.py --output my_visualization.png

Note: 
    The element IDs shown in the generated questions (e.g., [0], [1], [2]) are 
    indices within each similar group, NOT the global OmniParser indices.
    This script handles the mapping automatically by using the similar_groups 
    data from the omniparser_embeddings and detection results.
"""

import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from pprint import pprint

def load_omniparser_result(image_path: str) -> List[Dict]:
    """Load OmniParser parsing result for an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of parsed elements from OmniParser
    """
    # Derive the OmniParser result file path
    p = Path(image_path).resolve()
    stem = p.stem
    parts = list(p.parts)
    base_dir = p.parent
    
    # Find the base directory (parent of 'images' folder)
    if 'images' in parts:
        idx = parts.index('images')
        base_dir = Path(*parts[:idx])
    
    omniparser_file = base_dir / 'omniparser' / f'{stem}.json'
    
    if not omniparser_file.exists():
        raise FileNotFoundError(f"OmniParser result not found: {omniparser_file}")
    
    with open(omniparser_file, 'r') as f:
        omniparser_result = json.load(f)
    
    return omniparser_result


def load_similar_groups_mapping(image_path: str) -> Dict:
    """Load the similar groups mapping to get real element indices
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary mapping group indices to element indices
    """
    p = Path(image_path).resolve()
    stem = p.stem
    parts = list(p.parts)
    base_dir = p.parent
    
    if 'images' in parts:
        idx = parts.index('images')
        base_dir = Path(*parts[:idx])
    
    npz_path = base_dir / 'omniparser_embeddings' / f'{stem}.npz'
    
    if not npz_path.exists():
        return {}
    
    data = np.load(str(npz_path), allow_pickle=True)
    similar_groups = data.get('similar_groups', [])
    
    return similar_groups.tolist() if hasattr(similar_groups, 'tolist') else similar_groups


def map_element_id_to_real_index(group_index: int, element_id: int, 
                                  similar_groups: List, 
                                  detection_result: Dict) -> int:
    """Map element ID in question to real OmniParser index
    
    Args:
        group_index: Index of the similar group
        element_id: Element ID within the group (from question)
        similar_groups: Similar groups from embeddings
        detection_result: Detection result containing element mappings
        
    Returns:
        Real index in OmniParser result
    """
    # First, try to get from detection result which has the mapping
    if 'similar_groups' in detection_result:
        groups = detection_result.get('similar_groups', {})
        if isinstance(groups, dict):
            group_key = str(group_index)
            if group_key in groups:
                group = groups[group_key]
                elements = group.get('elements', [])
                for elem in elements:
                    if elem.get('id') == element_id:
                        # The element might have original_index or we need to derive it
                        # from the similar_groups mapping
                        break
    
    # Use similar groups to get the real index
    if similar_groups and group_index < len(similar_groups):
        group = similar_groups[group_index]
        if element_id < len(group):
            return group[element_id]
    
    # Fallback: return element_id as-is
    return element_id


def draw_element_with_question(image: np.ndarray, bbox: List[float], 
                                element_id: int, question: str, 
                                color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a bounding box and question label on the image
    
    Args:
        image: RGB image
        bbox: Normalized bounding box [x_min, y_min, x_max, y_max] in 0-1 range
        element_id: Element ID to display
        question: Question text to display
        color: RGB color tuple
        
    Returns:
        Image with drawn box and text
    """
    H, W = image.shape[:2]
    
    # Convert normalized bbox to pixel coordinates
    x1 = int(bbox[0] * W)
    y1 = int(bbox[1] * H)
    x2 = int(bbox[2] * W)
    y2 = int(bbox[3] * H)
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Draw element ID label
    label = f"[{element_id}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = y1 - 5 if y1 - label_h - 5 > 0 else y1 + label_h + 5
    
    # Draw label background
    cv2.rectangle(image, (x1, label_y - label_h - 2), 
                  (x1 + label_w, label_y + 2), color, -1)
    cv2.putText(image, label, (x1, label_y), font, font_scale, (0, 0, 0), thickness)
    
    # Draw question text (wrapped if too long)
    max_width = W - 50
    words = question.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        (text_w, text_h), _ = cv2.getTextSize(test_line, font, 0.5, 1)
        
        if text_w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Limit to 3 lines
    lines = lines[:3]
    if len(question.split()) > len(" ".join(lines).split()):
        lines[-1] = lines[-1][:50] + "..."
    
    # Draw question text at bottom of image
    text_y_start = H - (len(lines) + 1) * 25
    for i, line in enumerate(lines):
        text_y = text_y_start + i * 25
        # Draw background for text
        (text_w, text_h), _ = cv2.getTextSize(line, font, 0.5, 1)
        cv2.rectangle(image, (20, text_y - text_h - 2), 
                     (20 + text_w + 10, text_y + 5), (255, 255, 255), -1)
        cv2.putText(image, line, (25, text_y), font, 0.5, color, 1)
    
    return image


def list_available_images(questions_file: str):
    """List all images with generated questions
    
    Args:
        questions_file: Path to grounding_questions.json
    """
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    questions_results = questions_data.get('results', {})
    
    print(f"\n{'='*80}")
    print(f"Available Images with Generated Questions")
    print(f"{'='*80}\n")
    
    count = 0
    for img_key, img_data in questions_results.items():
        if isinstance(img_data, dict) and 'generated' in img_data:
            generated = img_data['generated']
            if generated and len(generated) > 0:
                count += 1
                num_groups = len(generated)
                num_questions = sum(len(g.get('questions', [])) for g in generated)
                print(f"{count}. {img_key}")
                print(f"   Groups: {num_groups}, Total Questions: {num_questions}")
                print()
    
    print(f"{'='*80}")
    print(f"Total: {count} images with questions")
    print(f"{'='*80}\n")


def visualize_all_questions(questions_file: str, image_src_dir: str,
                           detection_file: str, output_dir: str = "visualizations"):
    """Visualize all questions from all groups across all samples
    
    Args:
        questions_file: Path to grounding_questions.json
        image_src_dir: Root directory containing images
        detection_file: Path to similar_elements_anno.json
        output_dir: Directory to save all visualizations
    """
    # Load questions
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    questions_results = questions_data.get('results', {})
    
    # Load detection results
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    detection_results = detection_data.get('results', {})
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Batch Visualization - Processing All Samples")
    print(f"{'='*80}\n")
    
    total_images = 0
    total_groups = 0
    total_questions = 0
    
    # Iterate through all images
    for img_idx, (img_key, img_data) in enumerate(questions_results.items()):
        if not isinstance(img_data, dict) or 'generated' not in img_data:
            continue
        
        generated = img_data['generated']
        if not generated or len(generated) == 0:
            continue
        
        total_images += 1
        
        # Get image path
        image_path = img_data.get('image_path')
        if not image_path or not os.path.exists(image_path):
            image_path = os.path.join(image_src_dir, img_key)
            if not os.path.exists(image_path):
                print(f"Skipping {img_key}: Image not found")
                continue
        
        print(f"\n[{total_images}] Processing: {os.path.basename(image_path)}")
        print(f"    Groups: {len(generated)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"    ERROR: Failed to load image")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process each group
        for group_data in generated:
            total_groups += 1
            group_index = group_data.get('group_index', 0)
            questions = group_data.get('questions', [])
            elements = group_data.get('elements', [])
            visual_sim = group_data.get('visual_similarity', 'N/A')
            
            if not questions:
                continue
            
            total_questions += len(questions)
            
            # Create visualization for this group
            image_copy = image_rgb.copy()
            H, W = image_copy.shape[:2]
            
            # Colors
            colors = [
                (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
                (255, 100, 255), (100, 255, 255), (255, 150, 100), (150, 100, 255),
            ]
            
            # Draw all elements
            for elem in elements:
                elem_id = elem.get('id', 0)
                bbox_norm = elem.get('revised bbox', [])
                
                if len(bbox_norm) != 4:
                    continue
                
                bbox = [b / 1000.0 for b in bbox_norm]
                color = colors[elem_id % len(colors)]
                
                x1 = int(bbox[0] * W)
                y1 = int(bbox[1] * H)
                x2 = int(bbox[2] * W)
                y2 = int(bbox[3] * H)
                
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)
                
                label = f"[{elem_id}]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (label_w, label_h), _ = cv2.getTextSize(label, font, 0.8, 2)
                label_y = y1 - 5 if y1 - label_h - 5 > 0 else y1 + label_h + 5
                
                # cv2.rectangle(image_copy, (x1, label_y - label_h - 2), (x1 + label_w, label_y + 2), (0, 0, 0), -1)
                cv2.putText(image_copy, label, (x1, label_y), font, 0.8, color, 2)

            # Draw info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 25

            # Save this group
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_group{group_index}.png"
            output_path = os.path.join(output_dir, output_filename)

            output_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test.png', output_image)

            print(f"    Group {group_index}: {len(questions)} questions → {output_filename}:")
            print(json.dumps(group_data['questions'], indent=4))
            1+1

    print(f"\n{'='*80}")
    print(f"Batch Visualization Complete!")
    print(f"  Total Images: {total_images}")
    print(f"  Total Groups: {total_groups}")
    print(f"  Total Questions: {total_questions}")
    print(f"  Output Directory: {output_dir}")
    print(f"{'='*80}\n")


def visualize_questions(questions_file: str, image_src_dir: str, 
                        detection_file: str, output_path: str = "test.png",
                        sample_image: str = None):
    """Visualize generated questions and targeted elements
    
    Args:
        questions_file: Path to grounding_questions.json
        image_src_dir: Root directory containing images
        detection_file: Path to similar_elements_anno.json
        output_path: Output path for visualization
        sample_image: Optional specific image key to visualize
    """
    # Load questions
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    questions_results = questions_data.get('results', {})
    
    # Load detection results
    with open(detection_file, 'r') as f:
        detection_data = json.load(f)
    
    detection_results = detection_data.get('results', {})
    
    # Get a sample image to visualize
    if sample_image is None:
        # Pick the first image with questions
        for img_key, img_data in questions_results.items():
            if isinstance(img_data, dict) and 'generated' in img_data:
                generated = img_data['generated']
                if generated and len(generated) > 0:
                    sample_image = img_key
                    break
        
        if sample_image is None:
            print("No images with generated questions found!")
            return
    
    print(f"Visualizing: {sample_image}")
    
    # Get image data
    img_data = questions_results.get(sample_image)
    if not img_data or 'generated' not in img_data:
        print(f"No questions found for image: {sample_image}")
        return
    
    # Get image path
    image_path = img_data.get('image_path')
    if not image_path or not os.path.exists(image_path):
        # Try to construct path
        image_path = os.path.join(image_src_dir, sample_image)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load OmniParser result
    try:
        omniparser_result = load_omniparser_result(image_path)
    except Exception as e:
        print(f"Error loading OmniParser result: {e}")
        return
    
    # Load similar groups mapping
    try:
        similar_groups = load_similar_groups_mapping(image_path)
    except Exception as e:
        print(f"Warning: Could not load similar groups mapping: {e}")
        similar_groups = []
    
    # Get detection result for this image
    detection_result = detection_results.get(sample_image, {})
    
    # Color palette
    colors = [
        (255, 100, 100),   # Red
        (100, 255, 100),   # Green
        (100, 100, 255),   # Blue
        (255, 255, 100),   # Yellow
        (255, 100, 255),   # Magenta
        (100, 255, 255),   # Cyan
        (255, 150, 100),   # Orange
        (150, 100, 255),   # Purple
    ]
    
    # Visualize each group
    generated = img_data['generated']
    print(f"Found {len(generated)} groups with questions")
    
    # Create a copy for each group
    for idx, group_data in enumerate(generated):
        image_copy = image_rgb.copy()
        
        group_index = group_data.get('group_index', idx)
        questions = group_data.get('questions', [])
        elements = group_data.get('elements', [])
        visual_sim = group_data.get('visual_similarity', 'N/A')
        
        print(f"\nGroup {group_index}: {len(questions)} questions, {len(elements)} elements")
        print(f"Visual Similarity: {visual_sim}")
        
        # Print element details
        print("\n  Elements in this group:")
        for elem in elements:
            elem_id = elem.get('id', 0)
            desc = elem.get('detailed desctiption', 'N/A')[:100]
            func = elem.get('unique functionality', 'N/A')[:100]
            print(f"    [{elem_id}] {desc}...")
            print(f"         Functionality: {func}...")
        
        # Draw all elements in this group
        for elem in elements:
            elem_id = elem.get('id', 0)
            bbox_norm = elem.get('revised bbox', [])
            
            if len(bbox_norm) != 4:
                continue
            
            # Convert normalized bbox (0-1000) to 0-1 range
            bbox = [b / 1000.0 for b in bbox_norm]
            
            color = colors[elem_id % len(colors)]
            
            # Get real index in OmniParser
            real_idx = map_element_id_to_real_index(
                group_index, elem_id, similar_groups, detection_result
            )
            
            # Draw bounding box
            H, W = image_copy.shape[:2]
            x1 = int(bbox[0] * W)
            y1 = int(bbox[1] * H)
            x2 = int(bbox[2] * W)
            y2 = int(bbox[3] * H)
            
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)
            
            # Draw element ID
            label = f"[{elem_id}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (label_w, label_h), _ = cv2.getTextSize(label, font, 0.8, 2)
            label_y = y1 - 5 if y1 - label_h - 5 > 0 else y1 + label_h + 5
            
            #cv2.rectangle(image_copy, (x1, label_y - label_h - 2), (x1 + label_w, label_y + 2), (0, 0, 0), -1)
            cv2.putText(image_copy, label, (x1, label_y), font, 0.8, color, 2)
        
        # Draw all questions with better formatting
        if questions:
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30

            y_offset += 30
            
            # Print questions to console with better formatting
            for q_idx, q_data in enumerate(questions):
                target_id = q_data.get('target_element_id', 0)
                referring_exprs = q_data.get('referring_expressions', {})
                
                print(f"\n  Question for Element [{target_id}]:")
                for action, action_data in referring_exprs.items():
                    question_text = action_data.get('question', '')
                    action_intent = action_data.get('action_intent', '')
                    print(f"    - {action}: {question_text}")
                    if action_intent:
                        print(f"      Intent: {action_intent[:100]}...")
            
            # Draw a sample question on the image (first one)
            first_q = questions[0]
            target_id = first_q.get('target_element_id', 0)
            referring_exprs = first_q.get('referring_expressions', {})
            
            if referring_exprs:
                action = list(referring_exprs.keys())[0]
                question_text = referring_exprs[action].get('question', '')
                
                # Wrap text
                max_width = W - 100
                words = question_text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    (text_w, text_h), _ = cv2.getTextSize(test_line, font, 0.5, 1)
                    
                    if text_w <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)

                for i, line in enumerate(lines):
                    y_pos = y_offset + 40 + i * 25
                    cv2.putText(image_copy, line, (20, y_pos), font, 0.45, (50, 50, 50), 1)
        
        # Convert back to BGR for saving
        output_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        
        # Save (overwrite test.png each time, or save all groups)
        if idx == 0:  # Only save the first group
            cv2.imwrite(output_path, output_image)
            print(f"\nSaved visualization to: {output_path}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Visualize generated questions and targeted elements"
    )
    
    parser.add_argument("--questions-file", 
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/grounding_questions.json",
                       help="Path to grounding_questions.json")
    parser.add_argument("--detection-file",
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/similar_elements_anno.json",
                       help="Path to similar_elements_anno.json")
    parser.add_argument("--image-src-dir",
                       default="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/images",
                       help="Root directory containing images")
    parser.add_argument("--output",
                       default="test.png",
                       help="Output path for visualization")
    parser.add_argument("--image",
                       default=None,
                       help="Optional: specific image key to visualize")
    parser.add_argument("--list",
                       action="store_true",
                       help="List all available images with questions")
    parser.add_argument("--output-dir",
                       default=os.path.join(os.path.dirname(__file__), "visualizations"),
                       help="Output directory for batch visualization (used with --all)")
    
    args, _ = parser.parse_known_args()
    
    # Check if files exist
    if not os.path.exists(args.questions_file):
        print(f"Questions file not found: {args.questions_file}")
        return
    
    # List mode
    if args.list:
        list_available_images(args.questions_file)
        return
    
    # Batch visualization mode
    if not os.path.exists(args.detection_file):
        print(f"Detection file not found: {args.detection_file}")
        return
    
    visualize_all_questions(
        args.questions_file,
        args.image_src_dir,
        args.detection_file,
        args.output_dir
    )



if __name__ == "__main__":
    main()

