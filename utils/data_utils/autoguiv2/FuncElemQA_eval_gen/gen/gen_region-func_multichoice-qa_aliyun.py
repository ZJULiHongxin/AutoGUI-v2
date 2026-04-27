#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multiple-choice QA tasks based on functional regions.

Simplified version with 2 generation modes:
- Grounding mode: Aliyun Embedding → Gemini vision verification → Text-based region grounding QA
- Captioning mode: Aliyun Embedding → Gemini vision verification → Image annotation + captioning QA

Pipeline:
1. Aliyun text-embedding-v4: Initial grouping based on semantic text similarity
2. Gemini Vision: Verify visual similarity, filter out invalid groups, refine regions
3. Question Generation: Generate QA based on verified groups

This two-stage approach ensures groups are both semantically AND visually similar.
"""

import os
import json
import cv2
import base64
import time
import argparse
import multiprocessing
import traceback
import random
import glob
import copy
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from PIL import Image
from io import BytesIO

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

# Import utilities
import sys
sys.path.append('/'.join(__file__.split('/')[:-4]))
from utils.openai_utils.openai import OpenAIModel

# Import OpenAI for embedding generation (Aliyun DashScope)
try:
    from openai import OpenAI as OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    OpenAIClient = None

# Set random seeds
random.seed(42)
np.random.seed(42)

# Global variables for multiprocessing
generator_instance = None
debug_flag = False
generation_mode = "grounding_mode"  # "grounding_mode" or "captioning_mode"
output_file_path = None
cache_dir_global = None
embedding_model_instance = None
only_corrected_regions = True
per_image_output_dir = None

# Prompt templates

# Gemini Visual Verification Prompt (validates and refines groups from Aliyun embedding)
VISUAL_VERIFICATION_PROMPT = """You are a GUI understanding expert. Your task is to verify and refine groups of visually similar UI regions.

**Background:**
We have identified a potential group of UI regions based on visual description similarity (using text embeddings of their visual appearance descriptions). However, text-based visual descriptions may not perfectly capture true visual similarity. Your role is to:
1. Verify if the initially grouped regions are truly visually similar
2. Check if any other candidate regions should be added to this group
3. Ensure the final group size is between 2-5 regions

**Criteria for Valid Group Regions:**
- Regions should be **visually similar** (similar icon type, similar appearance, similar color/shape)
- Regions should have **different functionalities** in their respective contexts
- Regions should be **confusing for AI agents** because they look alike but behave differently
- **CRITICAL**: Regions must NOT have overlapping bounding boxes (no parent-child or containment relationships)
  * Check that no region's bbox is contained within another region's bbox
  * If bbox A is inside bbox B, they cannot be in the same group

**Initially Identified Group (based on visual description embeddings from Qwen3-Embedding):**
{initial_group_info}

**Other Candidate Regions (not in the initial group):**
{other_candidates_info}

**Your Task (Two-Stage Process):**

**Stage 1: Validate Initial Group Members**
- Examine each region in the initial group
- Check if it truly meets all criteria (visually similar + different functionality + confusing + NO bbox overlap)
- Check for bbox overlaps: If any two regions have overlapping bboxes, keep only ONE of them
- Mark regions that don't fit as keep=false

**Stage 2: Supplement from Other Candidates**
- Examine all other candidate regions
- Check if any of them are visually similar to the group and should be added
- **BEFORE adding**: Verify the candidate's bbox does NOT overlap with ANY existing group member's bbox
- Only add candidates that pass ALL criteria requirement

**Stage 3: Size Control**
- After Stages 1 & 2, count the final group size
- If size < 2: Mark the entire group as INVALID
- If size = 2-5: Keep all regions, mark as VALID
- If size > 5: Select the 5 most visually similar and confusing regions (ensuring no bbox overlaps), mark others as keep=false

**IMPORTANT: Judging Logic**
- Judge the group as VALID if **at least 2 regions** (after refinement) meet ALL criteria
- It's acceptable if some initial regions don't meet the criteria - just set their "keep" to false
- You MUST check all other candidates to see if they should be added
- Example: If initial group has 3 regions but only 1 fits, check other candidates. If you find 1+ qualifying candidates, add them and mark as VALID
- Only mark as INVALID if fewer than 2 regions can meet all criteria after checking ALL candidates

**Output Format (JSON):**
{{
  "valid": true/false,
  "rejection_reason": "Explanation if invalid" (only if valid=false),
  "visual_similarity_description": "What makes these regions look similar" (only if valid=true),
  "kept_region_ids": [
    {{
      "region_id": "1-0",
      "keep": true,  // false to remove from group
      "source": "initial_group",  // or "added_from_candidates"
      "reason": "Brief reason for keeping or removing this region"
    }},
    {{
      "region_id": "1-2",
      "keep": true,
      "source": "initial_group",
      "reason": "Matches the visual pattern perfectly"
    }},
    {{
      "region_id": "1-5",
      "keep": false,
      "source": "initial_group",
      "reason": "Different visual style, doesn't match the group"
    }},
    {{
      "region_id": "3-8",
      "keep": true,
      "source": "added_from_candidates",
      "reason": "Found in other candidates, matches the visual pattern"
    }}
  ],
  "final_group_size": 3,
  "adjustments_made": "Removed 1 region from initial group (not visually similar), added 1 region from candidates"
}}

**Requirements:**
- Be strict: only approve groups that truly meet all criteria
- Visual similarity is critical - don't approve groups where regions just happen to have similar text
- Functionality differences must be clear and meaningful
- **NO bbox overlaps allowed**: Reject or remove any region whose bbox overlaps with another group member
- You MUST examine all other candidates, not just the initial group
- Final group size must be 2-5 regions (if >5, select the best 5 with no overlaps)
- **IMPORTANT**: Do NOT modify bbox coordinates or functionality descriptions - they are already manually corrected and accurate

Now analyze the screenshot and evaluate this group:"""

# Gemini Element Selection Prompt (for oversized groups that need refinement)
ELEMENT_SELECTION_PROMPT = """You are a GUI understanding expert. Your task is to select the BEST 2-5 regions from a large group of visually similar UI regions.

**Background:**
We have identified a group of UI regions that are visually similar and functionally different. However, the group might have MORE than 5 regions, which is too many for a multiple-choice question. Your role is to:
1. Select the 2-5 MOST representative and confusing regions
2. Ensure selected regions have NO bbox overlaps
3. Maximize visual similarity while maintaining functional diversity

**Group Information:**
{group_info}

**Selection Criteria (in order of priority):**
1. **Visual Similarity**: Select regions that look MOST similar to each other
2. **Functional Diversity**: Ensure selected regions have clearly DIFFERENT functionalities
3. **Confusion Potential**: Prioritize regions that would be MOST confusing for AI agents
4. **NO Bbox Overlaps**: Selected regions must NOT have overlapping bounding boxes
5. **Optimal Size**: Select 2-5 regions (prefer 3-4 if possible for better question quality)

**Your Task:**
1. Analyze all regions in the group (ONLY the regions listed in "Group Information" above)
2. Identify the core "visual pattern" (e.g., all are blue icons, all are text buttons)
3. Select 2-5 regions that BEST represent this pattern
4. **CRITICAL**: You MUST ONLY select regions from the group list provided above. Do NOT select any region_id that is NOT in the "Group Information" section.
5. Ensure NO two selected regions have overlapping bboxes
6. Provide clear reasoning for your selection

**Output Format (JSON):**
{{
  "selected_region_ids": [
    {{
      "region_id": "1-0",
      "selection_reason": "Why this region was selected"
    }},
    {{
      "region_id": "1-2",
      "selection_reason": "Why this region was selected"
    }},
    // ... 2-5 regions total
  ],
  "excluded_region_ids": [
    {{
      "region_id": "1-3",
      "exclusion_reason": "Why this region was excluded"
    }},
    // ... other excluded regions
  ],
  "visual_pattern": "Description of the common visual pattern",
  "final_count": 3,
  "selection_summary": "Brief summary of selection strategy"
}}

**Requirements:**
- **CRITICAL**: You MUST ONLY select region_ids from the "Group Information" list above. Any region_id NOT in that list will be rejected.
- You MUST select between 2-5 regions (no more, no less)
- Selected regions MUST have NO bbox overlaps
- Prioritize regions with highest visual similarity
- Ensure functional diversity among selected regions
- Provide clear reasoning for each selection/exclusion
- **IMPORTANT**: Only return region_id - do NOT copy bbox or functionality, they are already accurate in our database

Now analyze the screenshot and select the best regions:"""

# Grounding Mode: Generate question based on text information (Mode 1 logic)
GENERATE_QUESTION_GROUNDING_MODE_PROMPT = """Based on the following visually similar regions with different functionalities, generate a multiple-choice question to test if an agent can predict the outcome or purpose of interacting with them.

**Region Information:**
{element_info}

**Task:**
Generate a question asking the agent to predict what will happen when interacting with a specific region, or what purpose/goal the interaction serves.

**Question Format Guidelines:**
Focus on PREDICTION and OUTCOME, such as:
- "If you want to [achieve specific goal], which region should you click?"
- "Which region will [result/outcome] when clicked?"
- "To [specific purpose/intention], which region would you interact with?"
- "Clicking which region will lead to [specific result/interface]?"

**Output Format (JSON):**
{{
  "question": "If you want to [specific goal/purpose], which region should you click?",
  "options": [
    {{
      "label": "A",
      "region_id": "1-0"
    }},
    {{
      "label": "B",
      "region_id": "1-2"
    }},
    // ... include ALL regions from the group as options
  ],
  "correct_answer": "A",  // The label of correct option
  "explanation": "Why clicking this region will achieve the goal/produce the expected result"
}}

**Requirements:**
- **CRITICAL**: You MUST include ALL regions from the group information as options (do not select a subset)
- Each option only needs "label" and "region_id" - do NOT copy bbox or description from the input (we already have them)
- Question must focus on PREDICTION: what will happen, what goal will be achieved, what result will occur
- Avoid questions like "Which region is for X?" - instead ask "To achieve X, which region should you use?"
- Options should be ordered randomly (not by position)
- All options except the correct one should be plausible distractors
- Explanation should clearly describe the predicted outcome/result

Please generate the question now:"""

# Captioning Mode: Generate question with annotated image (Mode 5 logic)
GENERATE_QUESTION_CAPTIONING_MODE_PROMPT = """You are an expert in GUI analysis. I will show you a GUI screenshot with ONE UI region highlighted by a red bounding box.

**This region belongs to a group of visually similar regions with different functionalities:**
{group_elements_info}

**Target region (the one with red box):**
{target_element_info}

**Task:**
Create a multiple-choice question asking about what will happen or what goal will be achieved when clicking the circled region.
The options should describe the outcomes/purposes of ALL regions in this group (including the target region).

**Question Format:**
Focus on PREDICTION and OUTCOME:
- "If you click the circled region, what will happen?"
- "What is the expected result of clicking the circled region?"
- "What goal can be achieved by clicking the circled region?"

**Output Format (JSON):**
{{
  "question": "If you click the circled region, what will happen?",
  "options": [
    {{
      "label": "A",
      "region_id": "1-0",
      "functionality": "Description of expected outcome/result (rephrased from functionality)"
    }},
    {{
      "label": "B",
      "region_id": "1-2",
      "functionality": "Description of expected outcome/result (rephrased from functionality)"
    }},
    {{
      "label": "C",
      "region_id": "2-1",
      "functionality": "Description of expected outcome/result (rephrased from functionality)"
    }}
    // ... include all group regions as options
  ],
  "correct_answer": "A",  // The label of the option containing target region's outcome
  "explanation": "Why this outcome/result will occur when clicking the circled region, and why other outcomes (from similar-looking regions) are incorrect"
}}

**Requirements:**
- Question must focus on PREDICTION: what will happen, what result will occur, what goal will be achieved
- Rephrase each functionality as an outcome/result (e.g., "Save document" → "The document will be saved")
- Include ALL group regions' outcomes as options (the group may contain 2 or more regions)
- One option must be the target region's outcome (the circled one)
- Options should be shuffled randomly (not by spatial position)
- Each option must include the region_id to identify which region it corresponds to
- Explanation should describe the predicted outcome and highlight how to distinguish the target from other visually similar regions

Please generate the question now:"""


def debug_print(message: str, level: str = "info") -> None:
    """Colorized debug print using colorama.
    
    Levels: info (cyan), step (blue), success (green), warn (yellow), error (red), title (magenta)
    """
    level_to_color = {
        'info': Fore.CYAN,
        'step': Fore.BLUE,
        'success': Fore.GREEN,
        'warn': Fore.YELLOW,
        'error': Fore.RED,
        'title': Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def on_off(value: bool) -> str:
    """Format boolean value display"""
    return f"{Fore.GREEN}ON{Style.RESET_ALL}" if value else f"{Fore.YELLOW}OFF{Style.RESET_ALL}"


def draw_bbox_on_image(image_path: str, bbox: List[float], is_normalized: bool = True, 
                       save_path: str = None) -> str:
    """Draw bounding box on image and return base64 encoded result
    
    Args:
        image_path: Path to original image
        bbox: Bounding box [x1, y1, x2, y2]
        is_normalized: Whether bbox coordinates are normalized (0-1)
        save_path: Optional path to save the annotated image
    
    Returns:
        Base64 encoded image with bbox drawn
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    if is_normalized:
        x1, y1, x2, y2 = bbox
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
    else:
        x1, y1, x2, y2 = map(int, bbox)
    
    # Draw red bounding box (thick line for visibility)
    color = (0, 0, 255)  # Red in BGR
    thickness = max(3, int(min(w, h) * 0.003))  # Adaptive thickness
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Save annotated image if path is provided
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, img)
    
    # Convert to base64
    return image_to_base64(img)


def get_region_text_content(region_data: Dict) -> str:
    """Extract text content from region for similarity comparison"""
    texts = []
    
    # Get functionality text
    functionality = region_data.get('functionality', {})
    if isinstance(functionality, dict):
        func_text = functionality.get('wo_context') or functionality.get('with_context', '')
    else:
        func_text = str(functionality) if functionality else ''
    if func_text and func_text != 'Unknown':
        texts.append(func_text)
    
    # Get description text
    description = region_data.get('description', {})
    if isinstance(description, dict):
        desc_text = description.get('wo_context') or description.get('with_context', '')
    else:
        desc_text = str(description) if description else ''
    if desc_text and desc_text != 'Unknown':
        texts.append(desc_text)
    
    return ' '.join(texts)


def generate_text_embeddings_realtime(texts: List[str], embedding_client) -> List[np.ndarray]:
    """Generate embeddings for texts using Aliyun DashScope API
    
    Args:
        texts: List of text strings to embed
        embedding_client: OpenAI client instance configured for Aliyun DashScope
    
    Returns:
        List of embedding vectors (numpy arrays)
    
    Note:
        Aliyun text-embedding-v4 has a batch size limit of 10 texts per request,
        and each text must not exceed 8192 tokens.
    """
    if not texts or embedding_client is None:
        return []
    
    try:
        # Aliyun API batch size limit: 10
        BATCH_SIZE = 10
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            # Call Aliyun embedding API
            response = embedding_client.embeddings.create(
                model="text-embedding-v4",
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting: small delay between batches
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    except Exception as e:
        debug_print(f"Error generating embeddings from Aliyun API: {e}", level="error")
        traceback.print_exc()
        return []


def compute_embeddings_for_regions(regions_data: Dict, embedding_model, debug: bool = False) -> Dict[str, np.ndarray]:
    """Compute embeddings for all functional regions using Aliyun API
    
    Args:
        regions_data: Functional regions data (dict of region_id -> region_info)
        embedding_model: OpenAI client instance for Aliyun DashScope
        debug: Whether to print debug info
    
    Returns:
        Dictionary mapping region_id to embedding vector (wo_context description for visual similarity)
    """
    if embedding_model is None:
        return {}
    
    try:
        region_ids = list(regions_data.keys())
        texts = []
        
        # Extract wo_context description text for each region (use description for visual similarity, not functionality)
        for region_id in region_ids:
            region_data = regions_data[region_id]
            # Get description wo_context (for finding visually similar elements)
            description = region_data.get('description', {})
            if isinstance(description, dict):
                desc_text = description.get('wo_context') or description.get('with_context', '')
            else:
                desc_text = str(description) if description else ''
            
            texts.append(desc_text if desc_text else '')
        
        if debug:
            debug_print(f"   Generating embeddings for {len(texts)} regions using Aliyun text-embedding-v4...", level="info")
            debug_print(f"   Using description (visual) for similarity, not functionality", level="info")
        
        # Generate all embeddings in batch
        embeddings = generate_text_embeddings_realtime(texts, embedding_model)
        
        # Map region_id to embedding
        embeddings_map = {}
        for i, region_id in enumerate(region_ids):
            if i < len(embeddings) and embeddings[i] is not None:
                embeddings_map[region_id] = embeddings[i]
        
        if debug:
            debug_print(f"   Successfully generated {len(embeddings_map)} embeddings", level="success")
        
        return embeddings_map
    
    except Exception as e:
        if debug:
            debug_print(f"   Error computing embeddings: {e}", level="error")
        return {}


def compute_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0-100, scaled like fuzzy match)
    """
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Scale to 0-100 range (like fuzz.token_set_ratio)
    similarity_score = max(0, min(100, cosine_sim * 100))
    
    return similarity_score


def check_bbox_overlap(bbox1: List[float], bbox2: List[float], threshold: float = 0.0) -> bool:
    """Check if two bounding boxes overlap (IoU > threshold)
    
    Args:
        bbox1: First bbox [x1, y1, x2, y2] (normalized or absolute)
        bbox2: Second bbox [x1, y1, x2, y2] (normalized or absolute)
        threshold: IoU threshold (default 0.0 means any overlap is detected)
    
    Returns:
        True if boxes overlap (IoU > threshold), False otherwise
    """
    if not bbox1 or len(bbox1) < 4 or not bbox2 or len(bbox2) < 4:
        return False
    
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
    
    # Compute intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return False  # No overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area <= 0:
        return False
    
    # Compute IoU
    iou = intersection_area / union_area
    
    return iou > threshold


def validate_group_no_bbox_overlaps(region_ids: List[str], regions_data: Dict, 
                                   debug: bool = False) -> tuple[bool, List[str]]:
    """Validate that no two elements in the group have overlapping bboxes
    
    Args:
        region_ids: List of region IDs in the group
        regions_data: Complete region annotation data
        debug: Whether to print debug info
    
    Returns:
        Tuple of (is_valid, list_of_overlapping_pairs)
    """
    overlapping_pairs = []
    
    for i in range(len(region_ids)):
        for j in range(i + 1, len(region_ids)):
            region_id_i = region_ids[i]
            region_id_j = region_ids[j]
            
            if region_id_i not in regions_data or region_id_j not in regions_data:
                continue
            
            bbox_i = regions_data[region_id_i].get('bbox_global_norm') or regions_data[region_id_i].get('bbox_global', [])
            bbox_j = regions_data[region_id_j].get('bbox_global_norm') or regions_data[region_id_j].get('bbox_global', [])
            
            if check_bbox_overlap(bbox_i, bbox_j):
                overlapping_pairs.append((region_id_i, region_id_j))
                if debug:
                    debug_print(f"   ⚠️  Bbox overlap detected: {region_id_i} ↔ {region_id_j}", level="warn")
    
    is_valid = len(overlapping_pairs) == 0
    return is_valid, overlapping_pairs


def find_duplicate_elements_between_groups(groups: List[Dict]) -> List[tuple]:
    """Find groups that share 2 or more common elements
    
    Args:
        groups: List of group dicts with 'region_ids' key
    
    Returns:
        List of tuples (group_idx_1, group_idx_2, common_elements)
        where common_elements is a list of region_ids present in both groups
    """
    duplicate_pairs = []
    
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_i_ids = set(groups[i]['region_ids'])
            group_j_ids = set(groups[j]['region_ids'])
            
            # Find common elements
            common = group_i_ids & group_j_ids
            
            # If 2 or more common elements, mark for merging
            if len(common) >= 2:
                duplicate_pairs.append((i, j, list(common)))
    
    return duplicate_pairs


def filter_parent_child_from_group(region_ids: List[str], parent_child_map: Dict[str, set], 
                                  debug: bool = False) -> List[str]:
    """Filter out parent-child pairs from a group, keeping only non-related elements
    
    Args:
        region_ids: List of region IDs in the group
        parent_child_map: Dictionary mapping parent_id to set of children IDs
        debug: Whether to print debug info
    
    Returns:
        Filtered list of region IDs with parent-child pairs removed
    """
    if not parent_child_map or len(region_ids) <= 1:
        return region_ids
    
    # Build set of all parent-child pairs for quick lookup
    parent_child_pairs = set()
    for parent_id, children_ids in parent_child_map.items():
        for child_id in children_ids:
            parent_child_pairs.add((parent_id, child_id))
            parent_child_pairs.add((child_id, parent_id))
    
    # Check all pairs in the group
    filtered_ids = []
    removed_ids = set()
    
    for i, region_id in enumerate(region_ids):
        # Skip if already removed
        if region_id in removed_ids:
            continue
        
        # Check if this element has parent-child relationship with any kept element
        has_parent_child_relation = False
        for kept_id in filtered_ids:
            if (region_id, kept_id) in parent_child_pairs:
                has_parent_child_relation = True
                removed_ids.add(region_id)
                if debug:
                    debug_print(f"      Filtered out {region_id} (parent-child relationship with {kept_id})", level="info")
                break
        
        if not has_parent_child_relation:
            filtered_ids.append(region_id)
    
    return filtered_ids


def merge_groups(groups: List[Dict], merge_indices: List[tuple], 
                parent_child_map: Dict[str, set] = None, debug: bool = False) -> List[Dict]:
    """Merge groups that share 2+ common elements, and filter out parent-child pairs
    
    Args:
        groups: List of group dicts
        merge_indices: List of tuples (idx1, idx2, common_elements) indicating which groups to merge
        parent_child_map: Dictionary mapping parent_id to set of children IDs (for filtering)
        debug: Whether to print debug info
    
    Returns:
        List of merged groups with parent-child pairs filtered out
    """
    if not merge_indices:
        return groups
    
    # Build a union-find structure to identify all groups that should be merged together
    parent = {i: i for i in range(len(groups))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union all groups that should be merged
    for idx1, idx2, _ in merge_indices:
        union(idx1, idx2)
    
    # Group indices by their root parent
    merged_groups_map = {}
    for i in range(len(groups)):
        root = find(i)
        if root not in merged_groups_map:
            merged_groups_map[root] = []
        merged_groups_map[root].append(i)
    
    # Create merged groups
    merged_groups = []
    for root, indices in merged_groups_map.items():
        if len(indices) == 1:
            # No merging needed, keep original group
            merged_groups.append(groups[indices[0]])
        else:
            # Merge multiple groups
            all_region_ids = set()
            first_group = groups[indices[0]]
            
            for idx in indices:
                all_region_ids.update(groups[idx]['region_ids'])
            
            # Filter out parent-child pairs from merged group
            filtered_region_ids = list(all_region_ids)
            if parent_child_map:
                original_count = len(filtered_region_ids)
                filtered_region_ids = filter_parent_child_from_group(
                    filtered_region_ids, parent_child_map, debug
                )
                removed_count = original_count - len(filtered_region_ids)
                
                if debug and removed_count > 0:
                    debug_print(f"   Filtered out {removed_count} parent-child elements from merged group", level="info")
            
            merged_group = {
                'group_id': first_group['group_id'],
                'region_ids': filtered_region_ids,
                'description': f"Merged from {len(indices)} groups with overlapping elements",
                'similarity_reason': first_group.get('similarity_reason', ''),
                'merged_from': indices,
                'needs_refinement': True  # Flag: merged groups MUST be refined by Gemini
            }
            merged_groups.append(merged_group)
            
            if debug:
                debug_print(f"   Merged groups {indices} into one group with {len(filtered_region_ids)} elements (after parent-child filtering)", level="info")
    
    return merged_groups


def load_parent_child_relationships(image_path: str, cache_dir: str = None) -> Dict[str, set]:
    """Load parent-child relationships from tree.json in cache directory
    
    Args:
        image_path: Path to the image file
        cache_dir: Cache directory path (if None, will try to infer)
    
    Returns:
        Dictionary mapping region_id to set of its children IDs
        Returns empty dict if tree.json not found
    """
    try:
        # Infer cache directory if not provided
        if cache_dir is None:
            # Try to find cache from common paths
            possible_paths = [
                "/mnt/vdb1/hongxin_li/AutoGUIv2/cache",
                "/mnt/nvme0n1p1/hongxin_li/AutoGUIv2/cache",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    cache_dir = path
                    break
        
        if not cache_dir or not os.path.exists(cache_dir):
            return {}
        
        # Get image stem (filename without extension)
        image_stem = Path(image_path).stem
        
        # Try to find tree.json in cache directory structure
        tree_json_patterns = [
            os.path.join(cache_dir, "*", "*", "*", image_stem, "tree.json"),
            os.path.join(cache_dir, "**", image_stem, "tree.json"),
        ]
        
        tree_json_path = None
        for pattern in tree_json_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                # Filter out backup directories
                non_backup = [m for m in matches if '_bak' not in m]
                if non_backup:
                    tree_json_path = non_backup[0]
                    break
        
        if not tree_json_path or not os.path.exists(tree_json_path):
            return {}
        
        # Load tree.json
        with open(tree_json_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        # Build parent-child mapping
        parent_child_map = {}
        for node_id, node_data in tree_data.items():
            if isinstance(node_data, dict) and 'children' in node_data:
                children = node_data.get('children', [])
                if children:
                    parent_child_map[node_id] = set(children)
        
        return parent_child_map
    
    except Exception as e:
        # Silently fail and return empty dict
        return {}


def group_regions_by_text_similarity(regions_data: Dict, parent_child_map: Dict[str, set] = None, 
                                    embeddings_map: Dict[str, np.ndarray] = None,
                                    debug: bool = False) -> List[Dict]:
    """Group functional regions by visual description similarity using embedding-based method
    
    Note: Uses description (visual appearance) for similarity, not functionality.
    This finds visually similar elements that may have different functions.
    
    Args:
        regions_data: Functional regions data (dict of region_id -> region_info)
        parent_child_map: Dictionary mapping parent_id to set of children IDs (optional)
        embeddings_map: Dictionary mapping region_id to embedding vector from descriptions (required)
        debug: Whether to print debug info
    
    Returns:
        List of group dicts with keys: {'group_id': int, 'region_ids': [str, ...], 'description': str}
    """
    if debug:
        debug_print(f"Grouping {len(regions_data)} functional regions by visual description similarity", level="step")
        if parent_child_map:
            debug_print(f"   Using parent-child relationships to exclude direct parent-child pairs", level="info")
    
    if len(regions_data) < 2:
        if debug:
            debug_print(f"   Not enough regions to form groups", level="warn")
        return []
    
    # Build a set of all parent-child pairs for quick lookup
    parent_child_pairs = set()
    if parent_child_map:
        for parent_id, children_ids in parent_child_map.items():
            for child_id in children_ids:
                parent_child_pairs.add((parent_id, child_id))
                parent_child_pairs.add((child_id, parent_id))
        
        if debug:
            debug_print(f"   Found {len(parent_child_pairs) // 2} parent-child relationships to exclude", level="info")
    
    # Check embeddings_map
    if not embeddings_map:
        error_msg = "embeddings_map is empty. Embedding generation may have failed."
        if debug:
            debug_print(f"   Error: {error_msg}", level="error")
        raise RuntimeError(error_msg)
    
    # Extract region IDs
    region_ids = list(regions_data.keys())
    n = len(region_ids)
    
    if debug:
        loaded_count = len([rid for rid in region_ids if rid in embeddings_map])
        debug_print(f"   Loaded embeddings for {loaded_count}/{n} regions", level="info")
    
    # Calculate text similarity matrix using embeddings
    text_similarity = np.ones((n, n), dtype=np.int32) * 100  # Initialize with 100 (self-similarity)
    
    excluded_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            region_id_i = region_ids[i]
            region_id_j = region_ids[j]
            
            # Check if this is a parent-child pair
            if parent_child_pairs and (region_id_i, region_id_j) in parent_child_pairs:
                text_similarity[i, j] = 0
                text_similarity[j, i] = 0
                excluded_count += 1
                continue
            
            # Get embeddings
            embedding_i = embeddings_map.get(region_id_i)
            embedding_j = embeddings_map.get(region_id_j)
            
            if embedding_i is not None and embedding_j is not None:
                # Compute cosine similarity
                score = compute_embedding_similarity(embedding_i, embedding_j)
                text_similarity[i, j] = int(score)
                text_similarity[j, i] = int(score)
            else:
                # If embeddings missing, similarity = 0
                text_similarity[i, j] = 0
                text_similarity[j, i] = 0
    
    if debug:
        debug_print(f"   Computed text similarity matrix ({n}x{n})", level="info")
        if excluded_count > 0:
            debug_print(f"   Excluded {excluded_count} parent-child pairs from similarity calculation", level="info")
    
    # Find similar groups (threshold: text_sim >= 60)
    TEXT_SIM_THRESHOLD = 60
    
    visited = set()
    groups = []
    
    # Sort regions by maximum similarity to others (descending)
    max_similarities = np.max(text_similarity, axis=1)
    sorted_indices = np.argsort(max_similarities)[::-1]
    
    for idx in sorted_indices:
        if idx in visited:
            continue
        
        # Find all regions with text_sim >= threshold
        similar_indices = []
        for j in range(n):
            if j != idx and text_similarity[idx, j] >= TEXT_SIM_THRESHOLD:
                similar_indices.append(j)
        
        # Only create group if we have at least 1 similar region (total 2+ regions)
        if len(similar_indices) >= 1:
            # No size limit at this stage - let Gemini vision verification decide the optimal group size
            # Sort by similarity score (descending) for better organization
            similar_with_scores = [(j, text_similarity[idx, j]) for j in similar_indices]
            similar_with_scores.sort(key=lambda x: -x[1])
            similar_indices = [j for j, _ in similar_with_scores]
            
            group_indices = [idx] + similar_indices
            
            # Convert to region IDs
            group_region_ids = [region_ids[i] for i in group_indices]
            
            # Mark as visited
            visited.add(idx)
            visited.update(similar_indices)
            
            # Add to groups
            groups.append({
                'group_id': len(groups) + 1,
                'region_ids': group_region_ids,
                'description': f'Visually similar regions based on description embeddings',
                'similarity_reason': f'Regions have high visual description similarity (>= {TEXT_SIM_THRESHOLD})'
            })
            
            if debug:
                debug_print(f"   Group {len(groups)}: {len(group_region_ids)} regions", level="info")
    
    if debug:
        debug_print(f"   Created {len(groups)} groups with >= 2 regions", level="success")
    
    return groups


def image_to_base64(image_or_path):
    """Convert an image to a base64 data URL.
    
    Accepts:
      - str: filesystem path to the image
      - np.ndarray: OpenCV image (BGR or grayscale)
      - PIL.Image.Image: PIL image instance
    """
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }

    # Case 1: path string
    if isinstance(image_or_path, str):
        ext = Path(image_or_path).suffix.lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    # Case 2: OpenCV image (numpy array)
    if isinstance(image_or_path, np.ndarray):
        success, buf = cv2.imencode('.png', image_or_path)
        if not success:
            raise ValueError("Failed to encode numpy image to PNG")
        binary_data = buf.tobytes()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    # Case 3: PIL Image
    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        fmt = image_or_path.format if image_or_path.format else 'PNG'
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else 'image/png'
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str), numpy array, or PIL Image")


def parse_json_response(response: str, debug: bool = False) -> Any:
    """Parse JSON response from LLM"""
    try:
        # Remove <think> tags
        if '</think>' in response:
            response = response.split('</think>')[-1]
        
        # Remove markdown code blocks
        import re
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Find JSON content
        bracket_idx = response.find('[')
        brace_idx = response.find('{')
        
        if bracket_idx == -1 and brace_idx == -1:
            if debug:
                debug_print(f"No JSON found in response (length: {len(response)})", level="error")
            return None
        elif bracket_idx == -1:
            start_idx = brace_idx
            is_array = False
        elif brace_idx == -1:
            start_idx = bracket_idx
            is_array = True
        else:
            if bracket_idx < brace_idx:
                start_idx = bracket_idx
                is_array = True
            else:
                start_idx = brace_idx
                is_array = False
        
        end_idx = response.rfind(']' if is_array else '}')
        if end_idx == -1:
            if debug:
                debug_print(f"No closing bracket found", level="error")
            return None
        
        json_str = response[start_idx:end_idx+1]
        data = json.loads(json_str)
        return data
        
    except json.JSONDecodeError as e:
        if debug:
            debug_print(f"JSON parsing failed: {e}", level="error")
        return None
    except Exception as e:
        if debug:
            debug_print(f"Unexpected error during parsing: {e}", level="error")
        return None


def save_result_per_image(image_key: str, result: Dict, output_dir: str, metadata: Dict = None):
    """Save result for a single image to its own JSON file
    
    For "both" mode, saves to grounding_mode/ and captioning_mode/ subdirectories.
    For single modes, saves directly to output_dir.
    Also saves grouping information to grouping_info/ subdirectory.
    
    Args:
        image_key: Image identifier
        result: Processing result for this image
        output_dir: Output directory for individual JSON files
        metadata: Optional metadata to include
    
    Returns:
        List of output file paths
    """
    if metadata is None:
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    # Extract image basename (without extension) for filename
    image_basename = os.path.splitext(os.path.basename(image_key))[0]
    
    generation_mode = result.get('generation_mode', 'unknown')
    output_files = []
    
    # Check if processing failed
    if 'error' in result:
        # Skip saving errors for "No regions available after filtering" - this is expected behavior
        # when using --only-corrected-regions and all regions are filtered out
        error_msg = result.get('error', '')
        if 'No regions available after filtering' in error_msg:
            # Silently skip - this is not a real error, just normal filtering
            return output_files
        
        # Save error result for other errors
        error_dir = os.path.join(output_dir, "errors")
        os.makedirs(error_dir, exist_ok=True)
        error_output_file = os.path.join(error_dir, f"{image_basename}_error.json")
        
        error_output = {
            "metadata": metadata,
            "image_key": image_key,
            "result": result
        }
        
        with open(error_output_file, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=2, ensure_ascii=False)
        output_files.append(error_output_file)
        return output_files
    
    # Check if no questions were generated (but processing succeeded)
    # For "both" mode, check both grounding and captioning
    # For single mode, check num_questions
    if generation_mode == "both":
        num_grounding = result.get('num_questions_grounding', 0)
        num_captioning = result.get('num_questions_captioning', 0)
        has_questions = num_grounding > 0 or num_captioning > 0
    else:
        num_questions = result.get('num_questions', 0)
        has_questions = num_questions > 0
    
    # Check if there are final_groups (groups that passed all validation)
    final_groups = result.get('final_groups', [])
    has_final_groups = len(final_groups) > 0
    
    # Skip saving for expected filtering/warning cases (not real errors)
    # These are normal processing outcomes that should be silently skipped
    warning_msg = result.get('warning', '')
    skip_warnings = [
        "No text-similar groups found",
        "No groups passed Gemini first pass",
        "No groups passed size check",
        "No groups passed final validation"
    ]
    if warning_msg in skip_warnings:
        # Silently skip - these are expected filtering behaviors, not errors
        return output_files
    
    # If there are final_groups but no questions generated, this is an error
    # Save to errors/ folder for debugging
    if has_final_groups and not has_questions:
        error_dir = os.path.join(output_dir, "errors")
        os.makedirs(error_dir, exist_ok=True)
        error_output_file = os.path.join(error_dir, f"{image_basename}_error.json")
        
        error_output = {
            "metadata": metadata,
            "image_key": image_key,
            "error": "Question generation failed despite having valid groups",
            "result": {
                "image_path": result.get('image_path', image_key),
                "num_regions": result.get('num_regions', 0),
                "num_questions": result.get('num_questions', 0) if generation_mode != "both" else 0,
                "num_questions_grounding": result.get('num_questions_grounding', 0) if generation_mode == "both" else None,
                "num_questions_captioning": result.get('num_questions_captioning', 0) if generation_mode == "both" else None,
                "generation_mode": result.get('generation_mode', generation_mode),
                "grouping_method": result.get('grouping_method', 'unknown'),
                "final_groups_count": len(final_groups),
                "initial_groups": result.get('initial_groups', []),
                "gemini_first_pass": result.get('gemini_first_pass'),
                "after_bbox_cleanup": result.get('after_bbox_cleanup'),
                "after_merge": result.get('after_merge'),
                "gemini_second_pass": result.get('gemini_second_pass'),
                "final_groups": final_groups,
                "rejected_groups": result.get('rejected_groups', []),
                "stats": result.get('stats', {}),
                "processing_time": result.get('processing_time', 0),
                "warning": result.get('warning', 'No questions generated despite having final_groups')
            }
        }
        
        with open(error_output_file, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=2, ensure_ascii=False)
        output_files.append(error_output_file)
        return output_files
    
    # Save grouping information (always save, regardless of mode)
    grouping_dir = os.path.join(output_dir, "grouping_info")
    os.makedirs(grouping_dir, exist_ok=True)
    grouping_output_file = os.path.join(grouping_dir, f"{image_basename}_grouping.json")
    
    grouping_data = {
        "metadata": metadata,
        "image_key": image_key,
        "image_path": result.get('image_path', image_key),
        "num_regions": result.get('num_regions', 0),
        "grouping_method": result.get('grouping_method', 'unknown'),
        # 保存所有阶段状态
        "initial_groups": result.get('initial_groups', []),
        "gemini_first_pass": result.get('gemini_first_pass'),
        "after_bbox_cleanup": result.get('after_bbox_cleanup'),
        "after_merge": result.get('after_merge'),
        "gemini_second_pass": result.get('gemini_second_pass'),
        "final_groups": result.get('final_groups', []),
        "rejected_groups": result.get('rejected_groups', []),
        # 统计信息
        "stats": result.get('stats', {}),
        "processing_time": result.get('processing_time', 0)
    }
    
    with open(grouping_output_file, 'w', encoding='utf-8') as f:
        json.dump(grouping_data, f, indent=2, ensure_ascii=False)
    output_files.append(grouping_output_file)
    
    if generation_mode == "both":
        # Save Grounding results
        grounding_dir = os.path.join(output_dir, "grounding_mode")
        os.makedirs(grounding_dir, exist_ok=True)
        grounding_output_file = os.path.join(grounding_dir, f"{image_basename}_result.json")
        
        grounding_result = {
            "image_path": result.get('image_path', image_key),
            "num_questions": result.get('num_questions_grounding', 0),
            "num_regions": result.get('num_regions', 0),
            "questions": result.get('questions_grounding', []),
            "generation_mode": "grounding_mode",
            "grouping_method": result.get('grouping_method', 'unknown'),
            "stats": result.get('stats', {}),
            "processing_time": result.get('processing_time', 0)
        }
        
        grounding_metadata = metadata.copy()
        grounding_metadata['generation_mode'] = 'grounding_mode'
        
        grounding_output = {
            "metadata": grounding_metadata,
            "image_key": image_key,
            "result": grounding_result
        }
        
        with open(grounding_output_file, 'w', encoding='utf-8') as f:
            json.dump(grounding_output, f, indent=2, ensure_ascii=False)
        output_files.append(grounding_output_file)
        
        # Save Captioning results
        captioning_dir = os.path.join(output_dir, "captioning_mode")
        os.makedirs(captioning_dir, exist_ok=True)
        captioning_output_file = os.path.join(captioning_dir, f"{image_basename}_result.json")
        
        captioning_result = {
            "image_path": result.get('image_path', image_key),
            "num_questions": result.get('num_questions_captioning', 0),
            "num_regions": result.get('num_regions', 0),
            "questions": result.get('questions_captioning', []),
            "generation_mode": "captioning_mode",
            "grouping_method": result.get('grouping_method', 'unknown'),
            "stats": result.get('stats', {}),
            "processing_time": result.get('processing_time', 0)
        }
        
        captioning_metadata = metadata.copy()
        captioning_metadata['generation_mode'] = 'captioning_mode'
        
        captioning_output = {
            "metadata": captioning_metadata,
            "image_key": image_key,
            "result": captioning_result
        }
        
        with open(captioning_output_file, 'w', encoding='utf-8') as f:
            json.dump(captioning_output, f, indent=2, ensure_ascii=False)
        output_files.append(captioning_output_file)
    else:
        # Single mode - save directly
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{image_basename}_result.json")
        
        output = {
            "metadata": metadata,
            "image_key": image_key,
            "result": result
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        output_files.append(output_file)
    
    return output_files


def load_processed_images_from_dir(output_dir: str) -> set:
    """Load list of already processed images from per-image output directory
    
    Supports both single mode (files directly in output_dir) and "both" mode 
    (files in grounding_mode/, captioning_mode/, and grouping_info/ subdirectories).
    Images are marked as processed if they have result files AND grouping_info files.
    
    Args:
        output_dir: Directory containing individual result JSON files
    
    Returns:
        Set of processed image keys
    """
    if not os.path.exists(output_dir):
        return set()
    
    processed_images = set()
    
    # Check for both mode subdirectories
    grounding_dir = os.path.join(output_dir, "grounding_mode")
    captioning_dir = os.path.join(output_dir, "captioning_mode")
    grouping_dir = os.path.join(output_dir, "grouping_info")
    
    dirs_to_check = []
    if os.path.isdir(grounding_dir) and os.path.isdir(captioning_dir):
        # Both mode: check both subdirectories + grouping_info, mark as processed only if present in ALL THREE
        grounding_images = set()
        captioning_images = set()
        grouping_images = set()
        
        for filename in os.listdir(grounding_dir):
            if filename.endswith('_result.json'):
                filepath = os.path.join(grounding_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'image_key' in data:
                            grounding_images.add(data['image_key'])
                        elif 'result' in data and 'image_path' in data['result']:
                            grounding_images.add(data['result']['image_path'])
                except Exception:
                    continue
    
        for filename in os.listdir(captioning_dir):
            if filename.endswith('_result.json'):
                filepath = os.path.join(captioning_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'image_key' in data:
                            captioning_images.add(data['image_key'])
                        elif 'result' in data and 'image_path' in data['result']:
                            captioning_images.add(data['result']['image_path'])
                except Exception:
                    continue
        
        # Check grouping_info directory
        if os.path.isdir(grouping_dir):
            for filename in os.listdir(grouping_dir):
                if filename.endswith('_grouping.json'):
                    filepath = os.path.join(grouping_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'image_key' in data:
                                grouping_images.add(data['image_key'])
                            elif 'image_path' in data:
                                grouping_images.add(data['image_path'])
                    except Exception:
                        continue
        
        # Only mark as processed if present in ALL THREE directories
        processed_images = grounding_images & captioning_images & grouping_images
    else:
        # Single mode: check output_dir directly for result files and grouping_info for grouping files
        dirs_to_check = [output_dir]
        result_images = set()
        
        for dir_path in dirs_to_check:
            if not os.path.isdir(dir_path):
                continue
            
            for filename in os.listdir(dir_path):
                if filename.endswith('_result.json'):
                    filepath = os.path.join(dir_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'image_key' in data:
                                result_images.add(data['image_key'])
                            elif 'result' in data and 'image_path' in data['result']:
                                result_images.add(data['result']['image_path'])
                    except Exception:
                        continue
        
        # Check grouping_info directory
        grouping_images = set()
        if os.path.isdir(grouping_dir):
            for filename in os.listdir(grouping_dir):
                if filename.endswith('_grouping.json'):
                    filepath = os.path.join(grouping_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'image_key' in data:
                                grouping_images.add(data['image_key'])
                            elif 'image_path' in data:
                                grouping_images.add(data['image_path'])
                    except Exception:
                        continue
        
        # Mark as processed if present in both result files and grouping_info
        processed_images = result_images & grouping_images
    
    return processed_images


class MultiChoiceQAGenerator:
    """Multiple-choice QA generator with 2 modes: Grounding and Captioning"""
    
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o", 
                 max_retries: int = 3, temperature: float = 0.1):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=16000
        )
        self.max_retries = max_retries
    
    def select_best_elements_from_oversized_group(self, image_path: str, group: Dict, 
                                                  regions_data: Dict, debug: bool = False) -> Dict:
        """Use Gemini to select the best 2-5 elements from an oversized group (>5 elements)
        
        Args:
            image_path: Path to the screenshot
            group: Group with >5 elements
            regions_data: Complete region annotation data
            debug: Whether to output debug info
        
        Returns:
            Selection result dict with keys: {'selected_region_ids': list, 'visual_pattern': str, ...}
            Returns None if selection fails
        """
        if debug:
            debug_print(f"Selecting best elements from oversized group {group['group_id']} ({len(group['region_ids'])} elements)", level="step")
        
        # Convert original image to base64 (no annotation)
        try:
            original_image_base64 = image_to_base64(image_path)
        except Exception as e:
            if debug:
                debug_print(f"   Failed to read image: {e}", level="error")
            return None
        
        # Prepare group information text
        group_info_lines = []
        for i, region_id in enumerate(group['region_ids']):
            if region_id in regions_data:
                region = regions_data[region_id]
                
                # Get functionality
                functionality = region.get('functionality', {})
                if isinstance(functionality, dict):
                    func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
                else:
                    func_text = str(functionality)
                
                # Get description
                description = region.get('description', {})
                if isinstance(description, dict):
                    desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
                else:
                    desc_text = str(description)
                
                # Get bbox
                bbox = region.get('bbox_global_norm', region.get('bbox_global', []))
                
                # Get type
                region_type = region.get('type', 'Unknown')
                
                group_info_lines.append(f"Element {i+1}:")
                group_info_lines.append(f"  - Region ID: {region_id}")
                group_info_lines.append(f"  - Type: {region_type}")
                group_info_lines.append(f"  - BBox: {bbox}")
                group_info_lines.append(f"  - Functionality: {func_text}")
                group_info_lines.append(f"  - Description: {desc_text}")
                group_info_lines.append("")
        
        group_info_text = "\n".join(group_info_lines)
        group_info_text += f"\n\nTotal Elements: {len(group['region_ids'])}"
        group_info_text += f"\nRequired: Select 2-5 BEST elements"
        
        # Prepare prompt
        prompt = ELEMENT_SELECTION_PROMPT.format(group_info=group_info_text)
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': original_image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                if debug:
                    debug_print(f"   Selection attempt {attempt + 1}/{self.max_retries}", level="info")
                
                start_time = time.time()
                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.2 if attempt == 0 else 0.4, timeout=600
                )
                elapsed = time.time() - start_time
                
                if not success:
                    error_msg = str(response) if response else "Unknown error"
                    if debug:
                        debug_print(f"   API call failed ({elapsed:.2f}s): {error_msg}", level="warn")
                    
                    # Check if it's a timeout or connection error - these need retry with delay
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                    
                    # Add delay before retry for timeout/connection errors
                    if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                        # Longer delay for timeout (since it already waited 5 minutes)
                        delay = 10 if is_timeout else min(2 ** attempt, 10)
                        if debug:
                            debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                        time.sleep(delay)
                    continue
                
                # Parse response
                selection_result = parse_json_response(response, debug=debug)
                
                if selection_result is None:
                    if debug:
                        debug_print(f"   Failed to parse response ({elapsed:.2f}s)", level="warn")
                    continue
                
                # Validate response structure
                if 'selected_region_ids' not in selection_result:
                    if debug:
                        debug_print(f"   Invalid response structure", level="warn")
                    continue
                
                # Extract selected region IDs
                selected_items = selection_result.get('selected_region_ids', [])
                selected_region_ids = [item['region_id'] for item in selected_items if isinstance(item, dict) and 'region_id' in item]
                
                # CRITICAL: Validate that all selected region IDs are in the original group
                original_group_ids = set(group['region_ids'])
                invalid_ids = [rid for rid in selected_region_ids if rid not in original_group_ids]
                
                if invalid_ids:
                    if debug:
                        debug_print(f"   ⚠️  Warning: Gemini returned {len(invalid_ids)} region IDs not in original group: {invalid_ids}", level="warn")
                        debug_print(f"   Original group: {list(original_group_ids)}", level="info")
                        debug_print(f"   Filtering out invalid IDs...", level="info")
                    
                    # Filter out invalid region IDs
                    selected_region_ids = [rid for rid in selected_region_ids if rid in original_group_ids]
                
                # Validate count (must be 2-5) after filtering
                if len(selected_region_ids) < 2 or len(selected_region_ids) > 5:
                    if debug:
                        debug_print(f"   Invalid selection count after filtering: {len(selected_region_ids)} (expected 2-5)", level="warn")
                        if invalid_ids:
                            debug_print(f"   This may be due to invalid region IDs returned by Gemini", level="warn")
                    continue
                
                # Log result
                if debug:
                    debug_print(f"   Selected {len(selected_region_ids)} elements from {len(group['region_ids'])} ({elapsed:.2f}s)", level="success")
                    visual_pattern = selection_result.get('visual_pattern', 'N/A')
                    debug_print(f"   Visual pattern: {visual_pattern}", level="info")
                
                # Return selection result
                return {
                    'selected_region_ids': selected_region_ids,
                    'visual_pattern': selection_result.get('visual_pattern', ''),
                    'selection_summary': selection_result.get('selection_summary', ''),
                    'final_count': len(selected_region_ids),
                    'excluded_count': len(group['region_ids']) - len(selected_region_ids),
                    'raw_response': response  # Add raw response from Gemini
                }
                
            except Exception as e:
                error_msg = str(e)
                if debug:
                    debug_print(f"   Exception: {error_msg}", level="error")
                
                # Check if it's a timeout or connection error
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                
                # Add delay before retry for timeout/connection errors
                if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                    delay = 10 if is_timeout else min(2 ** attempt, 10)
                    if debug:
                        debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                    time.sleep(delay)
                continue
        
        if debug:
            debug_print("   Element selection failed after all retries", level="error")
        return None
    
    def verify_group_with_vision(self, image_path: str, group: Dict, regions_data: Dict, 
                                debug: bool = False) -> Dict:
        """Use Gemini to visually verify and refine a group identified by text similarity
        
        Args:
            image_path: Path to the screenshot
            group: Initial group from text similarity (contains region_ids)
            regions_data: Complete region annotation data (all regions, not just group members)
            debug: Whether to output debug info
        
        Returns:
            Verified group dict with keys: {'valid': bool, 'rejection_reason': str, 
                                           'visual_similarity_description': str, 'revised_elements': list}
            Returns None if verification fails
        """
        if debug:
            debug_print(f"Verifying group {group['group_id']} with Gemini vision", level="step")
        
        # Convert original image to base64 (no annotation)
        try:
            original_image_base64 = image_to_base64(image_path)
        except Exception as e:
            if debug:
                debug_print(f"   Failed to read image: {e}", level="error")
            return None
        
        # Prepare initial group information text
        group_info_lines = []
        group_region_ids = set(group['region_ids'])
        
        for i, region_id in enumerate(group['region_ids']):
            if region_id in regions_data:
                region = regions_data[region_id]
                
                # Get functionality
                functionality = region.get('functionality', {})
                if isinstance(functionality, dict):
                    func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
                else:
                    func_text = str(functionality)
                
                # Get description
                description = region.get('description', {})
                if isinstance(description, dict):
                    desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
                else:
                    desc_text = str(description)
                
                # Get bbox
                bbox = region.get('bbox_global_norm', region.get('bbox_global', []))
                
                # Get type
                region_type = region.get('type', 'Unknown')
                
                group_info_lines.append(f"Element {i+1}:")
                group_info_lines.append(f"  - Region ID: {region_id}")
                group_info_lines.append(f"  - Type: {region_type}")
                group_info_lines.append(f"  - BBox: {bbox}")
                group_info_lines.append(f"  - Functionality: {func_text}")
                group_info_lines.append(f"  - Description: {desc_text}")
                group_info_lines.append("")
        
        group_info_text = "\n".join(group_info_lines)
        group_info_text += f"\nGrouping Reason: {group.get('similarity_reason', 'High visual description similarity')}"
        
        # Prepare other candidate elements information (all regions NOT in the group)
        other_candidates_lines = []
        candidate_count = 0
        
        for region_id, region in regions_data.items():
            # Skip elements already in the group
            if region_id in group_region_ids:
                continue
            
            candidate_count += 1
            
            # Get functionality
            functionality = region.get('functionality', {})
            if isinstance(functionality, dict):
                func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
            else:
                func_text = str(functionality)
            
            # Get description
            description = region.get('description', {})
            if isinstance(description, dict):
                desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
            else:
                desc_text = str(description)
            
            # Get bbox
            bbox = region.get('bbox_global_norm', region.get('bbox_global', []))
            
            # Get type
            region_type = region.get('type', 'Unknown')
            
            other_candidates_lines.append(f"Candidate {candidate_count}:")
            other_candidates_lines.append(f"  - Region ID: {region_id}")
            other_candidates_lines.append(f"  - Type: {region_type}")
            other_candidates_lines.append(f"  - BBox: {bbox}")
            other_candidates_lines.append(f"  - Functionality: {func_text}")
            other_candidates_lines.append(f"  - Description: {desc_text}")
            other_candidates_lines.append("")
        
        other_candidates_text = "\n".join(other_candidates_lines) if other_candidates_lines else "No other candidates available."
        
        if debug:
            debug_print(f"   Initial group size: {len(group['region_ids'])}", level="info")
            debug_print(f"   Other candidates: {candidate_count}", level="info")
        
        # Prepare prompt
        prompt = VISUAL_VERIFICATION_PROMPT.format(
            initial_group_info=group_info_text,
            other_candidates_info=other_candidates_text
        )
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': original_image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                if debug:
                    debug_print(f"   Verification attempt {attempt + 1}/{self.max_retries}", level="info")
                
                start_time = time.time()
                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.2 if attempt == 0 else 0.4, timeout=600
                )
                elapsed = time.time() - start_time
                
                if not success:
                    error_msg = str(response) if response else "Unknown error"
                    if debug:
                        debug_print(f"   API call failed ({elapsed:.2f}s): {error_msg}", level="warn")
                    
                    # Check if it's a timeout or connection error - these need retry with delay
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                    
                    # Add delay before retry for timeout/connection errors
                    if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                        # Longer delay for timeout (since it already waited 5 minutes)
                        delay = 10 if is_timeout else min(2 ** attempt, 10)
                        if debug:
                            debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                        time.sleep(delay)
                    continue
                
                # Parse response
                verification_result = parse_json_response(response, debug=debug)
                
                if verification_result is None:
                    if debug:
                        debug_print(f"   Failed to parse response ({elapsed:.2f}s)", level="warn")
                    continue
                
                # Validate response structure
                if 'valid' not in verification_result:
                    if debug:
                        debug_print(f"   Invalid response structure", level="warn")
                    continue
                
                # Log result
                if debug:
                    if verification_result.get('valid', False):
                        debug_print(f"   Group VALIDATED ({elapsed:.2f}s)", level="success")
                    else:
                        reason = verification_result.get('rejection_reason', 'Unknown')
                        debug_print(f"   Group REJECTED: {reason} ({elapsed:.2f}s)", level="warn")
                
                # Add raw response from Gemini
                verification_result['raw_response'] = response
                return verification_result
                
            except Exception as e:
                error_msg = str(e)
                if debug:
                    debug_print(f"   Exception: {error_msg}", level="error")
                
                # Check if it's a timeout or connection error
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                
                # Add delay before retry for timeout/connection errors
                if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                    delay = 10 if is_timeout else min(2 ** attempt, 10)
                    if debug:
                        debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                    time.sleep(delay)
                continue
        
        if debug:
            debug_print("   Visual verification failed after all retries", level="error")
        return None
    
    def _format_regions_info(self, regions_data: Dict) -> str:
        """Format functional region information as text"""
        lines = []
        for region_id, region_info in sorted(regions_data.items()):
            # Get functionality description
            functionality = region_info.get('functionality', {})
            if isinstance(functionality, dict):
                func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
            else:
                func_text = str(functionality)
            
            # Get description
            description = region_info.get('description', {})
            if isinstance(description, dict):
                desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
            else:
                desc_text = str(description)
            
            # Get bbox
            bbox = region_info.get('bbox_global_norm', region_info.get('bbox_global', []))
            
            # Get type
            region_type = region_info.get('type', 'Unknown')
            
            lines.append(f"- Region ID: {region_id}")
            lines.append(f"  Type: {region_type}")
            lines.append(f"  BBox: {bbox}")
            lines.append(f"  Functionality: {func_text}")
            lines.append(f"  Description: {desc_text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _enrich_group_with_region_info(self, group: Dict, regions_data: Dict) -> Dict:
        """Add detailed region information to element group"""
        enriched_group = {
            "group_id": group['group_id'],
            "description": group.get('description', ''),
            "similarity_reason": group.get('similarity_reason', ''),
            "regions": []
        }
        
        for region_id in group['region_ids']:
            if region_id in regions_data:
                region_info = regions_data[region_id]
                
                # Get functionality description
                functionality = region_info.get('functionality', {})
                if isinstance(functionality, dict):
                    func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
                else:
                    func_text = str(functionality)
                
                # Get description
                description = region_info.get('description', {})
                if isinstance(description, dict):
                    desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
                else:
                    desc_text = str(description)
                
                enriched_group['regions'].append({
                    "region_id": region_id,
                    "bbox": region_info.get('bbox_global_norm', region_info.get('bbox_global', [])),
                    "type": region_info.get('type', 'Unknown'),
                    "functionality": func_text,
                    "description": desc_text
                })
        
        return enriched_group
    
    def _format_single_region_info(self, region_id: str, region_data: Dict) -> str:
        """Format single region information as text"""
        # Get functionality description
        functionality = region_data.get('functionality', {})
        if isinstance(functionality, dict):
            func_text = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
        else:
            func_text = str(functionality)
        
        # Get description
        description = region_data.get('description', {})
        if isinstance(description, dict):
            desc_text = description.get('wo_context') or description.get('with_context', 'Unknown')
        else:
            desc_text = str(description)
        
        # Get bbox
        bbox = region_data.get('bbox_global_norm', region_data.get('bbox_global', []))
        
        # Get type
        region_type = region_data.get('type', 'Unknown')
        
        info = f"""Region ID: {region_id}
Type: {region_type}
BBox: {bbox}
Functionality: {func_text}
Description: {desc_text}"""
        
        return info
    
    def _validate_question(self, question: Dict) -> bool:
        """Validate question validity"""
        try:
            # Check required fields
            if not all(k in question for k in ['question', 'options', 'correct_answer']):
                return False
            
            options = question['options']
            
            # Check option count (at least 2 options, no upper limit since text similarity may find many similar elements)
            if len(options) < 2:
                return False
            
            # Check each option
            labels = set()
            for opt in options:
                if not all(k in opt for k in ['label']):
                    return False
                labels.add(opt['label'])
            
            # Check correct answer
            if question['correct_answer'] not in labels:
                return False
            
            return True
        except Exception:
            return False
    
    def generate_question_grounding_mode(self, group: Dict, regions_data: Dict, debug: bool = False) -> Dict:
        """Generate Grounding mode question (text-based, Mode 1 logic)
        
        Args:
            group: Element group information (contains region_ids)
            regions_data: Complete region annotation data
            debug: Whether to output debug info
        """
        if debug:
            debug_print(f"Generating Grounding question (group {group['group_id']})", level="step")
        
        # Validate group size (must be 2-5 elements)
        region_ids = group.get('region_ids', [])
        if len(region_ids) < 2:
            if debug:
                debug_print(f"   Group has less than 2 regions, skipping", level="warn")
            return None
        if len(region_ids) > 5:
            if debug:
                debug_print(f"   Group has more than 5 regions ({len(region_ids)}), skipping", level="warn")
            return None
        
        # Extract detailed region information for this group
        group_with_details = self._enrich_group_with_region_info(group, regions_data)
        
        # Prepare element information
        element_info = json.dumps(group_with_details, indent=2, ensure_ascii=False)
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': GENERATE_QUESTION_GROUNDING_MODE_PROMPT.format(element_info=element_info)}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                if debug:
                    debug_print(f"   Attempt {attempt + 1}/{self.max_retries}", level="info")
                
                start_time = time.time()
                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3 if attempt == 0 else 0.6, timeout=600
                )
                elapsed = time.time() - start_time
                
                if not success:
                    error_msg = str(response) if response else "Unknown error"
                    if debug:
                        debug_print(f"   API call failed ({elapsed:.2f}s): {error_msg}", level="warn")
                    
                    # Check if it's a timeout or connection error - these need retry with delay
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                    
                    # Add delay before retry for timeout/connection errors
                    if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                        # Longer delay for timeout (since it already waited 5 minutes)
                        delay = 10 if is_timeout else min(2 ** attempt, 10)
                        if debug:
                            debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                        time.sleep(delay)
                    continue
                
                # Parse response
                question_data = parse_json_response(response, debug=debug)
                
                if question_data is None:
                    if debug:
                        debug_print(f"   Failed to parse response ({elapsed:.2f}s)", level="warn")
                    continue
                
                # Validate question
                if self._validate_question(question_data):
                    # Add raw response from Gemini
                    question_data['raw_response'] = response
                    if debug:
                        debug_print(f"Grounding question generated successfully ({elapsed:.2f}s)", level="success")
                    return question_data
                
            except Exception as e:
                error_msg = str(e)
                if debug:
                    debug_print(f"   Exception: {error_msg}", level="error")
                
                # Check if it's a timeout or connection error
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                
                # Add delay before retry for timeout/connection errors
                if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                    delay = 10 if is_timeout else min(2 ** attempt, 10)
                    if debug:
                        debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                    time.sleep(delay)
                continue
        
        if debug:
            debug_print("Grounding question generation failed", level="error")
        return None
    
    def generate_question_captioning_mode(self, image_path: str, group: Dict, regions_data: Dict,
                                        output_file_path: str = None, debug: bool = False) -> Dict:
        """Generate Captioning mode question (annotated image, Mode 5 logic)
        
        Args:
            image_path: Image file path
            group: Element group information (contains region_ids)
            regions_data: Complete region annotation data
            output_file_path: Output file path to determine where to save annotated images
            debug: Whether to output debug info
        """
        if debug:
            debug_print(f"Generating Captioning question (group {group['group_id']})", level="step")
        
        # Get all region IDs in the group
        region_ids = group.get('region_ids', [])
        # Validate group size (must be 2-5 elements)
        if len(region_ids) < 2:
            if debug:
                debug_print(f"   Group has less than 2 regions, skipping", level="warn")
            return None
        if len(region_ids) > 5:
            if debug:
                debug_print(f"   Group has more than 5 regions ({len(region_ids)}), skipping", level="warn")
            return None
        
        # Randomly select one element as the target (correct answer)
        target_region_id = random.choice(region_ids)
        
        if target_region_id not in regions_data:
            if debug:
                debug_print(f"   Target region {target_region_id} not found", level="error")
            return None
        
        target_region = regions_data[target_region_id]
        
        # Get target functionality
        functionality = target_region.get('functionality', {})
        if isinstance(functionality, dict):
            correct_func = functionality.get('wo_context') or functionality.get('with_context', 'Unknown')
        else:
            correct_func = str(functionality)
        
        if correct_func == 'Unknown' or not correct_func:
            if debug:
                debug_print(f"   No valid functionality found for region {target_region_id}", level="warn")
            return None
        
        # Get bbox
        bbox = target_region.get('bbox_global_norm', target_region.get('bbox_global', []))
        if not bbox or len(bbox) < 4:
            if debug:
                debug_print(f"   No valid bbox found for region {target_region_id}", level="warn")
            return None
        
        # Construct save path for annotated image
        save_path = None
        if per_image_output_dir:
            base_output_dir = per_image_output_dir
        elif output_file_path:
            base_output_dir = os.path.dirname(output_file_path) or '.'
        else:
            base_output_dir = '.'
        
        # Get original image name (without extension)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        image_ext = os.path.splitext(os.path.basename(image_path))[1] or '.png'
        
        # If generation_mode is "both", save to captioning_mode subdirectory
        # Otherwise save directly to base_output_dir
        if generation_mode == "both":
            annotated_dir = os.path.join(base_output_dir, 'captioning_mode', 'annotated_images', image_basename)
        else:
            annotated_dir = os.path.join(base_output_dir, 'annotated_images', image_basename)
        
        save_path = os.path.join(annotated_dir, f"group{group['group_id']}_{target_region_id}_{image_basename}{image_ext}")
        
        # Draw bbox on image
        try:
            is_normalized = 'bbox_global_norm' in target_region
            annotated_image_base64 = draw_bbox_on_image(image_path, bbox, is_normalized, save_path)
            if debug and save_path:
                debug_print(f"   Saved annotated image to: {save_path}", level="info")
        except Exception as e:
            if debug:
                debug_print(f"   Failed to draw bbox: {e}", level="error")
            return None
        
        # Format all group elements info
        group_elements_lines = []
        for rid in region_ids:
            if rid in regions_data:
                region = regions_data[rid]
                func = region.get('functionality', {})
                if isinstance(func, dict):
                    func_text = func.get('wo_context') or func.get('with_context', 'Unknown')
                else:
                    func_text = str(func)
                
                marker = " (TARGET - circled in image)" if rid == target_region_id else ""
                group_elements_lines.append(f"- Region {rid}{marker}: {func_text}")
        
        group_elements_info = "\n".join(group_elements_lines)
        
        # Format target element info
        target_info = self._format_single_region_info(target_region_id, target_region)
        
        # Prepare prompt
        prompt = GENERATE_QUESTION_CAPTIONING_MODE_PROMPT.format(
            group_elements_info=group_elements_info,
            target_element_info=target_info
        )
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': annotated_image_base64}},
                {'type': 'text', 'text': prompt}
            ]
        }]
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                if debug:
                    debug_print(f"   Attempt {attempt + 1}/{self.max_retries}", level="info")
                
                start_time = time.time()
                success, response, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=0.3 if attempt == 0 else 0.6, timeout=600
                )
                elapsed = time.time() - start_time
                
                if not success:
                    error_msg = str(response) if response else "Unknown error"
                    if debug:
                        debug_print(f"   API call failed ({elapsed:.2f}s): {error_msg}", level="warn")
                    
                    # Check if it's a timeout or connection error - these need retry with delay
                    is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                    is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                    
                    # Add delay before retry for timeout/connection errors
                    if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                        # Longer delay for timeout (since it already waited 5 minutes)
                        delay = 10 if is_timeout else min(2 ** attempt, 10)
                        if debug:
                            debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                        time.sleep(delay)
                    continue
                
                # Parse response
                question_data = parse_json_response(response, debug=debug)
                
                if question_data is None:
                    if debug:
                        debug_print(f"   Failed to parse response ({elapsed:.2f}s)", level="warn")
                    continue
                
                # Validate question
                if self._validate_question(question_data):
                    # Add metadata
                    question_data['target_region_id'] = target_region_id
                    question_data['annotated_image_path'] = save_path
                    question_data['image_path'] = image_path  # Add original image path for consistency
                    question_data['raw_response'] = response  # Add raw response from Gemini
                    
                    if debug:
                        debug_print(f"Captioning question generated successfully ({elapsed:.2f}s)", level="success")
                    return question_data
                
            except Exception as e:
                error_msg = str(e)
                if debug:
                    debug_print(f"   Exception: {error_msg}", level="error")
                
                # Check if it's a timeout or connection error
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_connection_error = "connection" in error_msg.lower() or "connect" in error_msg.lower() or "network" in error_msg.lower()
                
                # Add delay before retry for timeout/connection errors
                if (is_timeout or is_connection_error) and attempt < self.max_retries - 1:
                    delay = 10 if is_timeout else min(2 ** attempt, 10)
                    if debug:
                        debug_print(f"   Waiting {delay}s before retry (timeout/connection error)...", level="info")
                    time.sleep(delay)
                continue
        
        if debug:
            debug_print("Captioning question generation failed", level="error")
        return None


def init_worker(base_url: str, api_key: str, model: str, max_retries: int, debug: bool, 
                gen_mode: str, out_file_path: str, only_corrected: bool = True, 
                cache_dir: str = None, aliyun_api_key: str = None):
    """Initialize worker process"""
    global generator_instance, debug_flag, generation_mode, output_file_path, cache_dir_global, embedding_model_instance, only_corrected_regions
    generator_instance = MultiChoiceQAGenerator(base_url, api_key, model, max_retries)
    debug_flag = debug
    generation_mode = gen_mode
    output_file_path = out_file_path
    cache_dir_global = cache_dir
    only_corrected_regions = only_corrected
    
    # Initialize embedding client (Aliyun DashScope)
    if not OPENAI_CLIENT_AVAILABLE:
        error_msg = "OpenAI client is not available. Please install: pip install openai"
        if debug:
            debug_print(error_msg, level="error")
        raise RuntimeError(error_msg)
    
    try:
        if debug:
            debug_print("Initializing Aliyun DashScope embedding client (text-embedding-v4)...", level="info")
        
        # Get API key from environment or parameter
        dashscope_key = aliyun_api_key or os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables or arguments")
        
        # Initialize OpenAI client for Aliyun DashScope
        embedding_model_instance = OpenAIClient(
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        if debug:
            debug_print("Aliyun embedding client initialized successfully", level="success")
    except Exception as e:
        error_msg = f"Failed to initialize Aliyun embedding client: {e}"
        if debug:
            debug_print(error_msg, level="error")
        raise RuntimeError(error_msg) from e


def remove_bbox_overlaps_from_group(region_ids: List[str], regions_data: Dict, 
                                    debug: bool = False) -> List[str]:
    """Remove elements causing bbox overlaps from a group iteratively
    
    Strategy: When overlap detected, remove the element with largest bbox area
    (assuming larger elements are less specific)
    
    Args:
        region_ids: List of region IDs in the group
        regions_data: Complete region annotation data
        debug: Whether to print debug info
    
    Returns:
        List of region IDs with no bbox overlaps
    """
    if len(region_ids) <= 1:
        return region_ids
    
    cleaned_ids = region_ids.copy()
    removed_count = 0
    
    while True:
        # Check for overlaps
        is_valid, overlapping_pairs = validate_group_no_bbox_overlaps(cleaned_ids, regions_data, debug=False)
        
        if is_valid or not overlapping_pairs:
            break  # No overlaps, done
        
        # Find element to remove (largest bbox area among overlapping elements)
        overlapping_elements = set()
        for id1, id2 in overlapping_pairs:
            overlapping_elements.add(id1)
            overlapping_elements.add(id2)
        
        # Calculate bbox areas
        element_areas = {}
        for rid in overlapping_elements:
            if rid in regions_data:
                bbox = regions_data[rid].get('bbox_global_norm') or regions_data[rid].get('bbox_global', [])
                if bbox and len(bbox) >= 4:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    element_areas[rid] = area
        
        # Remove element with largest area
        if element_areas:
            to_remove = max(element_areas, key=element_areas.get)
            cleaned_ids.remove(to_remove)
            removed_count += 1
            
            if debug:
                debug_print(f"      Removed {to_remove} (largest bbox, area={element_areas[to_remove]:.4f})", level="info")
        else:
            # Fallback: remove first overlapping element
            to_remove = overlapping_pairs[0][0]
            cleaned_ids.remove(to_remove)
            removed_count += 1
            if debug:
                debug_print(f"      Removed {to_remove} (fallback)", level="info")
    
    if debug and removed_count > 0:
        debug_print(f"      Bbox cleanup: removed {removed_count} elements, {len(cleaned_ids)} remain", level="success")
    
    return cleaned_ids


def process_image(args) -> Dict:
    """Process functional region annotation data for a single image"""
    image_key, image_data, worker_id = args
    start_time = time.time()
    
    # Initialize variables that might be used in return statements
    initial_groups = []
    gemini_first_pass = []
    after_bbox_cleanup = []
    after_merge = None
    gemini_second_pass = []
    
    try:
        global generator_instance, debug_flag, generation_mode, output_file_path, cache_dir_global, embedding_model_instance, only_corrected_regions, per_image_output_dir
        
        # Only first worker outputs debug info
        effective_debug = debug_flag and (worker_id == 0)
        
        # Get image path
        image_path = image_data.get('root_image_path') or image_key
        
        # If image_path is relative, try to resolve it
        if not os.path.isabs(image_path):
            if cache_dir_global:
                cache_parent = os.path.dirname(cache_dir_global)
                candidate_path = os.path.join(cache_parent, image_path)
                if os.path.exists(candidate_path):
                    image_path = candidate_path
        
        if not effective_debug:
            print(f"[Worker {worker_id}] Processing image: {os.path.basename(image_path)} (Mode: {generation_mode})")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return {
                "image_path": image_path,
                "error": f"Image file not found: {image_path}",
                "processing_time": time.time() - start_time
            }
        
        # Extract all functional regions (exclude root node)
        regions_data = {}
        for node_id, node_data in image_data.items():
            if isinstance(node_data, dict) and node_id != '0-0':
                regions_data[node_id] = node_data
        
        # Filter to only use manually corrected regions (if enabled)
        original_region_count = len(regions_data)
        regions_data = filter_corrected_regions(regions_data, only_corrected=only_corrected_regions, debug=effective_debug)
        
        if not regions_data:
            # Always print warning when no regions available (even if not debug mode)
            if effective_debug:
                debug_print("=" * 60, level="title")
                debug_print("⚠️  No regions available after filtering", level="warn")
                debug_print("=" * 60, level="title")
                debug_print(f"   Original regions: {original_region_count}", level="info")
                debug_print(f"   Filtered regions: 0 (all regions were filtered out)", level="warn")
                debug_print(f"   Reason: Region filtering requires both bbox_corrected=True AND reannotated=True", level="info")
                debug_print(f"   Skipping grouping and question generation for this image", level="info")
            else:
                # Even in non-debug mode, print a brief warning
                print(f"[Worker {worker_id}] ⚠️  No regions available after filtering ({original_region_count} regions filtered out)")
            
            return {
                "image_path": image_path,
                "num_questions": 0,
                "num_regions": 0,
                "questions": [],
                "generation_mode": generation_mode,
                "processing_time": time.time() - start_time,
                "error": "No regions available after filtering (all regions were filtered out)"
            }
        
        if effective_debug:
            debug_print(f"   Available regions: {len(regions_data)}", level="info")
            debug_print(f"   Generation mode: {generation_mode}", level="info")
        
        # Load parent-child relationships to exclude from similarity calculation and merge filtering
        parent_child_map = load_parent_child_relationships(image_path, cache_dir_global)
        
        if effective_debug and parent_child_map:
            debug_print(f"   Loaded parent-child relationships for filtering", level="info")
        
        # Generate embeddings in real-time
        if embedding_model_instance is None:
            error_msg = "Embedding model is not initialized."
            if effective_debug:
                debug_print(error_msg, level="error")
            raise RuntimeError(error_msg)
        
        embeddings_map = compute_embeddings_for_regions(
            regions_data, 
            embedding_model_instance, 
            debug=effective_debug
        )
        
        # ============================================================
        # 阶段1: Aliyun Embedding 初始分组
        # ============================================================
        if effective_debug:
            debug_print("=" * 60, level="title")
            debug_print("阶段1: Aliyun Embedding 初始分组", level="title")
            debug_print("=" * 60, level="title")
        
        groups = group_regions_by_text_similarity(
            regions_data, 
            parent_child_map=parent_child_map,
            embeddings_map=embeddings_map,
            debug=effective_debug
        )
        
        # Save initial groups (阶段1输出)
        initial_groups = copy.deepcopy(groups)
        
        if not groups:
            if effective_debug:
                debug_print(f"   No similar groups found based on text similarity", level="warn")
            return {
                "image_path": image_path,
                "num_questions": 0,
                "num_regions": len(regions_data),
                "questions": [],
                "generation_mode": generation_mode,
                "grouping_method": "aliyun_embedding + gemini_vision",
                "processing_time": time.time() - start_time,
                "warning": "No text-similar groups found",
                "initial_groups": initial_groups,
                "initial_groups_count": len(initial_groups)
            }
        
        if effective_debug:
            debug_print(f"✅ 阶段1完成: {len(initial_groups)} 个初始组", level="success")
            debug_print("", level="info")
        
        # ============================================================
        # 阶段2: Gemini 视觉验证（第一轮）
        # ============================================================
        if effective_debug:
            debug_print("=" * 60, level="title")
            debug_print("阶段2: Gemini 视觉验证（第一轮 - 核心验证）", level="title")
            debug_print("=" * 60, level="title")
        
        gemini_first_pass = []
        rejected_groups = []
        
        for group in groups:
            if effective_debug:
                debug_print(f"   验证组 {group['group_id']} ({len(group['region_ids'])} 元素)...", level="info")
            
            verification_result = generator_instance.verify_group_with_vision(
                image_path, group, regions_data, effective_debug
            )
            
            if verification_result and verification_result.get('valid', False):
                # Gemini认为这个组是有效的，提取refined元素列表
                kept_items = verification_result.get('kept_region_ids', [])
                if kept_items:
                    refined_region_ids = [
                        item['region_id'] for item in kept_items 
                        if isinstance(item, dict) and item.get('keep', True) and item['region_id'] in regions_data
                    ]
                else:
                    refined_region_ids = group['region_ids']
                
                # CRITICAL: Validate that all refined region IDs are in the original group
                # (Gemini might return region_ids from other candidates that weren't in the initial group)
                original_group_ids = set(group['region_ids'])
                invalid_ids = [rid for rid in refined_region_ids if rid not in original_group_ids]
                
                if invalid_ids:
                    if effective_debug:
                        debug_print(f"   ⚠️  警告: Gemini返回了 {len(invalid_ids)} 个不在原始组中的region IDs: {invalid_ids}", level="warn")
                        debug_print(f"   原始组: {list(original_group_ids)}", level="info")
                        debug_print(f"   过滤无效的region IDs...", level="info")
                    
                    # Filter out invalid region IDs
                    refined_region_ids = [rid for rid in refined_region_ids if rid in original_group_ids]
                
                # Validate count (must be >= 2) after filtering
                if len(refined_region_ids) < 2:
                    if effective_debug:
                        debug_print(f"   ❌ 组 {group['group_id']} 被跳过: 过滤后元素不足 ({len(refined_region_ids)} < 2)", level="warn")
                        if invalid_ids:
                            debug_print(f"   原因: Gemini返回了无效的region IDs，过滤后元素数量不足", level="info")
                    
                    rejected_groups.append({
                        **group,
                        'rejection_reason': f'Too few elements after validation ({len(refined_region_ids)} < 2). Invalid region IDs returned by Gemini: {invalid_ids}' if invalid_ids else f'Too few elements after validation ({len(refined_region_ids)} < 2)',
                        'rejection_stage': 'gemini_first_pass_validation'
                    })
                    continue
                
                # 创建refined group
                refined_group = copy.deepcopy(group)
                refined_group['region_ids'] = refined_region_ids
                refined_group['visual_similarity_description'] = verification_result.get('visual_similarity_description', '')
                refined_group['gemini_verified'] = True
                refined_group['gemini_adjustments'] = verification_result.get('adjustments_made', 'None')
                refined_group['raw_response'] = verification_result.get('raw_response', '')  # Add raw Gemini response
                
                # Track added elements (only those that are actually in the original group)
                if kept_items:
                    added_from_candidates = [
                        item['region_id'] for item in kept_items 
                        if isinstance(item, dict) and item.get('keep', True) 
                        and item.get('source') == 'added_from_candidates'
                        and item['region_id'] in original_group_ids  # Only count if actually in original group
                    ]
                    if added_from_candidates:
                        refined_group['added_from_candidates'] = added_from_candidates
                
                gemini_first_pass.append(refined_group)
                
                if effective_debug:
                    if invalid_ids:
                        debug_print(f"   ✅ 组 {group['group_id']} 通过验证（已过滤无效IDs）: {len(group['region_ids'])} → {len(refined_region_ids)} 元素", level="success")
                    else:
                        debug_print(f"   ✅ 组 {group['group_id']} 通过验证: {len(group['region_ids'])} → {len(refined_region_ids)} 元素", level="success")
            else:
                reason = verification_result.get('rejection_reason', 'Unknown') if verification_result else 'Verification failed'
                rejected_groups.append({
                    **group,
                    'rejection_reason': reason,
                    'rejection_stage': 'gemini_first_pass'
                })
                if effective_debug:
                    debug_print(f"   ❌ 组 {group['group_id']} 被拒绝: {reason}", level="warn")
        
        if effective_debug:
            debug_print(f"✅ 阶段2完成: {len(gemini_first_pass)} 个组通过验证, {len(rejected_groups)} 个被拒绝", level="success")
            debug_print("", level="info")
        
        if not gemini_first_pass:
            if effective_debug:
                debug_print("   没有组通过Gemini验证", level="warn")
            return {
                "image_path": image_path,
                "num_questions": 0,
                "num_regions": len(regions_data),
                "questions": [],
                "generation_mode": generation_mode,
                "grouping_method": "aliyun_embedding + gemini_vision",
                "processing_time": time.time() - start_time,
                "warning": "No groups passed Gemini first pass",
                "initial_groups": initial_groups,
                "initial_groups_count": len(initial_groups),
                "gemini_first_pass": [],
                "gemini_first_pass_count": 0,
                "rejected_groups": rejected_groups,
                "rejected_groups_count": len(rejected_groups)
            }
        
        # ============================================================
        # 阶段3: Python 处理和检查
        # ============================================================
        if effective_debug:
            debug_print("=" * 60, level="title")
            debug_print("阶段3: Python 处理和检查", level="title")
            debug_print("=" * 60, level="title")
        
        # 3.1 Bbox重叠检查和清理
        if effective_debug:
            debug_print("   3.1 检查并删除bbox重叠元素...", level="step")
        
        after_bbox_cleanup = []
        for group in gemini_first_pass:
            original_count = len(group['region_ids'])
            cleaned_ids = remove_bbox_overlaps_from_group(group['region_ids'], regions_data, effective_debug)
            
            cleaned_group = copy.deepcopy(group)
            cleaned_group['region_ids'] = cleaned_ids
            if original_count != len(cleaned_ids):
                cleaned_group['bbox_cleanup'] = True
                cleaned_group['bbox_removed_count'] = original_count - len(cleaned_ids)
            
            after_bbox_cleanup.append(cleaned_group)
        
        if effective_debug:
            cleanup_count = sum(1 for g in after_bbox_cleanup if g.get('bbox_cleanup', False))
            debug_print(f"   ✅ Bbox清理完成: {cleanup_count} 个组被清理", level="success")
        
        # 3.2 检查元素数量 <2，直接拒绝
        if effective_debug:
            debug_print("   3.2 检查组大小 (必须 >=2)...", level="step")
        
        groups_pass_size_check = []
        for group in after_bbox_cleanup:
            if len(group['region_ids']) < 2:
                rejected_groups.append({
                    **group,
                    'rejection_reason': f'Too few elements after bbox cleanup ({len(group["region_ids"])})',
                    'rejection_stage': 'python_size_check'
                })
                if effective_debug:
                    debug_print(f"   ❌ 组 {group['group_id']} 被拒绝: 元素过少 ({len(group['region_ids'])})", level="warn")
            else:
                groups_pass_size_check.append(group)
        
        if effective_debug:
            debug_print(f"   ✅ 大小检查完成: {len(groups_pass_size_check)} 个组通过", level="success")
        
        if not groups_pass_size_check:
            if effective_debug:
                debug_print("   没有组通过大小检查", level="warn")
            return {
                "image_path": image_path,
                "num_questions": 0,
                "num_regions": len(regions_data),
                "questions": [],
                "generation_mode": generation_mode,
                "grouping_method": "aliyun_embedding + gemini_vision",
                "processing_time": time.time() - start_time,
                "warning": "No groups passed size check",
                "initial_groups": initial_groups,
                "initial_groups_count": len(initial_groups),
                "gemini_first_pass": gemini_first_pass,
                "gemini_first_pass_count": len(gemini_first_pass),
                "after_bbox_cleanup": after_bbox_cleanup,
                "after_bbox_cleanup_count": len(after_bbox_cleanup),
                "rejected_groups": rejected_groups,
                "rejected_groups_count": len(rejected_groups)
            }
        
        # 3.3 标记超大组 (>5 元素)
        if effective_debug:
            debug_print("   3.3 标记超大组 (>5 元素)...", level="step")
        
        oversized_groups = []
        normal_groups = []
        for group in groups_pass_size_check:
            if len(group['region_ids']) > 5:
                oversized_groups.append(group)
                if effective_debug:
                    debug_print(f"      组 {group['group_id']}: {len(group['region_ids'])} 元素 (需要精简)", level="warn")
            else:
                normal_groups.append(group)
        
        if effective_debug:
            debug_print(f"   ✅ 找到 {len(oversized_groups)} 个超大组", level="success")
        
        # 3.4 检查并合并重复元素组
        if effective_debug:
            debug_print("   3.4 检查并合并重复元素组 (>=2个共同元素)...", level="step")
        
        duplicate_pairs = find_duplicate_elements_between_groups(groups_pass_size_check)
        
        after_merge = None
        groups_for_second_pass = groups_pass_size_check  # 默认使用size check后的组
        
        # 初始化变量（确保在所有分支中都有定义）
        merged_groups_needing_check = []
        
        if duplicate_pairs:
            if effective_debug:
                debug_print(f"      发现 {len(duplicate_pairs)} 对组有 >=2 个共同元素", level="warn")
                for idx1, idx2, common in duplicate_pairs:
                    debug_print(f"         组 {idx1+1} 和组 {idx2+1} 共享 {len(common)} 个元素", level="info")
            
            # 合并组 + 过滤父子关系
            merged_groups = merge_groups(groups_pass_size_check, duplicate_pairs, parent_child_map=parent_child_map, debug=effective_debug)
            
            # 保存合并后的状态
            after_merge = copy.deepcopy(merged_groups)
            groups_for_second_pass = merged_groups
            
            if effective_debug:
                debug_print(f"   ✅ 合并完成: {len(groups_pass_size_check)} → {len(merged_groups)} 个组", level="success")
            
            # 重新检查超大组和合并组（合并后可能产生新的超大组）
            oversized_groups = []
            normal_groups = []
            
            for group in merged_groups:
                if len(group['region_ids']) > 5:
                    oversized_groups.append(group)
                elif group.get('needs_refinement', False):
                    # 合并的组需要Gemini再次检查
                    merged_groups_needing_check.append(group)
                else:
                    normal_groups.append(group)
            
            if effective_debug:
                if oversized_groups:
                    debug_print(f"      发现 {len(oversized_groups)} 个超大组 (>5 元素)", level="warn")
                if merged_groups_needing_check:
                    debug_print(f"      发现 {len(merged_groups_needing_check)} 个合并组需要Gemini检查", level="warn")
        else:
            if effective_debug:
                debug_print(f"      没有发现重复元素组", level="success")
            # 没有合并，使用阶段3.3中分类的组（oversized_groups 和 normal_groups 已经在3.3中定义）
            # merged_groups_needing_check 保持为空列表（已在上面初始化）
        
        if effective_debug:
            debug_print(f"✅ 阶段3完成", level="success")
            debug_print("", level="info")
        
        # ============================================================
        # 阶段4: Gemini 再次检查异常组
        # ============================================================
        abnormal_groups = oversized_groups + merged_groups_needing_check
        
        if not abnormal_groups:
            if effective_debug:
                debug_print("阶段4: 跳过（无异常组需要Gemini再次检查）", level="info")
            gemini_second_pass = groups_for_second_pass
        else:
            if effective_debug:
                debug_print("=" * 60, level="title")
                debug_print(f"阶段4: Gemini 再次检查异常组 ({len(abnormal_groups)} 个)", level="title")
                debug_print("=" * 60, level="title")
            
            gemini_second_pass = []
            
            # 4.1 统一处理所有异常组（超大组 + 合并组）：使用 select_best_elements
            # 统一策略：只删不增，从现有元素中选择最佳的 2-5 个
            if abnormal_groups:
                if effective_debug:
                    oversized_count = len(oversized_groups)
                    merged_count = len(merged_groups_needing_check)
                    debug_print(f"   4.1 统一精简异常组 ({len(abnormal_groups)} 个)...", level="step")
                    if oversized_count > 0:
                        debug_print(f"      - {oversized_count} 个超大组 (>5 元素)", level="info")
                    if merged_count > 0:
                        debug_print(f"      - {merged_count} 个合并组 (needs_refinement=True)", level="info")
                
                for group in abnormal_groups:
                    group_type = "超大组" if len(group['region_ids']) > 5 else "合并组"
                    if effective_debug:
                        debug_print(f"      处理 {group_type} {group['group_id']} ({len(group['region_ids'])} 元素)...", level="info")
                    
                    selection_result = generator_instance.select_best_elements_from_oversized_group(
                        image_path, group, regions_data, effective_debug
                    )
                    
                    if selection_result and selection_result.get('selected_region_ids'):
                        # 创建新的refined group，避免修改原始对象
                        refined_group = copy.deepcopy(group)
                        refined_group['region_ids'] = selection_result['selected_region_ids']
                        refined_group['visual_pattern'] = selection_result.get('visual_pattern', '')
                        refined_group['gemini_refined'] = True
                        refined_group['original_size'] = len(group['region_ids'])
                        refined_group['raw_response'] = selection_result.get('raw_response', '')  # Add raw Gemini response
                        gemini_second_pass.append(refined_group)
                        
                        if effective_debug:
                            debug_print(f"         ✅ 精简成功: {refined_group['original_size']} → {len(refined_group['region_ids'])} 元素", level="success")
                    else:
                        if effective_debug:
                            debug_print(f"         ❌ 精简失败，跳过该组", level="error")
            
            # 添加没有问题的正常组（直接通过，无需Gemini再次检查）
            gemini_second_pass.extend(normal_groups)
        
        if effective_debug:
            debug_print(f"✅ 阶段4完成: {len(gemini_second_pass)} 个组进入最终验证", level="success")
            debug_print("", level="info")
        
        # ============================================================
        # 阶段5: Python 最终验证（bbox重叠 + 大小检查）
        # ============================================================
        if effective_debug:
            debug_print("=" * 60, level="title")
            debug_print(f"阶段5: Python 最终验证 ({len(gemini_second_pass)} 个组)", level="title")
            debug_print("=" * 60, level="title")
        
        final_groups = []
        # 注意：不要重新初始化 rejected_groups，继续使用之前收集的
        
        for group in gemini_second_pass:
            region_ids = group['region_ids']
            group_id = group['group_id']
            
            # 5.1 大小检查
            if len(region_ids) < 2:
                if effective_debug:
                    debug_print(f"   组 {group_id} 被拒绝: 元素过少 ({len(region_ids)} < 2)", level="warn")
                rejected_groups.append({
                    **group,
                    'rejection_reason': f'Too few elements ({len(region_ids)} < 2)',
                    'rejection_stage': 'python_size_check'
                })
                continue
            
            if len(region_ids) > 5:
                if effective_debug:
                    debug_print(f"   组 {group_id} 被拒绝: 元素过多 ({len(region_ids)} > 5)", level="warn")
                rejected_groups.append({
                    **group,
                    'rejection_reason': f'Too many elements ({len(region_ids)} > 5)',
                    'rejection_stage': 'python_size_check'
                })
                continue
            
            # 5.2 Bbox重叠检查
            is_valid, overlapping_pairs = validate_group_no_bbox_overlaps(region_ids, regions_data, effective_debug)
            if not is_valid:
                if effective_debug:
                    debug_print(f"   组 {group_id} 被拒绝: 检测到 bbox 重叠 ({len(overlapping_pairs)} 对)", level="warn")
                rejected_groups.append({
                    **group,
                    'rejection_reason': f'Bbox overlaps detected ({len(overlapping_pairs)} pairs)',
                    'rejection_stage': 'python_bbox_check',
                    'overlapping_pairs': overlapping_pairs
                })
                continue
            
            # 5.3 所有检查通过
            final_groups.append(group)
            if effective_debug:
                debug_print(f"   组 {group_id} ✅ 通过验证 ({len(region_ids)} 元素)", level="success")
        
        if effective_debug:
            debug_print(f"✅ 阶段5完成: {len(final_groups)} 个组通过最终验证, {len(rejected_groups)} 个组被拒绝", level="success")
            debug_print("", level="info")
        
        # ============================================================
        # 问题生成
        # ============================================================
        if not final_groups:
            if effective_debug:
                debug_print(f"⚠️  没有组通过最终验证", level="warn")
            return {
                "image_path": image_path,
                "num_questions": 0,
                "num_regions": len(regions_data),
                "questions": [],
                "generation_mode": generation_mode,
                "grouping_method": "aliyun_embedding + gemini_vision + python_validation",
                "processing_time": time.time() - start_time,
                "warning": "No groups passed final validation",
                # 保存所有阶段状态
                "initial_groups": initial_groups,
                "gemini_first_pass": gemini_first_pass,
                "after_bbox_cleanup": after_bbox_cleanup,
                "after_merge": after_merge,
                "gemini_second_pass": gemini_second_pass,
                "final_groups": [],
                "rejected_groups": rejected_groups,
                # 统计信息
                "stats": {
                    "initial_groups_count": len(initial_groups),
                    "after_gemini_first_pass": len(gemini_first_pass) if gemini_first_pass else 0,
                    "after_bbox_cleanup": len(after_bbox_cleanup) if after_bbox_cleanup else 0,
                    "after_merge": len(after_merge) if after_merge else 0,
                    "after_gemini_second_pass": len(gemini_second_pass),
                    "final_groups_count": 0,
                    "rejected_groups_count": len(rejected_groups)
                }
            }
        
        # 生成问题
        questions_grounding = []
        questions_captioning = []
        
        if generation_mode in ["grounding_mode", "both"]:
            for group in final_groups:
                question = generator_instance.generate_question_grounding_mode(group, regions_data, effective_debug)
                if question:
                    question['group_id'] = group['group_id']
                    question['group_description'] = group.get('visual_similarity_description', group.get('description', ''))
                    question['region_ids'] = group.get('region_ids', [])
                    question['generation_mode'] = 'grounding_mode'
                    question['verified_by_vision'] = True
                    question['image_path'] = image_path
                    questions_grounding.append(question)
        
        if generation_mode in ["captioning_mode", "both"]:
            for group in final_groups:
                question = generator_instance.generate_question_captioning_mode(
                    image_path, group, regions_data, output_file_path, effective_debug
                )
                if question:
                    question['group_id'] = group['group_id']
                    question['group_description'] = group.get('visual_similarity_description', group.get('description', ''))
                    question['region_ids'] = group.get('region_ids', [])
                    question['generation_mode'] = 'captioning_mode'
                    question['verified_by_vision'] = True
                    questions_captioning.append(question)
        
        # 准备返回结果
        if generation_mode == "both":
            result = {
                "image_path": image_path,
                "num_questions_grounding": len(questions_grounding),
                "num_questions_captioning": len(questions_captioning),
                "num_questions_total": len(questions_grounding) + len(questions_captioning),
                "num_regions": len(regions_data),
                "questions_grounding": questions_grounding,
                "questions_captioning": questions_captioning,
                "generation_mode": "both",
                "grouping_method": "aliyun_embedding + gemini_vision + python_validation",
                "processing_time": time.time() - start_time,
                # 保存所有阶段状态
                "initial_groups": initial_groups,
                "gemini_first_pass": gemini_first_pass,
                "after_bbox_cleanup": after_bbox_cleanup,
                "after_merge": after_merge,
                "gemini_second_pass": gemini_second_pass,
                "final_groups": final_groups,
                "rejected_groups": rejected_groups,
                # 统计信息
                "stats": {
                    "initial_groups_count": len(initial_groups),
                    "after_gemini_first_pass": len(gemini_first_pass) if gemini_first_pass else 0,
                    "after_bbox_cleanup": len(after_bbox_cleanup) if after_bbox_cleanup else 0,
                    "after_merge": len(after_merge) if after_merge else 0,
                    "after_gemini_second_pass": len(gemini_second_pass),
                    "final_groups_count": len(final_groups),
                    "rejected_groups_count": len(rejected_groups)
                }
            }
            print(f"[Worker {worker_id}] Completed {os.path.basename(image_path)}: "
                  f"{len(regions_data)} regions, {len(initial_groups)} initial groups, "
                  f"{len(final_groups)} final groups, "
                  f"{len(questions_grounding)} Grounding + {len(questions_captioning)} Captioning questions ({time.time() - start_time:.2f}s)")
        elif generation_mode == "grounding_mode":
            result = {
                "image_path": image_path,
                "num_questions": len(questions_grounding),
                "num_regions": len(regions_data),
                "questions": questions_grounding,
                "generation_mode": "grounding_mode",
                "grouping_method": "aliyun_embedding + gemini_vision + python_validation",
                "processing_time": time.time() - start_time,
                # 保存所有阶段状态
                "initial_groups": initial_groups,
                "gemini_first_pass": gemini_first_pass,
                "after_bbox_cleanup": after_bbox_cleanup,
                "after_merge": after_merge,
                "gemini_second_pass": gemini_second_pass,
                "final_groups": final_groups,
                "rejected_groups": rejected_groups,
                # 统计信息
                "stats": {
                    "initial_groups_count": len(initial_groups),
                    "after_gemini_first_pass": len(gemini_first_pass) if gemini_first_pass else 0,
                    "after_bbox_cleanup": len(after_bbox_cleanup) if after_bbox_cleanup else 0,
                    "after_merge": len(after_merge) if after_merge else 0,
                    "after_gemini_second_pass": len(gemini_second_pass),
                    "final_groups_count": len(final_groups),
                    "rejected_groups_count": len(rejected_groups)
                }
            }
            print(f"[Worker {worker_id}] Completed {os.path.basename(image_path)}: "
                  f"{len(regions_data)} regions, {len(initial_groups)} initial groups, "
                  f"{len(final_groups)} final groups, {len(questions_grounding)} Grounding questions ({time.time() - start_time:.2f}s)")
        else:  # captioning_mode
            result = {
                "image_path": image_path,
                "num_questions": len(questions_captioning),
                "num_regions": len(regions_data),
                "questions": questions_captioning,
                "generation_mode": "captioning_mode",
                "grouping_method": "aliyun_embedding + gemini_vision + python_validation",
                "processing_time": time.time() - start_time,
                # 保存所有阶段状态
                "initial_groups": initial_groups,
                "gemini_first_pass": gemini_first_pass,
                "after_bbox_cleanup": after_bbox_cleanup,
                "after_merge": after_merge,
                "gemini_second_pass": gemini_second_pass,
                "final_groups": final_groups,
                "rejected_groups": rejected_groups,
                # 统计信息
                "stats": {
                    "initial_groups_count": len(initial_groups),
                    "after_gemini_first_pass": len(gemini_first_pass) if gemini_first_pass else 0,
                    "after_bbox_cleanup": len(after_bbox_cleanup) if after_bbox_cleanup else 0,
                    "after_merge": len(after_merge) if after_merge else 0,
                    "after_gemini_second_pass": len(gemini_second_pass),
                    "final_groups_count": len(final_groups),
                    "rejected_groups_count": len(rejected_groups)
                }
            }
            print(f"[Worker {worker_id}] Completed {os.path.basename(image_path)}: "
                  f"{len(regions_data)} regions, {len(initial_groups)} initial groups, "
                  f"{len(final_groups)} final groups, {len(questions_captioning)} Captioning questions ({time.time() - start_time:.2f}s)")
        
        return result
        
    except Exception as e:
        print(f"[Worker {worker_id}] Processing failed: {e}")
        traceback.print_exc()
        return {
            "image_path": image_key,
            "error": str(e),
            "processing_time": time.time() - start_time
        }


def load_annotation_results(annotation_file: str, cache_dir: str = None) -> Dict:
    """Load functional region annotation results
    
    Args:
        annotation_file: Main annotation JSON file path
        cache_dir: Cache directory path, will read manually corrected bbox if provided
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
    
    # If cache_dir provided, load reannotations (which include corrected bbox + revised functionality/description)
    if cache_dir and os.path.isdir(cache_dir):
        debug_print(f"Loading reannotations from cache directory...", level="step")
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
                
                # Load reannotations (from *_meta_reannotated*.json)
                # Note: reannotation files contain corrected_bbox, so we don't need to load *_meta_fix*.json separately
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
                            
                            # If corrected_bbox differs from original, update the bbox
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
                            
                            # Mark as corrected regardless of whether bbox changed
                            # The presence of corrected_bbox in reannotation file means it was verified/confirmed
                            image_data[node_id]['bbox_corrected'] = True
                            bbox_correction_count += 1
                        
                        # Extract revised functionality and description
                        new_functionality = reannotation_data.get('new_functionality', {})
                        if isinstance(new_functionality, dict):
                            revised_func = new_functionality.get('revised functionality')
                            revised_desc = new_functionality.get('revised description')
                            
                            # Debug: Check if fields are found
                            if not revised_func and not revised_desc:
                                debug_print(f"   Warning: Reannotation file {os.path.basename(latest_reannotation)} has new_functionality dict but missing 'revised functionality' and 'revised description' keys. Available keys: {list(new_functionality.keys())}", level="warn")
                            
                            # Update functionality if available
                            if revised_func:
                                # Store old functionality for reference (optional)
                                if 'functionality' in image_data[node_id]:
                                    old_func = image_data[node_id]['functionality']
                                    if isinstance(old_func, dict):
                                        image_data[node_id]['functionality_original'] = old_func.copy()
                                    else:
                                        image_data[node_id]['functionality_original'] = old_func
                                
                                # Update with revised functionality (store as wo_context)
                                image_data[node_id]['functionality'] = {
                                    'wo_context': revised_func,
                                    'with_context': revised_func  # Use same text for both
                                }
                            
                            # Update description if available
                            if revised_desc:
                                # Store old description for reference (optional)
                                if 'description' in image_data[node_id]:
                                    old_desc = image_data[node_id]['description']
                                    if isinstance(old_desc, dict):
                                        image_data[node_id]['description_original'] = old_desc.copy()
                                    else:
                                        image_data[node_id]['description_original'] = old_desc
                                
                                # Update with revised description (store as wo_context)
                                image_data[node_id]['description'] = {
                                    'wo_context': revised_desc,
                                    'with_context': revised_desc  # Use same text for both
                                }
                            
                            # Mark as reannotated
                            if revised_func or revised_desc:
                                image_data[node_id]['reannotated'] = True
                                image_data[node_id]['reannotation_file'] = os.path.basename(latest_reannotation)
                                reannotation_count += 1
                        else:
                            # Debug: new_functionality is not a dict
                            debug_print(f"   Warning: Reannotation file {os.path.basename(latest_reannotation)} has new_functionality but it's not a dict (type: {type(new_functionality)})", level="warn")
                    
                    except Exception as e:
                        debug_print(f"   Warning: Failed to load reannotation file {latest_reannotation}: {e}", level="warn")
                        continue
        
        debug_print(f"Successfully loaded {reannotation_count} reannotations (bbox + functionality/description)", level="success")
        if bbox_correction_count > 0:
            debug_print(f"  - {bbox_correction_count} regions with corrected bbox", level="info")
        if reannotation_count > 0:
            debug_print(f"  - {reannotation_count} regions with revised functionality/description", level="info")
    
    return results


def filter_corrected_regions(regions_data: Dict, only_corrected: bool = True, debug: bool = False) -> Dict:
    """Filter regions to only include those with manually corrected bbox AND reannotated functionality/description
    
    Args:
        regions_data: Dictionary of region_id -> region_data
        only_corrected: If True, only return regions with bbox_corrected=True AND reannotated=True.
                       If False, return all regions.
        debug: Whether to print debug info
    
    Returns:
        Filtered dictionary containing only corrected and reannotated regions (or all if only_corrected=False)
    """
    if not only_corrected:
        if debug:
            debug_print(f"   Using all {len(regions_data)} regions (no filtering)", level="info")
        return regions_data
    
    corrected_regions = {}
    bbox_only_count = 0
    reannotation_only_count = 0
    both_count = 0
    
    for region_id, region_data in regions_data.items():
        if not isinstance(region_data, dict):
            continue
        
        has_bbox_correction = region_data.get('bbox_corrected', False)
        has_reannotation = region_data.get('reannotated', False)
        
        # Require BOTH bbox correction AND reannotation
        if has_bbox_correction and has_reannotation:
            corrected_regions[region_id] = region_data
            both_count += 1
        elif has_bbox_correction:
            bbox_only_count += 1
        elif has_reannotation:
            reannotation_only_count += 1
    
    if debug:
        total = len(regions_data)
        valid = len(corrected_regions)
        skipped = total - valid
        debug_print(f"   Filtered regions: {valid}/{total} valid (bbox_corrected + reannotated)", level="info")
        if bbox_only_count > 0:
            debug_print(f"      Skipped {bbox_only_count} regions with bbox correction only (no reannotation)", level="warn")
        if reannotation_only_count > 0:
            debug_print(f"      Skipped {reannotation_only_count} regions with reannotation only (no bbox correction)", level="warn")
        if skipped - bbox_only_count - reannotation_only_count > 0:
            debug_print(f"      Skipped {skipped - bbox_only_count - reannotation_only_count} regions with neither correction", level="info")
    
    return corrected_regions


def main(args):
    """Main processing function"""
    
    # Check for OpenAI client availability
    if not OPENAI_CLIENT_AVAILABLE:
        debug_print("=" * 60, level="error")
        debug_print("⚠️  ERROR: OpenAI client not available", level="error")
        debug_print("=" * 60, level="error")
        debug_print("", level="info")
        debug_print("Aliyun embedding API requires OpenAI client library.", level="error")
        debug_print("Please install: pip install openai", level="error")
        debug_print("", level="info")
        debug_print("=" * 60, level="error")
        return
    
    # Check for DASHSCOPE_API_KEY
    dashscope_key = args.aliyun_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_key:
        debug_print("=" * 60, level="error")
        debug_print("⚠️  ERROR: DASHSCOPE_API_KEY not found", level="error")
        debug_print("=" * 60, level="error")
        debug_print("", level="info")
        debug_print("Please provide Aliyun DashScope API key via:", level="error")
        debug_print("  1. Command line: --aliyun-api-key YOUR_KEY", level="error")
        debug_print("  2. Environment variable: export DASHSCOPE_API_KEY='YOUR_KEY'", level="error")
        debug_print("", level="info")
        debug_print("=" * 60, level="error")
        return
    
    # Check for Gemini API key
    gemini_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not gemini_key:
        debug_print("=" * 60, level="error")
        debug_print("⚠️  ERROR: Gemini API key not found", level="error")
        debug_print("=" * 60, level="error")
        debug_print("", level="info")
        debug_print("Please provide Gemini API key via:", level="error")
        debug_print("  1. Command line: --api-key YOUR_KEY", level="error")
        debug_print("  2. Environment variable: export OPENAI_API_KEY='YOUR_KEY'", level="error")
        debug_print("", level="info")
        debug_print("=" * 60, level="error")
        return
    
    # Sequential mode is still recommended for rate limiting
    if not args.sequential:
        debug_print("=" * 60, level="warn")
        debug_print("⚠️  Forcing sequential mode", level="warn")
        debug_print("=" * 60, level="warn")
        debug_print("", level="info")
        debug_print("Reason: API rate limiting and batch size constraints", level="warn")
        debug_print("Aliyun text-embedding-v4 has batch size limit of 10 per request", level="warn")
        debug_print("", level="info")
        debug_print("=" * 60, level="warn")
        debug_print("", level="info")
        args.sequential = True
        args.workers = 1
    
    # Print configuration info
    debug_print("=" * 60, level="title")
    debug_print("Functional Region Multiple-Choice QA Generation (Simplified)", level="title")
    debug_print("=" * 60, level="title")
    
    debug_print("", level="info")
    debug_print("DATA CONFIGURATION", level="step")
    debug_print(f"   Input file: {Fore.CYAN}{args.input_file}{Style.RESET_ALL}", level="info")
    
    # Determine output mode
    use_per_image_output = args.output_dir is not None
    if use_per_image_output:
        debug_print(f"   Output mode: {Fore.GREEN}Per-image{Style.RESET_ALL} (one JSON file per image)", level="info")
        debug_print(f"   Output directory: {Fore.CYAN}{args.output_dir}{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Output mode: {Fore.YELLOW}Single file{Style.RESET_ALL} (all results in one JSON)", level="info")
        debug_print(f"   Output file: {Fore.CYAN}{args.output_file}{Style.RESET_ALL}", level="info")
    
    debug_print("", level="info")
    debug_print("MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")
    debug_print(f"   Embedding Model: {Fore.GREEN}Aliyun text-embedding-v4{Style.RESET_ALL} (DashScope API)", level="info")
    debug_print(f"   Embedding API: {Fore.BLUE}https://dashscope.aliyuncs.com{Style.RESET_ALL}", level="info")
    
    debug_print("", level="info")
    debug_print("PROCESSING CONFIGURATION", level="step")
    mode_text = "Sequential (1 worker)"
    debug_print(f"   Execution mode: {Fore.RED}{mode_text}{Style.RESET_ALL}", level="info")
    
    # Generation mode info
    gen_mode_desc = {
        "grounding_mode": "Aliyun embedding grouping → Gemini vision verification → Text-based region grounding QA",
        "captioning_mode": "Aliyun embedding grouping → Gemini vision verification → Image-based captioning QA",
        "both": "Aliyun embedding grouping → Gemini vision verification → Grounding QA + Captioning QA (saves tokens!)"
    }
    gen_mode_text = gen_mode_desc.get(args.generation_mode, "Unknown")
    debug_print(f"   Generation mode: {Fore.MAGENTA}{args.generation_mode}{Style.RESET_ALL} - {gen_mode_text}", level="info")
    debug_print(f"   Grouping pipeline:", level="info")
    debug_print(f"      Step 1: {Fore.CYAN}Aliyun text-embedding-v4{Style.RESET_ALL} - Visual description similarity grouping", level="info")
    debug_print(f"      Step 2: {Fore.GREEN}{args.model}{Style.RESET_ALL} - Visual verification & refinement", level="info")
    debug_print(f"      Step 3: {Fore.GREEN}{args.model}{Style.RESET_ALL} - Question generation", level="info")
    
    debug_print(f"   Max retries: {Fore.YELLOW}{args.max_retries}{Style.RESET_ALL}", level="info")
    debug_print(f"   Debug mode: {on_off(args.debug)}", level="info")
    debug_print(f"   Force reprocess: {on_off(args.force)}", level="info")
    
    use_only_corrected = not args.use_all_regions
    if use_only_corrected:
        debug_print(f"   Region filtering: {Fore.GREEN}ENABLED{Style.RESET_ALL} (requires bbox_corrected + reannotated)", level="info")
    else:
        debug_print(f"   Region filtering: {Fore.YELLOW}DISABLED{Style.RESET_ALL} (using all regions, including uncorrected)", level="warn")
    
    cache_status = f"{Fore.GREEN}Enabled{Style.RESET_ALL}" if args.cache_dir and os.path.isdir(args.cache_dir) else f"{Fore.YELLOW}Disabled{Style.RESET_ALL}"
    debug_print(f"   Load corrections & reannotations: {cache_status}", level="info")
    if args.cache_dir:
        debug_print(f"   Cache directory: {Fore.CYAN}{args.cache_dir}{Style.RESET_ALL}", level="info")
    
    debug_print("", level="info")
    debug_print("=" * 60, level="title")
    
    # Load data
    debug_print(f"Loading functional region annotation file...", level="step")
    annotation_results = load_annotation_results(args.input_file, args.cache_dir)
    
    if not annotation_results:
        debug_print(f"Error: No valid annotation data found", level="error")
        return
    
    debug_print(f"Loaded annotation data for {len(annotation_results)} images", level="success")
    
    # Test mode: Filter to single image if --test-image is specified
    if args.test_image:
        debug_print("", level="info")
        debug_print("=" * 60, level="warn")
        debug_print("⚠️  TEST MODE: Single Image Processing", level="warn")
        debug_print("=" * 60, level="warn")
        debug_print(f"   Searching for image matching: '{args.test_image}'", level="info")
        
        # Find matching image(s)
        matching_images = {}
        for image_key in annotation_results.keys():
            # Match by: exact key, basename, or substring
            if (args.test_image == image_key or 
                os.path.basename(image_key) == args.test_image or
                args.test_image in image_key or
                args.test_image in os.path.basename(image_key)):
                matching_images[image_key] = annotation_results[image_key]
        
        if not matching_images:
            debug_print("", level="info")
            debug_print(f"❌ Error: No image found matching '{args.test_image}'", level="error")
            debug_print("", level="info")
            debug_print("Available images:", level="info")
            for i, key in enumerate(sorted(annotation_results.keys())[:10]):
                debug_print(f"   {i+1}. {key}", level="info")
            if len(annotation_results) > 10:
                debug_print(f"   ... and {len(annotation_results) - 10} more", level="info")
            debug_print("", level="info")
            debug_print("=" * 60, level="error")
            return
        
        if len(matching_images) > 1:
            debug_print("", level="info")
            debug_print(f"⚠️  Warning: Multiple images matched '{args.test_image}':", level="warn")
            for i, key in enumerate(matching_images.keys()):
                debug_print(f"   {i+1}. {key}", level="info")
            debug_print(f"   Using first match: {list(matching_images.keys())[0]}", level="warn")
            debug_print("", level="info")
            # Keep only the first match
            first_key = list(matching_images.keys())[0]
            matching_images = {first_key: matching_images[first_key]}
        
        annotation_results = matching_images
        test_image_key = list(matching_images.keys())[0]
        debug_print(f"✅ Test image selected: {test_image_key}", level="success")
        debug_print("=" * 60, level="warn")
        debug_print("", level="info")
    
    # Count region info
    total_regions = sum(
        len([k for k in img_data.keys() if k != '0-0' and isinstance(img_data.get(k), dict)])
        for img_data in annotation_results.values()
    )
    debug_print(f"Total functional regions: {total_regions}", level="info")
    
    # Load existing results
    if use_per_image_output:
        processed_images = load_processed_images_from_dir(args.output_dir)
        debug_print(f"Found {len(processed_images)} already processed images in output directory", level="info")
    else:
        # Single file mode not implemented in simplified version, use per-image only
        debug_print("Error: Please use --output-dir (per-image mode) in simplified version", level="error")
        return
    
    # Pre-filter: Count images with valid regions (bbox_corrected + reannotated)
    use_only_corrected = not args.use_all_regions
    images_with_valid_regions = set()  # Use set for fast lookup
    
    if use_only_corrected:
        debug_print("Pre-filtering: Counting images with corrected and reannotated regions...", level="step")
        total_regions_count = 0
        valid_regions_count = 0
        
        for image_key, image_data in annotation_results.items():
            # Extract all functional regions (exclude root node)
            regions_data = {}
            for node_id, node_data in image_data.items():
                if isinstance(node_data, dict) and node_id != '0-0':
                    regions_data[node_id] = node_data
            
            total_regions_count += len(regions_data)
            
            # Filter to only corrected and reannotated regions
            filtered_regions = filter_corrected_regions(regions_data, only_corrected=True, debug=False)
            valid_regions_count += len(filtered_regions)
            
            if len(filtered_regions) > 0:
                images_with_valid_regions.add(image_key)
        
        debug_print(f"   Total images: {len(annotation_results)}", level="info")
        debug_print(f"   Images with valid regions (bbox_corrected + reannotated): {len(images_with_valid_regions)}", level="success")
        debug_print(f"   Total regions: {total_regions_count}", level="info")
        debug_print(f"   Valid regions: {valid_regions_count}", level="success")
        debug_print(f"   Images without valid regions: {len(annotation_results) - len(images_with_valid_regions)} (will be skipped)", level="info")
        debug_print("", level="info")
    else:
        # If using all regions, all images are valid
        images_with_valid_regions = set(annotation_results.keys())
    
    # Filter already processed images AND images without valid regions
    images_to_process = []
    skipped_no_valid_regions = 0
    
    for image_key, image_data in annotation_results.items():
        # Skip if already processed (unless force)
        if not args.force and image_key in processed_images:
            continue
        
        # Skip if no valid regions (when filtering is enabled)
        if use_only_corrected and image_key not in images_with_valid_regions:
            skipped_no_valid_regions += 1
            continue
        
        images_to_process.append((image_key, image_data))
    
    debug_print(f"Skipping {len(processed_images)} already processed images", level="info")
    if skipped_no_valid_regions > 0:
        debug_print(f"Skipping {skipped_no_valid_regions} images without valid regions", level="info")
    debug_print(f"Total images to process: {len(images_to_process)}", level="success")
    
    if not images_to_process:
        debug_print("All images already processed", level="success")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sequential processing only
    processed_count = len(processed_images)
    total_time = 0.0
    start_time = time.time()
    
    debug_print("Starting sequential processing...", level="info")
    
    global generator_instance, debug_flag, generation_mode, output_file_path, cache_dir_global, embedding_model_instance, only_corrected_regions, per_image_output_dir
    
    generator_instance = MultiChoiceQAGenerator(
        args.base_url, gemini_key, args.model, args.max_retries
    )
    debug_flag = args.debug
    generation_mode = args.generation_mode
    output_file_path = None  # Not used in per-image mode
    cache_dir_global = args.cache_dir
    only_corrected_regions = use_only_corrected
    per_image_output_dir = args.output_dir
    
    # Initialize Aliyun embedding client
    try:
        debug_print("Initializing Aliyun DashScope embedding client (text-embedding-v4)...", level="info")
        
        # Get API key
        dashscope_key = args.aliyun_api_key or os.getenv("DASHSCOPE_API_KEY")
        
        # Initialize OpenAI client for Aliyun DashScope
        embedding_model_instance = OpenAIClient(
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        debug_print("Aliyun embedding client initialized successfully", level="success")
        debug_print(f"   Using model: text-embedding-v4 (batch size: 10, max tokens: 8192)", level="info")
    except Exception as e:
        error_msg = f"Failed to initialize Aliyun embedding client: {e}"
        debug_print(error_msg, level="error")
        raise RuntimeError(error_msg) from e
    
    with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
        for i, (image_key, image_data) in enumerate(images_to_process):
            try:
                result = process_image((image_key, image_data, i))
                
                # Save result per image
                result_metadata = {
                    "model": args.model,
                    "generation_mode": args.generation_mode,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_result_per_image(image_key, result, args.output_dir, result_metadata)
                
                processed_count += 1
                total_time += result.get('processing_time', 0)
                
                pbar.update(1)
            
            except Exception as e:
                debug_print(f"Processing failed {image_key}: {e}", level="error")
                processed_count += 1
    
    # Save summary
    summary_metadata = {
        "model": args.model,
        "generation_mode": args.generation_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_images_processed": processed_count,
        "total_images": len(images_to_process) + len(processed_images),
        "total_processing_time": total_time,
        "avg_processing_time": total_time / processed_count if processed_count > 0 else 0,
        "total_regions": total_regions,
        "wall_time": time.time() - start_time,
        "output_mode": "per_image",
        "output_directory": args.output_dir
    }
    
    # Save main summary
    summary_file = os.path.join(args.output_dir, "_processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_metadata, f, indent=2, ensure_ascii=False)
    
    # If both mode, also save summaries in subdirectories
    if args.generation_mode == "both":
        grounding_summary_file = os.path.join(args.output_dir, "grounding_mode", "_processing_summary.json")
        grounding_summary = summary_metadata.copy()
        grounding_summary['generation_mode'] = 'grounding_mode'
        with open(grounding_summary_file, 'w', encoding='utf-8') as f:
            json.dump(grounding_summary, f, indent=2, ensure_ascii=False)
        
        captioning_summary_file = os.path.join(args.output_dir, "captioning_mode", "_processing_summary.json")
        captioning_summary = summary_metadata.copy()
        captioning_summary['generation_mode'] = 'captioning_mode'
        with open(captioning_summary_file, 'w', encoding='utf-8') as f:
            json.dump(captioning_summary, f, indent=2, ensure_ascii=False)
    
    debug_print("", level="info")
    debug_print("=" * 60, level="title")
    debug_print("Processing complete!", level="success")
    debug_print(f"Processed images: {processed_count}", level="info")
    debug_print(f"Total time: {time.time() - start_time:.2f}s", level="info")
    debug_print(f"Results saved to: {args.output_dir}", level="success")
    debug_print(f"Summary saved to: {summary_file}", level="info")
    debug_print("=" * 60, level="title")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate functional region multiple-choice QA tasks (Simplified Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # RECOMMENDED: Both Modes (saves tokens by reusing grouping and verification!)
    python gen_region-func_multichoice-qa.py \
      --input-file /path/to/annotation.json \
      --output-dir /path/to/output \
      --generation-mode both \
      --cache-dir /path/to/cache \
        --model gemini-2.5-pro-thinking \
      --api-key "YOUR_API_KEY" \
        --base-url "https://xiaoai.plus/v1" \
        --debug

  # Output structure for "both" mode:
  #   /path/to/output/
  #     grounding_mode/
  #       IMAGE_result.json
  #       _processing_summary.json
  #     captioning_mode/
  #       IMAGE_result.json
  #       annotated_images/IMAGE/group*_*.png
  #       _processing_summary.json

  # Grounding Mode only: Embedding grouping + text-based region grounding QA
    python gen_region-func_multichoice-qa.py \
      --input-file /path/to/annotation.json \
      --output-dir /path/to/output/grounding_mode \
      --generation-mode grounding_mode \
      --cache-dir /path/to/cache \
        --model gemini-2.5-pro-thinking \
      --api-key "YOUR_API_KEY" \
        --base-url "https://xiaoai.plus/v1" \
      --debug

  # Captioning Mode only: Embedding grouping + annotated image captioning QA
    python gen_region-func_multichoice-qa.py \
      --input-file /path/to/annotation.json \
      --output-dir /path/to/output/captioning_mode \
      --generation-mode captioning_mode \
      --cache-dir /path/to/cache \
        --model gemini-2.5-pro-thinking \
      --api-key "YOUR_API_KEY" \
        --base-url "https://xiaoai.plus/v1" \
        --debug

Generation Pipeline (3 stages):
  1. Aliyun text-embedding-v4: Initial grouping based on visual description similarity (not functionality)
  2. Gemini Vision: Verify visual similarity, reject invalid groups, refine elements
  3. Question Generation: Create QA tasks from verified groups (tests functionality understanding)

Generation Modes:
  - both (RECOMMENDED): Generate both Grounding and Captioning questions simultaneously
        Saves tokens by reusing the same grouping and verification results
        Results saved to grounding_mode/ and captioning_mode/ subdirectories
        
  - grounding_mode: Text-based region grounding QA without images
        Questions focus on "Which element will achieve X goal?"
        Tests ability to ground functionality descriptions to correct regions
        
  - captioning_mode: Image-based captioning QA with visual annotation
        Shows image with ONE element circled, asks "What will happen when clicking it?"
        Options describe outcomes of all visually similar elements in the group
        Tests ability to understand and caption visual elements

Key Features:
  - Two-stage verification: Visual description similarity (Aliyun API) + Visual verification (Gemini)
  - Finds visually similar but functionally different elements (confusing for agents)
  - Parent-child relationship filtering (excludes hierarchical UI elements)
  - Sequential processing (recommended for API rate limiting)
  - Per-image output (one JSON file per image)
  - High-quality groups: Only groups validated by both text-based visual descriptions AND vision models
  - NO GPU required: Uses cloud APIs for all heavy computation

Requirements:
  - OpenAI client library: pip install openai
  - Aliyun DashScope API key (set DASHSCOPE_API_KEY environment variable or use --aliyun-api-key)
  - Gemini API key for vision verification (set via --api-key or OPENAI_API_KEY)
  - Cache directory with the following files:
    * tree.json: Parent-child relationship filtering (required)
    * *_meta_reannotated*.json: Contains corrected bbox + revised functionality + revised description (required)
  - By default, script only uses regions with BOTH bbox_corrected AND reannotated flags
  - Use --use-all-regions to disable filtering (not recommended)

API Limits:
  - Aliyun text-embedding-v4: batch size 10, max 8192 tokens per text
  - Automatic batching and rate limiting handled by the script
        """
    )
    
    parser.add_argument("--input-file", required=True,
                       help="Input functional region annotation JSON file")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for per-image mode (one JSON file per image)")
    parser.add_argument("--output-file", default=None, help=argparse.SUPPRESS)  # Hidden, not used
    parser.add_argument("--cache-dir", type=str, default="/mnt/vdb1/hongxin_li/AutoGUIv2/cache",
                       help="Cache directory path for reading reannotations (*_meta_reannotated*.json containing corrected bbox + revised functionality/description) and tree.json")
    parser.add_argument("--generation-mode", type=str, choices=["grounding_mode", "captioning_mode", "both"], default="both",
                       help="Question generation mode: grounding_mode=text-based region grounding QA, captioning_mode=annotated image captioning QA, both=generate both modes simultaneously (recommended to save tokens)")
    parser.add_argument("--api-key", default=None,
                       help="OpenAI API key (for Gemini vision and QA generation). If not provided, will use OPENAI_API_KEY environment variable")
    parser.add_argument("--base-url", default="https://xiaoai.plus/v1",
                       help="OpenAI API base URL (for Gemini vision and QA generation)")
    parser.add_argument("--model", default="gemini-2.5-pro-thinking",
                       help="Model to use for question generation")
    parser.add_argument("--aliyun-api-key", default=None,
                       help="Aliyun DashScope API key (for text-embedding-v4). If not provided, will use DASHSCOPE_API_KEY environment variable")
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)  # Hidden, always 1
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum API call retries")
    parser.add_argument("--sequential", action="store_true", default=True,
                       help=argparse.SUPPRESS)  # Hidden, always sequential
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--force", action="store_true",
                       help="Force reprocess already processed images")
    parser.add_argument("--use-all-regions", action="store_true",
                       help="Use all regions including those without bbox correction or reannotation (not recommended - only fully corrected and reannotated regions guarantee quality)")
    parser.add_argument("--test-image", type=str, default=None,
                       help="Test with a single image only (provide image key, filename, or path substring for matching)")
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    main(args)
