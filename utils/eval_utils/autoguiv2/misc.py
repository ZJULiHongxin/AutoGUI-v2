from typing import List

# Canonical action type names (snake_case gerunds).
# Rule: lowercase, spaces and hyphens replaced with underscores.
CANONICAL_ACTION_TYPES = {
    "clicking",
    "hovering",
    "typing",
    "dragging",
    "scrolling",
    "selecting",
    "swiping",
    "pressing",
    "double_clicking",
    "right_clicking",
    "long_pressing",
    "middle_clicking",
    "clicking_and_holding",
}

def normalize_action_type(action_type: str) -> str:
    """Normalize action_type to snake_case gerund form.

    Handles variants like 'double-clicking', 'double clicking', 'double_clicking'
    -> 'double_clicking'.
    """
    return action_type.strip().lower().replace("-", "_").replace(" ", "_")


def adjust_bbox(model_path: str, bbox: List[int]) -> List[int]:
    """
    Adjust the bbox to the model's coordinate system.
    """
    
    # https://ai.google.dev/gemini-api/docs/image-understanding
    if 'gemini' in model_path:
        return [bbox[1], bbox[0], bbox[3], bbox[2]] if len(bbox) == 4 else [bbox[1], bbox[0]] # [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax]
    else:
        return bbox