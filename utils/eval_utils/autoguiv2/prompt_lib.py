from typing import Dict, List, Any, Optional, Literal, TypeAlias
from utils.data_utils.make_autogui_data.qwen2vl_to_other_formats import get_ui_tars_question_template, get_uground_question_template

# AutoGUI++
def make_autoguiplus_prompt(refexp, model_name):
    if 'qwen2' in model_name:
        prompt = f"Point to the control that facilitates the action: {refexp}"
    elif 'tars' in model_name:
        prompt = get_ui_tars_question_template(refexp)
    elif 'uground' in model_name:
        prompt = get_uground_question_template(refexp)
    return prompt

# UGround Prompt
UGROUND_PROMPT = """Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- Your answer should be a single string (x, y) corresponding to the point of the interest.

Description: {instruction}

Answer:"""

# InfiGUI-G1
INFIGUIG1_SYSPROMPT = 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags.'
INFIGUIG1_PROMPT = '''The screen's resolution is {new_width}x{new_height}.
Locate the UI element(s) for "{instruction}", output the coordinates using JSON format: [{{"point_2d": [x, y]}}, ...]'''

# GUI-R1
GUIR1_PROMPT=(
        "You are RUN1-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, I want you to continue executing the command '{instruction}', with the action history being 'None'.\n"
        "Please provide the action to perform (enumerate from ['click']), the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum[ 'click'], 'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
        "Example:\n"
        "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
)

# UI-Venus
UIVENUS_PROMPT = "Outline the position corresponding to the instruction: {instruction}. The output should be only [x1,y1,x2,y2]."


# HOLO Prompt
from pydantic import BaseModel, Field
class ClickAbsoluteAction(BaseModel):
    """Click at absolute coordinates."""

    action: Literal["click_absolute"] = "click_absolute"
    x: int = Field(description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(description="The y coordinate, number of pixels from the top edge.")
    
HOLO_PROMPT = f"""Localize an element on the GUI image according to the provided target and output a click position.
     * You must output a valid JSON following the format: {ClickAbsoluteAction.model_json_schema()}"""

HOLO_BBOX_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI element, you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax]. Note that the X-axis runs horizontally from left to right, and the Y-axis runs vertically from top to bottom.

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:"""

# OpenCUA
OPENCUA_SYSPROMPT = (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task."
    )
