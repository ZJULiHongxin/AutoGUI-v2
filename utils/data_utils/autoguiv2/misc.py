
# Canonical action type names (snake_case gerunds).
# Rule: lowercase, spaces and hyphens replaced with underscores.

# Canonical action type names: base verb form, snake_case.
CANONICAL_ACTION_TYPES = {
    "click", "double_click", "right_click", "middle_click",
    "hover", "type", "drag", "scroll", "select",
    "swipe", "press", "long_press",
}

# Many-to-one map: every known surface form -> canonical base verb.
# Keys are pre-lowercased; spaces and hyphens are kept as-is so the
# normalization function can handle them before lookup.
ACTION_TYPE_ALIASES = {
    # click
    "click": "click",
    "clicking": "click",
    "clicked": "click",
    "clicks": "click",
    "click_and_hold": "click",
    "single_click": "click",
    "single-click": "click",
    "single click": "click",
    "single_clicking": "click",
    "left_click": "click",
    "left-click": "click",
    "left click": "click",
    "left_clicking": "click",
    "tap": "click",
    "tapping": "click",
    "tapped": "click",
    # double_click
    "double_click": "double_click",
    "double-click": "double_click",
    "double click": "double_click",
    "double_clicking": "double_click",
    "double-clicking": "double_click",
    "double clicking": "double_click",
    "double_clicked": "double_click",
    "dblclick": "double_click",
    "dbl_click": "double_click",
    "dbl-click": "double_click",
    "double_tap": "double_click",
    "double-tap": "double_click",
    "double tap": "double_click",
    # right_click
    "right_click": "right_click",
    "right-click": "right_click",
    "right click": "right_click",
    "right_clicking": "right_click",
    "right-clicking": "right_click",
    "right clicking": "right_click",
    "right_clicked": "right_click",
    "context_click": "right_click",
    "context click": "right_click",
    "context_menu": "right_click",
    # middle_click
    "middle_click": "middle_click",
    "middle-click": "middle_click",
    "middle click": "middle_click",
    "middle_clicking": "middle_click",
    "middle-clicking": "middle_click",
    "middle clicking": "middle_click",
    "wheel_click": "middle_click",
    "wheel-click": "middle_click",
    "wheel click": "middle_click",
    # hover
    "hover": "hover",
    "hovering": "hover",
    "hovered": "hover",
    "hovers": "hover",
    "mouse_over": "hover",
    "mouse-over": "hover",
    "mouse over": "hover",
    "mouseover": "hover",
    "mouse_hover": "hover",
    "mouse-hover": "hover",
    # type
    "type": "type",
    "typing": "type",
    "typed": "type",
    "types": "type",
    "input": "type",
    "inputting": "type",
    "input_text": "type",
    "input text": "type",
    "enter_text": "type",
    "enter text": "type",
    "text_input": "type",
    "keyboard_input": "type",
    "fill": "type",
    "filling": "type",
    "write": "type",
    "writing": "type",
    # drag
    "drag": "drag",
    "dragging": "drag",
    "dragged": "drag",
    "drags": "drag",
    "drag_and_drop": "drag",
    "drag-and-drop": "drag",
    "drag and drop": "drag",
    "drag_drop": "drag",
    "dragdrop": "drag",
    # scroll
    "scroll": "scroll",
    "scrolling": "scroll",
    "scrolled": "scroll",
    "scrolls": "scroll",
    "scroll_up": "scroll",
    "scroll_down": "scroll",
    "scroll_left": "scroll",
    "scroll_right": "scroll",
    # select
    "select": "select",
    "selecting": "select",
    "selected": "select",
    "selects": "select",
    "choose": "select",
    "choosing": "select",
    "chosen": "select",
    "pick": "select",
    "picking": "select",
    # swipe
    "swipe": "swipe",
    "swiping": "swipe",
    "swiped": "swipe",
    "swipes": "swipe",
    "swipe_up": "swipe",
    "swipe_down": "swipe",
    "swipe_left": "swipe",
    "swipe_right": "swipe",
    "flick": "swipe",
    "flicking": "swipe",
    "flicked": "swipe",
    # press
    "press": "press",
    "pressing": "press",
    "pressed": "press",
    "presses": "press",
    "key_press": "press",
    "key-press": "press",
    "key press": "press",
    "keypress": "press",
    "hotkey": "press",
    "shortcut": "press",
    "keyboard_shortcut": "press",
    # long_press
    "long_press": "long_press",
    "long-press": "long_press",
    "long press": "long_press",
    "long_pressing": "long_press",
    "long-pressing": "long_press",
    "long pressing": "long_press",
    "long_pressed": "long_press",
    "long_click": "long_press",
    "long-click": "long_press",
    "long click": "long_press",
    "long_tap": "long_press",
    "long-tap": "long_press",
    "long tap": "long_press",
    "press_and_hold": "long_press",
    "press-and-hold": "long_press",
    "press and hold": "long_press",
    # click_and_hold
    "click_and_hold": "click",
    "click-and-hold": "click",
    "click and hold": "click",
    "clicking_and_holding": "click",
    "clicking-and-holding": "click",
    "clicking and holding": "click",
    "hold": "click",
    "hold_click": "click",
    "hold click": "click",
    "click_hold": "click",
}



def normalize_action_type(action_type: str) -> str:
    """Normalize action_type to snake_case gerund form.

    Handles variants like 'double-clicking', 'double clicking', 'double_clicking'
    -> 'double_clicking'.
    """
    return action_type.strip().lower().replace("-", "_").replace(" ", "_")


OFFICIAL_TO_LOCALNAME_MAP = {
    "HongxinLi/ScreenSpot-Pro": "screenspot_pro",
    "MMInstruction/OSWorld-G": "osworld_g",
    "MMInstruction/MMBenchGUI": "mmbenchgui",
    "sujr/autogui-agentnet": "agentnet",
}