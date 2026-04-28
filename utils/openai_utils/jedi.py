import json, base64, openai, random
from PIL import Image
from utils.data_utils.task_prompt_lib import Qwen3VL_SYS_PROMPT
from utils.data_utils.misc import encode_image, get_image_mime, get_image_dimensions
from qwen_agent.tools.base import BaseTool, register_tool
from typing import List, Tuple, Union

FN_CALL_TEMPLATE = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return <tool_call>\n{{"name": "computer_use", "arguments": {{"action": "wait", "time": 10}}}}
</tool_call>
"""

class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "scroll",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.name = cfg["name"]
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click"]:
            return self._mouse_click(action)
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str):
        raise NotImplementedError()

    def _key(self, keys: List[str]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _mouse_move(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _scroll(self, pixels: int):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()

# https://github.com/xlang-ai/OSWorld-G/blob/main/evaluation/qwen25_vllm_osworld_g_jedi.py
class JEDI:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 4096):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.p_tokens = self.cmpl_tokens = 0
            
    def get_completion(self, prompt: str, logprobs: int = 20, temperature: float=0.0, max_new_tokens: int = 256):
        
        logprob_args = {"logprobs": logprobs} if logprobs != -1 else {}

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            timeout=30,
            **logprob_args
        )
        
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].text
            
            logprob_info = []
            if logprobs != -1:
                for token, text_offset, token_logprob, top_logprobs in zip(response.choices[0].logprobs.tokens, response.choices[0].logprobs.text_offset, response.choices[0].logprobs.token_logprobs, response.choices[0].logprobs.top_logprobs):
                    logprob_info.append({
                        'token': token, # str
                        'text_offset': text_offset, # int
                        'token_logprob': token_logprob, # float
                        'top_logprobs': top_logprobs # dict[<candidate token>: <logprob>]
                        })
        else:
            status, resp, logprob_info = False, response.error["message"], []
        
        return status, resp, logprob_info

    def get_model_response(self, prompt: str, images: list = None, use_img_url: bool = True, logprob: bool = False, use_system: bool = True, sys_prompt: str = '', temperature: float | None = None, num_gen: int = 1, timeout: int = 60, **kwargs: dict):
        img = images[0]

        W, H = get_image_dimensions(img)
        computer_use = ComputerUse(
            cfg={
                "name": "computer_use",
                "display_width_px": W,
                "display_height_px": H,
            }
        ) # The display_width_px and display_height_px are not used.

        tools = [computer_use.function]

        tool_descs = [{"type": "function", "function": f} for f in tools]
        tool_descs = "\n".join(
            [json.dumps(f, ensure_ascii=False) for f in tool_descs]
        )

        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": FN_CALL_TEMPLATE.format(tool_descs=tool_descs)},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{get_image_mime(img)};base64,{encode_image(img)}" if use_img_url else f"data:image/jpeg;base64,{encode_image_np(img)}"},
                    },
                    
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": '<tool_call>\n{"name": "computer_use", "arguments": {"action": ',
                    }
                ],
            }
        ]

        
        logprob_args = {"logprobs": True, "top_logprobs": 4} if logprob else {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=self.max_tokens,
            timeout=timeout,
            n=num_gen,
            **logprob_args
        )

        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            
            resp_list, logprobs_list, reasoning_content_list = [], [], []

            for i in range(len(response.choices)):
                reasoning_content = getattr(response.choices[i].message, 'reasoning_content', '')
                resp = response.choices[i].message.content

                if reasoning_content:
                    resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"

                logprobs = response.choices[i].logprobs.content if logprob and response.choices[i].logprobs is not None else None
                
                resp_list.append(resp)
                logprobs_list.append(logprobs)
                reasoning_content_list.append(reasoning_content)
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        if num_gen == 1:
            return status, resp_list[0], logprobs_list[0]
        else:
            return status, resp_list, logprobs_list

    def get_model_response_with_prepared_messages(self, messages: list, logprob: bool = False, temperature: float | None = None, max_new_tokens: int = None, timeout: int = 60):
        logprob_args = {"top_logprobs": 4} if logprob else {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_new_tokens if max_new_tokens is not None else self.max_tokens,
            timeout=timeout,
            logprobs=logprob,
            **logprob_args
        )

        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].message.content

            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            if reasoning_content:
                resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"

            logprobs = response.choices[0].logprobs.content if logprob and response.choices[0].logprobs is not None else None
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        return status, resp, logprobs
    
    def get_model_response_history(self, prompt: str, images: list = None, history: list = None, use_img_url: bool = True, logprob: bool = False, use_system: bool = True):
        messages = [{"role": "system", "content": "You are a helpful assistant."}] if use_system else []

        # 1st turn
        first_turn_content = [
            {
                "type": "text",
                "text": history[0]
            }
        ]

        if images is not None:
            for img in images:
                first_turn_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{get_image_mime(img)};base64,{encode_image(img)}" if use_img_url else f"data:image/jpeg;base64,{encode_image_np(img)}"
                    }
                })

        first_turn = {
                    "role": "user",
                    "content": first_turn_content
                }

        messages.append(first_turn)

        # subsequent history
        for turn_idx, turn in enumerate(history[1:]):
            messages.append(
                {
                    'role': 'assistant' if turn_idx % 2 == 0 else 'user',
                    'content': turn
                }
            )

        # add the latest prompt
        messages.append({
                    "role": "user",
                    "content": prompt
                })

        logprob_args = {"top_logprobs": 4} if logprob else {}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=300,
            logprobs=logprob,
            **logprob_args
        )
        
        # response = requests.post(self.base_url, headers=headers, json=payload).json()
        if "error" not in response:
            usage = response.usage
            self.p_tokens += usage.prompt_tokens
            self.cmpl_tokens += usage.completion_tokens
            status = True
            resp = response.choices[0].message.content
            logprobs = response.choices[0].logprobs.content if logprob and response.choices[0].logprobs is not None else None
        else:
            status, resp, logprobs = False, response.error["message"], None
        
        return status, resp, logprobs