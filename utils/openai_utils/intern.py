import base64, requests, random
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_np(np_img):
    return base64.b64encode(Image.fromarray(np_img).tobytes()).decode('utf-8')

class INTERN:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 4096):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

        self.p_tokens = self.cmpl_tokens = 0

    def get_completion(self, prompt: str, logprobs: int = 20, temperature: float=0.0, max_new_tokens: int = 256):
        
        messages = [{"role": "user", "content": prompt}]
        
        data = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if logprobs != -1:
            data["logprobs"] = logprobs

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "error" not in result:
                # Extract usage if available
                if "usage" in result:
                    usage = result["usage"]
                    self.p_tokens += usage.get("prompt_tokens", 0)
                    self.cmpl_tokens += usage.get("completion_tokens", 0)
                
                status = True
                resp = result["content"][0]["text"] if "content" in result and len(result["content"]) > 0 else result.get("text", "")
                
                logprob_info = []
                # Note: logprob extraction may need adjustment based on actual API response format
                if logprobs != -1 and "logprobs" in result:
                    logprob_info = result["logprobs"]
            else:
                status, resp, logprob_info = False, result.get("error", {}).get("message", "Unknown error"), []
        except Exception as e:
            status, resp, logprob_info = False, str(e), []
        
        return status, resp, logprob_info

    def get_model_response(self, prompt: str, images: list = None, use_img_url: bool = True, logprob: bool = False, use_system: bool = True, sys_prompt: str = '', temperature: float | None = None, num_gen: int = 1, timeout: int = 360):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        if images is not None:
            for img in images:
                img_data = encode_image(img) if use_img_url else encode_image_np(img)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data
                    }
                })
        
        messages = [{"role": "system", "content": sys_prompt if sys_prompt else "You are a helpful assistant."}] if use_system else []
        messages.append({
                    "role": "user",
                    "content": content
                })

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }
        
        if logprob:
            data["logprobs"] = True
            data["top_logprobs"] = 4

        try:
            resp_list, logprobs_list, reasoning_content_list = [], [], []
            
            # Make multiple requests if num_gen > 1
            for _ in range(num_gen):
                response = requests.post(self.base_url, headers=self.headers, json=data, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                
                if "error" not in result:
                    # Extract usage if available
                    if "usage" in result:
                        usage = result["usage"]
                        self.p_tokens += usage.get("prompt_tokens", 0)
                        self.cmpl_tokens += usage.get("completion_tokens", 0)
                    
                    resp = result["content"][0]["text"] if "content" in result and len(result["content"]) > 0 else result.get("text", "")
                    reasoning_content = result.get("reasoning_content", "")
                    
                    if reasoning_content:
                        resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"
                    
                    logprobs = result.get("logprobs") if logprob else None
                    
                    resp_list.append(resp)
                    logprobs_list.append(logprobs)
                    reasoning_content_list.append(reasoning_content)
                else:
                    status, resp, logprobs = False, result.get("error", {}).get("message", "Unknown error"), None
                    if num_gen == 1:
                        return status, resp, logprobs
                    else:
                        return status, resp_list, logprobs_list
            
            status = True
        except Exception as e:
            status, resp, logprobs = False, str(e), None
            if num_gen == 1:
                return status, resp, logprobs
            else:
                return status, resp_list, logprobs_list
        
        if num_gen == 1:
            return status, resp_list[0], logprobs_list[0]
        else:
            return status, resp_list, logprobs_list

    def get_model_response_with_prepared_messages(self, messages: list, logprob: bool = False, temperature: float | None = None, max_new_tokens: int = None, timeout: int = 60):
        data = {
            "model": self.model,
            "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": messages
        }
        
        if logprob:
            data["logprobs"] = True
            data["top_logprobs"] = 4

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            if "error" not in result:
                # Extract usage if available
                if "usage" in result:
                    usage = result["usage"]
                    self.p_tokens += usage.get("prompt_tokens", 0)
                    self.cmpl_tokens += usage.get("completion_tokens", 0)
                
                status = True
                resp = result["content"][0]["text"] if "content" in result and len(result["content"]) > 0 else result.get("text", "")
                
                reasoning_content = result.get("reasoning_content", "")
                if reasoning_content:
                    resp = f"<think>\n{reasoning_content}\n</think>\n{resp}"
                
                logprobs = result.get("logprobs") if logprob else None
            else:
                status, resp, logprobs = False, result.get("error", {}).get("message", "Unknown error"), None
        except Exception as e:
            status, resp, logprobs = False, str(e), None
        
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
                img_data = encode_image(img) if use_img_url else encode_image_np(img)
                first_turn_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data
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

        data = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
        
        if logprob:
            data["logprobs"] = True
            data["top_logprobs"] = 4

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            if "error" not in result:
                # Extract usage if available
                if "usage" in result:
                    usage = result["usage"]
                    self.p_tokens += usage.get("prompt_tokens", 0)
                    self.cmpl_tokens += usage.get("completion_tokens", 0)
                
                status = True
                resp = result["content"][0]["text"] if "content" in result and len(result["content"]) > 0 else result.get("text", "")
                logprobs = result.get("logprobs") if logprob else None
            else:
                status, resp, logprobs = False, result.get("error", {}).get("message", "Unknown error"), None
        except Exception as e:
            status, resp, logprobs = False, str(e), None
        
        return status, resp, logprobs