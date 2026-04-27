import os
import re
import time
import json
import ast
import random
import logging
import argparse
import multiprocessing
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

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

import transformers.data.metrics.squad_metrics as squad_metrics

from utils.eval_utils.eval_utils import mind2web_action2step
from utils.data_utils.task_prompt_lib import (
    ATLAS_PROMPT,
    OSATLAS_MIND2WEB_PROMPT,
    Qwen3VL_SYS_PROMPT,
    constants,
    make_actionplanning_prompt,
    parse_atlas_action,
)
from utils.data_utils.misc import keep_unique_actions, remove_redundant_spaces
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.qwen3vl import Qwen3VL
from utils.openai_utils.misc import extract_thought_components_qwen3vl
from uipro.model.builder import load_pretrained_model
from uipro.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from uipro.conversation import conv_templates


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


ROOT_CANDIDATES = [
    "/mnt/nvme0n1p1/hongxin_li/UI_training_data/Mind2Web",
    "/data2/hongxin_li/UI_training_data/Mind2Web",
    "/data0/jingran/workspace/UI_training_data/Mind2Web",
    "/mnt/shared-storage/groups/stepone_mm/lhx/ui_data/Mind2Web",
]

DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TIMEOUT = 120


def debug_print(message: str, level: str = "info") -> None:
    level_to_color = {
        "info": Fore.CYAN,
        "warn": Fore.YELLOW,
        "error": Fore.RED,
        "success": Fore.GREEN,
        "title": Fore.MAGENTA,
    }
    color = level_to_color.get(level, Fore.CYAN)
    print(f"{color}{message}{Style.RESET_ALL}")


def select_dataset_root(data_root: Optional[str]) -> str:
    if data_root:
        if os.path.isdir(data_root):
            return data_root
        raise FileNotFoundError(f"Specified data root does not exist: {data_root}")

    for candidate in ROOT_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "No valid Mind2Web data root found. Please provide --data-root explicitly."
    )


def format_action_repr(action_repr: str) -> str:
    elem, act = action_repr.split("->")
    act = act.strip()
    elem = elem.replace("  ", " ").strip()

    if "TYPE:" in act:
        split_id = act.find(":")
        act, text = act[:split_id], act[split_id + 1 :]
        text = text.strip(' \n\\').replace('"', '\\"').replace("\n", "\\n")
        prev_act = f"type \"{text}\" into the {elem}"
    elif act == "ENTER":
        prev_act = f"press enter on {elem}"
    elif act == "CLICK":
        prev_act = f"click on {elem}"
    elif act == "HOVER":
        prev_act = f"hover over {elem}"
    elif "SELECT:" in act:
        split_id = act.find(":")
        value = act[split_id + 1 :].strip()
        prev_act = f"select {value} in the {elem}"
    else:
        raise ValueError(f"unknown action: {act}")

    return prev_act


def build_history_variants(
    prev_actions: List[str], step_idx: int, max_prev_acts: int
) -> Dict[str, str]:
    window = prev_actions[max(0, step_idx - max_prev_acts) : step_idx]

    if window:
        default_history = " ".join(
            f"Step {i}. {action.strip(' .')}." for i, action in enumerate(window, start=1)
        )
        atlas_history = "\n".join(
            f"Step {i}: {action.strip(' .')}." for i, action in enumerate(window, start=1)
        )
        qwen3_history = "; ".join(
            f"Step {i}: {action.strip(' .').replace('\n', ' ').replace('\"', '')}"
            for i, action in enumerate(window, start=1)
        )
    else:
        default_history = atlas_history = qwen3_history = "None"

    retained_idxs, retained_history = keep_unique_actions(prev_actions[:step_idx])
    retained_history = retained_history[-max_prev_acts:]
    if retained_history:
        start_index = max(1, len(retained_idxs) - len(retained_history) + 1)
        qwen2_history = " ".join(
            f"Step {start_index + idx}. {remove_redundant_spaces(hist.replace('  ', ' ').replace('[', ' ', 1).replace(']', ' ', 1).strip(' .'))}."
            for idx, hist in enumerate(retained_history)
        )
    else:
        qwen2_history = "None"

    return {
        "default": default_history,
        "atlas": atlas_history,
        "qwen2": qwen2_history,
        "qwen3": qwen3_history,
    }


def safe_literal_eval(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def extract_action_from_response(
    response: str, runtime_type: str, is_atlas: bool
) -> Optional[Dict[str, Any]]:
    if is_atlas:
        return parse_atlas_action(response, device="desktop")

    if runtime_type == "qwen3":
        try:
            components = extract_thought_components_qwen3vl(response)
            action_raw = components.get("action")
            if isinstance(action_raw, dict):
                return action_raw
            if isinstance(action_raw, str):
                action_raw = action_raw.strip()
                if "{" in action_raw and "}" in action_raw:
                    action_raw = action_raw[action_raw.find("{") : action_raw.rfind("}") + 1]
                    return safe_literal_eval(action_raw)
        except Exception:
            return None

    candidates: List[str] = []
    if "Action:" in response:
        action_section = response.split("Action:")[-1]
        lines = [line.strip().strip("`") for line in action_section.splitlines() if line.strip()]
        candidates.extend(lines)

    json_like = re.findall(r"\{[^{}]*\}", response)
    candidates.extend(json_like)

    for candidate in reversed(candidates):
        if "{" not in candidate or "}" not in candidate:
            continue
        fragment = candidate[candidate.find("{") : candidate.rfind("}") + 1]
        if "action_type" not in fragment:
            continue
        parsed = safe_literal_eval(fragment)
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed

    return None


def normalize_click_point(action_pred: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    point = action_pred.get("target")
    if point is None:
        point = action_pred.get("click_point")

    if point is None:
        return None

    if isinstance(point, str):
        point = point.strip().strip("()").replace(" ", "")
        parts = point.split(",")
        if len(parts) >= 2:
            try:
                point = (float(parts[0]), float(parts[1]))
            except ValueError:
                return None

    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])

    return None


def prepare_entries(
    dataset: List[Dict[str, Any]],
    images_dir: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for ep_idx, episode in enumerate(dataset):
        if args.debug and ep_idx > 5:
            break

        goal = episode.get("confirmed_task", "")
        annot_id = episode.get("annotation_id", f"episode_{ep_idx}")

        prev_actions: List[str] = []
        for action_repr in episode.get("action_reprs", []):
            try:
                prev_actions.append(format_action_repr(action_repr))
            except Exception as exc:
                debug_print(f"⚠️ Failed to parse previous action: {exc}", level="warn")
                prev_actions.append("")

        for step_idx, step in enumerate(episode.get("actions", [])):
            if "bbox" not in step:
                debug_print("⚠️ action missing bbox; skip", level="warn")
                continue

            filename = f"{annot_id}-{step.get('action_uid', step_idx)}.jpg"
            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                alt_path = img_path.replace("mind2web_images/", "")
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    debug_print(f"⚠️ Image not found: {img_path}", level="warn")
                    continue

            try:
                with Image.open(img_path) as img:
                    image_size = img.size
            except Exception as exc:
                debug_print(f"⚠️ Failed to open image {img_path}: {exc}", level="warn")
                continue

            try:
                action_step, bbox_ref = mind2web_action2step(
                    step,
                    image_size,
                    scale=args.scale,
                    return_bbox=True,
                )
            except Exception as exc:
                debug_print(f"⚠️ Failed to convert GT action: {exc}", level="warn")
                continue

            try:
                action_step_ref = ast.literal_eval(action_step)
            except Exception as exc:
                debug_print(f"⚠️ Failed to parse GT action dict: {exc}", level="warn")
                continue

            history = build_history_variants(prev_actions, step_idx, args.max_prev_acts)

            entry = {
                "entry_id": f"{annot_id}-{step.get('action_uid', step_idx)}",
                "episode_idx": ep_idx,
                "step_idx": step_idx,
                "goal": goal,
                "annot_id": annot_id,
                "image_path": img_path,
                "history": history,
                "action_step": action_step,
                "action_step_ref": action_step_ref,
                "bbox_ref": bbox_ref,
                "image_size": image_size,
            }

            entries.append(entry)

    return entries


worker_runtime: Dict[str, Any] = {}


def init_worker(model_args: Dict[str, Any], shared_config: Dict[str, Any]) -> None:
    global worker_runtime

    worker_runtime = {
        "type": None,
        "model": None,
        "shared": shared_config,
    }

    model_path = model_args.get("model_path", "")
    provider = model_args.get("provider", "") or "openai"
    pretrained = model_args.get("pretrained", model_path)

    model_lower = model_path.lower()

    if "slime" in model_lower:
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, use_flash_attn=True
        )
        model.generation_config.eos_token_id = 107
        worker_runtime.update(
            {
                "type": "slime",
                "tokenizer": tokenizer,
                "model": model,
                "image_processor": image_processor,
                "gen_kwargs": {
                    "temperature": shared_config.get("temperature", 0.0),
                    "top_p": shared_config.get("top_p"),
                    "num_beams": shared_config.get("num_beams", 1),
                    "max_new_tokens": shared_config.get(
                        "max_new_tokens", DEFAULT_MAX_NEW_TOKENS
                    ),
                },
            }
        )
    elif "qwen2" in model_lower:
        worker_runtime.update(
            {
                "type": "qwen2",
                "model": QWen2VL(device="cuda", model_name=model_path),
            }
        )
    elif "qwen3" in model_lower or provider == "aliyun":
        base_url = model_args.get("base_url") or os.environ.get(
            "DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        api_key = model_args.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "")
        worker_runtime.update(
            {
                "type": "qwen3",
                "model": Qwen3VL(
                    base_url=base_url,
                    api_key=api_key,
                    model=pretrained,
                    temperature=0.0,
                    max_tokens=shared_config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                ),
            }
        )
    else:
        base_url = model_args.get("base_url") or os.environ.get(
            "OPENAI_API_BASE_XIAOAI", os.environ.get("OPENAI_API_BASE", "")
        )
        api_key = model_args.get("api_key") or os.environ.get(
            "OPENAI_API_KEY_XIAOAI", os.environ.get("OPENAI_API_KEY", "")
        )
        if not base_url or not api_key:
            debug_print("⚠️ Missing base URL or API key for OpenAI-compatible provider", "warn")

        worker_runtime.update(
            {
                "type": "openai",
                "model": OpenAIModel(
                    base_url=base_url,
                    api_key=api_key,
                    model=pretrained,
                    temperature=0.0,
                    max_tokens=shared_config.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                ),
            }
        )


def run_slime_inference(runtime: Dict[str, Any], prompt: str, image_path: str) -> str:
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    image_processor = runtime["image_processor"]
    gen_kwargs = runtime.get("gen_kwargs", {}).copy()

    conv = conv_templates["gemma"].copy()
    conv.append_message(conv.roles[0], f"{constants.DEFAULT_IMAGE_TOKEN}\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    with Image.open(image_path) as img:
        image = img.convert("RGB")
    img_tensor = process_images([image], image_processor, model.config).to(
        dtype=model.dtype, device=model.device
    )

    gen_kwargs["image_sizes"] = [image.size]

    input_ids = (
        tokenizer_image_token(
            prompt_formatted,
            tokenizer,
            constants.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(device=model.device)
    )

    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=img_tensor,
            image_sizes=gen_kwargs["image_sizes"],
            do_sample=gen_kwargs.get("temperature", 0) > 0,
            temperature=gen_kwargs.get("temperature", 0),
            top_p=gen_kwargs.get("top_p"),
            num_beams=gen_kwargs.get("num_beams", 1),
            max_new_tokens=gen_kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            use_cache=True,
        )

    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    return response


def process_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    global worker_runtime

    shared = worker_runtime.get("shared", {})
    runtime_type = worker_runtime.get("type", "openai")
    is_atlas = shared.get("is_atlas", False)

    history_map = entry.get("history", {})
    if is_atlas:
        history_str = history_map.get("atlas", "None")
    elif runtime_type == "qwen2":
        history_str = history_map.get("qwen2", history_map.get("default", "None"))
    elif runtime_type == "qwen3":
        history_str = history_map.get("qwen3", history_map.get("default", "None"))
    else:
        history_str = history_map.get("default", "None")

    if is_atlas:
        prompt = ATLAS_PROMPT.format(global_task=entry["goal"], history=history_str)
    else:
        prompt = make_actionplanning_prompt(
            entry["goal"],
            history_str,
            device_tag=shared.get("device_tag", ""),
            prompt_format_type="simple",
            with_cot=bool(shared.get("with_cot", False)),
            without_action_space=True,
            use_action_refexp=bool(shared.get("use_action_refexp", False)),
        )

    response = ""
    status = "ok"
    error_message = None

    try:
        if runtime_type == "slime":
            response = run_slime_inference(worker_runtime, prompt, entry["image_path"])
        elif runtime_type == "qwen2":
            response = worker_runtime["model"].get_model_response(
                prompt,
                f"file://{entry['image_path']}",
                max_new_tokens=shared.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                sys_prompt=shared.get("sys_prompt", ""),
            )
        elif runtime_type == "qwen3":
            success, response, _ = worker_runtime["model"].get_model_response(
                prompt,
                [entry["image_path"]],
                use_img_url=True,
                temperature=0.0,
                sys_prompt=Qwen3VL_SYS_PROMPT,
                timeout=shared.get("timeout", DEFAULT_TIMEOUT),
            )
            if not success:
                raise RuntimeError(response)
        else:
            success, response, _ = worker_runtime["model"].get_model_response(
                prompt,
                [entry["image_path"]],
                use_img_url=True,
                temperature=0.0,
                timeout=shared.get("timeout", DEFAULT_TIMEOUT),
                sys_prompt=shared.get("sys_prompt", ""),
            )
            if not success:
                raise RuntimeError(response)
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        response = str(exc)

    action_step_ref = dict(entry["action_step_ref"])
    action_step_ref["box"] = entry["bbox_ref"]

    step_result = {
        "episode_idx": entry["episode_idx"],
        "step_idx": entry["step_idx"],
        "img_path": os.path.basename(entry["image_path"]),
        "task": entry["goal"],
        "prompt": prompt,
        "response": response,
        "GT_action": entry["action_step"],
        "GT_box": entry["bbox_ref"],
        "Op_match": False,
        "Ele_match": False,
        "Op_F1": [0.0, action_step_ref.get("action_type")],
        "status": status,
    }

    if status != "ok":
        step_result["error"] = error_message
        return step_result

    action_pred = extract_action_from_response(response, runtime_type, is_atlas)
    if action_pred is None:
        step_result["status"] = "wrong_format"
        step_result["error"] = "Failed to parse action"
        return step_result

    step_result["action_pred"] = action_pred

    action_type_pred = action_pred.get("action_type")
    if isinstance(action_type_pred, str):
        action_type_pred_norm = action_type_pred.lower()
    else:
        action_type_pred_norm = action_type_pred

    if action_type_pred_norm in ["click", "hover"]:
        action_type_pred_norm = "click"
        action_pred["action_type"] = "click"

    if (
        action_type_pred_norm == action_step_ref.get("action_type")
        or action_type_pred_norm == action_step_ref.get("ori_act", "").lower()
    ):
        step_result["Op_match"] = True

    click_point = normalize_click_point(action_pred)

    if action_type_pred_norm == "enter":
        step_result["Ele_match"] = step_result["Op_match"]
        step_result["Op_F1"][0] = 1.0 if step_result["Op_match"] else 0.0
    else:
        if click_point is not None:
            scale_value = worker_runtime.get("shared", {}).get("scale", 1000)
            cp_x = click_point[0] / scale_value
            cp_y = click_point[1] / scale_value
            bbox_ref = entry["bbox_ref"]
            if (
                bbox_ref[0] <= cp_x <= bbox_ref[2]
                and bbox_ref[1] <= cp_y <= bbox_ref[3]
            ):
                step_result["Ele_match"] = True

        pred_str = str(action_type_pred_norm)
        if action_type_pred_norm in [3, "input_text", 2, "select"]:
            text_val = action_pred.get("text", action_pred.get("value", ""))
            if isinstance(text_val, str):
                pred_str += f" {text_val.lower()}"

        ref_action_type = action_step_ref.get("action_type")
        ref_str = str(ref_action_type)
        if ref_action_type in [3, "input_text", 2, "select"]:
            ref_value = action_step_ref.get("value", "")
            ref_str += f" {ref_value.lower()}"

        try:
            op_f1 = squad_metrics.compute_f1(pred_str, ref_str)
        except Exception:
            op_f1 = 0.0
        step_result["Op_F1"][0] = float(op_f1)

    if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
        step_result["Step_success"] = True
    else:
        step_result["Step_success"] = False

    print(
        (
            Fore.GREEN if step_result["Op_match"] and step_result["Ele_match"] else Fore.RED
        )
        + f"Op: {step_result['Op_match']} | Elem: {step_result['Ele_match']} | F1: {step_result['Op_F1'][0]:.2f}"
        + Style.RESET_ALL
        + f": GT: {action_step_ref} <=> Pred: {action_pred}"
    )

    return step_result


def process_entries_with_multiprocessing(
    entries: List[Dict[str, Any]],
    model_args: Dict[str, Any],
    shared_config: Dict[str, Any],
    max_workers: int,
) -> List[Dict[str, Any]]:
    manager = multiprocessing.Manager()
    processed = manager.Value("i", 0)
    lock = manager.Lock()
    start_time = time.time()

    def update_throughput() -> None:
        elapsed = time.time() - start_time
        count = processed.value
        throughput = count / elapsed if elapsed > 0 else 0.0
        print(
            f"\rThroughput: {throughput:.2f} steps/s | Processed: {count}/{len(entries)} | Elapsed: {elapsed:.1f}s",
            end="",
            flush=True,
        )

    shared_config = dict(shared_config)
    shared_config.setdefault("timeout", DEFAULT_TIMEOUT)

    results: List[Dict[str, Any]] = []

    with multiprocessing.Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(model_args, shared_config),
    ) as pool:
        for result in pool.imap(process_entry, entries):
            results.append(result)
            with lock:
                processed.value += 1
            update_throughput()

    print()
    return results


def restructure_results(
    step_results: List[Dict[str, Any]], num_episodes: int
) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = [[] for _ in range(num_episodes)]
    for result in step_results:
        idx = result.get("episode_idx", 0)
        if 0 <= idx < num_episodes:
            episodes[idx].append(result)

    for episode in episodes:
        episode.sort(key=lambda x: x.get("step_idx", 0))

    return episodes


def compute_metrics(results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    num_step_success = 0
    num_episode_success = 0

    op_f1: Dict[str, List[float]] = {"click": [], "select": [], "input_text": []}
    macro_ele_acc: Dict[int, List[int]] = {}
    macro_step_acc: Dict[int, List[int]] = {}
    macro_action_f1: Dict[int, List[float]] = {}

    for ep_idx, episode in enumerate(results):
        num_episode += 1
        episode_success = True
        macro_ele_acc[ep_idx] = []
        macro_step_acc[ep_idx] = []
        macro_action_f1[ep_idx] = []

        for step_result in episode:
            num_step += 1

            if step_result.get("Op_match"):
                num_op += 1

            if step_result.get("Ele_match"):
                num_ele += 1
                macro_ele_acc[ep_idx].append(1)
            else:
                macro_ele_acc[ep_idx].append(0)

            op_type = step_result.get("Op_F1", [0, None])[1]
            if op_type in op_f1:
                macro_action_f1[ep_idx].append(step_result["Op_F1"][0])
                op_f1[op_type].append(step_result["Op_F1"][0])
            else:
                macro_action_f1[ep_idx].append(step_result["Op_F1"][0])

            if step_result.get("Op_F1", [0])[0] == 1.0 and step_result.get("Ele_match"):
                num_step_success += 1
                macro_step_acc[ep_idx].append(1)
            else:
                macro_step_acc[ep_idx].append(0)
                episode_success = False

        if episode_success and episode:
            num_episode_success += 1

    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    marco_op_f1 = safe_mean([safe_mean(v) for v in op_f1.values() if v])
    element_acc = num_ele / num_step if num_step > 0 else 0.0
    step_success = num_step_success / num_step if num_step > 0 else 0.0
    episode_success_rate = num_episode_success / num_episode if num_episode > 0 else 0.0

    macro_ele_acc_values = [safe_mean(v) for v in macro_ele_acc.values() if v]
    macro_step_acc_values = [safe_mean(v) for v in macro_step_acc.values() if v]
    macro_action_f1_values = [safe_mean(v) for v in macro_action_f1.values() if v]

    macro_ele_acc_result = safe_mean(macro_ele_acc_values)
    macro_step_acc_result = safe_mean(macro_step_acc_values)
    macro_action_f1_result = safe_mean(macro_action_f1_values)

    metrics = {
        "Operation F1": marco_op_f1,
        "Element Acc": element_acc,
        "Step Success": step_success,
        "Episode Success": episode_success_rate,
        "Operation F1 cate": [safe_mean(v) for v in op_f1.values()],
        "Macro Ele Acc": macro_ele_acc_result,
        "Macro Op F1": macro_action_f1_result,
        "Macro Step SR": macro_step_acc_result,
    }

    metrics["counts"] = {
        "total_steps": num_step,
        "total_episodes": num_episode,
        "op_matches": num_op,
        "ele_matches": num_ele,
    }

    return metrics


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    model_path = args.pretrained.rstrip("/ ")

    postfix = model_path.replace("lora/", "").replace("merged/", "")
    if "snapshots" in postfix:
        postfix = postfix[postfix.find("models--") + 8 : postfix.find("snapshots") - 1]
    elif len(postfix.split("/")) == 2:
        postfix = postfix.replace("/", "--")
    elif "checkpoint-" in postfix:
        postfix = "/".join(postfix.split("/")[-2:])
    else:
        postfix = postfix.replace("/", "-")

    is_atlas = "atlas" in postfix.lower()
    if is_atlas:
        args.scale = 1000
        args.max_prev_acts = min(args.max_prev_acts, 6)

    data_root = select_dataset_root(args.data_root)
    mind2web_imgs_dir = os.path.join(data_root, "mind2web_images")
    dataset_path = os.path.join(data_root, f"mind2web_data_test_{args.task}.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Mind2Web dataset split not found: {dataset_path}")

    debug_print("═" * 60, "title")
    debug_print("🔍 Mind2Web Evaluation (Multiprocessing)", "title")
    debug_print("═" * 60, "title")

    debug_print(f"📁 Data root: {data_root}")
    debug_print(f"🖼️ Images dir: {mind2web_imgs_dir}")
    debug_print(f"📄 Dataset file: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        mind2web_test = json.load(f)

    entries = prepare_entries(mind2web_test, mind2web_imgs_dir, args)
    if args.sample_limit is not None:
        entries = entries[: args.sample_limit]

    if not entries:
        debug_print("❌ No valid entries to evaluate", "error")
        return

    debug_print(f"✅ Prepared {len(entries)} evaluation steps", "success")

    shared_config = {
        "device_tag": args.device_tag,
        "with_cot": bool(args.cot),
        "use_action_refexp": bool(args.action_refexp),
        "scale": args.scale,
        "max_prev_acts": args.max_prev_acts,
        "postfix": postfix,
        "is_atlas": is_atlas,
        "sys_prompt": OSATLAS_MIND2WEB_PROMPT if is_atlas else "",
        "max_new_tokens": args.max_new_tokens,
        "timeout": args.timeout,
        "temperature": 0.0,
        "args": vars(args),
    }

    model_args = {
        "model_path": model_path,
        "pretrained": args.pretrained,
        "provider": args.provider,
        "base_url": args.base_url,
        "api_key": args.api_key,
    }

    step_results = process_entries_with_multiprocessing(
        entries, model_args, shared_config, args.max_workers
    )

    results = restructure_results(step_results, len(mind2web_test))
    metrics = compute_metrics(results)

    debug_print(f"Operation F1: {metrics['Operation F1']:.4f}")
    debug_print(f"Element Acc: {metrics['Element Acc']:.4f}")
    debug_print(f"Step Success: {metrics['Step Success']:.4f}")
    debug_print(f"Episode Success: {metrics['Episode Success']:.4f}")
    debug_print(f"Macro Element Acc: {metrics['Macro Ele Acc']:.4f}")
    debug_print(f"Macro Op F1: {metrics['Macro Op F1']:.4f}")
    debug_print(f"Macro Step SR: {metrics['Macro Step SR']:.4f}")

    eval_result_dir = os.path.join(
        os.path.dirname(__file__), "eval_results", "mind2web", postfix
    )
    os.makedirs(eval_result_dir, exist_ok=True)

    save_file = os.path.join(
        eval_result_dir,
        f"{args.task}-{datetime.now().strftime('%m-%d-%H-%M-%S')}.json",
    )

    meta = vars(args).copy()
    meta.update({
        "model_postfix": postfix,
        "data_root": data_root,
        "num_entries": len(entries),
    })

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "overall_results": {k: v for k, v in metrics.items() if k != "counts"},
                "counts": metrics.get("counts", {}),
                "log": results,
            },
            f,
            indent=2,
        )

    debug_print(f"💾 Results saved to: {save_file}", "info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on Mind2Web with multiprocessing"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="gemini-2.5-pro-thinking",
        help="Model name or path",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model provider (openai, aliyun, etc.)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
        help="API base URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
        help="API key",
    )
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--action_refexp", type=bool, default=False)
    parser.add_argument(
        "--task",
        type=str,
        default="website",
        choices=["website", "task", "domain"],
    )
    parser.add_argument("--device_tag", type=str, default="Web")
    parser.add_argument("--max_prev_acts", type=int, default=66)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    main(args)

