import os, time, traceback
import cv2
import random
import json
import logging
import ast
import argparse
import uuid
from PIL import Image
from datetime import datetime
from utils.eval_utils.action_matching import *
from pprint import pprint
from colorama import Fore, Style
import transformers.data.metrics.squad_metrics as squad_metrics
from utils.data_utils.task_prompt_lib import *
from utils.openai_utils.openai import OpenAIModel
from utils.openai_utils.qwen2vl import QWen2VL
from utils.openai_utils.qwen3vl import Qwen3VL
from utils.data_utils.misc import keep_unique_actions, get_image_dimensions, get_swipe_direction, resize_image
from utils.openai_utils.misc import  extract_protocol_components, extract_all_action_jsons, extract_gemini_cua_protocol_components, extract_claude_cua_protocol_components, extract_thought_components_qwen3vl
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial

logging.basicConfig(level=logging.INFO)

def scroll2swipe(direction):
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'

# calculate action f1 following androidcontrol
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default=['qwen3-vl-8b-thinking', 'doubao-seed-1-6-251015', 'gpt-5', 'claude-sonnet-4-5-20250929-thinking', 'gemini-2.5-pro-thinking', 'o3'][-1])
parser.add_argument('--provider', type=str, default=['openai', 'aliyun', ''][0])
parser.add_argument('--base_url', type=str, default=os.environ.get('OPENAI_API_BASE_XIAOAI'))
parser.add_argument('--max_prev_acts', type=int, default=999)
parser.add_argument('--device_tag', type=str, default='Android')
parser.add_argument('--prompt_format_type', type=str, default='outcome_pred', choices=['reflec', 'outcome_pred', 'wo_outcome_pred', 'direct'])
parser.add_argument('--preset_id_file', type=str, default='utils/eval_utils/androidcontrol_test/selected_andcon_idx.json')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of parallel workers')
parser.add_argument('--debug', type=bool, default=False)

args, _ = parser.parse_known_args()

# # {'click': 7050, 'swipe': 1685, 'navigate_back': 620, 'open_app': 1190, 'input_text': 1241, 'wait': 899}

# Define a global variable for the worker processes
worker_model = None

def init_worker(model_args):
    """Initialize each worker with the model based on given arguments"""
    global worker_model
    model_path = model_args['model_path']
    provider = model_args['provider']
    pretrained = model_args['pretrained']

    # Initialize the model based on the provider and model_path
    if provider in ['openai']:
        url = args.base_url or os.environ.get('OPENAI_API_BASE_XIAOAI', 'https://models-proxy.stepfun-inc.com/v1')
        print(f"Requesting the cloud server at {url}")
        
        key = os.environ.get("OPENAI_API_KEY_XIAOAI", "EMPTY")
        worker_model = OpenAIModel(
            base_url=url,
            api_key=key,
            model=pretrained,
            temperature=0.0,
            max_tokens=4096
        )
    elif provider in ['aliyun']:
        url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        print(f"Requesting the cloud server at {url}")
        
        key = os.environ.get("DASHSCOPE_API_KEY", "EMPTY")
        worker_model = Qwen3VL(
            base_url=url,
            api_key=key,
            model=pretrained,
            temperature=0.0,
            max_tokens=4096
        )
    elif provider in ['volcano']:
        url = "https://ark.cn-beijing.volces.com/api/v3"
        print(f"Requesting the cloud server at {url}")
        
        key = os.environ.get("ARK_API_KEY", "EMPTY")
        worker_model = DOUBAO(
            base_url=url,
            api_key=key,
            model=pretrained,
            temperature=0.0,
            max_tokens=4096
        )
    elif 'qwen2' in model_path.lower():
        worker_model = QWen2VL(device='cuda', model_name=model_path)
    # Note: 'slime' models are not supported in multiprocessing mode

def process_sample(step, mode, model_path, ROOT, args, SCALE, MAX_PREV_ACT):
    """Process a single sample with multiprocessing using the global worker_model"""
    global worker_model

    goal = step["task"]
    action_type_ref = step['action_type']

    img_path = os.path.join(ROOT, step["image"])
    if not os.path.exists(img_path):
        img_path = img_path.replace("images/", "")

    image = Image.open(img_path)
    W, H = image.size

    # Prompt making and parsing function
    LONGEST = -1
    if 'gemini' in model_path:
        prompt_making_func = make_gemini_cua_protocol
        parsing_func = extract_gemini_cua_protocol_components
    elif 'claude' in model_path:
        prompt_making_func = make_claude_cua_protocol
        parsing_func = extract_claude_cua_protocol_components
        LONGEST = 1280
    elif 'doubao' in model_path:
        parsing_func = extract_thought_components_doubao
    elif 'qwen3' in model_path:
        parsing_func = extract_thought_components_qwen3vl
    else:
        prompt_making_func = make_planning_protocol
        parsing_func = extract_protocol_components

    # Prepare history string based on model type
    if 'qwen3' in model_path.lower():
        history_str = '; '.join(f"Step {i}: " + action.strip(' .').replace('\n', '').replace('"', '') for i, action in enumerate(step['history'][-MAX_PREV_ACT:], start=1)) if step['step_id'] > 0 else 'None'
    else:
        history_str = ' '.join(f"Step {i}: {action.strip(' .')}." for i, action in enumerate(step['history'][-MAX_PREV_ACT:], start=1)) if step['step_id'] > 0 else 'None'

    # Prepare prompt based on model type
    if model_path in ['Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-2B-Instruct']:
        prompt = ANDROIDCONTROL_PLANNING_PROMPT_COT.format(global_task=goal, history=history_str, step_instruction=f"The next step instruction: {step['step_instruction']}\n" if mode == 'HL' else '')
    elif 'qwen3' in model_path.lower():
        prompt = Qwen3VL_QUERY_TEMPLATE.format(instruction=goal, history=history_str)
    elif 'doubao' in model_path:
        prompt = DOUBAO_QUERY_TEMPLATE.format(instruction=goal, history=history_str)
    elif args.prompt_format_type in ['outcome_pred', 'wo_outcome_pred']:
        prompt = prompt_making_func('AndroidControl', goal, history_str, device_type='smartphone', use_obs=True, use_progress=False, use_intent=True, protocol_type='structured', use_outcome=args.prompt_format_type == 'outcome_pred')
    elif 'direct' in args.prompt_format_type: 
        prompt = make_qwen2p5_planning_prompt(
            'AndroidControl',
            goal, history_str,
            device_type='smartphone',
            use_unnorm_xy='Qwen2.5' in args.pretrained,
            actspace_type='' if 'wo' in args.prompt_format_type else 'qwen2p5', use_guidelines=False)


    step_result = {
        "id": step['id'],
        "img_path": img_path, 
        "task": goal, 
        "step_instruction": step['step_instruction'],
        "prompt": prompt, 
        "response": None, 
        "GT_action": step['task_attr'], 
        "action_pred": None, 
        "metrics": {k: [] for k in ['action_match', 'type_match', 'elem_acc', 'click_match', 'input_text_match', 'swipe_match', 'enter_match', 'status_match', 'navigate_home_match', 'navigate_back_match', 'open_app_match', 'wait_match', 'long_press_match', 'need_gnd']},
        'eval_status': 'single_turn_correct'
    }

    for k in step_result['metrics'].keys():
        if k == 'need_gnd': step_result['metrics'][k].append(action_type_ref in ['click', 'long_press'])
        else: step_result['metrics'][k].append(False)

    temp_file_path = None
    actions_pred = []
    final_action = None
    response = ""
    try:
        s = time.time()

        if LONGEST > 0:
            img, _ = resize_image(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), LONGEST)
            # Generate unique temp file name for each worker
            temp_file_path = f'temp_{os.getpid()}_{uuid.uuid4().hex[:8]}.png'
            cv2.imwrite(temp_file_path, img)
            img_path = temp_file_path

        retry = 0
        while retry < 4:
            retry += 1
            try:
                if isinstance(worker_model, OpenAIModel):
                    _, response, _ = worker_model.get_model_response(prompt, [img_path], use_img_url=True, temperature=0.0, timeout=360)
                elif 'qwen3' in model_path.lower():
                    _, response, _ = worker_model.get_model_response(
                        prompt, [img_path], use_img_url=True, temperature=0.0, sys_prompt=Qwen3VL_SYS_PROMPT.replace('{format_requirements}', Qwen3VL_SYS_PROMPT_FORMAT_REQUIREMENTS_OUTCOME_PRED if args.prompt_format_type == 'outcome_pred' else Qwen3VL_SYS_PROMPT_FORMAT_REQUIREMENTS))
                elif 'seed' in model_path.lower():
                    _, response, _ = worker_model.get_model_response(
                        prompt, [img_path], use_img_url=True, temperature=0.0, sys_prompt=DOUBAO_SYS_PROMPT)
                else:
                    response = worker_model.get_model_response(
                        prompt, 
                        f"file://{img_path}", 
                        max_new_tokens=4096, 
                        sys_prompt=OSATLAS_ANDROIDCONTROL_SYS_PROMPT if 'atlas' in model_path.lower() else OSATLAS_SYS_PROMPT
                    )
                duration = time.time() - s

                step_result["response"] = response

                # Parse the response
                resp_parts = parsing_func(response)
                action_raw = resp_parts['action']
                final_action = ast.literal_eval(action_raw) if isinstance(action_raw, str) else action_raw
                actions_pred = [final_action]
                step_result['thought_parts'] = resp_parts

                break
            except Exception as e:
                print(f"Error parsing response: {e}")
        
        step_result["actions_candidates"] = actions_pred
        step_result["action_pred"] = final_action

        # Matching logic
        for action_pred_idx, action_pred in enumerate(actions_pred):
            if action_pred_idx > 0:
                for k in step_result['metrics'].keys():
                    if k == 'need_gnd': step_result['metrics'][k].append(action_type_ref in ['click', 'long_press'])
                    else: step_result['metrics'][k].append(False)

            if action_pred is None:
                continue

            action_type_pred = action_pred.get('action_type', action_pred.get('action'))

            special_match = False
            # Special handling for navigate_back
            if action_type_ref == 'navigate_back' and action_pred['action_type'] == 'click' and any([k in response for k in ['back arrow', 'back button']]):
                special_match = True

            # Special handling for enter
            if action_type_ref == 'enter' and action_pred['action_type'] == 'press_key' and action_pred['key'].lower() == 'enter':
                special_match = True

            # Special handling for terminate
            if action_type_pred == 'terminate' and action_type_ref == 'status':
                special_match = True
            
            # Special handling for scroll
            if action_type_pred == 'scroll' and action_type_ref == 'swipe':
                special_match = True
                        
            if action_type_pred in ['open']:
                action_type_pred = action_pred['action'] = 'open_app'

            if action_type_pred in ['type', 'input_text']:
                action_type_pred = action_pred['action_type'] = 'input_text'

            if action_type_pred in ['answer']:
                action_type_pred = action_pred['action_type'] = 'status'
                action_pred['goal_status'] = 'successful' if 'complete' in action_pred['text'] else 'infeasible'

            if action_type_pred == 'system_button':
                if action_pred['button'].lower() == 'home':
                    action_type_pred = 'navigate_home'
                elif action_pred['button'].lower() == 'back':
                    action_type_pred = 'navigate_back'


            if action_type_ref == action_type_pred or special_match:
                step_result['metrics']['type_match'][-1] = True

                if action_type_ref in ['click', 'long_press']:
                    step_result['metrics']['need_gnd'][-1] = True

                    target = action_pred.get('target', action_pred.get('coordinate'))
                    if SCALE == -1:
                        target_pred = [target[0] / W, target[1] / H]
                    else:
                        target_pred = list(map(lambda p: p / SCALE, target))

                    gt_box = step['task_attr']['bbox']
                    gt_box_normalized = list(map(lambda p:round(p, 3), [gt_box[0]/W, gt_box[1]/H, gt_box[2]/W, gt_box[3]/ H]))

                    assert all(0 <= p <= 1.0 for p in gt_box_normalized)
                    #step['task_attr']['bbox'] = gt_box_normalized
                    if gt_box_normalized[0] <= target_pred[0] <= gt_box_normalized[2] and gt_box_normalized[1] <= target_pred[1] <= gt_box_normalized[3]:
                        step_result['metrics']['action_match'][-1] = True
                        step_result['metrics']['elem_acc'][-1] = True
                        step_result['metrics'][f'{action_type_ref}_match'][-1] = True

                elif action_type_ref == 'input_text':
                    text_ref, text_pred = step['task_attr']['text'].lower().strip(), action_pred['text'].lower().strip()
                    
                    step_result['metrics']['action_match'][-1] = step_result['metrics']['input_text_match'][-1] = squad_metrics.compute_f1(text_pred, text_ref) > 0.5
                
                elif action_type_ref == 'swipe':
                    direction_ref = step['task_attr']['direction']
                    if 'direction' in action_pred:
                        direction_pred = action_pred['direction']
                    elif 'scroll_direction' in action_pred:
                        direction_pred = action_pred['scroll_direction']
                    elif 'coordinate' in action_pred: # 'action': 'swipe', 'coordinate': [345, 279], 'coordinate2': [86, 290]}
                        direction_pred, distance = get_swipe_direction(action_pred['coordinate'], action_pred['coordinate2'], is_swipe=True)
                    
                    direction_ref = scroll2swipe(direction_ref)
                    if direction_ref == direction_pred:
                        step_result['metrics']['action_match'][-1] = step_result['metrics']['swipe_match'][-1] = True
                
                elif action_type_ref == 'status':
                    status_ref, status_pred = step['task_attr']['goal_status'], action_pred['goal_status']
                    
                    if status_ref == status_pred:
                        step_result['metrics']['action_match'][-1] = step_result['metrics']['status_match'][-1] = True
                elif action_type_ref == 'open_app':
                    app_name_ref, app_name_pred = step['task_attr']['app_name'], action_pred.get('app_name', action_pred.get('text', None))
                    
                    if app_name_ref == app_name_pred:
                        step_result['metrics']['action_match'][-1] = step_result['metrics']['open_app_match'][-1] = True
                else:
                    step_result['metrics']['action_match'][-1] = step_result['metrics'][f'{action_type_ref}_match'][-1] = True

            is_match = step_result['metrics']['action_match'][-1]

        matches = step_result['metrics']['action_match']

        if matches[-1]:
            step_result['eval_status'] = 'single_turn_correct'
        else:
            step_result['eval_status'] = 'single_turn_incorrect'

        
        print((Fore.GREEN if is_match else Fore.RED) + f"{is_match} | {step_result['eval_status']}" + Style.RESET_ALL + f": GT: {step['task_attr']} <=> Pred: {action_pred}")

    except Exception as e:
        print(f"Error processing sample: {e}")
        logging.info("format wrong")
        step_result['wrong_format'] = True
        step_result['wrong_format_reason'] = traceback.format_exc()
    finally:
        # Delete temp file after getting response
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_file_path}: {e}")
    
    # Return both the result and the action type for counting
    return step_result, action_type_ref

def process_samples_with_multiprocessing(samples, model_args, mode, model_path, ROOT, args, SCALE, MAX_PREV_ACT):
    """Process samples using multiprocessing with initialized workers"""
    # Create a process-safe counter for action types and throughput tracking
    manager = Manager()
    counts = manager.dict({'total': 0, 'click': 0, 'input_text': 0, 'swipe': 0, 'long_press': 0, 'enter': 0, 
                          'navigate_home': 0, 'navigate_back': 0, 'status': 0, 'open_app': 0, 'wait': 0})
    
    # Add throughput tracking variables using Value and Lock
    start_time = time.time()
    processed_samples = manager.Value('i', 0)  # 'i' for integer type
    lock = manager.Lock()  # Create a process-safe lock
    
    def update_throughput():
        current_time = time.time()
        elapsed_time = current_time - start_time
        samples_processed = processed_samples.value  # Access the value attribute
        throughput = samples_processed / elapsed_time if elapsed_time > 0 else 0
        print(f"\rThroughput: {throughput:.2f} samples/second | "
              f"Processed: {samples_processed}/{len(samples)} samples | "
              f"Elapsed: {elapsed_time:.1f}s", end='', flush=True)
    
    # Create a partial function without the model argument
    process_func = partial(process_sample, mode=mode, model_path=model_path, 
                          ROOT=ROOT, args=args, SCALE=SCALE, MAX_PREV_ACT=MAX_PREV_ACT)
    
    results = []
    
    # Initialize pool with the model arguments
    with Pool(processes=args.max_workers, initializer=init_worker, initargs=(model_args,)) as pool:
        # Process samples and update counts
        
        # for i, (result, action_type) in enumerate(tqdm(pool.imap(process_func, samples), total=len(samples), desc=f'Processing {mode}')):
        for i, (result, action_type) in enumerate(pool.imap(process_func, samples)):
            results.append(result)
            counts['total'] += 1
            counts[action_type] += 1
            
            # Update processed samples count and throughput
            with lock:  # Use the Lock directly
                processed_samples.value += 1
            update_throughput()

            # Print some results for debugging
            if i % 1 == 0 and result['response']:
                prompt_main = result['prompt'].split("The user's ")[1].split("Your output")[0].strip() if "The user's " in result['prompt'] else result['prompt']
                print(f"\n{Fore.YELLOW}User: <img>{result['img_path']}</img> {prompt_main}\n"
                      f"{Fore.CYAN}GPT: {result['response']}{Style.RESET_ALL}")

    # Print final throughput
    print("\nFinal throughput statistics:")
    update_throughput()
    print()  # New line after final stats
    
    # Convert manager.dict back to regular dict
    counts_dict = dict(counts)
    
    return results, counts_dict

# Main function to run the evaluation
def run_evaluation():
    model_path = args.pretrained.rstrip('/ ')
    print(f"Loading model from {model_path}")
    
    # Define SCALE and MAX_PREV_ACT based on model type
    SCALE = -1  # default
    MAX_PREV_ACT = 999  # default
    
    if args.provider == 'step':
        MAX_PREV_ACT = 6
    elif 'qwen3' in model_path.lower():
        MAX_PREV_ACT = 999
        SCALE = 999
    elif 'claude' in model_path.lower():
        MAX_PREV_ACT = 999
        SCALE = -1
    elif any(x in model_path.lower() for x in ['gemini', 'gpt']):
        SCALE = 1000
        # Note: Slime models are not supported in multiprocessing mode

    # special case
    if any(k in args.pretrained.lower() for k in ['qwen2.5', 'qwen2p5']):
        SCALE = -1

    model_path = model_path.replace("merged/", "").replace("lora/","")
    if "snapshots" in model_path:
        postfix = model_path[model_path.find("models--") + 8: model_path.find("snapshots") - 1]
    else:
        postfix = '/'.join(model_path.split('/')[-2:])

    index=0

    ROOT = ["/mnt/vdb1/hongxin_li"][index]
    REMOVE_RARE_ACTIONS = False

    androidcontrol_test_raw = json.load(open(f'{ROOT}/AndroidControl_test/AndroidControl_test-test_12685.json', 'r'))

    # 筛选出HL样本
    if args.preset_id_file:
        preset_ids = json.load(open(args.preset_id_file))
    else: preset_ids = {}

    h_ids = set(x['id'].split('-H')[0] for x in androidcontrol_test_raw if '-HL' not in x['id'] and not (len(preset_ids) > 0 and f"{x['task']}-{x['step_id']}" not in preset_ids['H']))

    # Prepare model arguments for worker initialization
    model_args = {
        'model_path': model_path,
        'provider': args.provider,
        'pretrained': args.pretrained
    }

    REPEAT = 1

    results_all_repeats = []
    metrics_all_repeats = []
    
    s1 = time.time()

    for repeat in range(REPEAT):
        selected_ids = random.sample(list(h_ids), min(len(h_ids), 500 if not args.debug else 16))
        

        h_samples = [x for x in androidcontrol_test_raw if '-HL' not in x['id'] and x['id'].split('-H')[0] in selected_ids]

        metrics_this_repeat = {'H': []}
        results = {'H': []}

        for mode, samples in zip(['H'], [h_samples]):
            # Process samples using multiprocessing with model_args

            # Debug actions
            # samples = [x for x in samples if x['action_type'] in [ 'open_app']]

            results[mode], counts = process_samples_with_multiprocessing(
                samples, model_args, mode, model_path, ROOT, args, SCALE, MAX_PREV_ACT
            )

            # Calculate metrics
            num_sample = counts['total']
            num_need_gnd = sum(x['metrics']['need_gnd'][-1] for x in results[mode])
            
            num_action_match = sum(x['metrics']['action_match'][-1] for x in results[mode])
            num_type_match = sum(x['metrics']['type_match'][-1] for x in results[mode])
            num_elem_match = sum(x['metrics']['elem_acc'][-1] for x in results[mode])
            
            final_metrics = {
                'step_acc': [num_action_match / num_sample if num_sample > 0 else 0, num_action_match, num_sample], 
                'action_type_acc': [num_type_match / num_sample if num_sample > 0 else 0, num_type_match, num_sample], 
                'elem_acc': [num_elem_match / num_need_gnd if num_need_gnd > 0 else 0, num_elem_match, num_need_gnd],
                'eval_status_ratios': {
                    'correct': {
                        'single_turn_correct': sum(x['eval_status'] == 'single_turn_correct' for x in results[mode]) / num_sample
                    },
                    'incorrect': {
                        'single_turn_incorrect': sum(x['eval_status'] == 'single_turn_incorrect' for x in results[mode]) / num_sample
                    }
                }
            }
            
            for k in counts.keys():
                if k=='total': continue
                
                cnt = counts[k]
                acc_cnt = sum(x['metrics'][f'{k}_match'][-1] for x in results[mode])
                
                final_metrics[f'{k}_acc'] = [round(acc_cnt / cnt, 4) if cnt > 0 else 0, acc_cnt, cnt]
            
            final_metrics['num_wrong_format'] = sum(1 for x in results[mode] if 'wrong_format' in x)
            
            pprint(final_metrics)
            
            metrics_this_repeat[mode] = final_metrics
        
        results_all_repeats.append(results)
        metrics_all_repeats.append(metrics_this_repeat)

    # aggr
    aggr_metrics = {'H': {}}

    for mode in aggr_metrics.keys():
        for repeat_result in metrics_all_repeats:
            for metric_name, info in repeat_result[mode].items():
                if metric_name in ['num_wrong_format', 'eval_status_ratios']: continue
                if metric_name not in aggr_metrics[mode]: aggr_metrics[mode][metric_name] = [0,0,0]
                aggr_metrics[mode][metric_name][1] += info[1]
                aggr_metrics[mode][metric_name][2] += info[2]

        for metric_name in aggr_metrics[mode].keys():
            if metric_name in ['num_wrong_format', 'eval_status_ratios']: continue
            acc_cnt, cnt = aggr_metrics[mode][metric_name][1], aggr_metrics[mode][metric_name][2]
            aggr_metrics[mode][metric_name][0] = acc_cnt / cnt if cnt > 0 else 0

    print("\nFinal:")
    pprint(aggr_metrics)

    eval_result_dir = os.path.join(os.path.dirname(__file__), 'eval_results/androidcontrol')
    os.makedirs(eval_result_dir, exist_ok=True)

    save_to = os.path.join(eval_result_dir, postfix)

    os.makedirs(save_to, exist_ok=True)
    save_file = os.path.join(save_to, datetime.now().strftime("%m-%d-%H-%M-%S")) + '.json'
    with open(save_file, "w") as f:
        meta = vars(args)
        meta['max_prev_actions'] = MAX_PREV_ACT
        meta['time_elapse'] = time.time() - s1
        json.dump(
            {
                "meta": meta,
                "overall_results": aggr_metrics,
                "metrics_each_repeat": metrics_all_repeats,
                "logs": results_all_repeats,
            },
            f,
            indent=2
        )

    print(f"Finised evaluation {args.pretrained} on AndroidControl. Save to {save_file}")

if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn')
    run_evaluation()