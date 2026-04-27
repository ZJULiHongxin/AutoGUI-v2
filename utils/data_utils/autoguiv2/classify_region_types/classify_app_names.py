"""
Classify app names from test images using VLM

This script loads test images from HuggingFace datasets (using cached images)
and prompts a VLM to identify the main software/application displayed in each image.
"""

import os
import json
import time
import argparse
import multiprocessing
import hashlib
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, Manager
from PIL import Image
from tqdm import tqdm

# Optional color output (safe fallback)
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

# Note: We use cached images from eval_elemgnd_mp.py, so we don't need to import datasets

import sys
sys.path.append('/'.join(__file__.split('/')[:-3]))
from utils.openai_utils.openai import OpenAIModel

from utils.data_utils.misc import resize_pil_image

# ---------------------------
# Pretty printing helpers
# ---------------------------
def debug_print(message: str, level: str = "info") -> None:
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
    return f"{Fore.GREEN}ON{Style.RESET_ALL}" if value else f"{Fore.YELLOW}OFF{Style.RESET_ALL}"


# ---------------------------
# Image helpers
# ---------------------------
def image_to_base64(image_or_path):
    """Convert an image to a base64 data URL.
    
    Accepts:
      - str: filesystem path to the image
      - PIL.Image.Image: PIL image instance
    """
    import base64
    from io import BytesIO
    
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
    }

    if isinstance(image_or_path, str):
        ext = os.path.splitext(image_or_path)[1].lower()
        with open(image_or_path, "rb") as f:
            binary_data = f.read()
        base64_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_types.get(ext, 'image/png')};base64,{base64_data}"

    if isinstance(image_or_path, Image.Image):
        output = BytesIO()
        fmt = image_or_path.format if image_or_path.format else 'PNG'
        image_or_path.save(output, format=fmt)
        binary_data = output.getvalue()
        mime = f"image/{fmt.lower()}" if fmt else 'image/png'
        base64_data = base64.b64encode(binary_data).decode('utf-8')
        return f"data:{mime};base64,{base64_data}"

    raise TypeError("image_to_base64 expects a file path (str) or PIL Image")


# ---------------------------
# Dataset loading (using cache from eval_elemgnd_mp.py)
# ---------------------------
def get_cache_images(hf_dataset_id: str) -> List[str]:
    """Get all cached images from eval_elemgnd_mp.py cache directories
    
    Scans all dataset cache directories for images.
    
    Returns:
        List of image file paths
    """
    # Get eval_elemgnd_mp.py script directory
    # Current file: utils/data_utils/autoguiv2/classify_region_types/classify_app_names.py
    # Target: utils/eval_utils/autoguiv2/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels: classify_region_types -> autoguiv2 -> data_utils -> utils
    utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    # Then go to eval_utils/autoguiv2
    script_dir = os.path.join(utils_dir, 'eval_utils', 'autoguiv2')
    
    # Find all dataset cache directories
    cache_pattern = os.path.join(script_dir, '*dataset_cache')
    cache_dirs = glob.glob(cache_pattern)
    
    all_images = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    
    for cache_dir in cache_dirs:
        if 'osworld' not in cache_dir: continue
        images_dir = os.path.join(cache_dir, 'images')
        if os.path.exists(images_dir):
            # Recursively find all images
            for ext in image_extensions:
                pattern = os.path.join(images_dir, '**', f'*{ext}')
                images = glob.glob(pattern, recursive=True)
                all_images.extend(images)
    
    # Deduplicate
    dedup_images = []
    visited = []
    
    for img in all_images:
        base = os.path.basename(img)
        
        parts = base.split('_')
        
        maybe_unique_id = parts[-1].split('.')[0]
        if 0 < len(maybe_unique_id) <= 3:
            base = base.replace(f"_{maybe_unique_id}.png", ".png")

        if base in visited: continue
        visited.append(base)
        dedup_images.append(img)
    
    return sorted(dedup_images)


def get_hf_dataset_cache_paths(
    hf_dataset_id: str,
    hf_split: str,
    task_type: str,
    cache_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Get cache file and image directory for HuggingFace dataset entries"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    default_cache_root = os.path.join(
        utils_dir, 'eval_utils', 'autoguiv2', 'elemgnd_hf_dataset_cache'
    )
    cache_root = cache_dir if cache_dir else default_cache_root

    cache_key = f"{hf_dataset_id}_{hf_split}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_file = os.path.join(cache_root, f"{cache_hash}_{task_type}.json")
    image_cache_dir = os.path.join(cache_root, 'images', cache_hash)
    return cache_file, image_cache_dir


def load_hf_cache_entries(
    hf_dataset_id: str,
    hf_split: str,
    task_type: str,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Load cached HuggingFace entries pre-generated by eval_elemgnd_mp."""
    cache_file, image_cache_dir = get_hf_dataset_cache_paths(
        hf_dataset_id, hf_split, task_type, cache_dir
    )

    if not os.path.exists(cache_file):
        debug_print(f"⚠️  HF cache file not found: {cache_file}", level="warn")
        return [], image_cache_dir

    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('entries', [])
        debug_print(
            f"📂 Loaded {len(entries)} cached HF entries from {cache_file}", level="success"
        )
        debug_print(f"📁 Image cache path: {image_cache_dir}", level="info")
        return entries, image_cache_dir
    except Exception as e:
        debug_print(f"⚠️  Failed to load HF cache: {e}", level="warn")
        return [], image_cache_dir


def build_image_entries_from_dataset(
    dataset_entries: List[Dict[str, Any]],
    default_task_type: str,
    image_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build unique image entries from HuggingFace dataset samples."""
    image_groups: Dict[str, Dict[str, Any]] = {}
    valid_samples = 0
    skipped_samples = 0

    for sample in dataset_entries:
        image_path = sample.get('image_path')
        if not image_path or not os.path.exists(image_path):
            skipped_samples += 1
            continue

        valid_samples += 1
        group = image_groups.setdefault(
            image_path,
            {
                'image_name': sample.get('image_name') or os.path.basename(image_path),
                'dataset_name': sample.get('dataset_name', 'unknown'),
                'linked_entries': [],
            },
        )

        sample_record = {
            'entry_id': sample.get('entry_id'),
            'question': sample.get('question', ''),
            'action_type': sample.get('action_type', ''),
            'group_index': sample.get('group_index'),
            'target_elem_id': sample.get('target_elem_id'),
            'density_class': sample.get('density_class'),
            'num_similar_elements': sample.get('num_similar_elements'),
            'dataset_name': sample.get('dataset_name', group['dataset_name']),
            'task_type': sample.get('task_type', default_task_type),
        }
        group['linked_entries'].append(sample_record)

    entries = []
    for image_path, group in image_groups.items():
        entries.append({
            'entry_id': image_path,
            'image_path': image_path,
            'image_name': group['image_name'],
            'dataset_name': group['dataset_name'],
            'linked_entries': group['linked_entries'],
        })

        if image_limit and len(entries) >= image_limit:
            break

    return entries, valid_samples, skipped_samples


def snapshot_processed_ids(processed_ids) -> set:
    if isinstance(processed_ids, set):
        return set(processed_ids)
    if hasattr(processed_ids, 'items'):
        return {k for k, v in processed_ids.items() if v}
    return set(processed_ids)


def build_sample_results_from_image_results(
    image_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Flatten image-level results into per-dataset-sample records."""
    sample_results: Dict[str, Dict[str, Any]] = {}
    for image_id, data in image_results.items():
        linked_entries = data.get('linked_entries', [])
        if not isinstance(linked_entries, list):
            continue

        for idx, sample in enumerate(linked_entries):
            base_id = sample.get('entry_id') or f"{image_id}_{idx}"
            sample_id = base_id
            if sample_id in sample_results:
                sample_id = f"{base_id}_{idx}"

            sample_results[sample_id] = {
                'entry_id': sample_id,
                'image_entry_id': image_id,
                'image_path': data.get('image_path', ''),
                'image_name': data.get('image_name', ''),
                'dataset_name': sample.get('dataset_name') or data.get('dataset_name', 'unknown'),
                'question': sample.get('question', ''),
                'action_type': sample.get('action_type', ''),
                'group_index': sample.get('group_index'),
                'target_elem_id': sample.get('target_elem_id'),
                'density_class': sample.get('density_class'),
                'num_similar_elements': sample.get('num_similar_elements'),
                'task_type': sample.get('task_type'),
                'app_name': data.get('app_name', 'Unknown'),
                'app_category': data.get('app_category', 'Unknown'),
                'notes': data.get('notes', ''),
                'raw_response': data.get('raw_response', ''),
                'timestamp': data.get('timestamp'),
                'processing_time': data.get('processing_time'),
                'image_size': data.get('image_size', []),
            }

    return sample_results




# ---------------------------
# App name classification prompt
# ---------------------------
APP_NAME_PROMPT = """You are an expert at identifying the software applications and programs involved in GUI understanding tasks.

Given a screenshot and a task description, identify the MAIN software applications or programs that are primarily involved in the task.

Guidelines:
- Provide the official or commonly used names of the involved software (e.g., "Visual Studio Code", "Chrome", "Photoshop", "Excel")
- If one involved app is a web browser, specify both the browser name AND the website if it's a specific web app (e.g., "Chrome - Gmail", "Firefox - YouTube")
- For system-level interfaces, use descriptive names (e.g., "Windows File Explorer", "macOS Finder")
- Be specific: prefer "Microsoft Word" over just "Word", "Adobe Photoshop" over "Photoshop" when possible

Reference app categories and names:
Office: Apple Notes, Apple Reminders, Calendar, Docs, Document Viewer, Evince, Gedit, Google Calendar, Google Docs, Google Keep, Keynote, Lark, Libreoffice, Notability, Notetaking App, Notepad, Notes, Notion, Numbers, Office, Overleaf, Pages, Powerpoint, Spreadsheet, Text Editor, VS Code, WPS Office, Microsoft Word, Xcode, Freeform.
Media: Amazon Music, Amazon Prime Video, Iheartradio, Likee, Music, Music Player, Pandora, Pocket FM, Podcast Player, Quicktime, Roku, Sofascore, Spotify, TikTok, Tubi, VLC media player, YouTube, YouTube Music.
Game: Arena_of_valor, CS2, Chess, Defense_of_the_ancients_2, Dream, Genshin_impact, Minecraft, Nintendo, Pubg, Red_dead_redemption_2, Steam, The Legend Of Zelda Breath Of The Wild.
Editing: 3dviewer, Adobe Acrobat, Adobe After Effects, Adobe Express, Adobe Photoshop, Adobe Photoshop Express, Adobe Premiere Pro, CapCut, Davinci Resolve, Draw.io, Gimp, Paint, PDF Editor, Photo Editing Tool, Photo Editor, Picsart, Procreate, Runway, Snapseed, Video Editing Software.
Social & Communication: Discord, Facebook, Flickr, Gmail, Google Meet, Google Messages, Imessage, Instagram, LinkedIn, Mail, Messenger, Outlook, Phone, Pinterest, Quora, Reddit, Signal, Slack, Teams Live, Telegram, Threads, Thunderbird, Tumblr, WeChat, Weibo, WhatsApp, X (Twitter), Zoom.
Shopping: 12306, Alibaba, Aliexpress, Amazon Shopping, Apartments.com, Applestore, Autoscout24, Autouncle, Booking.com, Car Marketplace, Cars.co.za, Ebay, Edmunds, Expedia, Magento, Offerup, Onestopmarket, Product Listing App, Realtor.com, Redfin, Shop, Taobao, Tripadvisor, Walmart, Wish.
AI & Tools: AI Art Generator, Align-anything-dev-omni, Amazon Alexa, Chatbot AI, Chatgpt, Chaton AI, DeepL Translate, Google Translate, Grammarly, Microsoft Copilot, Microsoft Translator, Remix AI Image Creator, Stable Diffusion, Translate, WOMBO Dream, Yandex Translate, Zhiyun Translate.
Browser & Search: Bing, DuckDuckGo, Firefox, Google App, Google Chrome, Google Search, Opera, Safari, Web Browser, Web.
Tools: Accerciser, Activities, Activity Monitor, App Lock, App Locker, Applock Pro, Automator, Baidu Netdisk, Bluetoothnotificationareaiconwindowclass, Calculator, Camera, Clean, ClevCalc - Calculator, Color Management Utility, Colorsync_utility, Contacts, Control Center, Cursor, Desktop, Dictionary, Digital Color Meter, Disk Utility, Drops, Electron, Email Client, File, File Explorer, File Manager, Files, Filezilla, Finder, Font Book, GPS, Image Viewer, Iphonelockscreen, Kid3, Launcher, Mi Mover, Microsoft Store, Preview, Recorder, Rosetta Stone, Scientific Calculator Plus 991, Script_editor, Search, Shortcuts, Spotlight, Stickies, System Information, System Search, System Settings, Task Manager, Terminal, Totem, ToDesk, Trash, Vim, Voicememos, Vottak, Wallpaper Picker.
Productivity: Any.do, Drive, Dropbox Paper, Google Drive, Onedrive, Paperflux, Things, TickTick, Todoist.
News & Reading: AP News, BBC News, BBC Sport, Bloomberg, Crimereads, Espn, Forbes, Goodreads, Google News, Google Play Books, Google Scholar, Kindle, Kobo Books, Metacritic, Microsoft News, Newsbreak, Wikidata, Wikipedia, Yahoo Sports, Apple News, Travel Guide App, Travel Review App.
Weather & Navigation: Accuweather, Apple Maps, Citymapper, Google Maps, Mapillary, Miuiweather, Msnweather, Navigation App, Openstreetmap, Waze, Weather, Windy.
Finance: Alipay, Budgeting App, Investing.com, Paymore, Stocks, Wallet For Your Business, Wallet: Budget Money Manager, Yahoo Finance.
Health & Fitness: Fitbit, Fiton, Mideaair, Mifitness.
Job Search: Indeed, Job Search By Ziprecruiter, Ziprecruiter.
Transportation: Didi, Ryanair, Uber.
System & Tools: System Status Bar, Application Launcher, Application Window, Android Home Screen, Android Launcher, Android Settings, Android Share Sheet, App Store, Apple, Applibrary, Gnome, Mobile Home Launcher, Mobile Launcher, Mobile Web Browser, OS, Ubuntu, Ubuntu Desktop.
Ohters: Other software names you believe are most suitable for the application.

Output your response in JSON format:
[
    {
        "app_category": "Application Category",
        "app_name": "Application Name"
    },
    ...
]

The GUI screenshot has been given to you and the task is: {task}

Now provide your answer:
"""


def parse_app_name_response(raw: str) -> Dict[str, Any]:
    """Extract app name from model response"""
    import re
    
    # Try to find JSON in response
    fenced = re.findall(r"```json\s*([\s\S]*?)```", raw, re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.extend(fenced)
    
    # Fallback: first JSON object
    if not candidates:
        obj_match = re.search(r"\{[\s\S]*?\}", raw)
        if obj_match:
            candidates.append(obj_match.group(0))
    
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                app_name = str(data.get('app_name', '')).strip()
                app_category = str(data.get('app_category', '')).strip()
                if app_name:
                    return {
                        'app_name': app_name,
                        'app_category': app_category if app_category else 'Unknown',
                        'notes': '',
                        'raw_response': raw
                    }
        except Exception:
            continue
    
    # Fallback: try to extract app name from text
    lines = raw.strip().splitlines()
    for line in lines:
        line = line.strip()
        if 'app_name' in line.lower() or 'application' in line.lower():
            # Try to extract quoted string
            match = re.search(r'["\']([^"\']+)["\']', line)
            if match:
                return {
                    'app_name': match.group(1),
                    'app_category': 'Unknown',
                    'notes': 'Extracted from text',
                    'raw_response': raw
                }
    
    # Final fallback
    first_line = lines[0] if lines else raw[:100]
    return {
        'app_name': 'Unknown',
        'app_category': 'Unknown',
        'notes': first_line,
        'raw_response': raw
    }


# ---------------------------
# Classifier
# ---------------------------
class AppNameClassifier:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = OpenAIModel(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=1024,
        )

    def classify(self, image_path: str, retries: int = 3) -> Dict[str, Any]:
        """Classify app name from image"""
        
        # resize the image to max_size = 1920
        image = Image.open(image_path)
        
        W, H = image.size
        
        if max(W, H) > 1600:
            image, _ = resize_pil_image(image, max_size=1600)
        
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': image_to_base64(image)}},
                {'type': 'text', 'text': APP_NAME_PROMPT},
            ]
        }]

        sleep_time, time_out = 1, 120
        for i in range(retries):
            sleep_time = int(1.5 * sleep_time)
            time_out *= 2
            try:
                temperature = 0.0 if i == 0 else 0.4
                success, raw_resp, _ = self.model.get_model_response_with_prepared_messages(
                    messages, temperature=temperature, timeout=time_out
                )
                if not success:
                    continue

                if '</think>' in raw_resp:
                    resp = raw_resp.split('</think>')[-1]
                else:
                    resp = raw_resp

                parsed = parse_app_name_response(resp)

                parsed['image_size'] = [W, H]
                if parsed['app_name'] != 'Unknown' or i == retries - 1:
                    return parsed
            except Exception as e:
                if i == retries - 1:
                    return {
                        'app_name': 'Unknown',
                        'app_category': 'Unknown',
                        'image_size': [W, H],
                        'notes': f'Error: {str(e)}',
                        'raw_response': str(e)
                    }
                time.sleep(sleep_time)
                continue

        return {
            'app_name': 'Unknown',
            'app_category': 'Unknown',
            'image_size': [W, H],
            'notes': 'Failed after retries',
            'raw_response': ''
        }


# ---------------------------
# Multiprocessing glue
# ---------------------------
classifier_instance: AppNameClassifier = None  # type: ignore


def init_worker(base_url: str, api_key: str, model: str):
    global classifier_instance
    classifier_instance = AppNameClassifier(base_url, api_key, model)


def process_image(args) -> Dict[str, Any]:
    """Process a single image to classify app name"""
    entry, worker_id = args
    t0 = time.time()

    try:
        image_path = entry['image_path']
        if not os.path.exists(image_path):
            return {
                'entry_id': entry['entry_id'],
                'image_path': image_path,
                'error': 'Image file not found',
                'processing_time': time.time() - t0,
            }

        result = classifier_instance.classify(image_path, retries=3)

        output = {
            'entry_id': entry['entry_id'],
            'image_path': image_path,
            'image_name': entry.get('image_name', ''),
            'dataset_name': entry.get('dataset_name') or get_dataset_name(image_path),
            'app_name': result['app_name'],
            'app_category': result.get('app_category', 'Unknown'),
            'notes': result.get('notes', ''),
            'raw_response': result.get('raw_response', ''),
            'linked_entries': entry.get('linked_entries', []),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': time.time() - t0,
        }
        
        image_name_short = os.path.basename(entry.get('image_name', 'unknown'))
        debug_print(f"[Worker {worker_id}] ✅ {image_name_short} -> {result['app_name']}", level="info")
        
        return output
    except Exception as e:
        import traceback
        return {
            'entry_id': entry.get('entry_id', 'unknown'),
            'image_path': entry.get('image_path', ''),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'processing_time': time.time() - t0,
        }


# ---------------------------
# I/O helpers
# ---------------------------
def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load classification checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint JSON file (can be checkpoint or full result file)
    
    Returns:
        Dictionary with processed entry IDs and results
        Note: Only successfully completed entries (has valid app_name and no error) are included in processed_ids.
        Failed entries will be retried on resume.
    """
    if not os.path.exists(checkpoint_file):
        return {'processed_ids': set(), 'results': {}}

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Normalize results to dict format for consistency
        results = data.get('results', {})
        if isinstance(results, list):
            # Convert list format to dict
            results = {r['entry_id']: r for r in results if 'entry_id' in r}
        elif not isinstance(results, dict):
            results = {}
        
        # Check if processed_ids exists in the checkpoint
        if 'processed_ids' in data:
            # Use existing processed_ids
            processed_ids = set(data.get('processed_ids', []))
            inferred = False
        else:
            # Backward compatibility: infer processed_ids from results
            # All entries in results are considered "processed" (attempted)
            processed_ids = set(results.keys())
            inferred = True
            if inferred:
                debug_print(f"📝 processed_ids not found in checkpoint, inferring from {len(processed_ids)} results", level="info")
        
        # Filter out failed entries from processed_ids so they can be retried
        # Only entries with valid app_name (not 'Unknown') and no error should be considered successfully processed
        successful_ids = set()
        failed_ids = set()
        
        for entry_id in processed_ids:
            result = results.get(entry_id)
            
            if result:
                # Check if classification was successful
                app_name = result.get('app_name', '')
                has_error = 'error' in result and result.get('error')
                # Successful if: has app_name, app_name is not 'Unknown', and no error
                if app_name and app_name != 'Unknown' and not has_error:
                    successful_ids.add(entry_id)
                else:
                    failed_ids.add(entry_id)
            else:
                # If we can't find the result, assume it needs to be retried
                failed_ids.add(entry_id)

        # Update processed_ids to only include successful entries
        processed_ids = successful_ids

        if failed_ids:
            debug_print(f"⚠️  Found {len(failed_ids)} failed entries that will be retried", level="warn")
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} successful entries, {len(failed_ids)} failed entries to retry", level="success")
        else:
            debug_print(f"✅ Loaded checkpoint: {len(successful_ids)} processed entries", level="success")

        return {'processed_ids': processed_ids, 'results': results}
    except Exception as e:
        debug_print(f"⚠️  Error loading checkpoint: {e}", level="warn")
        return {'processed_ids': set(), 'results': {}}


def save_checkpoint(
    results: Dict[str, Any],
    processed_ids: set,
    checkpoint_file: str,
    metadata: Dict[str, Any] = None,
    sample_results: Optional[Dict[str, Any]] = None,
):
    """Save classification checkpoint"""
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'metadata': metadata or {},
        'processed_ids': list(processed_ids),
        'results': results,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if sample_results is not None:
        checkpoint['sample_results'] = sample_results
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def find_latest_checkpoint(result_dir: str, model_name: str) -> Optional[str]:
    """Find the latest checkpoint file for a model
    
    Args:
        result_dir: Directory containing classification results
        model_name: Model identifier
    
    Returns:
        Path to latest checkpoint or None
    """
    # Clean model name for filesystem
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    pattern = os.path.join(result_dir, f"*{safe_model_name}*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def get_dataset_name(path: str) -> str:
    if '/elem' in path:
        return 'autogui-v2'
    elif 'osworld' in path:
        return 'osworldg'
    elif 'screenspot' in path:
        return 'screenspot'
    elif 'agentnet' in path:
        return 'agentnet'
    elif 'amex' in path:
        return 'amex'
    else:
        return 'unknown'
# ---------------------------
# Main
# ---------------------------
def main(args):
    debug_print("════════════════════════════════════════════════════════════", level="title")
    debug_print("📱 App Name Classification - Run Configuration", level="title")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    debug_print("", level="info")
    debug_print("📁 DATA & OUTPUT CONFIGURATION", level="step")
    debug_print(f"   HuggingFace Dataset: {Fore.CYAN}{args.hf_dataset_id}{Style.RESET_ALL}", level="info")
    debug_print(f"   Split: {Fore.CYAN}{args.hf_split}{Style.RESET_ALL}", level="info")
    debug_print(f"   Using Cache: {Fore.GREEN}eval_elemgnd_mp.py cache{Style.RESET_ALL}", level="info")
    debug_print(f"   Output File: {Fore.CYAN}{args.output_file}{Style.RESET_ALL}", level="info")
    
    debug_print("", level="info")
    debug_print("💾 CHECKPOINT CONFIGURATION", level="step")
    if args.checkpoint_file:
        debug_print(f"   Checkpoint File: {Fore.CYAN}{args.checkpoint_file}{Style.RESET_ALL}", level="info")
    elif args.load_latest:
        debug_print(f"   Load Latest: {Fore.GREEN}YES{Style.RESET_ALL}", level="info")
    else:
        debug_print(f"   Checkpoint: {Fore.YELLOW}Auto-detect from output file{Style.RESET_ALL}", level="info")

    debug_print("", level="info")
    debug_print("🤖 MODEL CONFIGURATION", level="step")
    debug_print(f"   Model: {Fore.GREEN}{args.model}{Style.RESET_ALL}", level="info")
    debug_print(f"   API Base URL: {Fore.BLUE}{args.base_url or 'Default'}{Style.RESET_ALL}", level="info")

    debug_print("", level="info")
    debug_print("⚙️  PROCESSING CONFIGURATION", level="step")
    mode_text = "SEQUENTIAL" if args.sequential else f"PARALLEL ({args.workers} workers)"
    mode_color = Fore.RED if args.sequential else Fore.GREEN
    debug_print(f"   Execution Mode: {mode_color}{mode_text}{Style.RESET_ALL}", level="info")
    debug_print(f"   Task Timeout: {Fore.YELLOW}{args.task_timeout}s{Style.RESET_ALL}", level="info")
    if args.sample_limit and args.sample_limit > 0:
        debug_print(f"   Sample Limit: {Fore.YELLOW}{args.sample_limit}{Style.RESET_ALL} (debugging mode)", level="info")
    debug_print(f"   Debug: {on_off(args.debug)}", level="info")
    debug_print("════════════════════════════════════════════════════════════", level="title")

    # Load unique images from eval_elemgnd_mp.py cache
    debug_print(f"\n📂 Loading cached images from eval_elemgnd_mp.py cache...", level="step")
    image_paths = get_cache_images(args.hf_dataset_id)
    
    if not image_paths:
        debug_print("❌ No cached images found. Please run eval_elemgnd_mp.py first to create the cache.", level="error")
        return
    
    debug_print(f"✅ Found {len(image_paths)} cached images", level="success")
    
    # Apply sample limit if specified
    if args.sample_limit and args.sample_limit > 0:
        image_paths = image_paths[:args.sample_limit]
        debug_print(f"🔬 Sample limit applied: processing {len(image_paths)} images (limit: {args.sample_limit})", level="info")
    
    # Convert image paths to entry dictionaries
    entries = []
    for idx, image_path in enumerate(image_paths):
        image_name = '/'.join(image_path.split('images/')[-1].split('/')[1:])
        entry_id = image_path.split('autoguiv2/')[-1]
        entries.append({
            'entry_id': entry_id,
            'image_path': image_path,
            'image_name': image_name,
            'dataset_name': 'unknown',
        })
    
    debug_print(f"✅ Prepared {len(entries)} entries for classification", level="success")
    
    # Setup checkpoint file path
    checkpoint_file = args.checkpoint_file
    if checkpoint_file is None:
        if args.load_latest:
            # Find latest checkpoint
            script_dir = os.path.dirname(os.path.abspath(__file__))
            latest = find_latest_checkpoint(script_dir, args.model)
            if latest:
                checkpoint_file = latest
                debug_print(f"📂 Found latest checkpoint: {latest}", level="info")
            else:
                checkpoint_file = args.output_file
                debug_print(f"📂 No existing checkpoint found, using output file: {checkpoint_file}", level="info")
        else:
            checkpoint_file = args.output_file
    
    # Load checkpoint
    checkpoint = {'processed_ids': set(), 'results': {}}
    if os.path.exists(checkpoint_file) and not args.force:
        checkpoint = load_checkpoint(checkpoint_file)
    
    existing_results = checkpoint.get('results', {})
    processed_ids = checkpoint.get('processed_ids', set())
    
    if existing_results:
        debug_print(f"📋 Found {len(existing_results)} existing classifications ({len(processed_ids)} successful)", level="info")
    
    # Filter out already successfully processed entries (but keep failed ones for retry)
    if not args.force:
        entries = [e for e in entries if e['entry_id'] not in processed_ids]
    
    if not entries:
        debug_print("✅ All images already successfully classified. Use --force to recompute.", level="success")
        return
    
    debug_print(f"📋 Processing {len(entries)} images", level="info")
    
    # Prepare results
    if args.sequential:
        results: Dict[str, Any] = dict(existing_results)
        processed_ids_set = set(processed_ids)
    else:
        manager = Manager()
        results = manager.dict({k: v for k, v in existing_results.items()})  # type: ignore
        processed_ids_set = manager.dict({pid: True for pid in processed_ids})  # type: ignore
    
    start_time = time.time()
    
    # Process images
    if args.sequential:
        global classifier_instance
        classifier_instance = AppNameClassifier(args.base_url, args.api_key, args.model)
        
        with tqdm(total=len(entries), desc=f"Classifying {len(entries)} images | Model: {args.model}", dynamic_ncols=True) as pbar:
            for i, entry in enumerate(entries):
                try:
                    output = process_image((entry, 0))
                    entry_id = entry['entry_id']
                    results[entry_id] = output
                    
                    # Mark as processed if successful (has valid app_name and no error)
                    app_name = output.get('app_name', '')
                    has_error = 'error' in output and output.get('error')
                    if app_name and app_name != 'Unknown' and not has_error:
                        processed_ids_set.add(entry_id)
                    
                    pbar.update(1)
                    
                    # Periodic checkpoint
                    if (i + 1) % 10 == 0:
                        meta = {
                            "model": args.model,
                            "hf_dataset_id": args.hf_dataset_id,
                            "hf_split": args.hf_split,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "num_images_processed": len(processed_ids_set),
                            "total_images": len(existing_results) + len(entries),
                            "processing_time_so_far": time.time() - start_time,
                        }
                        save_checkpoint(results, processed_ids_set, checkpoint_file, meta)
                        pbar.set_postfix_str(f"checkpoint @ {len(processed_ids_set)}")
                except Exception as e:
                    print(f"Error processing {entry.get('image_name', 'unknown')}: {e}")
                    pbar.update(1)
    else:
        with Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.base_url, args.api_key, args.model)
        ) as pool:
            try:
                # Submit tasks
                tasks = [(entry, i % args.workers) for i, entry in enumerate(entries)]
                async_results = [pool.apply_async(process_image, args=(task,)) for task in tasks]

                # Collect with timeout per task and progress bar
                completed_count = 0
                with tqdm(total=len(async_results), desc=f"Classifying {len(entries)} images (parallel)", dynamic_ncols=True) as pbar:
                    for async_res in async_results:
                        try:
                            output = async_res.get(timeout=args.task_timeout)
                            entry_id = output['entry_id']
                            results[entry_id] = output  # type: ignore
                            
                            # Mark as processed if successful (has valid app_name and no error)
                            app_name = output.get('app_name', '')
                            has_error = 'error' in output and output.get('error')
                            if app_name and app_name != 'Unknown' and not has_error:
                                processed_ids_set[entry_id] = True  # type: ignore
                            
                            completed_count += 1
                            pbar.update(1)

                            if completed_count % 10 == 0:
                                # Convert manager.dict to regular dict for processed_ids
                                successful_ids = {k for k, v in processed_ids_set.items() if v}  # type: ignore
                                meta = {
                                    "model": args.model,
                                    "hf_dataset_id": args.hf_dataset_id,
                                    "hf_split": args.hf_split,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "num_images_processed": len(successful_ids),
                                    "total_images": len(existing_results) + len(entries),
                                    "processing_time_so_far": time.time() - start_time,
                                }
                                # Convert manager.dict to regular dict for saving
                                results_dict = dict(results)  # type: ignore
                                save_checkpoint(results_dict, successful_ids, checkpoint_file, meta)
                                pbar.set_postfix_str(f"checkpoint @ {len(successful_ids)}")
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"Task failed: {e}")
                            pbar.update(1)
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Terminating workers...")
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                print(f"Parallel processing error: {e}")
                pool.terminate()
                pool.join()

    # Final save
    if args.sequential:
        successful_ids = processed_ids_set
        results_dict = results
    else:
        # Convert manager objects to regular Python objects
        successful_ids = {k for k, v in processed_ids_set.items() if v}  # type: ignore
        results_dict = dict(results)  # type: ignore
    
    meta = {
        "model": args.model,
        "hf_dataset_id": args.hf_dataset_id,
        "hf_split": args.hf_split,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_images_processed": len(successful_ids),
        "total_images": len(existing_results) + len(entries),
        "total_processing_time_wall": time.time() - start_time,
    }
    save_checkpoint(results_dict, successful_ids, checkpoint_file, meta)
    
    # Also save to output_file if different from checkpoint_file
    if checkpoint_file != args.output_file:
        save_checkpoint(results_dict, successful_ids, args.output_file, meta)
    
    debug_print("\n✅ Classification complete.", level="success")
    debug_print(f"💾 Results saved to: {checkpoint_file}", level="info")
    if checkpoint_file != args.output_file:
        debug_print(f"💾 Also saved to output file: {args.output_file}", level="info")
    
    # Print summary statistics
    app_name_counts = {}
    for result in results_dict.values():
        app_name = result.get('app_name', 'Unknown')
        app_name_counts[app_name] = app_name_counts.get(app_name, 0) + 1
    
    unique_apps_count = len(app_name_counts)
    debug_print(f"\n📊 Summary: {len(results_dict)} images classified ({len(successful_ids)} successful)", level="info")
    debug_print(f"🔢 Unique apps detected: {unique_apps_count}", level="info")
    debug_print(f"📱 Top 10 apps:", level="info")
    for app_name, count in sorted(app_name_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        debug_print(f"   {app_name}: {count}", level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify app names from test images using VLM")
    parser.add_argument("--hf-dataset-id", type=str, default=['HongxinLi/AutoGUIv2-FuncElemGnd', 'MMInstruction/OSWorld-G'][-1],
                       help="HuggingFace dataset ID")
    parser.add_argument("--hf-split", type=str, default='test',
                       help="Dataset split to load (default: 'test')")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output JSON file for app name classifications")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY_XIAOAI"),
                       help="API key")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_XIAOAI"),
                       help="API base URL")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro-thinking",
                       help="Model to use for classification")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true",
                       help="Run sequentially (debug mode)")
    parser.add_argument("--debug", action="store_true",
                       help="Verbose debug output")
    parser.add_argument("--force", action="store_true",
                       help="Recompute even if already present in output file")
    parser.add_argument("--task-timeout", type=int, default=1800,
                       help="Timeout per task in seconds (parallel mode)")
    parser.add_argument("--sample-limit", type=int, default=None,
                       help="Limit the number of images to process (for debugging)")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                       help="Path to checkpoint file to load/save")
    parser.add_argument("--load-latest", action="store_true",
                       help="Load the latest checkpoint for this model")

    args, _ = parser.parse_known_args()
    
    if args.output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        safe_model_name = args.model.replace('/', '_').replace('\\', '_')
        safe_dataset_name = args.hf_dataset_id.replace('/', '_').replace('\\', '_')
        args.output_file = os.path.join(
            script_dir, 
            f"app_names_{safe_dataset_name}_{safe_model_name}.json"
        )

    multiprocessing.set_start_method('spawn', force=True)
    main(args)

