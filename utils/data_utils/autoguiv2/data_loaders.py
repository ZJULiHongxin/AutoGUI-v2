import os, glob, random, json, shutil
import hashlib

from torch.utils.data import Dataset
import datasets
from tqdm import tqdm

import random
random.seed(999)

class ScreenSpotPro(Dataset):
    def __init__(self, data_dir: str = "HongxinLi/ScreenSpot-Pro", cache_dir: str = 'func_region_anno_results/screenspot_pro/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = datasets.load_dataset(data_dir, split='test')
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        sample_cache_file = os.path.join(os.path.dirname(self.cache_dir), 'samples.json')
        need_reload = True
        if os.path.exists(sample_cache_file):
            print(f"Loading ScreenSpot-Pro data from cache file: {sample_cache_file}")
            with open(sample_cache_file, 'r') as f:
                content = json.load(f)
            #self.samples = content['samples']
            self.image_paths = content['image_paths']
            
            if len(self.image_paths) == len(self.data): need_reload = False


        if need_reload:
            self.samples, self.image_paths = [], []
            for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Caching ScreenSpot-Pro data"):
                # if debug and idx > 0:
                #     break
                # id, file-name, image: PIL.Image, bbox, instruction, instruction_cn, application, data_type, data_source, image_size(wxh)
                sample_id, file_name, image, bbox, instruction, instruction_cn, application, data_type, data_source, image_size = item['id'], item['file_name'], item['image'], item['bbox'], item['instruction'], item['instruction_cn'], item['application'], item['data_type'], item['data_source'], item['image_size(wxh)']
                image_path = os.path.join(self.cache_dir, file_name)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                if not os.path.exists(image_path):
                    item['image'].save(image_path)
                # self.samples.append({
                #     'sample_id': sample_id,
                #     'image_path': image_path,
                #     'image': image,
                #     'image_size': image_size,
                #     'bbox': bbox,
                #     'instruction': instruction,
                #     'instruction_cn': instruction_cn,
                # })
                self.image_paths.append(image_path)
            
            with open(sample_cache_file, 'w') as f:
                json.dump({'image_paths': self.image_paths}, f, indent=2)

        if random_sample is not None:
            rand_indices = random.sample(range(len(self.data)), random_sample)
            self.image_paths = [x for i, x in enumerate(self.image_paths) if i in rand_indices]
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class OSWORLDG(Dataset):
    """
    OsWorldG dataset has 250 unique images.
    """
    def __init__(self, data_dir: str = "MMInstruction/OSWorld-G", cache_dir: str = 'func_region_anno_results/osworld_g/images', debug: bool = False):
        self.data_dir = data_dir
        self.data = datasets.load_dataset(data_dir, split='test')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Caching OSWorld-G data"):
            # if debug and idx > 1:
            #     break
            # id, instruction, image, mimo_bbox, GUI_types, image_path
            sample_id, instruction, image, unnorm_bbox, gui_types, image_path = item['id'], item['instruction'], item['image'], item['mimo_bbox'], item['GUI_types'], item['image_path']
            cached_image_path = os.path.join(self.cache_dir, image_path)
            
            if cached_image_path in self.image_paths:
                continue

            os.makedirs(os.path.dirname(cached_image_path), exist_ok=True)
            if not os.path.exists(cached_image_path):
                item['image'].save(cached_image_path)
            self.samples.append({
                'sample_id': sample_id,
                'image_path': cached_image_path,
                'image': image,
                'image_size': image.size,
                'unnorm_bbox': unnorm_bbox,
                'instruction': instruction,
                'gui_types': gui_types,
            })
            self.image_paths.append(cached_image_path)
        
        print(f"Cached {len(self.samples)} samples from {len(self.data)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class MMBenchGUI(Dataset):
    def __init__(self, data_dir: str = "/mnt/jfs/copilot/lhx/ui_data/MMBenchGUI/offline_images/", cache_dir: str = 'func_region_anno_results/mmbenchgui/images', debug: bool = False):
        self.data_dir = data_dir
        self.image_paths = []
        raw_iamge_paths = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
        
        for path in tqdm(raw_iamge_paths, total=len(raw_iamge_paths), desc="Caching MMBenchGUI data"):
            image_path = os.path.join(cache_dir, path.split('offline_images/')[-1])
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            if not os.path.exists(image_path):
                shutil.copy(path, image_path)
            self.image_paths.append(image_path)
        # if debug:
        #     self.image_paths = self.image_paths[:1]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        return self.image_paths[index]

class AgentNet(Dataset):
    def __init__(self, data_dir: str = "sujr/autogui-agentnet", cache_dir: str = 'func_region_anno_results/agentnet/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = datasets.load_dataset(data_dir, split='train')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Caching AgentNet data"):
            if random_sample is not None and idx >= random_sample:
                break
            image_path = os.path.join(self.cache_dir, item['task_id'] + '.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            if not os.path.exists(image_path):
                item['image'].save(image_path)
            
            self.samples.append({
                'sample_id': item['task_id'],
                'image_path': image_path,
                'image': item['image'],
                'image_size': item['image'].size,
                'application_name': item['application_name'],
                'category': item['category'],
                'os': item['os']
            })
            self.image_paths.append(image_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class AndroidControl(Dataset):
    def __init__(self, data_dir: str = "/mnt/vdb1/hongxin_li/AutoGUIv2/raw_image_path.json", cache_dir: str = 'func_region_anno_results/androidcontrol/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = json.load(open(data_dir))
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, image_path in tqdm(enumerate(self.data), total=len(self.data), desc="Caching AndroidControl data"):
            if random_sample is not None and idx >= random_sample:
                break
            
            dest_image_path = os.path.join(cache_dir, '/'.join(image_path.split('/')[-3:]))
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            if not os.path.exists(dest_image_path):
                shutil.copy(image_path, dest_image_path)
            
            self.samples.append({'image_path': dest_image_path})
            self.image_paths.append(dest_image_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class GUIOdyssey(Dataset):
    def __init__(self, data_dir: str = "/mnt/vdb1/hongxin_li/AutoGUIv2/guiodyssey/raw_image_path.json", cache_dir: str = 'func_region_anno_results/guiodyssey/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = json.load(open(data_dir))
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, image_info in tqdm(enumerate(self.data), total=len(self.data), desc="Caching GUIOdyssey data"):
            image_path = image_info['path']
            if random_sample is not None and idx >= random_sample:
                break

            dest_image_path = os.path.join(cache_dir, os.path.basename(image_path))
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            if not os.path.exists(dest_image_path):
                shutil.copy(image_path, dest_image_path)
            
            self.samples.append({'image_path': dest_image_path})
            self.image_paths.append(dest_image_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class AMEX(Dataset):
    def __init__(self, data_dir: str = "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/raw_image_path.json", cache_dir: str = 'func_region_anno_results/amex/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = json.load(open(data_dir))
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, image_info in tqdm(enumerate(self.data), total=len(self.data), desc="Caching AMEX data"):
            image_path = image_info['path']
            if random_sample is not None and idx >= random_sample:
                break

            dest_image_path = os.path.join(cache_dir, os.path.basename(image_path))
            os.makedirs(os.path.dirname(dest_image_path), exist_ok=True)
            if not os.path.exists(dest_image_path):
                shutil.copy(image_path, dest_image_path)
            
            self.samples.append({'image_path': dest_image_path})
            self.image_paths.append(dest_image_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class MagicUI(Dataset):
    def __init__(self, data_dir: str = "GUIAgent/Magic-RICH", cache_dir: str = 'func_region_anno_results/magicui/images', debug: bool = False, random_sample: int = None):
        self.data_dir = data_dir
        self.data = datasets.load_dataset(data_dir, split='train')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples, self.image_paths = [], []
        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Caching MagicUI data"):
            if random_sample is not None and idx >= random_sample:
                break
            
            image = item['images']
            image_hash = hashlib.md5(image.tobytes()).hexdigest()

            
            image_path = os.path.join(self.cache_dir, f'{image_hash}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            if not os.path.exists(image_path):
                image.save(image_path)

            self.samples.append({
                'sample_id': image_hash,
                'image_path': image_path,
                'image': image,
                'image_size': image.size,
                'type': item['type']
            })
            self.image_paths.append(image_path)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
