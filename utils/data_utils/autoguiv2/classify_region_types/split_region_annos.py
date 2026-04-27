import os, json
from tqdm import tqdm
from utils.data_utils.autoguiv2.misc import OFFICIAL_TO_LOCALNAME_MAP

ROOT = "/mnt/vdb1/hongxin_li/AutoGUIv2"
anno_file_postfix = "osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G_region_types_gemini-2.5-flash-thinking.json"

file = os.path.join(ROOT, anno_file_postfix)

data = json.load(open(file))

bmk_local_name, anno_model, version = anno_file_postfix.split('/')[:3]

for img_filename, anno in tqdm(data['results'].items(), total=len(data['results']), desc="Splitting region annos"):
    img_name = img_filename.split('/')[-1].split('.')[0]
    
    cache_dir = os.path.join(ROOT, 'cache', bmk_local_name, anno_model, version, img_name, "nodes")
    
    if 'region_types' not in anno:
        print(f"No region types for {img_name}")
        continue
    for node_id, retion_anno in anno['region_types'].items():
        node_anno_file = os.path.join(cache_dir, node_id + '_region-type.json')
        with open(node_anno_file, 'w') as f:
            json.dump(retion_anno, f, indent=2, ensure_ascii=False)