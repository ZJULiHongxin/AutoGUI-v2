import os, json, shutil, glob
from PIL import Image
from tqdm import tqdm

ROOT = "/mnt/vdb1/hongxin_li/AutoGUIv2"
anno_file_postfix = [
    "screenspot_pro/gemini-2.5-pro-thinking/v2/HongxinLi-ScreenSpot-Pro.json",
    "osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G.json"][1]

bmk_name, model, version = anno_file_postfix.split('/')[:3]

cache_dir = os.path.join(ROOT, 'cache', os.path.dirname(anno_file_postfix))

file = os.path.join(ROOT, anno_file_postfix)

data = json.load(open(file))

for sample_filename, anno in tqdm(data['results'].items(), total=len(data['results']), desc="Splitting region annos"):
    sample_name = sample_filename.split('/')[-1].split('.')[0]
    sample_folder = sorted(glob.glob(os.path.join(cache_dir, sample_name + '*')))[0]
    if sample_name == '0NVmz0b7L0':
        1+1
    raw_img_path = os.path.join(ROOT, sample_filename)
    img = Image.open(raw_img_path)

    for node_id, node_anno in anno['result'].items():
        region_bbox = node_anno['bbox_global']
        if region_bbox[2] - region_bbox[0] < 8 or region_bbox[3] - region_bbox[1] < 8:
            continue

        node_anno_file = os.path.join(sample_folder, 'nodes', node_id + '_meta.json')
        with open(node_anno_file, 'w') as f:
            json.dump(node_anno, f, indent=2, ensure_ascii=False)

        dest = os.path.join(sample_folder, 'nodes', node_id + '_crop.png')

        if node_id == '0-0':
            src = os.path.join(ROOT, node_anno['root_image_path']) if ROOT not in node_anno['root_image_path'] else node_anno['root_image_path']
            shutil.copy(src, dest)
        else:
            src = os.path.join(ROOT, 'cache', bmk_name, model, version, sample_name, 'nodes', os.path.basename(node_anno['node_image_path']))


        #if not os.path.exists(src):

            region_crop = img.crop(region_bbox)
            region_crop.save(dest)
        # else:
        #     shutil.copy(src, dest)