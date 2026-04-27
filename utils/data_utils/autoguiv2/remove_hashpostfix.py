import os, json

ROOT = "/mnt/vdb1/hongxin_li/AutoGUIv2"
anno_file_postfix = "osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G.json"

file = os.path.join(ROOT, anno_file_postfix)

data = json.load(open(file))

for img_filename, anno in data['results'].items():
    for node_id, node_info in anno['result'].items():
        if node_id == '0-0':
            continue

        node_img_path = node_info['node_image_path']
        cache_dir, bmk_name, model, version, sample_name = node_img_path.split('/')[-7:-2]

        if '-' not in sample_name:
            continue

        main, hash = sample_name.rsplit('-', 1)
        if len(hash) != 8: continue

        node_info['node_image_path'] = os.path.join(cache_dir, bmk_name, model, version, main, 'nodes', node_id + '.png')

with open(file, 'w') as f:
    json.dump(data, f, indent=2)