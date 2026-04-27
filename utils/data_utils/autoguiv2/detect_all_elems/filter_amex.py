# Pick the samples with non-empty similar groups
import os, json, numpy as np

file = "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/raw_image_path.json"
autoguiv2_dir = "/mnt/vdb1/hongxin_li/AutoGUIv2"
with open(file, "r") as f:
    data = json.load(f)

filtered_paths = []

for image_info in data:
    path = image_info['path']
    emb_file = os.path.join(autoguiv2_dir, "amex/omniparser_embeddings", path.split('AMEX/')[-1].replace(".png", ".npz"))
    if not os.path.exists(emb_file):
        continue
    
    try:
        emb_result = np.load(emb_file, allow_pickle=True)
    except Exception as e:
        continue
    if len(emb_result['similar_groups']) > 0:
        filtered_paths.append(image_info)

print(f"Filtered {len(filtered_paths)} images")
with open(file.replace(".json", "_filtered.json"), "w") as f:
    json.dump(filtered_paths, f, indent=2)