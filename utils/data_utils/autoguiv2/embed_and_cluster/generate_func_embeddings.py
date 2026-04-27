import os, json
import torch
import vllm
from vllm import LLM
from tqdm import tqdm

EMBED_MODEL = ["Qwen/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-0.6B"][0]

model = LLM(model=EMBED_MODEL, task="embed")

file = "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/gemini-2.5-pro-thinking/v2/MMInstruction-OSWorld-G.json"

with open(file, 'r') as f:
    data = json.load(f)['results']

embed_output_dir_root = os.path.join(os.path.dirname(file), "embeddings")

for img_idx, (img_filename, anno_info) in tqdm(enumerate(data.items()), total=len(data), desc="Generating embeddings " + EMBED_MODEL):
    img_name = os.path.basename(img_filename).split('.')[0]
    embed_output_dir = os.path.join(embed_output_dir_root, img_name)
    os.makedirs(embed_output_dir, exist_ok=True)

    for node_id, node_info in anno_info['result'].items():
        node_embed_file = os.path.join(embed_output_dir, f"{node_id}.json")
        if os.path.exists(node_embed_file): continue

        texts = [node_info['description']['with_context'],
        node_info['description']['wo_context'] or '',
        node_info['functionality']['with_context'],
        node_info['functionality']['wo_context'] or '']

        outputs = model.embed(texts)
        embeddings = [o.outputs.embedding for o in outputs]

        embed_result = {
            'description': {
                'with_context': embeddings[0],
                'wo_context': embeddings[1]
            },
            'functionality': {
                'with_context': embeddings[2],
                'wo_context': embeddings[3]
            },
        }
    
        with open(node_embed_file, 'w') as f:
            json.dump(embed_result, f, indent=2)
