import os, json

# elemfuncgnd
file = "utils/eval_utils/autoguiv2/elemgnd_hf_dataset_cache/a136b59f5a5f5e2809a96d5798412c8a_funcgnd.json"

data = json.load(open(f))

num_images = len(set([x['image_name'] for x in data['entries']]))
