import glob
import json
import os
from tqdm import tqdm
from utils.data_utils.misc import get_image_dimensions

data_dir = "/mnt/nvme1n1p1/hongxin_li/hf_home/hub/GUIKnowledgeBench/Image"
checkpoint_file = os.path.join(os.path.dirname(__file__), "guiknowledgebench_dimensions_cache.json")

# Load existing checkpoint if it exists
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        cache = json.load(f)
    print(f"Loaded checkpoint with {len(cache)} cached image dimensions")
else:
    cache = {}

images = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
print(f"Found {len(images)} images")

# Process images with checkpointing
resolutions = []
for image in tqdm(images, desc="Processing images"):
    # Skip if already processed
    if image in cache:
        resolutions.append(cache[image])
    else:
        try:
            resolution = get_image_dimensions(image)
            cache[image] = resolution
            resolutions.append(resolution)
            # Save checkpoint periodically (every 100 images)
            if len(cache) % 100 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Error processing {image}: {e}")
            continue

# Final checkpoint save
with open(checkpoint_file, 'w') as f:
    json.dump(cache, f, indent=2)
print(f"Saved checkpoint with {len(cache)} image dimensions")

# Count resolutions
d = {}
for resolution in resolutions:
    dim_str = f"{resolution[0]}x{resolution[1]}"
    if dim_str not in d:
        d[dim_str] = 0
    d[dim_str] += 1

# Calculate total count for proportions
total_count = sum(d.values())

# Rank by total pixels (width * height) and calculate proportions
ranked_resolutions = []
for dim_str, count in d.items():
    width, height = map(int, dim_str.split('x'))
    total_pixels = width * height
    proportion = (count / total_count) * 100
    ranked_resolutions.append({
        'resolution': dim_str,
        'width': width,
        'height': height,
        'total_pixels': total_pixels,
        'count': count,
        'proportion': proportion
    })

# Sort by total pixels (descending)
ranked_resolutions.sort(key=lambda x: x['total_pixels'], reverse=True)

# Print ranked results with proportions
print("\n" + "="*80)
print("Resolution Statistics (Ranked by Total Pixels)")
print("="*80)
print(f"{'Rank':<6} {'Resolution':<20} {'Total Pixels':<15} {'Count':<10} {'Proportion':<12}")
print("-"*80)
for idx, res in enumerate(ranked_resolutions, 1):
    print(f"{idx:<6} {res['resolution']:<20} {res['total_pixels']:<15,} {res['count']:<10} {res['proportion']:<12.2f}%")
print("="*80)
print(f"Total images: {total_count}")