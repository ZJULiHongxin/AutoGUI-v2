import os, json
from glob import glob
from tqdm import tqdm
from utils.data_utils.misc import get_image_dimensions


image_pattern = os.path.join("/mnt/vdb1/hongxin_li/AutoGUIv2", 
                   ["amex", "androidcontrol", "agentnet", "screenspot_pro", "osworld_g", "mmbenchgui"][-1], "images", "**/*.png")

images = glob(image_pattern, recursive=True)

resolutions = [get_image_dimensions(image) for image in tqdm(images, total=len(images))]

d = {}

for resolution in tqdm(resolutions, total=len(resolutions)):
    dim_str = f"{resolution[0]}x{resolution[1]}"
    if dim_str not in d:
        d[dim_str] = 0
    d[dim_str] += 1

print(d)