import os, torch, time, json, io, base64, glob
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
from utils.data_utils.misc import get_image_dimensions

device = 'cuda'
OMNIPARSER_MODEL_DIR = "/mnt/nvme0n1p1/hongxin_li/OmniParser"
model_path=os.path.join(OMNIPARSER_MODEL_DIR, "weights/icon_detect/model.pt")

som_model = get_yolo_model(model_path)

som_model.to(device)
print('model to {}'.format(device))

caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path=os.path.join(OMNIPARSER_MODEL_DIR, "weights/icon_caption_florence"), device=device)


BOX_TRESHOLD = 0.05

ROOT = "/mnt/vdb1/hongxin_li/AutoGUIv2/"
bmk_source = [
    "osworld_g",
    "screenspot_pro",
    "mmbenchgui",
    "agentnet",
    "androidcontrol",
    "guiodyssey",
    "amex",
    "magicui"][2]

cache_dir = os.path.join(ROOT, bmk_source, 'omniparser')
os.makedirs(cache_dir, exist_ok=True)

image_filenames = glob.glob(os.path.join(ROOT, bmk_source, 'images', '**', '*.png'), recursive=True)


for img_path in tqdm(image_filenames, total=len(image_filenames), desc="Detecting all elements"):
    img_name = img_path.split('images/')[-1].rsplit('.', 1)[0]
    
    save_to = os.path.join(cache_dir, f"{img_name}.json")
    img_save_to = save_to.replace(".json", ".png")

    if os.path.exists(save_to) and os.path.exists(img_save_to):
        continue

    save_to_dir = os.path.dirname(save_to)
    os.makedirs(save_to_dir, exist_ok=True)

    W, H = get_image_dimensions(img_path)
    box_overlay_ratio = max(W, H) / 3200

    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    start = time.time()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(img_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=False)

    text, ocr_bbox = ocr_bbox_rslt

    if len(ocr_bbox) == 0:
        continue

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(img_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
    if dino_labled_img is None:
        continue

    cur_time_caption = time.time()

    # plot dino_labled_img it is in base64
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    image.save(img_save_to)

    with open(save_to, 'w') as f:
        json.dump(parsed_content_list, f, indent=2, ensure_ascii=False)

    print(f"Time taken for {save_to}: {cur_time_caption - start} seconds")

start = time.time()

