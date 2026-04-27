"""
Generate DINO-v3 embeddings for all elements detected by OmniParser.

This script reads element annotations from the `omniparser` directory located
next to the dataset root, crops each element from the corresponding image, and
uses a HuggingFace feature-extraction pipeline (DINO-v3) to generate embeddings.

Embeddings are saved per image to: {base_dir}/omniparser_embeddings/{image_name}.npz
with keys: 'embeddings' (float32 array [N, D]) and 'indices' ([N] element indices).
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple
from colorama import Fore, Style
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
from rapidfuzz import fuzz

from utils.data_utils.misc import is_pure_color, merge_lists_with_shared_elements


def find_image_by_stem(images_root: Path, stem: str) -> Path:
    exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    for ext in exts:
        matches = list(images_root.rglob(stem + ext))
        if matches:
            return matches[0]
    return None


def clamp_bbox_to_image(bbox: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(bbox[0] * width)))
    y1 = max(0, min(height - 1, int(bbox[1] * height)))
    x2 = max(0, min(width, int(np.ceil(bbox[2] * width))))
    y2 = max(0, min(height, int(np.ceil(bbox[3] * height))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2

def check_elem_validity(screenshot, bbox):
    x1, y1, x2, y2 = bbox
    W, H = screenshot.size

    if x1 < 0 or x2 > 1 or y1 < 0 or y2 > 1 or x1 >= x2 or y1 >= y2:
        return False
    if x2 - x1 <= 0.005 or y2 - y1 <= 0.005:
        return False
    if (x2-x1) * (y2-y1) >= 0.25:
        return False
                
    if is_pure_color(screenshot, [x1, y1, x2, y2]):
        return False
   
    return True

def main():
    parser = argparse.ArgumentParser(description="Embed all OmniParser elements with DINO-v3")
    parser.add_argument("--images-root", type=str, default=[
        "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/mmbenchgui/",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/images/",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/androidcontrol/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/guiodyssey/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/images",
        "/mnt/vdb1/hongxin_li/AutoGUIv2/magicui/images/"][2],
                        help="Root directory containing images (will search recursively)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Dataset base directory. If None and images-root contains 'images', uses its parent")
    parser.add_argument("--model", type=str, default=[
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "facebook/dinov3-vith16plus-pretrain-lvd1689m"][-1],
                        help="HuggingFace model id for image feature extraction")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding pipeline")
    parser.add_argument("--device", type=str, default='cuda:7', help="Device: 'cuda' or 'cpu' (auto if None)")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for number of images")
    parser.add_argument("--draw-debug", type=bool, default=False, help="Draw debug images")
    args, _ = parser.parse_known_args()

    images_root = Path(args.images_root).resolve()
    if args.base_dir is None:
        parts = list(images_root.parts)
        if "images" in parts:
            idx = parts.index("images")
            base_dir = Path(*parts[:idx])
        else:
            base_dir = images_root
    else:
        base_dir = Path(args.base_dir).resolve()

    omniparser_dir = base_dir / "omniparser"
    out_dir = base_dir / "omniparser_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not omniparser_dir.exists():
        print(f"OmniParser directory not found: {omniparser_dir}")
        return

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model, 
        device_map="auto", 
    )
    model.eval()

    json_files = sorted(list(omniparser_dir.glob("**/*.json")))
    if args.max_images is not None:
        json_files = json_files[: args.max_images]

    for json_path in tqdm(json_files, desc="Embedding elements"):
        stem = str(json_path).split('omniparser/')[-1].rsplit('.', 1)[0]
        
        out_path = out_dir / f"{stem}.npz"
        if out_path.exists():
            continue
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        image_path = find_image_by_stem(images_root, stem)
        if image_path is None or not image_path.exists():
            print(f"Image not found for {stem}, skipping")
            continue

        try:
            with open(json_path, "r") as f:
                elements = json.load(f)
        except Exception as e:
            print(f"Failed to read {json_path}: {e}")
            continue

        try:
            img = Image.open(str(image_path)).convert("RGB")
            W, H = img.size
        except Exception as e:
            print(f"Failed to open image {image_path}: {e}")
            continue

        crops = []
        indices = []
        
        # Calculate the edit distances among all pairs to remove those pairs that are absolutely dissimilar.
        # element attrs: {'type': 'text', 'bbox': [0.008333333767950535, 0.003703703638166189, 0.04270833358168602, 0.024074073880910873], 'interactivity': False, 'content': 'Activities', 'source': 'box_ocr_content_ocr'}
        # NOTE: Do not use fuzz.ratio; use fuzz.token_set_ratio instead.
        # Because fuzz.ratio('Close', 'Comment') = 50.0 <=> fuzz.ratio('Close', 'Close (Ctrl+F4)') = 50.0, this is not a good way to measure the similarity.
        # NOTE: A better method is fuzz.token_set_ratio. This function is powerful because it splits the strings into words (tokens), ignores word order, and then compares the sets of tokens. This is highly effective for matching things like UI elements, where order might not matter (e.g., 'Save File As' vs. 'File Save As').

        text_similarity = np.ones((len(elements), len(elements)))
        for i in range(len(elements)):
            for j in range(len(elements)):
                if i == j:
                    text_similarity[i, j] = 100
                else:
                    text_similarity[i, j] = fuzz.token_set_ratio(elements[i]['content'], elements[j]['content'])
        
        for idx, elem in enumerate(elements):
            bbox = elem.get("bbox") or elem.get("bbox_global")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = clamp_bbox_to_image(bbox, W, H)
            try:
                crop = img.crop((x1, y1, x2, y2))
            except Exception:
                continue
            crops.append(crop)
            indices.append(idx)

        if not crops:
            # Save empty file to avoid retrying forever
            np.savez_compressed(
                str(out_path),
                embeddings=np.zeros((0, 0), dtype=np.float32),
                indices=np.array([], dtype=np.int32),
                similarity=np.zeros((0, 0), dtype=np.float32),
                text_similarity=np.zeros((0, 0), dtype=np.int32),
                similar_groups=np.array([], dtype=object),
            )
            continue

        # Batched embedding
        embeddings: List[np.ndarray] = []
        batch_size = max(1, args.batch_size)
        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model(**inputs)
            pooled_output = outputs.pooler_output
            # Convert to numpy
            embeddings_batch = pooled_output.detach().cpu().numpy()
            for i in range(embeddings_batch.shape[0]):
                embeddings.append(embeddings_batch[i])

        try:
            emb_matrix = np.stack(embeddings).astype(np.float32)
        except Exception as e:
            print(f"Failed to stack embeddings for {stem}: {e}")
            continue

        # Compute cosine similarity matrix using L2-normalized embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
        emb_norm = emb_matrix / norms
        sim_mat = emb_norm @ emb_norm.T

        # Find the top-5 similar pairs
        ## Set the diagonal to 0
        np.fill_diagonal(sim_mat, 0)
        
        # Handle special bad cases
        for i in range(len(elements)):
            # Case 1: Meaningless annotation output by OmniParser.
            # Case 2: Non-interactable elements.
            # Case 3: Elements on the notification bar at the top of mobile devices.
            if 'M0,0L9,0' in elements[i]['content'] \
                or not elements[i]['interactivity'] \
                or (W < H and elements[i]['bbox'][-1] < 0.055):
                sim_mat[i, :] = 0
                sim_mat[:, i] = 0

        sim_sorted_idxs = np.argsort(np.max(sim_mat, axis=1))[::-1] # From the highest to lowest
        top_pair_idxs = sim_sorted_idxs[:20]
        mask =  text_similarity >= 60 #np.logical_or(sim_mat > 0.6, text_similarity >= 60)
        visited, groups = set(), []

        # Debugging: Mark top-10 similar elements using OpenCV
        for idx in top_pair_idxs:
            if idx in visited: continue
            visited.add(idx)

            similar_idxs = [x for x in np.argsort(sim_mat[idx])[::-1] if sim_mat[idx, x] > 0.5]
            similar_idxs_w_mask = [x for x in np.argsort((sim_mat *mask)[idx])[::-1] if (sim_mat * mask)[idx, x] > 0.58]
            
            # Too many similar elements is also not acceptable as this is likely caused by OmniParser detection errors.
            if not 0 < len(similar_idxs_w_mask) <= 10: continue
            
            if args.draw_debug:
                debug_img = np.array(img)
                debug_img_w_mask = np.array(img)
                print(Fore.CYAN + f"\nCur node: {idx} - {elements[idx]['content']}\nSimilar nodes: " + Style.RESET_ALL)

                x1, y1, x2, y2 = clamp_bbox_to_image(elements[idx]['bbox'], W, H)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(debug_img, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.rectangle(debug_img_w_mask, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(debug_img, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            for pair_idx in similar_idxs[:10]:
                if pair_idx == idx: continue

                if args.draw_debug:
                    print(f"{pair_idx} - {elements[pair_idx]['content']}", end='\t')
                    x1, y1, x2, y2 = clamp_bbox_to_image(elements[pair_idx]['bbox'], W, H)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_img, str(pair_idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            visited.update(similar_idxs_w_mask)

            valid = True
            for elem_idx in [idx] + similar_idxs_w_mask:
                if not check_elem_validity(img, elements[elem_idx]['bbox']):
                    valid = False
                    break
            
            # Skip numeric elements as they cannot be used to generate meaningful tasks.
            valid = sum([elements[i]['content'].replace('.','').isnumeric() for i in similar_idxs_w_mask]) / len(similar_idxs_w_mask) < 0.7

            if valid:
                groups.append([idx] + similar_idxs_w_mask)

                if args.draw_debug:
                    print("\nSimilar nodes with mask: ")

                    for pair_idx in similar_idxs_w_mask:
                        if pair_idx == idx: continue
                        print(f"{pair_idx} - {elements[pair_idx]['content']}", end='\t')
                        x1, y1, x2, y2 = clamp_bbox_to_image(elements[pair_idx]['bbox'], W, H)
                        cv2.rectangle(debug_img_w_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_img_w_mask, str(pair_idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    output_image = cv2.cvtColor(np.concatenate([debug_img, debug_img_w_mask], axis=1), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(out_dir, f"{stem}_debug.png"), output_image)
                    cv2.imwrite("test.png", output_image)

                    1+1

        merged = merge_lists_with_shared_elements(groups)
        np.savez_compressed(
            str(out_path),
            embeddings=emb_matrix.astype(np.float32),
            indices=np.asarray(indices, dtype=np.int32),
            similarity=sim_mat.astype(np.float32),
            text_similarity=text_similarity.astype(np.int32),
            similar_groups=np.asarray(merged, dtype=object)
        )


if __name__ == "__main__":
    main()