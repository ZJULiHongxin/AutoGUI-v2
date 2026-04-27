import cv2

img_path = "/mnt/vdb1/hongxin_li/AutoGUIv2/cache/osworld_g/gemini-2.5-pro-thinking/v2/1y0CmHiyPQ/root.png"
out_path = "/mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/data_utils/autoguiv2/FuncElemQA_eval_gen/test/root_with_bboxes.png"

image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"找不到图片：{img_path}")

# 红色 bbox
cv2.rectangle(image, (360, 210), (680, 247), color=(0, 0, 255), thickness=2)
# 绿色 bbox
cv2.rectangle(image, (520, 71), (1442, 101), color=(0, 128, 0), thickness=2)

cv2.imwrite(out_path, image)
print(f"结果已保存到 {out_path}")