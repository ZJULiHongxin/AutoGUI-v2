#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import Any, Dict, List, Tuple, Union, Optional

try:
	from PIL import Image
except ImportError:
	Image = None  # 延迟报错，运行时提示安装 pillow


TARGET_DATASET_NAMES = {"osworld_g", "screenspot_pro", "agentnet", "amex"}

# 需要处理的 4 个目录（grounding_mode）
DEFAULT_INPUT_DIRS = [
	"/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/grounding_mode",
	"/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/grounding_mode",
	"/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/grounding_mode",
	"/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/grounding_mode",
]


def infer_dataset_name_from_path(path: str) -> Optional[str]:
	"""
	从路径中推断数据集名称（osworld_g、screenspot_pro、agentnet、amex）
	策略：寻找路径段中第一个命中 TARGET_DATASET_NAMES 的名字
	"""
	for part in os.path.normpath(path).split(os.sep):
		if part in TARGET_DATASET_NAMES:
			return part
	return None


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(path: str, data: Any) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)
		f.write("\n")


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
	"""
	返回 (height, width)。如果失败返回 None。
	"""
	if not image_path:
		return None
	if not os.path.isabs(image_path):
		# 尝试展开用户/相对路径
		image_path = os.path.abspath(os.path.expanduser(image_path))
	if not os.path.exists(image_path):
		return None
	if Image is None:
		raise RuntimeError("未安装 pillow，请先安装: pip install pillow")
	try:
		with Image.open(image_path) as img:
			width, height = img.size
			return height, width
	except Exception:
		return None


def extract_image_name_from_path(value: Optional[str]) -> Optional[str]:
	if not value or not isinstance(value, str):
		return None
	# 取最后一个 "/" 后的内容
	return value.rsplit("/", 1)[-1]


def ensure_fields_for_item(
	item: Dict[str, Any],
	root_obj: Union[Dict[str, Any], List[Any]],
	dataset_name: str,
) -> Tuple[bool, List[str]]:
	"""
	为单条问题补充字段：
	- image_size: [height, width]，基于 image_path 读取
	- image_name: 来自 image_path 的 basename（优先从 item，其次从 root）
	- dataset_name: 来自所在目录推断
	返回 (是否有变更, 错误信息列表)。当任一字段无法成功写入时，将在错误列表中记录原因。
	"""
	changed = False
	errors: List[str] = []

	# image_path: 优先 item，其次 root
	image_path_in_item = item.get("image_path")
	image_path_in_root = None
	if isinstance(root_obj, dict):
		image_path_in_root = root_obj.get("image_path")
	image_path = image_path_in_item or image_path_in_root

	# image_size
	size = None
	if not image_path:
		errors.append("image_size: 缺少 image_path（在 item 与根对象中均未找到）")
	else:
		size = get_image_size(image_path)
		if size is None:
			errors.append(f"image_size: 无法读取图片尺寸 image_path={image_path}")
	if size is not None:
		h, w = size
		if item.get("image_size") != [h, w]:
			item["image_size"] = [h, w]
			changed = True

	# image_name
	image_name = extract_image_name_from_path(image_path)
	if not image_name:
		errors.append("image_name: 缺少 image_path（在 item 与根对象中均未找到）")
	if image_name and item.get("image_name") != image_name:
		item["image_name"] = image_name
		changed = True

	# dataset_name
	if not dataset_name:
		errors.append("dataset_name: 传入的数据集名称为空")
	else:
		if item.get("dataset_name") != dataset_name:
			item["dataset_name"] = dataset_name
			changed = True

	return changed, errors


def process_json_file(json_path: str, dataset_name: str, *, dry_run: bool = False, write_path: Optional[str] = None) -> bool:
	"""
	处理单个 JSON 文件。
	支持两种常见结构：
	1) 列表：文件内容为 [ question_obj, ... ]
	2) 字典：文件内容为 { ..., "questions": [ ... ] } 或者 { question_fields... }（少见）
	返回是否有变更
	"""
	obj = load_json(json_path)
	changed = False

	def process_list(lst: List[Dict[str, Any]]) -> bool:
		local_changed = False
		for idx, q in enumerate(lst):
			if isinstance(q, dict):
				changed_item, errs = ensure_fields_for_item(q, obj, dataset_name)
				if errs:
					for e in errs:
						print(f"[ERROR] 字段补充失败: {json_path} [item #{idx}] -> {e}")
				if changed_item:
					local_changed = True
		return local_changed

	if isinstance(obj, list):
		changed = process_list(obj)
	elif isinstance(obj, dict):
		# 优先寻找 questions 列表
		if "questions" in obj and isinstance(obj["questions"], list):
			if process_list(obj["questions"]):
				changed = True
		# 兼容 result.questions 的结构
		elif (
			"result" in obj
			and isinstance(obj["result"], dict)
			and "questions" in obj["result"]
			and isinstance(obj["result"]["questions"], list)
		):
			if process_list(obj["result"]["questions"]):
				changed = True
		else:
			# 退化为“单题目对象”的情况
			changed_item, errs = ensure_fields_for_item(obj, obj, dataset_name)
			if errs:
				for e in errs:
					print(f"[ERROR] 字段补充失败: {json_path} -> {e}")
			if changed_item:
				changed = True
	else:
		# 非预期结构，跳过
		return False

	if changed:
		if dry_run:
			print(f"[DRY-RUN] 将更新: {json_path}")
		else:
			target = write_path if write_path else json_path
			os.makedirs(os.path.dirname(target), exist_ok=True)
			save_json(target, obj)
	return changed


def iter_json_files(root_dir: str) -> List[str]:
	results: List[str] = []
	for dirpath, _dirnames, filenames in os.walk(root_dir):
		for fn in filenames:
			if fn.lower().endswith(".json"):
				results.append(os.path.join(dirpath, fn))
	return results


def main(input_dirs: Optional[List[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="为 grounding 问题 JSON 补充 image_size / image_name / dataset_name")
	parser.add_argument(
		"--dirs",
		nargs="*",
		default=None,
		help="要处理的目录（可多个）。若不提供则使用脚本内置的 4 个目录。",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="试运行：仅打印将要更新的文件，不写回任何 JSON。",
	)
	parser.add_argument(
		"--output-root",
		default=None,
		help="将更新结果写入到该根目录（按输入目录的相对路径结构镜像输出）。未提供时默认就地写回（非 dry-run）。",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="每个输入目录最多处理的 JSON 文件数（用于小样本测试）。",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="打印详细信息（例如成功更新的文件）。默认仅打印错误/警告/跳过/汇总。",
	)
	args = parser.parse_args()

	target_dirs = args.dirs if args.dirs else (input_dirs if input_dirs else DEFAULT_INPUT_DIRS)
	total_files = 0
	updated_files = 0

	for directory in target_dirs:
		if not os.path.isdir(directory):
			print(f"[WARN] 目录不存在，跳过: {directory}")
			continue
		dataset_name = infer_dataset_name_from_path(directory) or "unknown"
		if dataset_name not in TARGET_DATASET_NAMES:
			print(f"[WARN] 未能从路径推断标准数据集名，使用: {dataset_name} 目录={directory}")
		json_files = iter_json_files(directory)
		if not json_files:
			print(f"[INFO] 未找到 JSON 文件: {directory}")
			continue
		if args.limit is not None and args.limit >= 0:
			json_files = json_files[: args.limit]
		for jf in json_files:
			total_files += 1
			try:
				write_path = None
				if args.output_root:
					rel_path = os.path.relpath(jf, directory)
					write_path = os.path.join(args.output_root, dataset_name, rel_path)
				changed = process_json_file(
					jf,
					dataset_name,
					dry_run=args.dry_run,
					write_path=write_path,
				)
				if changed:
					updated_files += 1
					# 成功信息：默认不打印；若 dry-run 或显式 verbose 则打印
					if args.dry_run or args.verbose:
						label = "[DRY-RUN]" if args.dry_run else ("[OK]" if not args.output_root else "[OK->OUT]")
						target_msg = f" -> {write_path}" if write_path else ""
						print(f"{label} 更新: {jf}{target_msg}")
				else:
					print(f"[SKIP] 无需更新: {jf}")
			except Exception as e:
				print(f"[ERROR] 处理失败: {jf} -> {e}")

	print(f"\n共扫描文件: {total_files}，更新文件: {updated_files}")


if __name__ == "__main__":
	main()


