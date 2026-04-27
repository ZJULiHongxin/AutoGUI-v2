#!/usr/bin/env python3
"""
脚本功能：将指定路径下JSON文件中每个问题的每个选项的functionality字段改名为option_context字段
"""

import json
from pathlib import Path

# 需要处理的4个路径
PATHS = [
    "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode",
    "/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode"
]


def rename_functionality_to_option_context(file_path):
    """
    将JSON文件中每个问题的每个选项的functionality字段改名为option_context
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        bool: 是否成功修改
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # 检查是否有result字段和questions字段
        if 'result' in data and 'questions' in data['result']:
            questions = data['result']['questions']
            
            # 遍历每个问题
            for question in questions:
                if 'options' in question:
                    # 遍历每个选项
                    for option in question['options']:
                        # 如果存在functionality字段，则改名为option_context
                        if 'functionality' in option:
                            option['option_context'] = option.pop('functionality')
                            modified = True
        
        # 如果有修改，保存文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        
        return False
        
    except (json.JSONDecodeError, IOError, KeyError) as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def process_directory(directory_path):
    """
    处理目录下的所有JSON文件
    
    Args:
        directory_path: 目录路径
        
    Returns:
        tuple: (处理的文件数, 成功修改的文件数)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"警告: 路径不存在: {directory_path}")
        return 0, 0
    
    # 查找所有JSON文件（排除_processing_summary.json）
    json_files = list(directory.glob("*.json"))
    json_files = [f for f in json_files if f.name != "_processing_summary.json"]
    
    total_files = len(json_files)
    modified_files = 0
    
    print(f"\n处理目录: {directory_path}")
    print(f"找到 {total_files} 个JSON文件")
    
    for json_file in json_files:
        if rename_functionality_to_option_context(json_file):
            modified_files += 1
            print(f"  ✓ 已修改: {json_file.name}")
    
    print(f"完成: 共处理 {total_files} 个文件，成功修改 {modified_files} 个文件")
    
    return total_files, modified_files


def main():
    """主函数"""
    print("=" * 60)
    print("开始批量重命名JSON文件中的functionality字段为option_context")
    print("=" * 60)
    
    total_processed = 0
    total_modified = 0
    
    # 处理每个路径
    for path in PATHS:
        processed, modified = process_directory(path)
        total_processed += processed
        total_modified += modified
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print(f"总计: 处理 {total_processed} 个文件，成功修改 {total_modified} 个文件")
    print("=" * 60)


if __name__ == "__main__":
    main()

