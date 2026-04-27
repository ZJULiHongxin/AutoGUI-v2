# FuncRegionGnd Task Revising UI

区域级别（Region-level）grounding任务修订工具。

## 功能特点

- 📋 浏览和修订区域级别的grounding问题
- 🎯 修改问题文本（question）
- ✅ 修改正确答案（correct_answer: A/B/C/D）
- 📦 修改正确答案对应的边界框（bbox）
- 🔍 可视化所有选项（options）及其区域类型（region_type）
- 🖱️ 交互式bbox修正工具（支持边缘/角点自动吸附）
- 🗑️ 标记废弃样本（abandoned）

## 数据结构

### 输入数据格式

期望数据位于：`{datasets_root}/{dataset}/FuncRegion/grounding_mode/*_result.json`

每个JSON文件结构：
```json
{
  "result": {
    "questions": [
      {
        "question": "问题文本",
        "correct_answer": "C",
        "options": [
          {
            "label": "A",
            "region_id": "2-9",
            "bbox": [x1, y1, x2, y2],
            "region_type": "Toolbar / Action Bar",
            "description": "区域描述",
            "functionality": "功能说明"
          },
          ...
        ],
        "explanation": "答案解释",
        "image_path": "/path/to/image.png",
        "image_size": [width, height]
      }
    ]
  }
}
```

### 修订数据格式

修订保存在：`{datasets_root}/{dataset}/FuncRegion/grounding_questions_corrections.json`

```json
{
  "filename_result.json__0": {
    "modified_question": "修改后的问题",
    "modified_correct_answer": "B",
    "modified_bbox": [x1, y1, x2, y2],
    "abandoned": false,
    "updated_at": "2025-12-28T10:30:00"
  }
}
```

## 使用方法

### 1. 安装依赖

```bash
cd /mnt/nvme0n1p1/hongxin_li/highres_autogui
pip install -r requirements-webui.txt
```

### 2. 启动服务

```bash
python3 -m utils.data_utils.autoguiv2.monitor.revise_regiongnd_tasks \
  --datasets-root /mnt/vdb1/hongxin_li/AutoGUIv2 \
  --host 0.0.0.0 \
  --port 17806
```

### 3. 访问界面

打开浏览器访问：`http://localhost:17806`

## 界面操作

### 主界面

1. **Dataset选择器**：选择数据集（osworld_g, screenspot_pro等）
2. **Sample选择器**：选择具体样本（每个问题是一个样本）
   - 🆕 = 未修改
   - ✏️ = 已修改
   - 🗑️ = 已废弃
3. **导航按钮**：Previous/Next快速切换样本

### 修订操作

1. **修改问题文本**：直接在"Question Text"文本框中编辑
2. **修改正确答案**：点击A/B/C/D按钮选择新的正确答案
   - 选中的答案会高亮显示（绿色）
   - 切换答案时，bbox会自动更新为对应选项的bbox
3. **查看所有选项**：Options区域显示所有候选区域及其类型
4. **修改BBox**：
   - 手动编辑：在"Modified Box"输入框中输入，点击"Apply"
   - 交互式修正：点击"Fix BBox"按钮打开精确标注工具
5. **标记废弃**：勾选"Abandoned"复选框
6. **自动保存**：切换样本时自动保存当前修改

### BBox修正工具

点击"Fix BBox"按钮后：

1. **基本操作**：
   - 点击"Select Top-Left"，然后在图像上点击左上角
   - 点击"Select Bottom-Right"，然后在图像上点击右下角
   - 或直接在输入框中输入坐标

2. **智能吸附**（需要OpenCV.js加载完成）：
   - **长按（500ms）**：自动吸附到最近的边缘线
   - **Ctrl + 长按**：自动吸附到最近的角点

3. **缩放预览**：鼠标移动时显示3倍放大的局部区域

4. **确认/取消**：
   - 点击"Confirm"应用修改
   - 点击"Cancel"放弃修改

## 与元素级别工具的区别

| 特性 | 元素级别 (revise_elemgnd_tasks) | 区域级别 (revise_regiongnd_tasks) |
|------|--------------------------------|----------------------------------|
| 数据源 | FuncElemGnd/grounding_questions.json | FuncRegion/grounding_mode/*_result.json |
| 样本单位 | 每个element的每个question | 每个question（包含多个options） |
| 修改内容 | 多个action_type的问题文本 + bbox | 单个问题文本 + correct_answer + bbox |
| 答案格式 | 无选项概念 | A/B/C/D选择题 |
| 端口 | 17805 | 17806 |

## 技术细节

- **后端**：FastAPI + Uvicorn
- **前端**：原生JavaScript + D3.js + OpenCV.js
- **图像处理**：PIL (Python) + OpenCV.js (前端)
- **bbox格式**：
  - 原始：像素坐标 [x1, y1, x2, y2]
  - 归一化：0-1000范围 [x1, y1, x2, y2]

## 注意事项

1. **自动保存**：切换样本时会自动保存，无需手动保存
2. **无变化检测**：如果修改后与原始数据完全相同，会自动删除修订记录
3. **OpenCV加载**：首次打开页面时OpenCV.js需要几秒钟加载，加载完成前无法使用智能吸附功能
4. **并发编辑**：不支持多人同时编辑同一数据集，会导致覆盖问题

## 故障排查

### 问题：No datasets found

**原因**：数据目录结构不正确

**解决**：确保数据位于 `{datasets_root}/{dataset}/FuncRegion/grounding_mode/` 且包含 `*_result.json` 文件

### 问题：OpenCV is not loaded yet

**原因**：OpenCV.js尚未加载完成

**解决**：等待几秒钟或刷新页面

### 问题：图像无法显示

**原因**：JSON文件中的 `image_path` 不正确

**解决**：检查 `image_path` 字段是否指向存在的图像文件

## 开发者信息

- 基于 `revise_elemgnd_tasks.py` 改造
- 保持了相同的UI风格和交互逻辑
- 适配了区域级别数据结构


