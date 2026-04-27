# 评测详细说明

本文档详细说明 Caption 和 Grounding 任务的评测流程和指标计算方法。

## 一、任务类型概览

### 1. Grounding 任务（定位任务）
- **funcgnd**: 根据功能问题定位 UI 元素
- **descgnd**: 根据视觉描述定位 UI 元素

**任务形式**: 给定问题/描述 → 模型预测边界框 (bbox)

### 2. Captioning 任务（选择任务）
- **desccap**: 给定 bbox，从多个描述中选择最匹配的
- **funccap**: 给定 bbox，从多个功能问题中选择最匹配的

**任务形式**: 给定 bbox → 模型从多个选项中选择正确答案

---

## 二、Grounding 任务评测流程

### 2.1 数据加载

从 JSON 文件中加载数据，每个问题（question）包含：
- `question`: 问题文本（funcgnd）或描述文本（descgnd）
- `options`: 选项列表，每个选项包含：
  - `label`: 选项标签（A, B, C, ...）
  - `region_id`: 区域ID
  - `bbox`: 边界框 `[x_min, y_min, x_max, y_max]`（**像素坐标，未归一化**）
  - `description`: 视觉描述
  - `functionality`: 功能描述
  - `metrics`: 包含面积、密度等指标
- `correct_answer`: 正确答案的标签（如 "C"）
- `image_path`: 截图路径
- `image_size`: 图像尺寸 `[height, width]`

**数据转换过程**：
1. 找到 `correct_answer` 对应的选项
2. 从该选项的 `bbox` 字段提取真实边界框
3. 将 bbox 从像素坐标归一化到 0-1000 范围：
   ```python
   # 假设图像尺寸为 [H, W]
   normalized_bbox = [
       bbox[0] * 1000 / W,  # x_min
       bbox[1] * 1000 / H,  # y_min
       bbox[2] * 1000 / W,  # x_max
       bbox[3] * 1000 / H   # y_max
   ]
   ```

**对于 descgnd 任务**:
- 从正确答案选项的 `description` 字段提取描述
- 使用模板: `"Which element matches the following visual description: {description}?"`

### 2.2 模型推理

#### Prompt 构建
根据模型类型使用不同的 prompt 模板：

**Gemini 模型**:
```
You are a GUI expert. Given a screenshot and a question about locating a specific UI element, 
you need to identify the bounding box of the target element, which should be [ymin, xmin, ymax, xmax] 
normalized to 0-1000.

Question: {question}

Now analyze the screenshot and provide the bounding box for the target element:
```

**Claude 模型**:
```
You are a GUI expert. Given a screenshot and a question about locating a specific UI element, 
you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax].

Question: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:
```

**通用模型**:
```
You are a GUI expert. Given a screenshot and a question about locating a specific UI element, 
you need to identify the bounding box of the target element, which should be [xmin, ymin, xmax, ymax] 
normalized to 0-1000.

Question: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target element:
```

#### 模型调用
- 输入: prompt + 截图
- 输出: 模型返回的文本响应
- 温度: 0.0（确定性输出）
- 超时: 360 秒

### 2.3 结果解析

从模型响应中提取边界框，支持多种格式：

1. **JSON 格式**: `{"box_2d": [x1, y1, x2, y2]}` 或 `[{"box_2d": [x1, y1, x2, y2]}]`
2. **GLM-4.5 格式**: `<|begin_of_box|>[x1, y1, x2, y2]<|end_of_box|>`
3. **通用解析**: 使用正则表达式提取 `[x1, y1, x2, y2]` 格式

解析后的 bbox 会被调整到标准格式 `[x_min, y_min, x_max, y_max]`，归一化到 0-1000。

### 2.4 指标计算

#### IoU (Intersection over Union)
计算预测 bbox 和真实 bbox 的交并比：

```python
# 转换为 0-1 归一化
bbox1_norm = [x/1000 for x in bbox1]
bbox2_norm = [x/1000 for x in bbox2]

# 计算交集
x1 = max(bbox1_norm[0], bbox2_norm[0])
y1 = max(bbox1_norm[1], bbox2_norm[1])
x2 = min(bbox1_norm[2], bbox2_norm[2])
y2 = min(bbox1_norm[3], bbox2_norm[3])

intersection = (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0

# 计算并集
area1 = (bbox1_norm[2] - bbox1_norm[0]) * (bbox1_norm[3] - bbox1_norm[1])
area2 = (bbox2_norm[2] - bbox2_norm[0]) * (bbox2_norm[3] - bbox2_norm[1])
union = area1 + area2 - intersection

iou = intersection / union if union > 0 else 0.0
```

#### Center Accuracy
检查预测 bbox 的中心点是否在真实 bbox 内：

```python
center = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
center_acc = (gt_bbox[0] <= center[0] <= gt_bbox[2] and 
              gt_bbox[1] <= center[1] <= gt_bbox[3])
```

#### IoU 阈值准确率
计算不同 IoU 阈值下的准确率：
- `iou@0.1`: IoU ≥ 0.1 的样本比例
- `iou@0.3`: IoU ≥ 0.3 的样本比例
- `iou@0.5`: IoU ≥ 0.5 的样本比例
- `iou@0.7`: IoU ≥ 0.7 的样本比例
- `iou@0.9`: IoU ≥ 0.9 的样本比例

### 2.5 最终指标

- **Total**: 总样本数
- **Successful**: 成功解析 bbox 的样本数
- **Success Rate**: 成功率 = Successful / Total
- **Average IoU**: 所有成功样本的平均 IoU
- **Center Accuracy**: 中心点准确率
- **IoU Thresholds**: 各阈值下的准确率

---

## 三、Captioning 任务评测流程

### 3.1 数据加载

从 JSON 文件中加载数据，每个问题（question）包含：
- `target_region_id`: 目标区域的ID（被标注的元素）
- `annotated_image_path`: **已标注的图像路径**（图像上已用红色框标注了目标元素）
- `options`: 选项列表，每个选项包含：
  - `label`: 选项标签（A, B, C, ...）
  - `region_id`: 区域ID
  - `option_context`: **选项的上下文描述**（这是选项的核心内容）
  - `description`: 视觉描述
  - `functionality`: 功能描述
  - `metrics`: 包含面积、密度等指标
- `correct_answer`: 正确答案的标签（如 "C"）
- `image_path`: 原始截图路径（未标注）
- `annotated_image_size`: 标注图像的尺寸 `[height, width]`

**关键区别**：
- **Grounding 模式**：选项中有 `bbox`，需要模型预测 bbox
- **Captioning 模式**：有 `target_region_id` 和 `annotated_image_path`，选项中有 `option_context`，需要模型从选项中选择

**数据转换过程**：
1. 使用 `annotated_image_path` 作为输入图像（已标注红色框）
2. 从 `target_region_id` 对应的选项获取 bbox（用于 prompt 中）
3. 将 bbox 归一化到 0-1000 范围
4. 找到 `correct_answer` 对应的选项索引作为正确答案

**对于 desccap 任务**:
- 选项的 `option_context` 是描述性文本
- 从同一组的所有问题中收集描述作为选项
- 每个选项对应一个不同的 UI 元素描述

**对于 funccap 任务**:
- 选项的 `option_context` 是功能性问题
- 从同一组的所有问题中收集功能问题作为选项
- 每个选项对应一个不同的功能问题

### 3.2 模型推理

#### Prompt 构建

**desccap 任务 (Gemini)**:
```
You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, 
you need to select the visual description that best matches the highlighted element from the provided options.

The bounding box coordinates are [ymin, xmin, ymax, xmax] normalized to 0-1000: {bbox}

Options:
A: {option_context_A}
B: {option_context_B}
C: {option_context_C}
...

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching description:
```

**funccap 任务 (Claude)**:
```
You are a GUI expert. Given a screenshot with a UI element highlighted by a red bounding box, 
you need to select the functionality question that best matches the highlighted element from the provided options.

The bounding box coordinates are [xmin, ymin, xmax, ymax] normalized to 0-1000: {bbox}

Options:
A: {option_context_A}
B: {option_context_B}
C: {option_context_C}
...

Output format:
Answer: [option_label]

Now analyze the screenshot and select the best matching functionality question:
```

**注意**：
- 选项内容来自 `option_context` 字段，不是 `description` 或 `functionality`
- 使用 `annotated_image_path` 作为输入图像（图像上已用红色框标注了目标元素）

#### 图像处理
- 对于 Claude/Seed 模型：需要将标注图像 resize 到最大 2560px 并保存为临时文件
- 对于其他模型：直接使用 `annotated_image_path`

#### 模型调用
- 输入: prompt + 截图（bbox 会在图像上以红色框标注）
- 输出: 模型返回的文本响应
- 温度: 0.0
- 超时: 360 秒

### 3.3 结果解析

从模型响应中提取选项标签（A, B, C, ...）：

1. **标准格式**: `Answer: A` 或 `Answer: B`
2. **行首格式**: 行首的单个字母，如 `A: description...`

使用正则表达式提取：
```python
answer_match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
if not answer_match:
    answer_match = re.search(r'^([A-Z]):', response, re.MULTILINE | re.IGNORECASE)
```

### 3.4 指标计算

#### Accuracy
计算预测选项是否与正确答案匹配：

```python
correct_label = options[correct_option_idx]['label']
pred_option = extracted_label  # 从响应中提取
correct = (pred_option.upper() == correct_label.upper())
```

### 3.5 最终指标

- **Total**: 总样本数
- **Successful**: 成功解析选项的样本数
- **Success Rate**: 成功率 = Successful / Total
- **Accuracy**: 准确率 = Correct / Total
- **Correct**: 正确预测的样本数

---

## 四、重试机制

所有任务都支持最多 4 次重试：

1. **API 调用失败**: 网络错误、超时等
2. **解析失败**: 无法从响应中提取 bbox 或选项标签
3. **其他异常**: 未预期的错误

每次重试会记录错误信息，4 次重试后仍失败则标记为失败。

---

## 五、多进程处理

- 使用 `multiprocessing.Pool` 并行处理多个样本
- 每个 worker 进程独立初始化模型
- 支持检查点保存和续跑
- 实时显示处理进度和吞吐量

---

## 六、检查点机制

### 保存时机
- 每处理完一个样本后立即保存检查点
- 检查点包含：
  - `results`: 所有结果（成功和失败）
  - `processed_ids`: 成功处理的样本 ID
  - `metadata`: 模型名称、总样本数等

### 续跑机制
- 使用 `--load-latest` 自动加载最新检查点
- 跳过已成功处理的样本
- 重试之前失败的样本

---

## 七、输出格式

### 结果文件
保存为 JSON 格式，包含：
- 每个样本的详细结果
- 模型响应
- 计算得到的指标
- 错误信息（如果有）

### 控制台输出
- 实时显示每个样本的处理状态
- 最终显示汇总指标表格
- 包含 IoU 阈值准确率（grounding 任务）
- 包含分解指标（按数据集、密度等）

---

## 八、实际数据格式示例

### Grounding 模式数据结构

```json
{
  "result": {
    "questions": [
      {
        "question": "If you want to find a specific site search entry...",
        "options": [
          {
            "label": "A",
            "region_id": "1-8",
            "bbox": [71, 149, 342, 1080],  // 像素坐标
            "description": "This is a vertical navigation sidebar...",
            "functionality": "This sidebar allows users to navigate..."
          },
          {
            "label": "B",
            "region_id": "1-7",
            "bbox": [70, 29, 1920, 149],
            ...
          }
        ],
        "correct_answer": "C",
        "image_path": "/path/to/image.png",
        "image_size": [1080, 1920]
      }
    ]
  }
}
```

### Captioning 模式数据结构

```json
{
  "result": {
    "questions": [
      {
        "question": "What goal can be achieved by interacting with the circled element?",
        "target_region_id": "3-4",
        "annotated_image_path": "/path/to/annotated_image.png",  // 已标注红色框
        "options": [
          {
            "label": "A",
            "region_id": "1-8",
            "option_context": "Switch to a different settings category...",  // 选项内容
            "description": "This is a vertical navigation sidebar...",
            "functionality": "This sidebar allows users to navigate..."
          },
          {
            "label": "B",
            "region_id": "1-7",
            "option_context": "Navigate to a different website...",
            ...
          }
        ],
        "correct_answer": "C",
        "image_path": "/path/to/original_image.png",  // 原始图像
        "annotated_image_size": [1080, 1920]
      }
    ]
  }
}
```

### 关键字段说明

| 字段 | Grounding 模式 | Captioning 模式 |
|------|---------------|----------------|
| **输入图像** | `image_path` (原始截图) | `annotated_image_path` (已标注红色框) |
| **选项内容** | 选项中有 `bbox` | 选项中有 `option_context` |
| **目标元素** | `correct_answer` 对应的选项的 `bbox` | `target_region_id` 对应的区域 |
| **正确答案** | `correct_answer` 标签 | `correct_answer` 标签 |

---

## 九、关键代码位置

- **IoU 计算**: `calculate_iou()` (line 206)
- **Grounding 处理**: `process_entry()` (line 1473)
- **Captioning 处理**: `process_entry_desccap()` (line 1132), `process_entry_funcap()` (line 1303)
- **指标计算**: `calculate_metrics()` (line 1877)
- **数据加载**: `load_dataset_from_json()` (line 244)

