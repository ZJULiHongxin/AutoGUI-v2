# eval_funcregion_mp.py 技术细节说明

## 模型配置详解

### 1. API 配置方式

所有模型都通过 OpenAI 兼容的 API 方式访问本地 vllm 服务：

```python
# 在 init_worker 函数中
cloud_model_class = OpenAIModel
base_url = "http://localhost:<PORT>/v1"
api_key = "NOT_REQUIRED"  # 本地服务不需要真实 key
```

### 2. Prompt 配置

#### Holo2-8B / Holo1.5-7B
```python
# Holo2-8B: 输出归一化坐标 (0-1000)
HOLO_BBOX_PROMPT = """You are a GUI expert. Given a screenshot and {ref_tag} a specific UI region, 
you need to identify the bounding box of the target region, which should be [xmin, ymin, xmax, ymax] 
normalized to 0-1000. Note that the X-axis runs horizontally from left (0) to right (999), 
and the Y-axis runs vertically from top (0) to bottom (999).

{ref_placeholder}: {question}

Output format:
Box: [xmin, ymin, xmax, ymax]

Now analyze the screenshot and provide the bounding box for the target region:"""
```

**注意**: Holo1.5-7B 使用相同的 prompt，但会根据其训练特性输出绝对坐标。

#### OpenCUA-7B
```python
system_prompt = OPENCUA_SYSPROMPT  # "You are a GUI agent..."
prompt = f"{question} Click on the target region"

# 输出格式: pyautogui.click(x=<x>, y=<y>)
# 坐标类型: 绝对像素坐标
```

#### InfiGUI-G1-7B
```python
system_prompt = INFIGUIG1_SYSPROMPT  # 包含思维链要求
prompt = INFIGUIG1_PROMPT.format(new_width=W, new_height=H, instruction=question)

# 输出格式: [{"point_2d": [x, y], "label": "..."}]
# 坐标类型: 绝对像素坐标
```

#### GUI-R1-7B
```python
prompt = GUIR1_PROMPT.replace('{instruction}', question)

# 输出格式: <think>...</think> <answer>[{'action': 'click', 'point': [x, y], ...}]</answer>
# 坐标类型: 绝对像素坐标
```

#### UI-Venus-Ground-7B
```python
prompt = UIVENUS_PROMPT.format(instruction=question)

# 输出格式: [x1, y1, x2, y2]
# 坐标类型: 绝对像素坐标
```

### 3. 图像输入顺序 (image_first)

某些模型需要特定的输入顺序：

```python
image_first = any(x in worker_model.model.lower() for x in ['opencua', 'holo', 'infigui-g1'])
```

**说明**:
- `image_first=True`: 图像在 system prompt 之后、文本指令之前
- `image_first=False`: 标准顺序（文本 + 图像）

**适用模型**:
- OpenCUA-7B: ✅ 需要 `image_first=True`
- Holo2-8B: ✅ 需要 `image_first=True`
- InfiGUI-G1: ✅ 需要 `image_first=True`
- 其他模型: ❌ 使用标准顺序

### 4. 坐标系统 (scale)

```python
# Determine the scale
if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'jedi', 'holo1.5', 'opencua', 'infigui-g1', 'gui-r1', 'venus']):
    scale = -1  # 绝对像素坐标
else:
    scale = 1000  # 归一化坐标 (0-1000)
```

**坐标系统总结**:

| 模型 | Scale | 坐标类型 | 输出范围 |
|------|-------|---------|---------|
| Holo2-8B | 1000 | 归一化 | 0-1000 |
| Holo1.5-7B | -1 | 绝对 | 0-像素宽/高 |
| UI-TARS-1.5-7B | -1 | 绝对 | 0-像素宽/高 |
| OpenCUA-7B | -1 | 绝对 | 0-像素宽/高 |
| InfiGUI-G1-7B | -1 | 绝对 | 0-像素宽/高 |
| GUI-R1-7B | -1 | 绝对 | 0-像素宽/高 |
| UI-Venus | -1 | 绝对 | 0-像素宽/高 |

### 5. 响应解析逻辑

#### 思维链处理
```python
# 检测并提取思维链
if '</think>' in response:
    thinking = response.split('</think>')[0].replace('<think>', '').strip()
    bbox_str = response.split('</think>')[1].strip()
else:
    thinking, bbox_str = '', response.strip()
```

#### 模型特定解析

##### Holo2-8B / Holo1.5-7B
```python
elif 'holo' in worker_model.model.lower():
    # Holo outputs bbox format: "Box: [x1, y1, x2, y2]" or direct list
    raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)
```

##### OpenCUA-7B
```python
elif 'opencua' in worker_model.model.lower():
    # Extract from: pyautogui.click(x=3167, y=360)
    raw_pred_bbox = [
        int(bbox_str.split('x=')[1].split(',')[0]),
        int(bbox_str.split('y=')[1].split(')')[0])
    ]
```

##### InfiGUI-G1-7B
```python
elif 'infigui-g1' in worker_model.model.lower():
    # Parse: [{"point_2d": [1007, 924], "label": "..."}]
    points = json.loads(bbox_str)
    raw_pred_bbox = points[0]['point_2d']
```

##### GUI-R1-7B
```python
elif 'gui-r1' in worker_model.model.lower():
    # Parse: [{'action': 'click', 'point': [2200, 354], ...}]
    act_dict = eval(bbox_str[bbox_str.find("{'action"):bbox_str.rfind('}')+1])
    raw_pred_bbox = act_dict['point']
```

##### UI-TARS-1.5-7B
```python
elif 'tars' in worker_model.model.lower():
    # UI-TARS has special parsing logic
    # Format: "Action: click(start_box='(x,y)')"
    pass  # Handled by general parsing
```

##### GLM-4.5V
```python
elif '<|begin_of_box|>' in bbox_str:
    # Parse: <|begin_of_box|>[816, 162, 838, 172]<|end_of_box|>
    raw_pred_bbox = bbox_str.split('<|begin_of_box|>')[1].split('<|end_of_box|>')[0].strip()
    raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale)
```

##### OS-Atlas
```python
elif 'atlas' in worker_model.model.lower():
    # Parse: (576,12),(592,42) or with text prefix
    import re
    coord_pattern = re.compile(r'\((\d+),(\d+)\),\((\d+),(\d+)\)')
    match = coord_pattern.search(bbox_str)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        raw_pred_bbox = [x1, y1, x2, y2]
        raw_pred_bbox = pred_2_point(raw_pred_bbox, scale=scale)
```

### 6. 坐标转换与归一化

```python
# pred_2_point 函数处理坐标转换
def pred_2_point(bbox_str, scale=1000, w=None, h=None):
    """
    scale = 1000: 输入是归一化坐标 (0-1000)，转换为 0-1
    scale = -1: 输入是绝对坐标，转换为 0-1 (需要 w, h)
    """
    # 详细实现见 utils.data_utils.misc.pred_2_point
```

### 7. 评估指标计算

#### IoU (Intersection over Union)
```python
def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的 IoU
    输入: [xmin, ymin, xmax, ymax] (任意 scale)
    输出: IoU 值 (0-1)
    """
    # 自动归一化到 0-1 scale
    bbox1_norm = _normalize_bbox_0_1(bbox1)
    bbox2_norm = _normalize_bbox_0_1(bbox2)
    
    # 计算交集和并集
    intersection = calculate_intersection(bbox1_norm, bbox2_norm)
    union = area(bbox1_norm) + area(bbox2_norm) - intersection
    
    return intersection / union
```

#### Center Accuracy
```python
# 计算预测框的中心点
if len(pred_bbox_n) == 4:
    center = [(pred_bbox_n[0] + pred_bbox_n[2]) / 2, 
              (pred_bbox_n[1] + pred_bbox_n[3]) / 2]
elif len(pred_bbox_n) == 2:
    center = pred_bbox_n  # 点坐标直接作为中心

# 判断中心点是否在 GT 框内
center_acc = (gt_bbox_n[0] <= center[0] <= gt_bbox_n[2] and 
              gt_bbox_n[1] <= center[1] <= gt_bbox_n[3])
```

## 常见问题

### Q1: 为什么某些模型需要 image_first=True？
**A**: 这些模型在训练时使用了特定的输入顺序，将图像放在 system prompt 之后、用户指令之前。这是模型架构和训练方式决定的。

### Q2: 如何判断模型输出的是归一化坐标还是绝对坐标？
**A**: 
1. 检查模型训练文档和 prompt 要求
2. 运行小样本测试，观察输出数值范围
3. 如果输出值在 0-1000 范围，通常是归一化坐标
4. 如果输出值超过 1000，通常是绝对坐标

### Q3: 为什么 Holo2-8B 和 Holo1.5-7B 的 scale 不同？
**A**: 这是两个不同版本的模型：
- Holo2-8B: 新版本，训练时使用归一化坐标
- Holo1.5-7B: 旧版本，训练时使用绝对坐标

### Q4: pred_2_point 函数的作用是什么？
**A**: 该函数负责：
1. 从文本中提取坐标数值
2. 根据 scale 参数进行坐标转换
3. 处理各种格式的坐标表示（括号、方括号、逗号分隔等）

### Q5: 为什么需要临时图片文件？
**A**: 某些模型需要对输入图片进行预处理：
- JEDI: 调整到 1080p
- Claude: 调整到最大 2560px
- Qwen3: 超过 3500px 时调整到 3200px

## 性能优化建议

### 1. 内存管理
```python
# 及时清理临时文件
if is_resized and temp_img_path != image_path:
    try:
        os.remove(temp_img_path)
    except Exception:
        pass
```

### 2. 并发控制
- 本地 vllm: `--max-workers 1-4`（取决于 GPU 数量）
- 避免过多并发导致 OOM

### 3. 断点续传
```python
# 自动跳过已处理的样本
if entry['entry_id'] not in processed_ids:
    process_entry(entry)
```

### 4. 错误重试
```python
# 自动重试失败的请求（最多 4 次）
retry = 0
while retry < 4:
    try:
        # 尝试推理
        ...
        break
    except Exception as e:
        retry += 1
        if retry >= 4:
            # 记录失败
            result['error'] = str(e)
```

## 调试技巧

### 1. 启用详细日志
```python
# 查看每次推理的详细信息
print(f"[Worker {worker_id}] 🚀 Starting query | Entry: {entry_id} | "
      f"Model: {worker_model.model} | Image: {image_name_short}")
```

### 2. 检查中间结果
```python
# 保存在 result 中
result['prompt'] = prompt
result['response'] = response
result['thinking'] = thinking
result['bbox_pred_str'] = bbox_str
```

### 3. 小样本测试
```bash
# 先用 10 个样本测试
python eval_funcregion_mp.py --model "Hcompany/Holo2-8B" \
    --base-url http://localhost:11627/v1 \
    --sample-limit 10 --max-workers 1
```

### 4. 查看评测结果
```python
# 结果保存在 JSON 文件中
{
    "metadata": {...},
    "metrics": {
        "total": 1000,
        "successful": 950,
        "avg_iou": 0.756,
        "center_acc": 0.823,
        ...
    },
    "results": [
        {
            "entry_id": "...",
            "prompt": "...",
            "response": "...",
            "pred_bbox": [...],
            "gt_bbox": [...],
            "iou": 0.85,
            "center_acc": true,
            ...
        },
        ...
    ]
}
```
