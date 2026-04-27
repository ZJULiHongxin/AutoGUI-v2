# eval_funcregion_mp.py 模型支持扩展总结

## 修改日期
2026-01-29

## 新增支持的模型

本次修改为 `eval_funcregion_mp.py` 脚本添加了以下 6 个模型的评测支持：

1. **Hcompany/Holo2-8B** - 使用归一化坐标 (scale=1000)
2. **Hcompany/Holo1.5-7B** - 使用绝对像素坐标 (scale=-1)
3. **xlangai/OpenCUA-7B** - 使用绝对像素坐标
4. **inclusionAI/UI-Venus-Ground-7B** - 使用绝对像素坐标
5. **ritzzai/GUI-R1-7B** - 使用绝对像素坐标，支持思维链
6. **InfiX-ai/InfiGUI-G1-7B** - 使用绝对像素坐标，支持思维链

## 主要修改内容

### 1. 导入模块 (第 68-75 行)
```python
from utils.openai_utils.jedi import JEDI
```
- 添加了 JEDI 类的导入

### 2. 类型导入 (第 20 行)
```python
from typing import Dict, List, Any, Optional, Literal
```
- 添加了 `Literal` 类型支持 (用于 Holo 模型的 pydantic 定义)

### 3. Prompt 定义 (第 166-209 行)
新增了以下 Prompt 模板：

- **JEDI_PROMPT**: JEDI 模型的任务指令格式
- **HOLO_PROMPT**: Holo 系列模型的 JSON schema 格式指令
- **HOLO_BBOX_PROMPT**: Holo 模型的边界框格式指令
- **INFIGUIG1_SYSPROMPT**: InfiGUI-G1 的系统提示词（包含思维链要求）
- **INFIGUIG1_PROMPT**: InfiGUI-G1 的具体任务格式
- **GUIR1_PROMPT**: GUI-R1 的推理格式（包含思维链和动作格式）
- **UIVENUS_PROMPT**: UI-Venus 的边界框输出格式

### 4. 模型初始化 (init_worker 函数, 第 811-863 行)
在 `init_worker` 函数中添加了对 JEDI 模型的判断：
```python
elif 'jedi' in model.lower():
    base_url = 'https://afs3uxirrk48y8q5.us-east-1.aws.endpoints.huggingface.cloud/v1/'
    api_key = api_key or os.environ.get("HF_INFER_API_KEY", "EMPTY")
    cloud_model_class = JEDI
```

### 5. Prompt 选择逻辑 (process_entry 函数, 第 923-986 行)
在 `process_entry` 函数中添加了针对不同模型的 prompt 选择：

- **JEDI**: 调整图片到 1080p，使用专用指令格式
- **Holo**: 使用 JSON schema 格式的 prompt
- **OpenCUA**: 已有支持，保持原有逻辑
- **InfiGUI-G1**: 使用系统提示词 + 具体任务格式
- **GUI-R1**: 使用推理链格式
- **UI-Venus**: 使用简单的边界框输出格式
- **Claude/Qwen3**: 图片大小调整逻辑

### 6. 响应解析逻辑 (process_entry 函数, 第 1049-1165 行)
为每个模型添加了专门的响应解析逻辑：

```python
# Case 6: JEDI - 从 tool_call JSON 中提取坐标
elif 'jedi' in worker_model.model.lower():
    bbox_str = bbox_str[bbox_str.rfind('['):bbox_str.rfind(']')+1]
    raw_pred_bbox = pred_2_point(bbox_str, scale=scale, w=W, h=H)

# Case 7: Holo - 解析 JSON 格式的 click_absolute
elif 'holo' in worker_model.model.lower():
    act_dict = json.loads(bbox_str)
    raw_pred_bbox = [act_dict['x'], act_dict['y']]

# Case 8: OpenCUA - 从 pyautogui.click 中提取坐标
elif 'opencua' in worker_model.model.lower():
    raw_pred_bbox = [
        int(bbox_str.split('x=')[1].split(',')[0]),
        int(bbox_str.split('y=')[1].split(')')[0])
    ]

# Case 9: InfiGUI-G1 - 从 JSON 数组中提取 point_2d
elif 'infigui-g1' in worker_model.model.lower():
    points = json.loads(bbox_str)
    raw_pred_bbox = points[0]['point_2d']

# Case 10: GUI-R1 - 从 answer 标签中提取 point
elif 'gui-r1' in worker_model.model.lower():
    act_dict = eval(bbox_str[bbox_str.find("{'action"):bbox_str.rfind('}')+1])
    raw_pred_bbox = act_dict['point']
```

### 7. 临时文件清理 (第 1215-1221, 1227-1232 行)
改进了临时文件的清理逻辑：
- 使用 `is_resized` 标志跟踪是否创建了临时文件
- 在成功完成和异常情况下都确保清理临时文件

### 8. Scale 设置 (第 1254 行)
保持了原有的 scale 判断逻辑：
```python
if any(x in model_args['model'].lower() for x in ['claude', 'tars', 'jedi', 'holo1.5', 'opencua', 'infigui-g1', 'gui-r1', 'venus']):
    scale = -1  # 使用绝对坐标
else:
    scale = 1000  # 使用归一化坐标 (0-1000)
```

### 9. 模型列表更新 (第 1836-1853 行)
在命令行参数的模型列表中添加了新支持的模型：
```python
'xlangai/Jedi-7B-1080p',
'Hcompany/Holo2-8B',
'Hcompany/Holo1.5-7B',
'inclusionAI/UI-Venus-Ground-7B',
'ritzzai/GUI-R1-7B',
'InfiX-ai/InfiGUI-G1-7B',
```

## 技术要点

### 坐标系统
- **归一化坐标 (0-1000)**: Holo2-8B, Gemini, Qwen 等
- **绝对像素坐标**: Holo1.5-7B, JEDI, OpenCUA, InfiGUI-G1, GUI-R1, UI-Venus, Claude

### 思维链支持
- **InfiGUI-G1**: 使用 `<think>` 标签
- **GUI-R1**: 使用 `<think>` 和 `<answer>` 标签

### 图片调整
- **JEDI**: 调整到最大 1080px
- **Claude**: 调整到最大 2560px  
- **Qwen3**: 超过 3500px 时调整到 3200px

### API 配置
所有模型都使用各自的 API endpoint 和认证方式：
- JEDI: HuggingFace Inference API
- 其他模型: 使用现有的 API 配置框架

## 兼容性
- 保持了与原有模型的完全兼容
- 不影响现有的评测功能
- 支持所有原有的任务类型 (funcgnd, descgnd)

## 使用示例

```bash
# 评测 Holo2-8B
python eval_funcregion_mp.py --model Hcompany/Holo2-8B --max-workers 4

# 评测 GUI-R1-7B
python eval_funcregion_mp.py --model ritzzai/GUI-R1-7B --max-workers 4

# 评测 InfiGUI-G1
python eval_funcregion_mp.py --model InfiX-ai/InfiGUI-G1-7B --max-workers 4
```

## 注意事项
1. 确保安装了 `pydantic` 库（用于 Holo 模型）
2. 需要配置相应的 API keys（如 HF_INFER_API_KEY）
3. 某些模型可能需要较长的响应时间
4. 建议先用小样本测试 (`--sample-limit 10`)
