# 数据使用指南

本指南说明如何使用本地 JSON 数据和 HuggingFace 缓存数据进行评估。

## 数据路径说明

### 本地 JSON 数据

#### Captioning 模式数据（用于 desccap 和 funccap 任务）
- `/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/captioning_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/captioning_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/captioning_mode`

#### Grounding 模式数据（用于 funcgnd 和 descgnd 任务）
- `/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/grounding_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/screenspot_pro/FuncRegion/grounding_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/agentnet/FuncRegion/grounding_mode`
- `/mnt/vdb1/hongxin_li/AutoGUIv2/amex/FuncRegion/grounding_mode`

### HuggingFace 缓存数据

- `/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionCap` - Captioning 数据集缓存
- `/mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionGnd` - Grounding 数据集缓存

## 使用示例

### 1. 使用本地 JSON 文件（Grounding 任务）

#### 功能定位任务 (funcgnd)
```bash
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/grounding_mode/*_result.json" \
  --task-type funcgnd \
  --field-type functionality \
  --model gpt-4o \
  --max-workers 4
```

#### 描述定位任务 (descgnd)
```bash
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/grounding_mode/*_result.json" \
  --task-type descgnd \
  --field-type description \
  --model gemini-2.5-pro-thinking \
  --max-workers 4
```

### 2. 使用本地 JSON 文件（Captioning 任务）

#### 描述选择任务 (desccap)
```bash
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/captioning_mode/*_result.json" \
  --task-type desccap \
  --field-type description \
  --model claude-sonnet-4-5 \
  --max-workers 4
```

#### 功能选择任务 (funccap)
```bash
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/captioning_mode/*_result.json" \
  --task-type funccap \
  --field-type functionality \
  --model gpt-4o \
  --max-workers 4
```

### 3. 使用单个数据集

#### osworld_g 数据集
```bash
# Grounding 任务
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/grounding_mode/*_result.json" \
  --task-type funcgnd \
  --field-type functionality \
  --model gpt-4o

# Captioning 任务
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncRegion/captioning_mode/*_result.json" \
  --task-type desccap \
  --field-type description \
  --model gpt-4o
```

### 4. 使用 HuggingFace 缓存数据

#### 使用 FuncRegionGnd 缓存
```bash
python eval_funcregion_mp.py \
  --hf-dataset-id HongxinLi/AutoGUIv2-FuncRegionGnd \
  --hf-cache-dir /mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionGnd \
  --task-type funcgnd \
  --field-type functionality \
  --model gpt-4o \
  --hf-split test
```

#### 使用 FuncRegionCap 缓存
```bash
python eval_funcregion_mp.py \
  --hf-dataset-id HongxinLi/AutoGUIv2-FuncRegionCap \
  --hf-cache-dir /mnt/vdb1/hongxin_li/AutoGUIv2/hf_dataset_cache/FuncRegionCap \
  --task-type desccap \
  --field-type description \
  --model gpt-4o \
  --hf-split test
```

### 5. 使用检查点续跑

```bash
python eval_funcregion_mp.py \
  --questions-file "/mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/grounding_mode/*_result.json" \
  --task-type funcgnd \
  --field-type functionality \
  --model gpt-4o \
  --load-latest  # 自动加载最新检查点
```

## 数据格式说明

### 新格式（您的数据）

脚本现在支持以下两种数据格式：

#### 格式 1: 新格式（您的数据）
```json
{
  "metadata": {...},
  "image_key": "osworld_g/images/0FOB4CLBT2.png",
  "result": {
    "image_path": "/path/to/image.png",
    "questions": [
      {
        "question": "If you want to open a new file...",
        "options": [
          {
            "label": "A",
            "region_id": "2-9",
            "bbox": [448, 1002, 1474, 1023],  // Grounding 模式有 bbox
            "metrics": {...},
            "description": "...",
            "functionality": "..."
          }
        ],
        "correct_answer": "A",
        "target_region_id": "2-9",
        "group_id": 1
      }
    ]
  }
}
```

#### 格式 2: 旧格式（兼容）
```json
{
  "results": {
    "image_name": {
      "image_path": "/path/to/image.png",
      "generated": [
        {
          "group_index": 1,
          "questions": [...],
          "elements": [...]
        }
      ]
    }
  }
}
```

## 注意事项

1. **Captioning 任务的 bbox**: Captioning 模式的数据中选项没有 bbox，脚本会自动从对应的 grounding 文件中查找 bbox。确保 grounding 和 captioning 数据在同一目录结构中。

2. **Glob 模式**: 使用 `*` 通配符可以匹配多个文件或目录，例如：
   - `/*/FuncRegion/grounding_mode/*_result.json` - 匹配所有数据集的 grounding 文件
   - `osworld_g/FuncRegion/captioning_mode/*_result.json` - 匹配单个数据集的所有文件

3. **任务类型与数据模式对应**:
   - `funcgnd`, `descgnd` → 使用 `grounding_mode` 数据
   - `desccap`, `funccap` → 使用 `captioning_mode` 数据

4. **字段类型**:
   - `functionality` → 使用功能问题
   - `description` → 使用视觉描述

5. **HuggingFace 缓存**: 如果使用 `--hf-cache-dir`，脚本会直接从缓存加载，无需重新下载。

## 输出结果

评估结果会保存在：
```
eval_results/{task_type}/{model_name}/{timestamp}_results.json
```

检查点文件会保存在：
```
eval_results/{task_type}/{model_name}/{timestamp}_checkpoint.json
```

