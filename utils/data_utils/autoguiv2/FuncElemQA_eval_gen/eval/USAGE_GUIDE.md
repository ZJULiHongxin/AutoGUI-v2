# eval_funcregion_mp.py 使用指南

## 环境准备

### 1. 启动 vllm 服务

首先需要激活 vllm 环境并启动模型服务：

```bash
conda activate vllm
```

### 2. 部署模型

根据要评测的模型，使用以下命令启动 vllm 服务：

#### Holo2-8B
```bash
vllm serve "Hcompany/Holo2-8B" --dtype bfloat16 --port 11627 --tensor-parallel-size 1 --trust-remote-code
```

#### Holo1.5-7B  
```bash
vllm serve "Hcompany/Holo1.5-7B" --dtype bfloat16 --port 11628 --tensor-parallel-size 1 --trust-remote-code
```

#### OpenCUA-7B
```bash
vllm serve "xlangai/OpenCUA-7B" --dtype bfloat16 --port 11629 --tensor-parallel-size 1 --trust-remote-code
```

#### InfiGUI-G1-7B
```bash
vllm serve "InfiX-ai/InfiGUI-G1-7B" --dtype bfloat16 --port 11630 --tensor-parallel-size 1 --trust-remote-code
```

#### GUI-R1-7B
```bash
vllm serve "ritzzai/GUI-R1-7B" --dtype bfloat16 --port 11631 --tensor-parallel-size 1 --trust-remote-code
```

#### UI-Venus-Ground-7B
```bash
vllm serve "inclusionAI/UI-Venus-Ground-7B" --dtype bfloat16 --port 11632 --tensor-parallel-size 1 --trust-remote-code
```

## 运行评测

### 基本命令格式

```bash
python eval_funcregion_mp.py \
    --model <MODEL_NAME> \
    --base-url http://localhost:<PORT>/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4 \
    --sample-limit 100
```

### 具体示例

#### 评测 Holo2-8B
```bash
python eval_funcregion_mp.py \
    --model "Hcompany/Holo2-8B" \
    --base-url http://localhost:11627/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

#### 评测 Holo1.5-7B
```bash
python eval_funcregion_mp.py \
    --model "Hcompany/Holo1.5-7B" \
    --base-url http://localhost:11628/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

#### 评测 OpenCUA-7B
```bash
python eval_funcregion_mp.py \
    --model "xlangai/OpenCUA-7B" \
    --base-url http://localhost:11629/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

#### 评测 InfiGUI-G1-7B
```bash
python eval_funcregion_mp.py \
    --model "InfiX-ai/InfiGUI-G1-7B" \
    --base-url http://localhost:11630/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

#### 评测 GUI-R1-7B
```bash
python eval_funcregion_mp.py \
    --model "ritzzai/GUI-R1-7B" \
    --base-url http://localhost:11631/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

#### 评测 UI-Venus-Ground-7B
```bash
python eval_funcregion_mp.py \
    --model "inclusionAI/UI-Venus-Ground-7B" \
    --base-url http://localhost:11632/v1 \
    --api-key NOT_REQUIRED \
    --max-workers 4
```

### 使用 HuggingFace 数据集
```bash
python eval_funcregion_mp.py \
    --model "Hcompany/Holo2-8B" \
    --base-url http://localhost:11627/v1 \
    --api-key NOT_REQUIRED \
    --hf-dataset-id HongxinLi/AutoGUIv2-FuncRegionGnd \
    --hf-split test \
    --max-workers 4
```

### 测试小样本
```bash
python eval_funcregion_mp.py \
    --model "Hcompany/Holo2-8B" \
    --base-url http://localhost:11627/v1 \
    --api-key NOT_REQUIRED \
    --sample-limit 10 \
    --max-workers 1
```

## 重要说明

### 1. API 配置
- **base_url**: 必须设置为 `http://localhost:<PORT>/v1`，其中 PORT 是 vllm 服务的端口号
- **api-key**: 可以设置为任意值（如 `NOT_REQUIRED`），vllm 本地服务不需要真实的 API key
- **model**: 必须使用完整的 HuggingFace 模型名称（如 `Hcompany/Holo2-8B`）

### 2. 坐标系统
不同模型输出的坐标系统不同，脚本会自动处理：

- **归一化坐标 (0-1000)**: Holo2-8B
- **绝对像素坐标**: UI-TARS-1.5-7B, OpenCUA-7B, Holo1.5-7B

其他模型（InfiGUI-G1, GUI-R1, UI-Venus）基于 QwenVL 微调，保留了通用指令跟随能力，能够根据 prompt 输出相应格式。

### 3. 图像输入顺序
某些模型需要特定的输入顺序：
- **OpenCUA-7B** 和 **Holo2-8B**: 图像必须在 system prompt 之后、文本指令之前
- 脚本已自动处理这些要求（通过 `image_first=True` 参数）

### 4. Prompt 选择
- **Holo 模型**: 使用 `HOLO_BBOX_PROMPT` 输出边界框
- **OpenCUA**: 使用专门的系统提示词
- **InfiGUI-G1**: 支持思维链输出（`<think>` 标签）
- **GUI-R1**: 支持推理链输出
- **其他模型**: 使用通用的 `GENERIC_PROMPT`

## 故障排查

### 错误: model_not_found
```
Error code: 503 - {'error': {'code': 'model_not_found', 'message': '分组 default 下模型 xxx 无可用渠道'}}
```

**原因**: 没有正确设置 `--base-url` 参数，系统尝试通过默认 API 访问模型

**解决**: 确保添加 `--base-url http://localhost:<PORT>/v1` 参数

### 错误: Connection refused
**原因**: vllm 服务未启动或端口错误

**解决**: 
1. 检查 vllm 服务是否正在运行
2. 确认端口号是否正确
3. 使用 `curl http://localhost:<PORT>/v1/models` 测试服务是否可访问

### 评测速度慢
**建议**:
1. 减少 `--max-workers` 数量（本地 vllm 服务，建议使用 1-4）
2. 使用 `--sample-limit` 先测试小样本
3. 检查 GPU 资源是否充足

## 输出结果

评测结果保存在：
```
utils/data_utils/autoguiv2/FuncElemQA_eval_gen/eval/eval_results/<task_type>/<model_name>/<timestamp>.json
```

结果包含：
- 整体指标（IoU, Center Accuracy 等）
- 按密度分类的指标
- 按区域类型分类的指标
- 按面积分类的指标
- 每个样本的详细结果

## 性能优化建议

1. **GPU 资源**:
   - 确保有足够的 GPU 内存
   - 根据 GPU 数量调整 `--tensor-parallel-size`

2. **并行度**:
   - 本地部署建议 `--max-workers 1-4`
   - 远程 API 可以增加 workers 数量

3. **批处理**:
   - 先用小样本测试（`--sample-limit 10`）
   - 确认没问题后再完整评测

4. **断点续传**:
   - 使用 `--load-latest` 自动加载最新 checkpoint
   - 或使用 `--checkpoint-file` 指定特定 checkpoint
