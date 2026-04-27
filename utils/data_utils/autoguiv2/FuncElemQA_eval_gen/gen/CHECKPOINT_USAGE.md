# 断点续传功能使用说明

## 功能说明

两个脚本现在都支持断点续传功能：
- `gen_region-func_multichoice-qa_hard.py`
- `gen_region-func_multichoice-qa_hard_multi.py`

当脚本运行被中断时（如容器重启、手动停止等），可以从上次保存的进度继续运行，避免重新处理已完成的问题。

## 使用方法

### 1. 首次运行（不使用断点续传）

```bash
python gen_region-func_multichoice-qa_hard.py \
    --api-key YOUR_API_KEY \
    --output-file /path/to/output.json \
    --debug

# 或者
python gen_region-func_multichoice-qa_hard_multi.py \
    --api-key YOUR_API_KEY \
    --output-file /path/to/output.json \
    --debug
```

### 2. 启用断点续传模式

添加 `--resume` 参数即可：

```bash
python gen_region-func_multichoice-qa_hard.py \
    --api-key YOUR_API_KEY \
    --output-file /path/to/output.json \
    --resume \
    --debug

# 或者
python gen_region-func_multichoice-qa_hard_multi.py \
    --api-key YOUR_API_KEY \
    --output-file /path/to/output.json \
    --resume \
    --debug
```

## 工作原理

### 自动保存
- 脚本每处理 **10 个问题** 就会自动保存一次进度
- 保存到 `--output-file` 指定的文件中
- 旧的文件会自动备份为 `.backup` 后缀

### 恢复机制
- 启用 `--resume` 后，脚本会检查输出文件是否存在
- 如果存在，会加载已处理的问题列表
- 自动跳过已处理的问题，只处理剩余的问题
- 最后将新处理的问题追加到已有结果中

### 安全机制
- 每次保存前会创建 `.backup` 备份文件
- 如果保存失败，可以从备份文件恢复

## 示例场景

### 场景1：容器即将重启

```bash
# 1. 在容器内运行脚本，启用断点续传
docker exec -it llama /bin/zsh
cd /mnt/nvme0n1p1/hongxin_li/highres_autogui/utils/data_utils/autoguiv2/FuncElemQA_eval_gen/gen

python gen_region-func_multichoice-qa_hard.py \
    --api-key YOUR_KEY \
    --output-file /mnt/vdb1/hongxin_li/AutoGUIv2/func_region_cap_hard.json \
    --resume \
    --debug

# 2. 容器重启后，再次运行相同命令
# 脚本会自动从上次保存的位置继续
```

### 场景2：手动中断后恢复

```bash
# 1. 运行中按 Ctrl+C 中断
# 输出: Saving checkpoint at 50/500

# 2. 再次运行（使用相同的命令），会看到：
# Resuming from checkpoint: 50 questions already processed
# Total questions: 500, Already processed: 50, To process: 450
```

## 查看进度

运行时会显示：

```
Resume mode enabled, loading checkpoint...
Resuming from checkpoint: 150 questions already processed

Loading questions from source directories...
Total questions: 500, Already processed: 150, To process: 350

Processing questions...
Processing: 100%|██████████| 350/350 [00:45<00:00,  7.71it/s]
```

## 注意事项

1. **必须使用相同的输出文件路径**
   - `--output-file` 参数必须与之前运行时相同
   
2. **API Key 和模型参数可以不同**
   - 只要输出文件路径相同，可以使用不同的 API key 或模型

3. **备份文件**
   - `.backup` 文件会在每次保存时更新
   - 如果最新的输出文件损坏，可以手动恢复备份

4. **完全重新运行**
   - 如果不想使用断点续传，删除 `--resume` 参数
   - 或者删除输出文件后重新运行

## 文件结构

```
输出目录/
├── func_region_cap_hard.json          # 主输出文件（包含checkpoint标记）
└── func_region_cap_hard.json.backup   # 自动备份文件
```

## 疑难解答

### Q: 为什么启用 --resume 但还是从头开始？
A: 检查输出文件路径是否正确，确保文件存在且可读

### Q: 如何查看已处理了多少问题？
A: 查看输出文件中的 `metadata.total_questions` 字段：
```bash
cat /path/to/output.json | jq '.metadata.total_questions'
```

### Q: 中断后丢失了多少进度？
A: 最多丢失 10 个问题的进度（checkpoint_interval = 10）

### Q: 如何修改保存间隔？
A: 修改脚本中的 `checkpoint_interval = 10` 为其他值（单位：问题数）

## 更新日期

2026-01-28
