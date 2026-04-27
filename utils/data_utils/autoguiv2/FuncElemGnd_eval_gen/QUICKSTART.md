# Quick Start Guide: Hugging Face Dataset Upload

The fastest way to convert your FuncElemGnd questions to a Hugging Face dataset.

## TL;DR

```bash
# Install dependencies
pip install datasets huggingface_hub

# Convert and upload to Hugging Face (with local cache)
./run_conversion.sh \
    -q /path/to/grounding_questions.json \
    -r username/dataset-name \
    -m
```

## Step-by-Step

### 1. Install Dependencies

```bash
pip install datasets huggingface_hub pillow
```

### 2. Setup Hugging Face Authentication

Choose one method:

**Option A: Login via CLI (easiest)**
```bash
huggingface-cli login
```

**Option B: Use token directly**
```bash
export HF_TOKEN=your_token_here
```

Get your token at: https://huggingface.co/settings/tokens

### 3. Run Conversion

**Using the shell script (easiest):**

```bash
./run_conversion.sh \
    --questions /path/to/grounding_questions.json \
    --repo-id username/my-dataset \
    --include-images
```

**Or using Python directly:**

```bash
python 3_convert_to_hf_dataset.py \
    --questions-file /path/to/grounding_questions.json \
    --repo-id username/my-dataset \
    --include-images
```

### 4. Verify Upload

Visit: `https://huggingface.co/datasets/username/my-dataset`

## Common Use Cases

### Case 1: Public Dataset with Images

```bash
./run_conversion.sh \
    -q /path/to/questions.json \
    -r username/dataset \
    -m
```

### Case 2: Private Dataset

```bash
./run_conversion.sh \
    -q /path/to/questions.json \
    -r username/dataset \
    -v -m
```

### Case 3: Custom Token

```bash
./run_conversion.sh \
    -q /path/to/questions.json \
    -r username/dataset \
    -t YOUR_TOKEN \
    -m
```

## Flags Explained

| Flag | Description |
|------|-------------|
| `-q, --questions` | Questions JSON from step 2 |
| `-r, --repo-id` | HuggingFace repo (username/name) |
| `-m, --include-images` | Embed images in dataset |
| `-v, --private` | Make dataset private |
| `-t, --token` | HF token (or use env var) |

## After Upload

### Load Your Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("username/my-dataset")

# View a sample
sample = dataset['train'][0]
print(sample['question'])
sample['image'].show()
```

### Explore with Example Script

```bash
python example_load_dataset.py \
    --from-hub \
    --repo-id username/my-dataset \
    --stats \
    --num-examples 3 \
    --visualize
```

## Typical Dataset Sizes

| Images | Questions | With Images | Without Images |
|--------|-----------|-------------|----------------|
| 100 | ~300 | ~500 MB | ~5 MB |
| 1,000 | ~3,000 | ~5 GB | ~50 MB |
| 10,000 | ~30,000 | ~50 GB | ~500 MB |

**Tip**: Always use `--include-images` when uploading to Hub for better usability.

## Troubleshooting

### "Authentication error"
- Run: `huggingface-cli login`
- Or set: `export HF_TOKEN=your_token`

### "Image not found"
- Check that `--image-dir` is correct
- Verify paths in questions JSON are valid

### "Slow upload"
- Normal for large datasets with images
- Upload happens in background, safe to wait

### "Out of memory"
- Try without `--include-images` first
- Process in smaller batches
- Use a machine with more RAM

## What's in the Dataset?

Each entry contains:
- ✅ GUI screenshot image
- ✅ Natural language question
- ✅ Action intent (precise localization)
- ✅ Bounding box coordinates
- ✅ Element functionality description
- ✅ Interaction outcomes (click, hover, etc.)
- ✅ OmniParser metadata (type, content, interactivity)
- ✅ Similar elements context

Perfect for:
- 🎯 Element grounding tasks
- 🤖 GUI agent training
- 📊 Visual reasoning benchmarks
- 🔍 UI understanding research

## Next Steps

1. ✅ Upload your dataset
2. 📝 Add a dataset card on HuggingFace
3. 🎉 Share with the community
4. 📊 Use for training/evaluation

For more details, see:
- `README_DATASET_CONVERSION.md` - Full documentation
- `WORKFLOW.md` - Complete pipeline guide
- `example_load_dataset.py` - Usage examples

## Example: OSWorld Dataset

```bash
# Assuming you've already run steps 1 and 2...

# Upload to HuggingFace
./run_conversion.sh \
    -q /mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/grounding_questions.json \
    -p -r myusername/osworld-funcelem \
    -m

# Done! 🎉
# View at: https://huggingface.co/datasets/myusername/osworld-funcelem
```

## Need Help?

- 📖 Read the full docs: `README_DATASET_CONVERSION.md`
- 🔧 Check the workflow: `WORKFLOW.md`
- 💻 Try examples: `example_load_dataset.py`
- 🐛 Debug: Check error messages and file paths

