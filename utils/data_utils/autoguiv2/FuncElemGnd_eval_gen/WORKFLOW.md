# Complete FuncElemGnd Dataset Generation Workflow

This document outlines the complete workflow for generating and publishing the FuncElemGnd (Functional Element Grounding) dataset.

## Overview

The pipeline consists of three main scripts:

1. **`1_make_func_elemgnd_samples.py`**: Detects visually similar elements with different functionality
2. **`2_generate_func_elemgnd_questions.py`**: Generates natural language questions for element grounding
3. **`3_convert_to_hf_dataset.py`**: Converts to Hugging Face dataset format and uploads

## Prerequisites

### Software Requirements

```bash
# Core dependencies
pip install opencv-python pillow numpy tqdm colorama

# For question generation (OpenAI/VLM)
pip install openai

# For Hugging Face upload
pip install datasets huggingface_hub
```

### Data Requirements

Your dataset directory should have this structure:

```
dataset_root/
├── images/
│   ├── screenshot1.png
│   ├── screenshot2.png
│   └── ...
├── omniparser/
│   ├── screenshot1.json
│   ├── screenshot2.json
│   └── ...
└── omniparser_embeddings/
    ├── screenshot1.npz
    ├── screenshot2.npz
    └── ...
```

- **images/**: GUI screenshots (PNG format)
- **omniparser/**: OmniParser detection results (JSON format)
- **omniparser_embeddings/**: DINO-v3 embeddings with similar groups (NPZ format)

## Step-by-Step Workflow

### Step 1: Detect Similar Elements

This step identifies visually similar GUI elements that have different functionality.

```bash
python 1_make_func_elemgnd_samples.py \
    --image-src-dir /path/to/dataset/images \
    --output-file /path/to/dataset/FuncElemGnd/similar_elements_anno.json \
    --api-key YOUR_API_KEY \
    --base-url YOUR_API_BASE_URL \
    --model gemini-2.5-pro-thinking \
    --workers 4 \
    --repeats 3 \
    --max-retries 3 \
    --debug
```

**Output**: `similar_elements_anno.json` containing detected similar element groups

**What it does**:
- Loads DINO-v3 embeddings to find initial similar element candidates
- Uses VLM to verify visual similarity and identify functional differences
- Extracts detailed functionality and interaction outcomes for each element
- Generates revised bounding boxes for accurate localization

**Key Parameters**:
- `--workers`: Number of parallel processes (4-8 recommended)
- `--repeats`: How many times to query the VLM per image (increases recall)
- `--model`: VLM model to use (e.g., gpt-4o, gemini-2.5-pro-thinking)

### Step 2: Generate Grounding Questions

This step generates natural language questions for identifying specific elements.

```bash
python 2_generate_func_elemgnd_questions.py \
    --image-src-dir /path/to/dataset/images \
    --detection-file /path/to/dataset/FuncElemGnd/similar_elements_anno.json \
    --output-file /path/to/dataset/FuncElemGnd/grounding_questions.json \
    --api-key YOUR_API_KEY \
    --base-url YOUR_API_BASE_URL \
    --model gemini-2.5-pro-thinking \
    --workers 4 \
    --max-retries 3 \
    --debug
```

**Output**: `grounding_questions.json` containing questions for each element

**What it does**:
- Takes the detected similar element groups from Step 1
- Generates natural, task-oriented questions for each element
- Creates action intents for precise element localization
- Covers multiple interaction types (clicking, hovering, typing, etc.)

**Key Parameters**:
- `--detection-file`: Output from Step 1
- `--workers`: Number of parallel processes

### Step 3: Convert to Hugging Face Dataset

This step converts the questions into a structured dataset and uploads to Hugging Face.

#### Option A: Upload to Hugging Face Hub

```bash
python 3_convert_to_hf_dataset.py \
    --image-src-dir /path/to/dataset/images \
    --questions-file /path/to/dataset/FuncElemGnd/grounding_questions.json \
    --push-to-hub \
    --repo-id username/funcelem-grounding \
    --hf-token YOUR_HF_TOKEN \
    --include-images
```

#### Option B: Save Locally

```bash
python 3_convert_to_hf_dataset.py \
    --image-src-dir /path/to/dataset/images \
    --questions-file /path/to/dataset/FuncElemGnd/grounding_questions.json \
    --save-local \
    --output-dir /path/to/dataset/FuncElemGnd/hf_dataset \
    --include-images
```

#### Option C: Both

```bash
python 3_convert_to_hf_dataset.py \
    --image-src-dir /path/to/dataset/images \
    --questions-file /path/to/dataset/FuncElemGnd/grounding_questions.json \
    --save-local \
    --output-dir /path/to/dataset/FuncElemGnd/hf_dataset \
    --push-to-hub \
    --repo-id username/funcelem-grounding \
    --hf-token YOUR_HF_TOKEN \
    --include-images
```

**Output**: Hugging Face dataset with all metadata

**What it does**:
- Loads questions and matches them with OmniParser detections
- Creates one dataset entry per question
- Includes images, bounding boxes, and all metadata
- Uploads to Hugging Face Hub (optional)

**Key Parameters**:
- `--include-images`: Embeds actual images (recommended for Hub upload)
- `--private`: Makes the dataset private on Hugging Face
- `--repo-id`: Your Hugging Face repository (username/dataset-name)

## Complete Example: OSWorld Dataset

Here's a complete example processing the OSWorld dataset:

```bash
# Set your API credentials
export OPENAI_API_KEY=your_key_here
export OPENAI_API_BASE=your_base_url_here

# Define paths
DATASET_ROOT="/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g"
IMAGE_DIR="${DATASET_ROOT}/images"
OUTPUT_DIR="${DATASET_ROOT}/FuncElemGnd"

# Step 1: Detect similar elements (takes ~1-2 hours for 1000 images with 4 workers)
python 1_make_func_elemgnd_samples.py \
    --image-src-dir ${IMAGE_DIR} \
    --output-file ${OUTPUT_DIR}/similar_elements_anno.json \
    --api-key ${OPENAI_API_KEY} \
    --base-url ${OPENAI_API_BASE} \
    --model gemini-2.5-pro-thinking \
    --workers 4 \
    --repeats 3 \
    --max-retries 3

# Step 2: Generate questions (takes ~30-60 min for 1000 images with 4 workers)
python 2_generate_func_elemgnd_questions.py \
    --image-src-dir ${IMAGE_DIR} \
    --detection-file ${OUTPUT_DIR}/similar_elements_anno.json \
    --output-file ${OUTPUT_DIR}/grounding_questions.json \
    --api-key ${OPENAI_API_KEY} \
    --base-url ${OPENAI_API_BASE} \
    --model gemini-2.5-pro-thinking \
    --workers 4 \
    --max-retries 3

# Step 3: Convert and upload (takes ~5-10 min depending on dataset size)
# Always saves locally first as cache, then uploads to HuggingFace
python 3_convert_to_hf_dataset.py \
    --questions-file ${OUTPUT_DIR}/grounding_questions.json \
    --repo-id myusername/osworld-funcelem-grounding \
    --hf-token ${HF_TOKEN} \
    --include-images

echo "✅ Dataset generation complete!"
echo "🔗 View at: https://huggingface.co/datasets/myusername/osworld-funcelem-grounding"
```

## Exploring the Dataset

After creating the dataset, you can explore it using the example script:

```bash
# Load from Hugging Face Hub
python example_load_dataset.py \
    --from-hub \
    --repo-id username/funcelem-grounding \
    --stats \
    --num-examples 3 \
    --visualize

# Load from local disk
python example_load_dataset.py \
    --local-path /path/to/dataset/FuncElemGnd/hf_dataset \
    --stats \
    --interactive
```

Or programmatically in Python:

```python
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("username/funcelem-grounding", split='train')

# Explore a sample
sample = dataset[0]
print(f"Question: {sample['question']}")
print(f"Action: {sample['action_type']}")
print(f"Bbox: {sample['target_element_bbox']}")

# Display image with bounding box
img = sample['image']
img.show()

# Parse JSON fields
interactions = json.loads(sample['target_element_interaction_outcomes'])
print(f"Interactions: {interactions}")
```

## Dataset Schema

Each entry in the final dataset contains:

| Field | Type | Description |
|-------|------|-------------|
| `image` | Image | The GUI screenshot |
| `image_path` | string | Path to the image file |
| `question` | string | Natural language question |
| `action_intent` | string | Precise action description |
| `action_type` | string | Type of interaction (clicking, hovering, etc.) |
| `target_element_bbox` | float[] | Bounding box [x_min, y_min, x_max, y_max] (0-1000) |
| `target_element_functionality` | string | Unique functionality description |
| `visual_similarity` | string | Why elements look similar |
| `num_similar_elements` | int | Number of similar elements in group |
| `omniparser_type` | string | Element type from OmniParser |
| `omniparser_content` | string | Text content from OmniParser |
| `omniparser_interactivity` | bool | Whether element is interactive |
| `similar_elements` | string (JSON) | Info about all similar elements |
| ... | ... | See README for complete schema |

## Troubleshooting

### Common Issues

**1. "OmniParser data not found"**
- Ensure OmniParser results are in `dataset_root/omniparser/`
- Check file naming matches image files (same stem)

**2. "DINO embeddings not found"**
- Run the embedding generation script first
- Embeddings should be in `dataset_root/omniparser_embeddings/`

**3. API rate limits**
- Reduce `--workers` to 1-2
- Increase delays between requests
- Use checkpoint/resume functionality (automatic)

**4. Out of memory errors**
- Reduce `--workers`
- Process images in smaller batches
- Don't use `--include-images` for very large datasets

**5. Authentication errors (HF)**
- Run `huggingface-cli login`
- Or provide `--hf-token` explicitly
- Ensure token has write permissions

## Performance Tips

1. **Parallel Processing**: Use 4-8 workers for optimal speed
2. **Checkpointing**: Scripts save progress automatically; safe to interrupt
3. **Batch Processing**: Process dataset in chunks if very large
4. **Model Selection**: Faster models reduce processing time but may affect quality
5. **Caching**: Reuse similarity detection results for different question generation runs

## Citation

If you use this dataset generation pipeline, please cite:

```bibtex
@article{funcelem2025,
  title={FuncElemGnd: Functional Element Grounding for GUI Understanding},
  author={Your Name},
  year={2025}
}
```

## Support

For issues or questions:
- Check the individual README files for each script
- Review the example scripts
- Open an issue on GitHub

