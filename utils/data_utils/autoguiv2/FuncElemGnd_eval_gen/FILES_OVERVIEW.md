# Files Overview

This document provides a quick reference for all scripts and documentation in the FuncElemGnd dataset generation pipeline.

## Scripts (Execution Order)

### Core Pipeline Scripts

1. **`1_make_func_elemgnd_samples.py`** ⭐
   - **Purpose**: Detect visually similar elements with different functionality
   - **Input**: GUI screenshots, OmniParser data, DINO embeddings
   - **Output**: `similar_elements_anno.json`
   - **Runtime**: ~1-2 hours for 1000 images (4 workers)
   
2. **`2_generate_func_elemgnd_questions.py`** ⭐
   - **Purpose**: Generate grounding questions for similar elements
   - **Input**: Output from script 1, GUI screenshots
   - **Output**: `grounding_questions.json`
   - **Runtime**: ~30-60 minutes for 1000 images (4 workers)

3. **`3_convert_to_hf_dataset.py`** ⭐ **[NEW]**
   - **Purpose**: Convert questions to HuggingFace dataset format and upload
   - **Input**: Output from script 2, GUI screenshots, OmniParser data
   - **Output**: Local cache + HuggingFace dataset
   - **Runtime**: ~5-10 minutes for 1000 images

### Utility Scripts

4. **`run_conversion.sh`** **[NEW]**
   - **Purpose**: Convenient shell wrapper for script 3
   - **Usage**: `./run_conversion.sh -q questions.json -r username/dataset -m`
   - **Benefit**: Simplified command-line interface with validation

5. **`example_load_dataset.py`** **[NEW]**
   - **Purpose**: Demonstrate how to load and explore the dataset
   - **Features**: Statistics, visualization, interactive mode
   - **Usage**: `python example_load_dataset.py --from-hub --repo-id username/dataset --stats`

6. **`visualize_questions.py`**
   - **Purpose**: Visualize generated questions on images
   - **Input**: Output from script 2
   - **Output**: Annotated images with questions

## Documentation

### Quick Reference

- **`QUICKSTART.md`** **[NEW]** 🚀
  - Start here! Fastest way to upload to HuggingFace
  - TL;DR examples and common use cases
  - Troubleshooting quick tips

### Detailed Guides

- **`README_DATASET_CONVERSION.md`** **[NEW]** 📖
  - Complete documentation for script 3
  - Dataset schema and field descriptions
  - Authentication and configuration options
  - Loading and usage examples

- **`WORKFLOW.md`** **[NEW]** 📋
  - Complete end-to-end pipeline guide
  - Step-by-step instructions for all 3 scripts
  - Performance tips and troubleshooting
  - Complete examples with actual paths

### Reference

- **`FILES_OVERVIEW.md`** **[NEW]** 📑
  - This file - quick reference for all files

## File Tree

```
FuncElemGnd_eval_gen/
├── Core Scripts (run in order)
│   ├── 1_make_func_elemgnd_samples.py          # Step 1: Detect similar elements
│   ├── 2_generate_func_elemgnd_questions.py    # Step 2: Generate questions
│   └── 3_convert_to_hf_dataset.py             # Step 3: Convert to HF dataset ⭐ NEW
│
├── Utility Scripts
│   ├── run_conversion.sh                       # Easy conversion wrapper ⭐ NEW
│   ├── example_load_dataset.py                # Dataset exploration tool ⭐ NEW
│   └── visualize_questions.py                 # Visualization helper
│
└── Documentation
    ├── QUICKSTART.md                           # Quick start guide ⭐ NEW
    ├── README_DATASET_CONVERSION.md            # Detailed conversion docs ⭐ NEW
    ├── WORKFLOW.md                             # Complete pipeline guide ⭐ NEW
    └── FILES_OVERVIEW.md                       # This file ⭐ NEW
```

## Quick Start Paths

### Just Want to Upload to HuggingFace?

1. Read: `QUICKSTART.md`
2. Run: `./run_conversion.sh -i /path/to/images -q questions.json -p -r username/dataset -m`
3. Done! 🎉

### Need Detailed Information?

1. Pipeline overview: `WORKFLOW.md`
2. Conversion details: `README_DATASET_CONVERSION.md`
3. Usage examples: `example_load_dataset.py`

### Running the Complete Pipeline?

1. Read: `WORKFLOW.md`
2. Run: `1_make_func_elemgnd_samples.py`
3. Run: `2_generate_func_elemgnd_questions.py`
4. Run: `3_convert_to_hf_dataset.py` or `run_conversion.sh`

## Common Tasks

### Upload Dataset to HuggingFace

```bash
# Quick way (with local cache)
./run_conversion.sh -q questions.json -r username/dataset -m

# Or detailed way
python 3_convert_to_hf_dataset.py \
    --questions-file questions.json \
    --repo-id username/dataset \
    --include-images
```

### Local Cache

The dataset is always saved locally as cache in `[questions_file_directory]/hf_dataset_cache/` before uploading to HuggingFace. This ensures you have a backup even if upload fails.

### Explore Dataset

```bash
# From HuggingFace
python example_load_dataset.py --from-hub --repo-id username/dataset --stats --interactive

# From local
python example_load_dataset.py --local-path ./output --stats --num-examples 5 --visualize
```

### Visualize Questions

```bash
python visualize_questions.py \
    --questions-file questions.json \
    --output-dir ./visualizations
```

## Input/Output Files

### Expected Input Structure

```
dataset_root/
├── images/                          # GUI screenshots
│   ├── screenshot1.png
│   └── screenshot2.png
├── omniparser/                      # OmniParser detections
│   ├── screenshot1.json
│   └── screenshot2.json
└── omniparser_embeddings/           # DINO embeddings
    ├── screenshot1.npz
    └── screenshot2.npz
```

### Generated Output Structure

```
dataset_root/
└── FuncElemGnd/
    ├── similar_elements_anno.json   # From script 1
    ├── grounding_questions.json     # From script 2
    └── hf_dataset/                  # From script 3 (if --save-local)
        ├── dataset_dict.json
        ├── data-00000-of-00001.arrow
        └── state.json
```

## Dependencies

### Required Python Packages

```bash
# Core dependencies (scripts 1 & 2)
pip install opencv-python pillow numpy tqdm colorama openai

# For HuggingFace upload (script 3)
pip install datasets huggingface_hub
```

### System Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended for large datasets)
- GPU not required (but helps for faster processing)
- Sufficient disk space (varies by dataset size)

## Key Concepts

### Dataset Format

- **One entry per question** (not per image or per group)
- Each entry includes full context (image, similar elements, etc.)
- Supports multiple action types per element
- Includes both detection and OmniParser metadata

### Bounding Box Formats

1. **Target Element BBox**: Normalized 0-1000 (from question generation)
2. **OmniParser BBox**: Normalized 0-1 (from OmniParser detection)

### Action Types

Common action types in the dataset:
- `clicking` - Most common
- `hovering` - Show tooltips/highlights
- `typing` - Input fields
- `dragging` - Sliders, scrollbars
- `selecting` - Checkboxes, dropdowns
- `swiping` - Mobile gestures
- `long pressing` - Context menus

## Version History

### v1.0 (Current)
- ✅ Similar element detection (script 1)
- ✅ Question generation (script 2)
- ✅ HuggingFace dataset conversion (script 3)
- ✅ Comprehensive documentation
- ✅ Example scripts and utilities

## Support and Contributions

### Getting Help

1. Check the appropriate documentation file
2. Review the example scripts
3. Look at the troubleshooting sections
4. Open an issue with details

### Contributing

When adding new features:
1. Update the relevant script
2. Update documentation
3. Add examples if applicable
4. Update this overview

## License

[Specify your license here]

## Citation

```bibtex
@article{funcelem2025,
  title={FuncElemGnd: Functional Element Grounding for GUI Understanding},
  author={Your Name},
  year={2025}
}
```

---

**Last Updated**: October 29, 2025

**Maintained By**: [Your Name/Team]

**Repository**: [GitHub URL if applicable]

