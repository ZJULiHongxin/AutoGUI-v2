# Functional Region Annotation Tool

A sophisticated tool for automatically annotating functional regions in GUI screenshots using Large Language Models (LLMs). The tool recursively decomposes UI images into hierarchical functional components with quality verification.

## 🎯 Overview

This script processes GUI screenshots and identifies functional regions (buttons, text fields, menus, etc.) by:

1. **Hierarchical Decomposition**: Recursively breaking down complex UIs into smaller, manageable components
2. **Quality Verification**: Using separate LLMs to validate region completeness and boundedness
3. **Intelligent Refinement**: Automatically retrying with improved prompts when quality thresholds aren't met
4. **Parallel Processing**: Supporting distributed processing for large datasets

## 🚀 Key Features

- **Multi-Model Support**: Works with GPT-4, Gemini, Claude, and other LLM providers
- **Hierarchical Annotation**: Tree-like structure capturing UI component relationships
- **Quality Assurance**: Dual-model verification (annotation + completeness checking)
- **Parallel Processing**: Multi-worker support with load balancing
- **Caching System**: Intelligent caching to avoid redundant processing
- **Configurable Depth**: Adjustable maximum hierarchy depth
- **Comprehensive Logging**: Detailed progress tracking and debugging
- **Dataset Integration**: Built-in support for ScreenSpot-Pro and OSWorld-G

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch` - For image processing
- `opencv-python` - Computer vision operations
- `PIL` (Pillow) - Image manipulation
- `colorama` - Colored terminal output
- `tqdm` - Progress bars
- `megfile` - Cloud storage support (S3, etc.)
- `openai` - LLM API client

## 🏗️ Architecture

### Processing Flow

```
Input Image
    ↓
Root Region Analysis
    ↓ (if divisible)
Child Region Generation
    ↓
Completeness Verification
    ↓ (if quality < threshold)
Refinement with Improved Prompts
    ↓
Quality Meets Criteria?
    ├─ Yes → Accept Children
    └─ No → Retry (up to 3 times)
```

### Key Components

1. **FunctionalRegionAnnotator**: Main annotation engine

   - `model`: Primary LLM for region identification
   - `checking_model`: Secondary LLM for quality verification
2. **Hierarchical Processing**:

   - Level 0: Root image
   - Level 1+: Child regions
   - Configurable maximum depth
3. **Quality Metrics**:

   - **Completeness**: Does region contain all necessary elements?
   - **Boundedness**: Is region properly bounded without excess background?

## 💻 Usage

### Basic Usage

```bash
python annotate_functional_regions.py \
    --data-path "HongxinLi/ScreenSpot-Pro" \
    --model "gpt-4o" \
    --output-dir "./output" \
    --debug
```

### Advanced Usage

```bash
python annotate_functional_regions.py \
    --data-path "HongxinLi/ScreenSpot-Pro" \
    --model "gemini-2.5-pro-preview-03-25" \
    --checking-model "gemini-2.5-flash-lite-preview-06-17" \
    --workers 4 \
    --max-level 3 \
    --max-refine 2 \
    --debug \
    --debug-draw
```

## ⚙️ Configuration Options

### Required Arguments

- `--data-path`: Dataset path (HuggingFace dataset or local directory)
- `--model`: Primary LLM model for annotation

### Optional Arguments

#### Model Configuration

- `--checking-model`: Separate model for quality checking (defaults to primary model)
- `--base-url`: Custom API endpoint
- `--api-key`: API key (or use environment variable)

#### Processing Control

- `--workers`: Number of parallel workers (default: 1)
- `--max-level`: Maximum hierarchy depth (-1 for unlimited, default: 3)
- `--max-refine`: Maximum refinement attempts (default: 3)
- `--max-retries`: Maximum API retry attempts (default: 3)

#### Quality Thresholds

- `--completeness-threshold`: Minimum completeness score (0-3, default: 2.5)
- `--boundedness-threshold`: Minimum boundedness ratio (default: 0.8)

#### Output & Debugging

- `--output-dir`: Output directory
- `--debug`: Enable detailed logging
- `--debug-draw`: Generate debug visualization images
- `--cache-dir`: Custom cache directory
- `--force`: Reprocess already completed images

## 📊 Output Format

### JSON Structure

```json
{
  "metadata": {
    "model": "gpt-4o",
    "timestamp": "2024-01-15 10:30:00",
    "total_annotation_queries": 1250,
    "total_completeness_queries": 890
  },
  "results": {
    "image_001": {
      "image_path": "/path/to/image.png",
      "result": {
        "0-0": {
          "root_image_path": "/path/to/image.png",
          "bbox_global": [0, 0, 1920, 1080],
          "functionality": {
            "with_context": "Main application window",
            "wo_context": "Application window"
          },
          "children": ["1-0", "1-1", "1-2"],
          "completeness_info": {...}
        },
        "1-0": {
          "bbox_global": [50, 100, 400, 200],
          "functionality": {
            "with_context": "Navigation menu",
            "wo_context": "Menu"
          },
          "children": ["2-0", "2-1"],
          "completeness_info": {
            "completeness_score": 2.8,
            "boundedness": true,
            "check_region_completeness_response": "..."
          }
        }
      },
      "processing_time": 45.23,
      "annotation_queries": 12,
      "completeness_queries": 8
    }
  }
}
```

### File Structure

```
output/
├── exp_config.json          # Complete configuration used
├── screenspot_pro/
│   └── gemini-2.5-pro-preview-03-25/
│       └── v3/
│           ├── functional_regions_gemini-2.5-pro-preview-03-25.json
│           ├── debug_images/        # Debug visualizations
│           └── cache/              # Cached images and metadata
```

## 🔧 How It Works

### 1. Image Loading & Preprocessing

- Load image from local path or cloud storage
- Resize if exceeds maximum dimensions
- Convert to appropriate format for LLM input

### 2. Hierarchical Decomposition

- **Root Analysis**: Process entire image to identify major functional areas
- **Child Generation**: For each region, recursively identify sub-components
