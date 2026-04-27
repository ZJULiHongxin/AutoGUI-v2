# Question Visualization Tool

This tool visualizes the generated questions and their targeted elements from the FuncElemGnd evaluation generation pipeline.

## Overview

The visualization script (`visualize_questions.py`) helps you:
- View generated questions overlaid on screenshots
- See which elements are targeted by each question
- Understand the mapping between question element IDs and actual OmniParser indices
- Explore all available images with generated questions

## Key Concept: Element Index Mapping

**Important:** The element IDs shown in the generated questions JSON file (e.g., `[0]`, `[1]`, `[2]`) are **NOT** the global OmniParser indices. They are local indices within each similar element group.

The script automatically handles the mapping:
1. Loads the similar groups from `omniparser_embeddings/{image_stem}.npz`
2. Maps local group indices to global OmniParser indices
3. Retrieves the correct bounding boxes for visualization

## Usage

### 1. List all available images with questions

```bash
python visualize_questions.py --list
```

This will show:
- All images that have generated questions
- Number of groups per image
- Total number of questions per image

### 2. Batch Mode: Visualize ALL questions from ALL groups across ALL samples

```bash
python visualize_questions.py --all
```

This is the **recommended mode** for comprehensive visualization. It will:
- Process all images with generated questions
- Create one visualization per group
- Save all visualizations to the `visualizations/` directory
- Each file is named: `{image_name}_group{group_index}.png`

Example output:
```
================================================================================
Batch Visualization Complete!
  Total Images: 86
  Total Groups: 271
  Total Questions: 666
  Output Directory: visualizations
================================================================================
```

To save to a custom directory:
```bash
python visualize_questions.py --all --output-dir my_visualizations
```

### 3. Visualize the first available image

```bash
python visualize_questions.py
```

By default, this saves the visualization to `test.png` in the project root.

### 4. Visualize a specific image

```bash
python visualize_questions.py --image /path/to/image.png
```

or using the image key:

```bash
python visualize_questions.py --image /mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/images/07JVNIBSB9.png
```

Note: This will visualize the first group of the specified image. To visualize all groups, use `--all` mode.

### 5. Save to a custom output path

```bash
python visualize_questions.py --output my_visualization.png
```

## Output Format

### Console Output

The script prints detailed information to the console:

```
Group 2: 4 questions, 4 elements
Visual Similarity: All elements are 'X' icons, typically used for closing or dismissing a UI component.

  Elements in this group:
    [0] Description of element 0...
         Functionality: What element 0 does...
    [1] Description of element 1...
         Functionality: What element 1 does...

  Question for Element [0]:
    - clicking: User's question about clicking...
      Intent: Precise action intent...
    - hovering: User's question about hovering...
      Intent: Precise action intent...
```

### Image Output

The visualization shows:
- Original screenshot
- Bounding boxes around each element (color-coded)
- Element IDs in colored boxes
- A sample question displayed at the top
- Group information and visual similarity description

## File Structure

The script expects the following file structure:

```
base_dir/
├── images/
│   └── {image_stem}.png
├── omniparser/
│   └── {image_stem}.json          # OmniParser parsing result
├── omniparser_embeddings/
│   └── {image_stem}.npz           # Similar groups mapping
└── FuncElemGnd/
    ├── similar_elements_anno.json  # Detection results
    └── grounding_questions.json    # Generated questions
```

## Example Output

When you run the script, you'll see something like:

```
Visualizing: /path/to/image.png
Found 4 groups with questions

Group 2: 4 questions, 4 elements
Visual Similarity: All elements are 'X' icons...

  Elements in this group:
    [1] A standard window control button with a white 'X'...
         Functionality: This element closes the entire application...
    [2] An 'X' icon located in the top-right corner of the dialog...
         Functionality: This element dismisses the dialog...

  Question for Element [1]:
    - clicking: How do I completely shut down the application?
      Intent: Click the 'X' icon in the top-right corner...

Saved visualization to: test.png
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--list` | List all images with questions | `--list` |
| `--all` | Batch mode: visualize all questions of all groups of all samples | `--all` |
| `--output-dir PATH` | Output directory for batch mode | `--output-dir visualizations` |
| `--image PATH` | Visualize specific image (first group only) | `--image /path/to/img.png` |
| `--output PATH` | Custom output path (single mode) | `--output viz.png` |
| `--questions-file` | Custom questions JSON | `--questions-file questions.json` |
| `--detection-file` | Custom detection JSON | `--detection-file detection.json` |

## Default File Paths

The script uses these default paths (can be overridden):

- Questions file: `/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/grounding_questions.json`
- Detection file: `/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/FuncElemGnd/similar_elements_anno.json`
- Image source dir: `/mnt/vdb1/hongxin_li/AutoGUIv2/osworld_g/images`
- Output file: `test.png`

## Tips

1. **Explore with --list first**: Use `--list` to see all available images before visualizing
2. **Check multiple groups**: If an image has multiple groups, visualize each one separately with `--group`
3. **Compare questions**: Look at the console output to see all questions for each element, not just the one shown in the image
4. **Verify mappings**: The script automatically handles index mapping, but you can verify by checking the element descriptions

## Troubleshooting

**Issue**: "Image not found"
- **Solution**: Make sure the image path exists and matches the structure in the questions JSON

**Issue**: "No questions found"
- **Solution**: Verify the questions JSON file contains data for the specified image

**Issue**: "OmniParser result not found"
- **Solution**: Ensure the `omniparser/` directory exists with the corresponding JSON file

**Issue**: "Could not load similar groups mapping"
- **Solution**: This is a warning. The script will attempt to work without the mapping, but results may be less accurate

