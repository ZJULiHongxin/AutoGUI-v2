# AutoGUIv2 Annotation Monitor

Run a small server to visualize the stack-based tree-search annotation progress live, from the cache directory written by the annotator.

## 1) Start annotator with caching

Annotator now writes crops and metadata to a cache directory automatically:

```bash
python utils/data_utils/autoguiv2/annotate_functional_regions.py \
  --data-path MMInstruction/OSWorld-G \
  --output-dir /mnt/jfs/copilot/lhx/ui_data/AutoGUIv2 \
  --cache-dir /mnt/jfs/copilot/lhx/ui_data/AutoGUIv2/cache \
  --model gemini-2.5-pro-thinking \
  --workers 2 --debug --sequential
```

If --cache-dir is not provided, it defaults to `<output-dir>/cache`.

Cache layout (by namespace `benchmark/model/version`):

```
cache/
  osworld_g/gemini-2.5-pro-thinking/v1/
    <image-id>/
      root.png
      tree.json
      stack.json
      nodes/
        0-0.png
        0-0.json
        1-0.png
        1-0.json
        ...
```

## 2) Install deps and run the monitor

```bash
pip install -r requirements-webui.txt
python utils/data_utils/autoguiv2/monitor/server.py --cache-dir /mnt/jfs/copilot/lhx/ui_data/AutoGUIv2/cache --port 8000
```

Then open `http://localhost:8000`.

The server auto-detects cache dir if `--cache-dir` is omitted (also checks `AUTOGUI_CACHE_DIR`).

## Notes
- The UI auto-refresh button pulls the latest `tree.json` and `stack.json`.
- Each node view shows the saved crop and metadata as soon as the annotator writes it.
- Multiple datasets/models/versions are grouped as namespaces; pick one to browse its images.

# AutoGUIv2 Monitor Tools

## BBox Correction Server
- Run: python server_bboxcorrection_v2.py
- Access: http://localhost:17800
- Features: Browse namespaces, models, versions, images; view tree and correct bboxes.

## Evaluation Visualizer
- Run: python visualize_bmk_and_eval_results.py --port 17801
- Access: http://localhost:17801
- Features:
  - Metrics tab: Overall stats, IoU thresholds, breakdowns.
  - Sample Inspector: View sample details, edit failure reasons with auto-save.
  - Failure Summary: Table with sorting/filtering, stats, pie chart, CSV export.
- Troubleshooting:
  - If no data: Check eval_results directory has task/model/timestamp.json files.
  - Check server logs for eval_root path.
  - Browser console for JS errors.


