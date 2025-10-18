## OCR API Guide

### Overview
This document explains how the current API processes images, what the `/ocr` endpoint expects and returns, and how to adapt the system to use a different OCR backend.

Primary entry point: Flask app in `gradio_demo_final.py`.

### Endpoints
- `POST /ocr`
  - Multipart form-data fields:
    - `image` (file): PNG/JPG screenshot
    - `box_threshold` (float, default 0.05): YOLO confidence threshold
    - `iou_threshold` (float, default 0.1): overlap/merge threshold
    - `use_paddleocr` (bool string, default `true`): select OCR backend (PaddleOCR=true, EasyOCR=false)
    - `imgsz` (int, default 640): YOLO input resolution

- `GET /health`
  - Returns JSON `{status, models_loaded}`

### Request → Processing → Response
1) `image` is saved to a temporary file.
2) OCR step (`check_ocr_box` in `util/utils.py`):
   - If `use_paddleocr=true`, runs PaddleOCR; else EasyOCR.
   - Returns: `text` list and `ocr_bbox` (xyxy coordinates).
3) Icon detection (`predict_yolo`): Ultralytics YOLO on the image to get icon boxes.
4) Merge (`remove_overlap_new`): Combines OCR text boxes with icon boxes, prioritizing text where overlapping.
5) Optional semantics: If enabled, crops icons and runs Florence-2 captioning to enrich content.
6) Response assembly: Joins parsed elements into a newline-separated string under `ocr_text`.
7) Temp file is deleted.

Response example (200):
```json
{
  "ocr_text": "icon 0: {\"type\": \"text\", ...}\nicon 1: {\"type\": \"icon\", ...}"
}
```

Error conditions:
- Models not loaded → 503 with `{error}`
- Invalid image → 400 with `{error}`
- Unexpected processing error → 500 with `{error}`

### Files and Responsibilities
- `gradio_demo_final.py`: Flask API, model loading, endpoint orchestration.
- `util/utils.py`:
  - `check_ocr_box`: OCR abstraction (Paddle/Easy switch)
  - `predict_yolo`: YOLO inference
  - `remove_overlap_new`: dedup / merge
  - `get_parsed_content_icon(_phi3v)`: captioning
  - `get_som_labeled_img`: pipeline orchestrator returning annotated image + elements
- `weights/`:
  - `icon_detect/model.pt`: YOLO checkpoint
  - `icon_caption_florence/`: Florence-2 directory

### Adapting to a New OCR Backend

Minimum files to copy for a new project:
- `gradio_demo_final.py` (API server)
- `util/utils.py` (pipeline)
- `requirements.txt` (baseline deps) — trim if not needed
- `weights/` (replace with your own model folders as applicable)
- `docs/architecture/system_architecture.md` (reference)

Where to integrate a new OCR:
- Replace or extend `check_ocr_box` in `util/utils.py`.
  - Contract:
    - Input: `image_source` (path or PIL), flags (e.g., `display_img`, `output_bb_format` = 'xyxy')
    - Output: `(text_list, bbox_list), goal_filtering`
      - `bbox_list` must be a list of 4-pt boxes matching `output_bb_format`.

Adapter outline (pseudo):
```python
def check_ocr_box(..., use_paddleocr: bool, easyocr_args=None, **kwargs):
    if use_new_backend:
        # new_backend_result = NEW_BACKEND.run(image_np, ...)
        # coord = ...  # list of boxes in xyxy
        # text = ...   # list of strings matching coord
        return (text, coord), goal_filtering
    elif use_paddleocr:
        # existing PaddleOCR branch
    else:
        # existing EasyOCR branch
```

Notes:
- Ensure boxes are consistent with later steps which expect `xyxy` for `remove_overlap_new` and normalization in `get_som_labeled_img`.
- Keep image mode normalization (convert RGBA→RGB) to avoid alpha-channel issues.

### Configuration & Tunables
- `box_threshold`, `iou_threshold`, and `imgsz` are passed from the API request down to detection/merge.
- OCR-specific settings can be exposed similarly via form fields or environment variables.

### Logging & Observability
- API logs to stdout (or to files when configured via Gunicorn flags).
- Consider adding a rotating file handler if you need a dedicated app log.

### Security & Resource Notes
- Temp files are deleted in a `finally` block after processing.
- For production, prefer a small number of workers to avoid multiple large model instances.

