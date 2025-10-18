## System Architecture

### Overview
This system provides two primary entry points:
- Flask API in `gradio_demo_final.py` exposing endpoints like `/ocr` and `/health`.
- Gradio UI in `gradio_demo.py` for interactive demonstration of the same pipeline.

Core capabilities:
- Text detection via OCR (PaddleOCR preferred; EasyOCR optional).
- Icon/object detection via Ultralytics YOLO checkpoint at `weights/icon_detect/model.pt`.
- Optional icon local semantics using Florence-2 captioning at `weights/icon_caption_florence`.
- Fusion of OCR and icon detections with overlap handling and serialization into structured elements.

### High-level Architecture (ASCII)
```
+-------------------+         +-----------------------+
|   Client (UI/CLI) |  HTTP   |   Flask API Server    |
| - Gradio frontend | <-----> |  (/ocr, /health)      |
+-------------------+         +-----------------------+
                                    |
                                    | invokes
                                    v
                         +-------------------------+
                         |  Screen Parse Pipeline  |
                         | (extract_ocr_from_image)|
                         +-----------+-------------+
                                     \
                                      \ orchestrates
                                       v
    +----------------+     +-------------------+     +---------------------+
    |   OCR Module   |     |  Icon Detector    |     |   Icon Captioner    |
    | (Paddle/Easy)  |     |  (Ultralytics)    |     |  (Florence-2)       |
    | check_ocr_box  |     | predict_yolo      |     | get_parsed_content_ |
    +-------+--------+     +---------+---------+     | icon(_phi3v)        |
            |                        |               +----------+----------+
            | text + boxes           | icon boxes               |
            +------------+-----------+--------------------------+
                         | merge & de-duplicate (remove_overlap_new)
                         v
                 +---------------------------+
                 |   Structured Elements     |
                 |  - text boxes (OCR)       |
                 |  - icon boxes (+captions) |
                 +-------------+-------------+
                               |
                               | annotate + serialize
                               v
                    +-------------------------------+
                    | API Response / Gradio Outputs |
                    +-------------------------------+
```

### Detailed Request/Response Flow (API)
```
Client                Flask (/ocr)                OCR               YOLO                 Florence-2
  |  POST image         |                          |                  |                        |
  |-------------------->|                          |                  |                        |
  |                     | save temp                |                  |                        |
  |                     |------------------------->|                  |                        |
  |                     | call check_ocr_box       |                  |                        |
  |                     |------------------------->|                  |                        |
  |                     |      text, ocr_bbox      |                  |                        |
  |                     |<-------------------------|                  |                        |
  |                     | call get_som_labeled_img |                  |                        |
  |                     |-------------------------------------------->|                        |
  |                     |                         icon boxes (xyxy)   |                        |
  |                     |<---------------------------------------------                        |
  |                     | merge OCR+icons, dedup                       |                        |
  |                     |-----------------------------------------------> caption crops         |
  |                     |                                 parsed icon texts <-------------------|
  |                     | compose outputs (annotated image, list)      |                        |
  |                     | delete temp                                  |                        |
  |           JSON/text/PNG response                                   |                        |
  |<--------------------|                                              |                        |
```

### Components and Key Files
- `gradio_demo_final.py`
  - Endpoints: `/ocr`, `/health`
  - Loads models via `get_yolo_model('weights/icon_detect/model.pt')` and `get_caption_model_processor('florence2', 'weights/icon_caption_florence')`
  - Device selection: CUDA if available; else CPU.

- `gradio_demo.py`
  - Interactive UI that wraps the same pipeline for demos.

- `util/utils.py`
  - `check_ocr_box`: OCR via PaddleOCR (preferred) or EasyOCR
  - `predict_yolo`: Ultralytics YOLO inference for icons
  - `remove_overlap_new`: merge OCR and icon boxes, prioritizing OCR where nested
  - `get_parsed_content_icon` / `_phi3v`: optional icon captioning (Florence-2 / phi3 variant)
  - `get_som_labeled_img`: orchestrates detection, merge, annotation, and serialization

- `weights/`
  - `icon_detect/model.pt`: YOLO detector checkpoint (AGPL lineage noted in README)
  - `icon_caption_florence/`: local Florence-2 model directory

### Configuration and Tunables
- Request parameters (POST `/ocr`):
  - `box_threshold` (float, default 0.05) — YOLO confidence threshold
  - `iou_threshold` (float, default 0.1) — overlap/merge behavior
  - `use_paddleocr` (bool string, default "true") — OCR backend selection
  - `imgsz` (int, default 640) — YOLO input size (higher aids small-icon recall)

- Model paths:
  - YOLO: `weights/icon_detect/model.pt`
  - Florence-2: `weights/icon_caption_florence`

### Known Limitations
- Cross-OS icon variance:
  - The YOLO checkpoint is specialized for UI icons; unfamiliar OS/icon styles may be missed.
  - Mitigations: lower `box_threshold`, raise `imgsz`, add an open-vocabulary fallback, or fine-tune YOLO on broader data.

- OCR backend stability (WSL2/CPU):
  - PaddleOCR may fail with oneDNN/MKL kernel errors (e.g., "could not execute a primitive").
  - Workarounds: set `FLAGS_use_mkldnn=false`, limit threads (`OMP_NUM_THREADS=1`), or temporarily use EasyOCR while investigating.

- Ultralytics cache warnings:
  - Non-fatal JSON decode errors for `~/.config/Ultralytics/persistent_cache.json`; reset to `{}` if noisy.

### Output Contracts
- `/ocr` (200):
  - JSON with `ocr_text`: newline-joined entries describing structured elements.

- Gradio UI:
  - Annotated PNG image and textual element list.

### Performance Notes
- Increasing `imgsz` improves small-object recall with additional compute cost.
- Florence-2 captioning is batched; tune batch size in `get_parsed_content_icon`.
- Prefer a single worker if moving behind Gunicorn initially to avoid multiple large model instances.

### Troubleshooting (Quick Reference)
- PaddleOCR runtime error on CPU/WSL2:
  - Try: `export FLAGS_use_mkldnn=false`, `export OMP_NUM_THREADS=1`
  - Validate installs: `python -c "import paddle; print(paddle.__version__, paddle.device.is_compiled_with_cuda())"`
  - Temporary fallback: set `use_paddleocr=false` (uses EasyOCR)

- Ultralytics cache warning:
  - Reset file: `printf '{}' > ~/.config/Ultralytics/persistent_cache.json`

### Roadmap (Optional Enhancements)
- Add a flag-gated fallback detector (open-vocabulary or Florence-2 grounded pass) unioned with YOLO to improve recall.
- Centralize configuration (env/YAML) for thresholds, model paths, and backends.
- Expanded observability: structured logs and lightweight metrics to aid diagnosing runtime issues.


