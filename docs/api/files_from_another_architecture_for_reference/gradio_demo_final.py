import os
import io
import base64 # Not directly used in the final output for this API, but was in original 'process'
import json
import logging
import tempfile # Using tempfile for more robust temporary file creation

from flask import Flask, request, jsonify
from PIL import Image
import torch
# import numpy as np # numpy might be a dependency of your utils or models

# --- Import your utility functions ---
# Ensure utils.py is in the same directory or accessible via PYTHONPATH
try:
    from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
except ImportError as e:
    print(f"Error importing from utils.py: {e}. Make sure utils.py is accessible.")
    # You might want to exit or handle this more gracefully if utils are critical at module level
    # For now, we'll let it potentially fail later if models can't be loaded.


# --- Global Variables & Model Loading ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine the directory of this script for relative path construction
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a directory for temporary images if you prefer a dedicated one,
# otherwise tempfile module handles temporary file creation securely.
# TEMP_IMAGE_DIR = os.path.join(current_dir, 'temp_api_images')
# os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


# Model paths (adjust if your directory structure is different)
yolo_model_path = os.path.join(current_dir, 'weights', 'icon_detect', 'model.pt')
# Use Florence-2 model
florence_model_path = os.path.join(current_dir, "weights", "icon_caption_florence")

# Load models
try:
    if not os.path.exists(yolo_model_path):
        logger.error(f"YOLO model file not found at: {yolo_model_path}")
        raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")
    if not os.path.exists(florence_model_path):
         logger.error(f"Florence-2 model path not found at: {florence_model_path}")
         raise FileNotFoundError(f"Florence-2 model path not found: {florence_model_path}")

    yolo_model = get_yolo_model(model_path=yolo_model_path)
    # Use Florence-2 model
    caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path=florence_model_path)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Fatal error loading models: {e}", exc_info=True)
    # If models fail to load, the API is not functional.
    # You might want to exit or have the health check endpoint report unhealthy.
    yolo_model = None
    caption_model_processor = None

# Device configuration (models should ideally handle their own device placement or be moved after loading)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")
# If your models need explicit moving:
# if yolo_model: yolo_model.to(DEVICE)
# if caption_model_processor and hasattr(caption_model_processor, 'model'): caption_model_processor.model.to(DEVICE)


# --- Flask App Definition ---
app = Flask(__name__)

# --- Helper function for OCR extraction (adapted from original 'process') ---
def extract_ocr_from_image(
    pil_image: Image.Image,
    box_threshold: float,
    iou_threshold: float,
    use_paddleocr: bool,
    imgsz: int
) -> str:
    """
    Processes a PIL Image object to extract OCR text.
    """
    if not yolo_model or not caption_model_processor:
        logger.error("Models not loaded. Cannot process image.")
        raise RuntimeError("Models are not loaded. OCR service is unavailable.")

    # Using tempfile for secure temporary file creation and cleanup
    # Suffix is important for some libraries that infer type from extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image_save_path = tmp_file.name
        pil_image.save(image_save_path)
    
    logger.info(f"Temporary image saved to {image_save_path}")

    try:
        # Calculate draw_bbox_config, assuming it's needed by get_som_labeled_img
        image_width = pil_image.size[0]
        if image_width == 0: # Should not happen with valid images
            box_overlay_ratio = 1.0 
        else:
            # Using 3200.0 from original script's logic
            box_overlay_ratio = image_width / 3200.0 

        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # Perform OCR
        # Default easyocr_args from original: {'paragraph': False, 'text_threshold':0.9}
        ocr_bbox_rslt, _ = check_ocr_box(
            image_save_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        logger.info(f"Raw OCR text extracted (first 50 chars): {text[:50]}")

        # Get structured labeled image data (we only need the parsed content list)
        _, _, parsed_content_list = get_som_labeled_img(
            image_save_path,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz
        )
        
        parsed_text_output = '\n'.join(
            f"icon {i}: {json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)}"
            for i, item in enumerate(parsed_content_list)
        )
        logger.info(f"Content parsing successful. Found {len(parsed_content_list)} items.")
        return parsed_text_output

    except Exception as e:
        logger.error(f"Error during OCR extraction pipeline: {e}", exc_info=True)
        raise # Re-raise to be caught by the API route's error handler
    finally:
        # Clean up the temporary image
        if os.path.exists(image_save_path):
            try:
                os.remove(image_save_path)
                logger.info(f"Temporary image {image_save_path} removed.")
            except Exception as e_rem:
                logger.error(f"Error removing temporary image {image_save_path}: {e_rem}")


# --- API Endpoint ---
@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if not yolo_model or not caption_model_processor:
        logger.critical("OCR endpoint called but models are not loaded. Service unavailable.")
        return jsonify({"error": "OCR models not loaded, service unavailable."}), 503 # Service Unavailable

    if 'image' not in request.files:
        logger.warning("API /ocr: No 'image' file part in request.")
        return jsonify({"error": "Missing 'image' file part in the request."}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        logger.warning("API /ocr: No file selected (empty filename).")
        return jsonify({"error": "No file selected."}), 400

    try:
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        # It's good practice to ensure a consistent format, e.g., RGB
        pil_image = pil_image.convert("RGB")
        logger.info(f"API /ocr: Received image '{file.filename}', size: {pil_image.size}, mode: {pil_image.mode}")

    except Exception as e:
        logger.error(f"API /ocr: Invalid image file provided. Error: {e}", exc_info=True)
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Get parameters from request form data or use defaults
    try:
        box_threshold = float(request.form.get('box_threshold', 0.05))
        iou_threshold = float(request.form.get('iou_threshold', 0.1))
        # For boolean, check against 'true' or '1' etc.
        use_paddleocr_str = request.form.get('use_paddleocr', 'true').lower()
        use_paddleocr = use_paddleocr_str == 'true'
        imgsz = int(request.form.get('imgsz', 640))
        logger.info(f"API /ocr: Processing with params: box_threshold={box_threshold}, iou_threshold={iou_threshold}, use_paddleocr={use_paddleocr}, imgsz={imgsz}")
    except ValueError as e:
        logger.error(f"API /ocr: Invalid parameter value. Error: {e}", exc_info=True)
        return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400

    try:
        # Process the image to get OCR text
        # The @torch.autocast context manager can be used here if needed for mixed precision
        # with torch.autocast(device_type=DEVICE.type if DEVICE.type != 'mps' else 'cpu', dtype=torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32, enabled=DEVICE.type == 'cuda'):
        ocr_text_result = extract_ocr_from_image(
            pil_image,
            box_threshold,
            iou_threshold,
            use_paddleocr,
            imgsz
        )
        return jsonify({"ocr_text": ocr_text_result})
    
    except RuntimeError as e: # Catch specific model loading issues from helper
        logger.error(f"API /ocr: Runtime error during processing (likely model issue): {e}", exc_info=True)
        return jsonify({"error": str(e)}), 503 # Service Unavailable
    except Exception as e:
        logger.error(f"API /ocr: Unexpected error processing image: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during image processing: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    if yolo_model and caption_model_processor:
        return jsonify({"status": "healthy", "models_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "models_loaded": False, "message": "One or more models failed to load."}), 503

# --- Main ---
if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn or uWSGI
    # Example: gunicorn --workers 4 --bind 0.0.0.0:58081 api_app:app
    # The port 58080 was used by Gradio, using a different one for the API.
    app.run(host='0.0.0.0', port=52000, debug=False, threaded=False) # Set debug=False for production-like behavior