import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download
import traceback

from prediction import split_image, make_predictions
from chat import get_response

# ----------------------
# Flask Setup
# ----------------------
app = Flask(__name__)
CORS(app, origins=["*"])

# ----------------------
# Create necessary directories
# ----------------------
def ensure_directories():
    """Create all necessary directories for file operations"""
    directories = [
        os.path.join(app.root_path, "static"),
        os.path.join(app.root_path, "static", "uploads"),
        os.path.join(app.root_path, "static", "output"),
        os.path.join(app.root_path, "static", "output", "discs"),
        os.path.join(app.root_path, "models")
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories on startup
ensure_directories()

# ----------------------
# Hugging Face Hub Config
# ----------------------
REPO_ID = "tudao01/spine"

# Local paths
NONBINARY_PATH = "models/nonBinaryIndividualPredictions.keras"
BINARY_PATH = "models/binaryIndividualPredictions.keras"
UNET_PATH = "models/unet_spine_segmentation.pth"

# ----------------------
# Download models if missing
# ----------------------
def download_model_if_missing(local_path, filename):
    try:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading {filename} from Hugging Face Hub...")
            # Download and return the path, then copy to desired location if needed
            downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            # If the downloaded file isn't in the right location, move it
            if downloaded_path != local_path:
                import shutil
                shutil.copy2(downloaded_path, local_path)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"Model {filename} already exists locally")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise

# Download models on startup with retry logic
def download_models_with_retry():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to download models...")
            download_model_if_missing(NONBINARY_PATH, "nonBinaryIndividualPredictions.keras")
            download_model_if_missing(BINARY_PATH, "binaryIndividualPredictions.keras")
            download_model_if_missing(UNET_PATH, "unet_spine_segmentation.pth")
            print("All models downloaded successfully!")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5)  # Wait 5 seconds before retry
            else:
                print(f"Failed to download models after {max_retries} attempts")
                return False

# Download models on startup
models_downloaded = download_models_with_retry()

# ----------------------
# Load models
# ----------------------
def load_models():
    """Load all models with error handling"""
    try:
        nonBinaryModel = load_model(NONBINARY_PATH)
        binaryModel = load_model(BINARY_PATH)
        
        def create_unet_model(num_classes=1, in_channels=3):
            return Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=num_classes,
            )

        def load_torch_model(weights_path):
            model = create_unet_model()
            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            return model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unetModel = load_torch_model(UNET_PATH).to(device)
        
        return nonBinaryModel, binaryModel, unetModel
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load models
try:
    if models_downloaded:
        nonBinaryModel, binaryModel, unetModel = load_models()
        print("All models loaded successfully")
    else:
        print("Models not downloaded, setting to None")
        nonBinaryModel = binaryModel = unetModel = None
except Exception as e:
    print(f"Failed to load models: {e}")
    nonBinaryModel = binaryModel = unetModel = None

# ----------------------
# Image Processing Functions
# ----------------------
def preprocess_image(image_path, target_size=(256, 256)):
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None
    resized_image = original_image.resize(target_size)
    image_np = np.array(resized_image) / 255.0
    return original_image, image_np

def calculate_bounding_boxes(binary_mask, original_size, scale_x, scale_y, margin=10):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 100:
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            bounding_boxes.append((
                max(0, x - margin),
                max(0, y - margin),
                min(original_size[0], x + w + margin),
                min(original_size[1], y + h + margin)
            ))
    return bounding_boxes

def infer_and_visualize(model, image_path, save_folder):
    try:
        os.makedirs(save_folder, exist_ok=True)
        original_image, image_np = preprocess_image(image_path)
        if original_image is None:
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            predicted_mask = torch.sigmoid(output).cpu().numpy()[0,0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

        draw_original = ImageDraw.Draw(original_image)
        scale_x, scale_y = original_image.size[0]/256.0, original_image.size[1]/256.0
        for x1,y1,x2,y2 in calculate_bounding_boxes(binary_mask, original_image.size, scale_x, scale_y):
            draw_original.rectangle([x1,y1,x2,y2], outline="red", width=5)

        save_path = os.path.join(save_folder, "processed_image.png")
        original_image.save(save_path)
        return save_path
    except Exception as e:
        print(f"Error in infer_and_visualize: {e}")
        traceback.print_exc()
        return None

# ----------------------
# Flask routes
# ----------------------
@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Ensure models are loaded
        if nonBinaryModel is None or binaryModel is None or unetModel is None:
            return jsonify({"error": "Models not loaded properly. Please try again later."}), 503

        # Save uploaded image
        upload_folder = os.path.join(app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        
        # Generate unique filename
        import uuid
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_folder, unique_filename)
        
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Run UNet inference
        output_folder = os.path.join(app.root_path, "static", "output")
        os.makedirs(output_folder, exist_ok=True)
        
        processed_image_path = infer_and_visualize(unetModel, file_path, save_folder=output_folder)

        if processed_image_path is None:
            return jsonify({"error": "Failed to process image with UNet"}), 500

        # Split and predict
        discs_folder = os.path.join(output_folder, "discs")
        os.makedirs(discs_folder, exist_ok=True)
        
        split_result = split_image(processed_image_path)
        if isinstance(split_result, dict) and "error" in split_result:
            return jsonify({"error": f"Failed to split image: {split_result['error']}"}), 500
        
        disc_messages = make_predictions(discs_folder, nonBinaryModel, binaryModel)
        if isinstance(disc_messages, dict) and "error" in disc_messages:
            return jsonify({"error": f"Failed to make predictions: {disc_messages['error']}"}), 500

        # Build response
        processed_image_url = "/static/output/processed_image.png"
        disc_image_urls = [
            f"/static/output/discs/{filename}"
            for filename in os.listdir(discs_folder)
            if filename.endswith((".png", ".jpg", ".jpeg"))
        ]

        return jsonify({
            "output_image_url": processed_image_url,
            "disc_images": [
                {"url": url, "message": msg}
                for url, msg in zip(disc_image_urls, disc_messages)
            ],
        })

    except Exception as e:
        print(f"Error in upload: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        response = get_response(user_message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    models_status = {
        "nonBinaryModel": nonBinaryModel is not None,
        "binaryModel": binaryModel is not None,
        "unetModel": unetModel is not None
    }
    
    all_models_loaded = all(models_status.values())
    
    return jsonify({
        "status": "healthy" if all_models_loaded else "degraded",
        "message": "MedVisor AI Backend is running",
        "models_loaded": all_models_loaded,
        "models_status": models_status
    })

# Add static file serving routes
@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory('static/uploads', filename)

@app.route('/static/output/<path:filename>')
def serve_output(filename):
    return send_from_directory('static/output', filename)

@app.route('/static/output/discs/<path:filename>')
def serve_discs(filename):
    return send_from_directory('static/output/discs', filename)

# ----------------------
# Hugging Face Spaces Integration
# ----------------------
def create_gradio_app():
    """Create and return the Gradio app for Hugging Face Spaces"""
    try:
        from app_gradio import create_interface
        return create_interface()
    except ImportError:
        print("Gradio app not available, falling back to Flask")
        return None

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    # Check if we're running in Hugging Face Spaces
    if os.environ.get("SPACE_ID"):
        # We're in Hugging Face Spaces, use Gradio
        demo = create_gradio_app()
        if demo:
            demo.launch(server_name="0.0.0.0", server_port=7860)
        else:
            # Fallback to Flask
            port = int(os.environ.get("PORT", 5000))
            host = os.environ.get("HOST", "0.0.0.0")
            print(f"Starting Flask server on {host}:{port}")
            app.run(host=host, port=port, debug=False)
    else:
        # Local development, use Flask
        port = int(os.environ.get("PORT", 5000))
        host = os.environ.get("HOST", "0.0.0.0")
        print(f"Starting Flask server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
