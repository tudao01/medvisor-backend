import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download

from prediction import split_image, make_predictions
from chat import get_response

# ----------------------
# Flask Setup
# ----------------------
app = Flask(__name__)
CORS(app)

# ----------------------
# Hugging Face Hub Config
# ----------------------
REPO_ID = "your-username/your-model-repo"  # TODO: replace with your repo

# Local paths
NONBINARY_PATH = "models/nonBinaryIndividualPredictions.keras"
BINARY_PATH = "models/binaryIndividualPredictions.keras"
UNET_PATH = "models/unet_spine_segmentation.pth"

# ----------------------
# Download models if missing
# ----------------------
def download_model_if_missing(local_path, filename):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {filename} from Hugging Face Hub...")
        # Download and return the path, then copy to desired location if needed
        downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        # If the downloaded file isn't in the right location, move it
        if downloaded_path != local_path:
            import shutil
            shutil.copy2(downloaded_path, local_path)

download_model_if_missing(NONBINARY_PATH, "nonBinaryIndividualPredictions.keras")
download_model_if_missing(BINARY_PATH, "binaryIndividualPredictions.keras")
download_model_if_missing(UNET_PATH, "unet_spine_segmentation.pth")

# ----------------------
# Load models
# ----------------------
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

# ----------------------
# Image Processing Functions (MISSING IN YOUR CODE)
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
    os.makedirs(save_folder, exist_ok=True)
    original_image, image_np = preprocess_image(image_path)
    if original_image is None:
        return None

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
        # Save uploaded image
        upload_folder = os.path.join(app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Run UNet inference
        output_folder = os.path.join(app.root_path, "static", "output")
        processed_image_path = infer_and_visualize(unetModel, file_path, save_folder=output_folder)

        if processed_image_path is None:
            return jsonify({"error": "Failed to process image"}), 500

        # Split and predict
        discs_folder = os.path.join(output_folder, "discs")
        split_image(processed_image_path, discs_folder)
        disc_messages = make_predictions(discs_folder, nonBinaryModel, binaryModel)

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
        return jsonify({"error": str(e)}), 500

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
    return jsonify({"status": "healthy", "message": "MedVisor AI Backend is running"})

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
