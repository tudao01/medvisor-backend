import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from segmentation_models_pytorch import Unet
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from prediction import split_image
from prediction import make_predictions
from chat import get_response

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Model setup
def create_unet_model(num_classes=1, in_channels=3):
    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=num_classes,
    )
    return model

# Load the model and weights
def load_torch_model(weights_path):
    model = create_unet_model()
    try:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint, strict=False)
    except FileNotFoundError:
        print(f"Error: Model weights not found at {weights_path}")
        return None
    model.eval()
    return model

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_torch_model("unet_spine_segmentation.pth")
if model:
    model = model.to(device)

# Image preprocessing and bounding box visualization
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
            # Scale and enlarge bounding box
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            bounding_boxes.append((
                max(0, x - margin),
                max(0, y - margin),
                min(original_size[0], x + w + margin),
                min(original_size[1], y + h + margin)
            ))
    return bounding_boxes

# Inference and visualization
def infer_and_visualize(model, image_path, save_folder):
    original_image, image_np = preprocess_image(image_path)
    if original_image is None:
        return None
    
    # Prepare image tensor
    image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    # Draw bounding boxes
    draw_original = ImageDraw.Draw(original_image)
    scale_x, scale_y = original_image.size[0] / 256.0, original_image.size[1] / 256.0
    for x1, y1, x2, y2 in calculate_bounding_boxes(binary_mask, original_image.size, scale_x, scale_y):
        draw_original.rectangle([x1, y1, x2, y2], outline="red", width=5)

    # Save and display results directly in static/output
    output_filename = "processed_image.png"
    save_path = os.path.join(save_folder, output_filename)
    original_image.save(save_path)
    
    return save_path

# Flask route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        output_folder = os.path.join(app.root_path, 'static', 'output')
        os.makedirs(output_folder, exist_ok=True)
        output_path = infer_and_visualize(model, file_path, save_folder=output_folder)

        split_image()
        disc_messages = make_predictions()

        processed_image_url = "/static/output/processed_image.png"
        discs_folder = "./static/output/discs"
        disc_image_urls = [
            f"/static/output/discs/{filename}" 
            for filename in os.listdir(discs_folder)
            if filename.endswith(('.png', '.jpg', '.jpeg'))
        ]
        if output_path:
            return jsonify({
                "output_image_url": processed_image_url,
                "disc_images": [
                    {"url": url, "message": message}
                    for url, message in zip(disc_image_urls, disc_messages)
                ]
            })
        else:
            return jsonify({"error": "Failed to process the image"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/get", methods=["GET"])
def chat():
    user_message = request.args.get('msg')  # Get user input from URL query string
    print(f"Received message: {user_message}")
    if user_message:
        response = get_response(user_message)  # Get chatbot response
        print(f"Generated response: {response}")
        return jsonify({"response": response})
    return jsonify({"response": "I do not understand..."})

if __name__ == '__main__':
    app.run(debug=True)
