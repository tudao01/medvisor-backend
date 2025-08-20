import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download
import traceback
import random
import json

from prediction import split_image, make_predictions
from chat import get_response

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
            downloaded_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
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
                time.sleep(5)
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
def preprocess_image(image, target_size=(256, 256)):
    try:
        if isinstance(image, str):
            original_image = Image.open(image).convert("RGB")
        else:
            original_image = image.convert("RGB")
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

def infer_and_visualize(model, image):
    try:
        original_image, image_np = preprocess_image(image)
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

        return original_image
    except Exception as e:
        print(f"Error in infer_and_visualize: {e}")
        traceback.print_exc()
        return None

# ----------------------
# Gradio Interface Functions
# ----------------------
def process_image(image):
    """Process uploaded image and return results"""
    if image is None:
        return None, "Please upload an image first."
    
    if nonBinaryModel is None or binaryModel is None or unetModel is None:
        return None, "Models not loaded properly. Please try again later."
    
    try:
        # Run UNet inference
        processed_image = infer_and_visualize(unetModel, image)
        
        if processed_image is None:
            return None, "Failed to process image with UNet"
        
        # Convert to numpy for OpenCV processing
        img_array = np.array(processed_image)
        
        # Split image into discs
        discs_folder = "temp_discs"
        os.makedirs(discs_folder, exist_ok=True)
        
        # Save processed image temporarily
        temp_path = "temp_processed.png"
        processed_image.save(temp_path)
        
        split_result = split_image(temp_path)
        if isinstance(split_result, dict) and "error" in split_result:
            return processed_image, f"Failed to split image: {split_result['error']}"
        
        # Make predictions
        disc_messages = make_predictions(discs_folder, nonBinaryModel, binaryModel)
        if isinstance(disc_messages, dict) and "error" in disc_messages:
            return processed_image, f"Failed to make predictions: {disc_messages['error']}"
        
        # Create result text
        result_text = "Analysis Results:\n\n"
        for i, msg in enumerate(disc_messages):
            result_text += f"Disc {i+1}:\n{msg}\n\n"
        
        # Clean up temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return processed_image, result_text
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        traceback.print_exc()
        return None, f"Error processing image: {str(e)}"

def chat_with_bot(message, history):
    """Chat with the bot"""
    if not message.strip():
        return "", history
    
    try:
        response = get_response(message)
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history.append((message, error_msg))
        return "", history

# ----------------------
# Create Gradio Interface
# ----------------------
def create_interface():
    with gr.Blocks(title="MedVisor AI - Medical Image Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏥 MedVisor AI
            ## Medical Image Analysis Powered by Artificial Intelligence
            
            Upload your medical images for automated analysis and chat with our AI assistant.
            """
        )
        
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Upload Medical Image", type="pil")
                    analyze_btn = gr.Button("Analyze Image", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(label="Processed Image")
                    output_text = gr.Textbox(label="Analysis Results", lines=10)
            
            analyze_btn.click(
                fn=process_image,
                inputs=[input_image],
                outputs=[output_image, output_text]
            )
        
        with gr.Tab("AI Chat Assistant"):
            chatbot = gr.Chatbot(
                label="Chat with MedVisor AI",
                height=400,
                show_label=True
            )
            msg = gr.Textbox(
                label="Type your message",
                placeholder="Ask me about medical imaging, spine conditions, or general health questions...",
                lines=2
            )
            clear = gr.Button("Clear Chat")
            
            msg.submit(
                fn=chat_with_bot,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        with gr.Tab("About"):
            gr.Markdown(
                """
                ## About MedVisor AI
                
                MedVisor AI is an advanced medical image analysis platform that combines:
                
                - **UNet Segmentation**: Automatically detects and segments spine structures
                - **Multi-class Classification**: Analyzes disc conditions using Pfirrman grading
                - **Binary Classification**: Detects various spine pathologies
                - **AI Chat Assistant**: Provides medical information and guidance
                
                ### Supported Analysis:
                - Pfirrman Grade Classification
                - Modic Changes Detection
                - Disc Herniation Analysis
                - Spinal Stenosis Assessment
                - Endplate Changes Detection
                
                ### Team:
                - Love Bhusal
                - Tu Dao
                - Elden Delguia
                - Riley Mckinney
                - Sai Peram
                - Rishil Uppaluru
                
                **Note**: This tool is for educational and research purposes only. Always consult with healthcare professionals for medical decisions.
                """
            )
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
