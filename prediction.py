import pandas as pd
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from collections import OrderedDict
from tensorflow.keras.models import load_model
import cv2

def split_image(input_image_path):
    """
    Splits the uploaded image into discs and saves them into a subfolder.
    """
    try:
        # Create output discs folder relative to input image
        output_dir = os.path.join(os.path.dirname(input_image_path), "discs")
        os.makedirs(output_dir, exist_ok=True)

        # Clear previous disc images
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Load and process the image
        image = cv2.imread(input_image_path)
        if image is None:
            return {"error": f"Could not load image from {input_image_path}"}
            
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color mask
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # top to bottom

        # Save each contour as a separate image
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            disc = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f"disc_{i + 1}.png"), disc)

        return output_dir  # return the folder for predictions

    except Exception as e:
        print(f"Error in split_image: {e}")
        return {"error": str(e)}


def make_predictions(discs_folder, nonBinaryModel, binaryModel):
    """
    Runs predictions on all disc images in the folder.
    """
    try:
        messages = []
        disc_files = [f for f in os.listdir(discs_folder) if f.endswith(".png")]
        
        if not disc_files:
            return ["No disc images found for prediction"]
            
        for filename in disc_files:
            img_path = os.path.join(discs_folder, filename)

            # Preprocess
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img)
            img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
            img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

            # Predictions
            nonBinaryPred = nonBinaryModel.predict(img_preprocessed, verbose=0)
            binaryPred = binaryModel.predict(img_preprocessed, verbose=0)

            predicted_pfirrman = np.argmax(nonBinaryPred[0]) + 1
            predicted_modic = np.argmax(nonBinaryPred[1])

            msg = (
                f"Pfirrman Grade: {predicted_pfirrman}\n"
                f"Modic: {predicted_modic}\n"
                f"Up Endplate: {binaryPred[0][0][0]:.2f}\n"
                f"Low Endplate: {binaryPred[1][0][0]:.2f}\n"
                f"Disc Herniation: {binaryPred[2][0][0]:.2f}\n"
                f"Disc Narrowing: {binaryPred[3][0][0]:.2f}\n"
                f"Disc Bulging: {binaryPred[4][0][0]:.2f}\n"
                f"Spondylilisthesis: {binaryPred[5][0][0]:.2f}\n"
            )
            messages.append(msg)

        return messages

    except Exception as e:
        print(f"Error in make_predictions: {e}")
        return {"error": str(e)}