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

def split_image():      
    nonBinaryModel = load_model('nonBinaryIndividualPredictions.keras')
    binaryModel = load_model('binaryIndividualPredictions.keras')

    # Specify the input image path
    input_image = "static/output/processed_image.png"  # Replace with your specific image name
    output_dir = "static/output/discs"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Load and process the image
    image = cv2.imread(input_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Apply the red color mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from top to bottom by y-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Loop through each contour and save the bounding boxes as separate images
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        red_box = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f"disc_{i + 1}.png")
        cv2.imwrite(output_path, red_box)


def make_predictions():

    nonBinaryModel = load_model('nonBinaryIndividualPredictions.keras')
    binaryModel = load_model('binaryIndividualPredictions.keras')
    # Define the directory containing images to process
    input_dir = "static/output/discs"
    message_array = []
    # Loop through each image in the directory
    disc_num = 1
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):  # Adjust the extension if needed
            test_image_path = os.path.join(input_dir, filename)
            
            # Load and preprocess the image
            img = Image.open(test_image_path).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)

            # Preprocess the image for ResNet50
            img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array)
            img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension

            # Run predictions with your models
            nonBinaryPredictions = nonBinaryModel.predict(img_preprocessed)
            binaryPredictions = binaryModel.predict(img_preprocessed)

            # Interpret non-binary predictions
            predicted_pfirrman = np.argmax(nonBinaryPredictions[0]) + 1  # Add 1 if classes are 1-5
            predicted_modic = np.argmax(nonBinaryPredictions[1])
     
            message = (
                f"Pfirrman Grade: {predicted_pfirrman}\n"
                f"Modic: {predicted_modic}\n"
                f"Up Endplate: {binaryPredictions[0][0][0]:.2f}\n"
                f"Low Endplate: {binaryPredictions[1][0][0]:.2f}\n"
                f"Disc Herniation: {binaryPredictions[2][0][0]:.2f}\n"
                f"Disc Narrowing: {binaryPredictions[3][0][0]:.2f}\n"
                f"Disc Bulging: {binaryPredictions[4][0][0]:.2f}\n"
                f"Spondylilisthesis: {binaryPredictions[5][0][0]:.2f}\n"
            )
            message_array.append(message)
            disc_num += 1

    print(message_array)
    return message_array
