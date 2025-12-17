import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "C:/DeepFake_Defenders/xception_final_model.h5"
model = load_model(model_path)

# Define image preprocessing function
def preprocess_image(image_path, target_size=(299, 299)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# Function to predict if an image is real or fake
def predict_image(image_path, threshold=0.5):  # Lowering threshold to 0.4
    image = preprocess_image(image_path)
    prediction = model.predict(image)[0][0]  
    result = "Fake" if prediction >= threshold else "Real"
    print(f"Prediction: {result} (Confidence: {prediction:.4f})")
    return result, prediction

# Test the model on sample images
if __name__ == "__main__":
    test_image_path = r"C:\Users\chand\Desktop\originalphotos\WhatsApp Image 2025-03-18 at 22.22.50_c9004eff.jpg" #test image here
    if os.path.exists(test_image_path):
        result, confidence = predict_image(test_image_path)
    else:
        print("Test image not found. Place a sample image in the correct directory.")
