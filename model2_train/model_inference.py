import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Ensure class_names is defined correctly
# Example: class_names = ['Apple', 'Banana', 'Orange']
from data_preprocessing import class_names  # Update with your correct import path

# Load the saved model
try:
    model = tf.keras.models.load_model(
        'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/model2_train/Result Train/fruit_model.h5'
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Function to predict the class of an image
def predict(model, img_path):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(100, 100))  # Ensure the size matches your model input
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image (if model requires normalization)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Predict on a new image
img_path = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/Predict/image1.jpg'  # Update with your image path
if os.path.exists(img_path):
    predicted_class, confidence = predict(model, img_path)
    if predicted_class:
        print(f"Predicted class: {predicted_class} with {confidence}% confidence.")
    else:
        print("Prediction failed.")
else:
    print(f"Image file not found: {img_path}")
