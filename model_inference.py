import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from data_preprocessing import class_names

# Load the saved model
model = tf.keras.models.load_model('fruit_model.h5')

# Function to predict the class of an image
def predict(model, img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 0)
    return predicted_class, confidence

# Example: Predict on a new image
img_path = 'path_to_your_image.jpg'  # Replace with an actual image path
predicted_class, confidence = predict(model, img_path)

print(f"Predicted class: {predicted_class} with {confidence}% confidence.")
