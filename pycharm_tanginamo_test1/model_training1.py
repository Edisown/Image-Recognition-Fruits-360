import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the model
model_path = '/pycharm_tanginamo_test1/fruit_recognition.keras'  # Replace with your actual model path
model = tf.keras.models.load_model(model_path, custom_objects={'RandomRotation': tf.keras.layers.RandomRotation})

# Load the image for prediction
img_path = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/prediction_test/images/image2.jpg'  # Replace with your image file path
img = image.load_img(img_path, target_size=(100, 100))  # Resize image to match model input size

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand dimensions to match the batch size (model expects a batch of images)
img_array = tf.expand_dims(img_array, 0)

# Apply the same preprocessing as done during model training (e.g., for ResNet)
preprocess_input = tf.keras.applications.resnet.preprocess_input
img_array = preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Get class names (ensure this matches your training data's class names)
train_dir = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/data2/fruits-360_dataset/fruits-360/Training'  # Adjust path if needed
class_names = sorted(os.listdir(train_dir))
predicted_class = class_names[np.argmax(predictions[0])]
confidence = round(100 * np.max(predictions[0]), 0)

# Print the result
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")

# Display the image and prediction
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} with {confidence}% confidence")
plt.axis('off')
plt.show()
