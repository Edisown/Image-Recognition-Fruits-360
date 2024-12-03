from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Path to your uploaded image
img_path = '/content/image3.jpg'  # Replace with your image file name

# Load the image and resize to (100, 100) to match model's input size
img = image.load_img(img_path, target_size=(100, 100))

# Convert the image to a numpy array
img_array = image.img_to_array(img)

# Expand dimensions to match the batch size (model expects a batch of images)
img_array = tf.expand_dims(img_array, 0)

# Apply the same preprocessing used during model training (e.g., ResNet preprocessing)
preprocess_input = tf.keras.applications.resnet.preprocess_input
img_array = preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Apply softmax to convert logits to probabilities
predictions = tf.nn.softmax(predictions[0])

# Get predicted class names from your training directory
# Ensure the class names match what was used during training
class_names = sorted(os.listdir(train_dir))  # Replace with the directory where your classes are stored
predicted_class = class_names[np.argmax(predictions)]
confidence = round(100 * np.max(predictions), 0)

# Print the result
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")

# Display the image and prediction
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} with {confidence}% confidence")
plt.axis('off')
plt.show()