import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from data_preprocessing import class_names

# load the saved model
model = tf.keras.models.load_model('C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/fruit_model.h5')

# function to predict the class of an image
def predict(model, img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 0)
    return predicted_class, confidence

# Predict on a new image
img_path = '/Predict/image1.jpg'  # path of the image you want to predict
predicted_class, confidence = predict(model, img_path)

print(f"Predicted class: {predicted_class} with {confidence}% confidence.")
