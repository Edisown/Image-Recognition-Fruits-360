import tensorflow as tf

try:
    model = tf.keras.models.load_model('C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/fruit_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
