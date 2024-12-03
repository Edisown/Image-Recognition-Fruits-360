import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import os

# Assuming class_names is already available or you can manually define them based on your dataset
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
               'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3',
               'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado',
               'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry',
               'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1',
               'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow',
               'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2',
               'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White',
               'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White',
               'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer',
               'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo',
               'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled',
               'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2',
               'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams',
               'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis',
               'Physalis with Husk',
               'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate',
               'Pomelo Sweetie',
               'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
               'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2',
               'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow',
               'Tomato not Ripened', 'Walnut', 'Watermelon']

# Load the trained model
model = load_model(
    'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/fruit_recognition.keras')  # Path to your saved model

# Directory containing the images
image_dir = '/prediction_test/images'  # Update with the path to your images folder

# Get a list of image paths in the directory
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png'))]

# List to store processed image arrays
img_arrays = []


# Define the data augmentation function
def augment_image(image):
    # Apply random flip
    image = tf.image.random_flip_left_right(image)
    # Apply random rotation
    image = tf.image.random_rotation(image, 0.2)
    return image


# Preprocess each image
for image_path in image_paths:
    # Load image and resize it to the size expected by the model
    img = load_img(image_path, target_size=(100, 100))  # Resize to match model's input size
    img_array = img_to_array(img)  # Convert to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 100, 100, 3)
    img_array = img_array / 255.0  # Scale pixel values to [0, 1] if needed

    # Manually augment the image
    img_array = augment_image(img_array)

    img_arrays.append(img_array)

# Stack all images into a single batch (batch_size, height, width, channels)
img_batch = np.vstack(img_arrays)

# Predict the classes of the images
predictions = model.predict(img_batch)

# Iterate over predictions and print results for each image
for i, prediction in enumerate(predictions):
    predicted_index = np.argmax(prediction, axis=-1)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Print results
    print(f"Image {image_paths[i]}: Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
