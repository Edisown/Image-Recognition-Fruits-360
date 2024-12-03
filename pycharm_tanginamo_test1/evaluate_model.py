import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define class names based on your dataset
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']

# Load the trained model
model = load_model('C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/fruit_recognition.keras')  # Replace with the path to your saved model

# Function to preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))  # Resize to match model's input size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Load test images and labels
test_images_dir = '/prediction_test/images'  # Replace with your test images directory
test_image_paths = [os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.endswith(('.jpg', '.png'))]

# Prepare test data
X_test = []
y_true = []

# Assuming you have a way to get true labels for your test images, for example:
# Map filenames or paths to class indices
for image_path in test_image_paths:
    img_array = preprocess_image(image_path)
    X_test.append(img_array)

    # Extract the true label from the image filename (customize this based on how labels are stored)
    label = image_path.split('_')[0]  # Modify based on your image naming convention
    y_true.append(class_names.index(label))  # Find the index of the true label

X_test = np.vstack(X_test)  # Stack all images into a batch
y_true = np.array(y_true)  # Convert true labels into numpy array

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)  # Get the predicted class index

# Print the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Evaluate accuracy
accuracy = np.sum(y_true == y_pred_classes) / len(y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize a few predictions
for i in range(5):  # Display first 5 test images
    plt.imshow(X_test[i])  # Display the image
    plt.title(f"Predicted: {class_names[y_pred_classes[i]]}, Actual: {class_names[y_true[i]]}")
    plt.show()
