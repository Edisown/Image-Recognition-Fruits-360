import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Load dataset paths
train_dir = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/data/fruits-360_dataset/fruits-360/Training'  # Path to your training dataset
val_dir = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/data/fruits-360_dataset/fruits-360/Validation'  # Path to your validation dataset
test_dir = 'C:/Users/ediso/Desktop/Image-Recognition-Fruits-360/data/fruits-360_dataset/fruits-360/Test'  # Path to your test dataset

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    batch_size=32,
    image_size=(100, 100),
    seed=123,
    shuffle=True,
)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    batch_size=32,
    image_size=(100, 100),
    seed=42,
    shuffle=False,
)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    batch_size=32,
    image_size=(100, 100),
    seed=123,
    shuffle=False,
)

# Cache and prefetch datasets for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Get class names from the directory structure
class_names = sorted(os.listdir(train_dir))  # This assumes your class labels are the subfolder names in the train_dir
num_classes = len(class_names)
print("Number of Classes:", num_classes)

# Function to display a batch of images
def show_images(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):  # Take one batch
        for i in range(25):  # Display 25 images
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

# Show some images from the training dataset
show_images(train_ds, class_names)
