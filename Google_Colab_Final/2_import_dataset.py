import kagglehub

# Download latest version
path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")

print("Path to dataset files:", path)

# U CAN IMPORT YOUR OWN DATASET IN default directory which is /content