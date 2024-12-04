def explore_data(data_path):
    """Explore and visualize the dataset"""
    print("\nExploring Dataset Structure:")
    print("-" * 50)

    splits = ['train', 'validation', 'test']
    for split in splits:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            classes = sorted(os.listdir(split_path))
            total_images = sum(len(os.listdir(os.path.join(split_path, cls)))
                             for cls in classes)

            print(f"\n{split.capitalize()} Set:")
            print(f"Number of classes: {len(classes)}")
            print(f"Total images: {total_images}")
            print(f"Example classes: {', '.join(classes[:5])}...")

    # Visualize sample images
    print("\nVisualizing Sample Images...")
    train_path = os.path.join(data_path, 'train')
    classes = sorted(os.listdir(train_path))

    plt.figure(figsize=(15, 10))
    for i in range(9):
        class_name = random.choice(classes)
        class_path = os.path.join(train_path, class_name)
        img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)

        img = Image.open(img_path)
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(f'Class: {class_name}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    plt.show()

# Explore dataset
data_path = "/root/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/8" # can be change depending on the path of your data set
explore_data(data_path)