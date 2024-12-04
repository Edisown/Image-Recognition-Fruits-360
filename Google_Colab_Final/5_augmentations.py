class FruitVegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Visualize augmentations
def show_augmentations(dataset, num_augments=5):
    """Show original image and its augmented versions"""
    idx = random.randint(0, len(dataset)-1)
    img_path = dataset.images[idx]
    original_img = Image.open(img_path).convert('RGB')

    plt.figure(figsize=(15, 5))

    # Show original
    plt.subplot(1, num_augments+1, 1)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')

    # Show augmented versions
    for i in range(num_augments):
        augmented = train_transform(original_img)
        augmented = augmented.permute(1, 2, 0).numpy()
        augmented = (augmented * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])
        augmented = np.clip(augmented, 0, 1)

        plt.subplot(1, num_augments+1, i+2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/augmentations.png')
    plt.show()

# Create datasets and show augmentations
train_dataset = FruitVegDataset(data_path, 'train', train_transform)
show_augmentations(train_dataset)