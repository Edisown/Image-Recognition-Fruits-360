class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class FruitVegCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Function to visualize feature maps
def visualize_feature_maps(model, sample_image):
    """Visualize feature maps after each conv block"""
    model.eval()

    # Get feature maps after each conv block
    feature_maps = []
    x = sample_image.unsqueeze(0).to(device)

    for block in model.features:
        x = block(x)
        feature_maps.append(x.detach().cpu())

    # Plot feature maps
    plt.figure(figsize=(15, 10))
    for i, fmap in enumerate(feature_maps):
        # Plot first 6 channels of each block
        fmap = fmap[0][:6].permute(1, 2, 0)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())

        for j in range(min(6, fmap.shape[-1])):
            plt.subplot(5, 6, i*6 + j + 1)
            plt.imshow(fmap[:, :, j], cmap='viridis')
            plt.title(f'Block {i+1}, Ch {j+1}')
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/feature_maps.png')
    plt.show()

# Initialize model and visualize feature maps
model = FruitVegCNN(num_classes=len(train_dataset.classes)).to(device)
sample_image, _ = train_dataset[0]
visualize_feature_maps(model, sample_image)