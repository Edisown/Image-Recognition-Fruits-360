import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# Improved training configurations
class OptimizedConfig:
    def __init__(self):
        self.image_size = 256  # Increased from 224
        self.batch_size = 16   # Smaller batch size for better generalization
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.epochs = 50
        self.dropout = 0.3

# Enhanced data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Optimized model architecture
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Use pretrained ResNet50 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0',
                                     'resnet50', pretrained=True)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        # Modified classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Optimized training function
def train_with_optimization(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    # One Cycle Learning Rate Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    # Gradient Scaler for mixed precision training
    scaler = GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch+1}/{config.epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
            }, 'optimized_model.pth')
            print(f'New best validation accuracy: {best_val_acc:.2f}%')

    return history

# Create dataloaders with optimized configuration
config = OptimizedConfig()
train_dataset = FruitVegDataset(data_path, 'train', train_transform)
val_dataset = FruitVegDataset(data_path, 'validation', val_transform)

train_loader = DataLoader(train_dataset,
                         batch_size=config.batch_size,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=True)
val_loader = DataLoader(val_dataset,
                       batch_size=config.batch_size,
                       shuffle=False,
                       num_workers=4,
                       pin_memory=True)

# Initialize and train optimized model
model = OptimizedCNN(num_classes=len(train_dataset.classes)).to(device)
history = train_with_optimization(model, train_loader, val_loader, config)