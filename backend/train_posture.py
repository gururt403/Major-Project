"""
Automated MHAVH Posture Model Training Script
Trains the posture classifier for better posture analysis
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostureDataset(Dataset):
    """Dataset for posture classification"""
    
    CLASSES = {
        'standing': 0,
        'sitting': 1,
        'lying_down': 2,
        'unusual_posture': 3
    }
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class folders
        for class_name, class_idx in self.CLASSES.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.jpg') | class_dir.glob('*.png'):
                    self.images.append(img_file)
                    self.labels.append(class_idx)
        
        logger.info(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return {'image': img, 'label': label, 'path': str(img_path)}
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return None


class SimplePostureModel(nn.Module):
    """Simple CNN-based posture classifier"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predictions = torch.max(logits, 1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / total_samples
        })
    
    return total_loss / max(len(dataloader), 1), total_correct / total_samples


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predictions = torch.max(logits, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(len(dataloader), 1)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train posture model')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test-split', type=float, default=0.1, help='Test split')
    parser.add_argument('--augmentation', type=bool, default=True, help='Use data augmentation')
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = PostureDataset(data_dir, transform=train_transform)
    
    # Split data
    n = len(dataset)
    test_size = int(n * args.test_split)
    val_size = int(n * args.val_split)
    train_size = n - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Update val and test datasets to use val_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    logger.info(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Model
    model = SimplePostureModel(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    best_val_acc = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'mhavh_best.pth')
            logger.info("✅ Best model saved!")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")
    
    # Classification report
    class_names = ['Standing', 'Sitting', 'Lying Down', 'Unusual Posture']
    logger.info("\nClassification Report:")
    logger.info(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Save results
    results = {
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'training_history': training_history,
        'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'posture_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
