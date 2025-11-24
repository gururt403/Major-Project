"""
Improved Training Script with Advanced Techniques
- Mixed precision training
- Learning rate scheduling (Cosine Annealing)
- Gradient accumulation
- Model checkpointing
- Early stopping
- Data augmentation
- Label smoothing
- Mixup augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import json

from mhavh_model import MHAVH
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixupDataset:
    """Apply Mixup augmentation"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target)
        
        return (1 - self.epsilon) * nll + self.epsilon * loss / n_classes


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, mixup, accumulation_steps=1):
    """Train for one epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        # Apply mixup
        if mixup:
            images, labels_a, labels_b, lam = mixup(images, labels)
        
        # Mixed precision training
        try:
            amp_context = autocast('cuda' if torch.cuda.is_available() else 'cpu')
        except TypeError:
            amp_context = autocast()  # Fallback
        
        with amp_context:
            outputs, _ = model(images)
            
            if mixup:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        _, predictions = torch.max(outputs, 1)
        if mixup:
            total_correct += (lam * (predictions == labels_a).sum().item() + 
                            (1 - lam) * (predictions == labels_b).sum().item())
        else:
            total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': total_correct / total_samples
        })
    
    return total_loss / len(dataloader), total_correct / total_samples


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue
            
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            try:
                amp_context = autocast('cuda' if device.type == 'cuda' else 'cpu')
            except TypeError:
                amp_context = autocast()
            
            with amp_context:
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(len(dataloader), 1)
    
    return avg_loss, accuracy, all_preds, all_labels


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Config.MODEL_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model with improved architecture
    model = MHAVH(
        num_classes=Config.NUM_CLASSES,
        seq_length=Config.SEQUENCE_LENGTH,
        feature_dim=512,
        backbone='resnet50',  # Can be changed to 'efficientnet_b0' etc.
        dropout=Config.DROPOUT
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(epsilon=Config.LABEL_SMOOTHING)
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6
    )
    
    # Mixed precision training
    try:
        scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    except TypeError:
        scaler = GradScaler()  # Fallback for older PyTorch versions
    
    # Mixup augmentation
    mixup = MixupDataset(alpha=0.2)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # Create synthetic data for demonstration (replace with actual data)
    logger.info("Creating synthetic training data for demonstration...")
    logger.info("⚠️ IMPORTANT: Replace this with your actual dataset!")
    
    from torch.utils.data import TensorDataset
    
    # Generate synthetic training data
    num_train_samples = 320  # Should be divisible by batch_size
    num_val_samples = 80
    
    # Synthetic images: (batch, seq_len, channels, height, width)
    train_images = torch.randn(num_train_samples, Config.SEQUENCE_LENGTH, 3, 224, 224)
    train_labels = torch.randint(0, Config.NUM_CLASSES, (num_train_samples,))
    
    val_images = torch.randn(num_val_samples, Config.SEQUENCE_LENGTH, 3, 224, 224)
    val_labels = torch.randint(0, Config.NUM_CLASSES, (num_val_samples,))
    
    # Create datasets
    class VideoDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {
                'images': self.images[idx],
                'labels': self.labels[idx]
            }
    
    train_dataset = VideoDataset(train_images, train_labels)
    val_dataset = VideoDataset(val_images, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {num_train_samples}, Val samples: {num_val_samples}")
    
    # Training loop
    best_val_acc = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    logger.info("Starting training...")
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            scaler, mixup, accumulation_steps=2
        )
        
        # Validation phase
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, output_dir / 'best_model.pth')
            logger.info("✅ Best model saved!")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered!")
            break
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
