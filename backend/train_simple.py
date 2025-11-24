"""
Simplified but improved training script that works reliably
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from datetime import datetime
import json
from pathlib import Path

from mhavh_model import MHAVH
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for better generalization"""
    
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = self.log_softmax(preds)
        loss = -log_preds.sum(dim=-1).mean()
        nll = nn.functional.nll_loss(log_preds, target)
        return (1 - self.epsilon) * nll + self.epsilon * loss / n_classes


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['images'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / total_samples:.2%}'
        })
    
    return total_loss / len(dataloader), total_correct / total_samples


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['images'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(len(dataloader), 1)
    
    return avg_loss, accuracy


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_dir = Config.MODEL_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MHAVH(
        num_classes=Config.NUM_CLASSES,
        seq_length=Config.SEQUENCE_LENGTH,
        feature_dim=512,
        backbone='resnet50',
        dropout=0.3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Create synthetic data for demonstration
    logger.info("\n⚠️ Creating synthetic training data for demonstration")
    logger.info("Replace this with your actual dataset!\n")
    
    class VideoDataset(Dataset):
        def __init__(self, num_samples, seq_length, num_classes):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate random video sequence
            images = torch.randn(self.seq_length, 3, 224, 224)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return {'images': images, 'labels': label}
    
    train_dataset = VideoDataset(320, Config.SEQUENCE_LENGTH, Config.NUM_CLASSES)
    val_dataset = VideoDataset(80, Config.SEQUENCE_LENGTH, Config.NUM_CLASSES)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Smaller batch size for stability
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    best_val_acc = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    logger.info("\nStarting training...\n")
    
    num_epochs = 10  # Reduced for quick demonstration
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, output_dir / 'best_model.pth')
            logger.info("✅ Best model saved!\n")
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
