"""
Automated rPPG (Heart Rate) Model Training Script
Trains the rPPG estimator for better heart rate detection
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class rPPGDataset(Dataset):
    """Dataset for rPPG training"""
    
    def __init__(self, video_dir, annotations_file, transform=None):
        self.video_dir = Path(video_dir)
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = self.video_dir / row['video_file']
        ground_truth_hr = float(row['ground_truth_hr'])
        
        # Extract frames from video
        frames = self._extract_frames(str(video_path), num_frames=30)
        
        if frames is None:
            return None
        
        # Convert to tensor
        frames = torch.from_numpy(frames).float() / 255.0
        
        return {
            'frames': frames,
            'heart_rate': torch.tensor(ground_truth_hr, dtype=torch.float32),
            'video_file': row['video_file']
        }
    
    def _extract_frames(self, video_path, num_frames=30):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while len(frames) < num_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize to 128x128 for efficiency
                frame = cv2.resize(frame, (128, 128))
                frames.append(frame)
            
            cap.release()
            
            if len(frames) < num_frames:
                return None
            
            return np.array(frames)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return None


class SimpleRPPGModel(nn.Module):
    """Simple CNN-based rPPG model"""
    
    def __init__(self):
        super().__init__()
        
        # 3D CNN for temporal-spatial learning
        self.conv3d_1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 2, 2))
        
        self.conv3d_2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv3d_3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        
        self.fc1 = nn.Linear(64 * 32 * 32 * 30, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        # x shape: (batch, 30, 3, 128, 128)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, 3, 30, 128, 128)
        
        x = self.relu(self.conv3d_1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3d_2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3d_3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    valid_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue
        
        frames = batch['frames'].to(device)
        hr = batch['heart_rate'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        pred_hr = model(frames)
        loss = criterion(pred_hr, hr)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_samples += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / max(valid_samples, 1)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    mae = 0
    valid_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            frames = batch['frames'].to(device)
            hr = batch['heart_rate'].to(device).unsqueeze(1)
            
            pred_hr = model(frames)
            loss = criterion(pred_hr, hr)
            
            total_loss += loss.item()
            mae += torch.abs(pred_hr - hr).mean().item()
            valid_samples += 1
    
    avg_loss = total_loss / max(valid_samples, 1)
    avg_mae = mae / max(valid_samples, 1)
    
    return avg_loss, avg_mae


def main():
    parser = argparse.ArgumentParser(description='Train rPPG model')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test-split', type=float, default=0.1, help='Test split')
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    annotations_file = data_dir / 'annotations.csv'
    if not annotations_file.exists():
        logger.error("annotations.csv not found!")
        return
    
    logger.info("Loading dataset...")
    dataset = rPPGDataset(data_dir, str(annotations_file))
    
    # Split data
    n = len(dataset)
    test_size = int(n * args.test_split)
    val_size = int(n * args.val_split)
    train_size = n - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    logger.info(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Model
    model = SimpleRPPGModel().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} BPM")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'rppg_best.pth')
            logger.info("✅ Best model saved!")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_mae = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f} BPM")
    
    # Save results
    results = {
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'training_history': training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'rppg_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
