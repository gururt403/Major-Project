import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.class_names = Config.CLASS_NAMES
        
    def evaluate(self, save_dir='results'):
        """Run complete evaluation"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("Running model evaluation...")
        
        # Get predictions
        all_preds, all_labels, all_probs = self.get_predictions()
        
        # Generate reports
        self.print_classification_report(all_labels, all_preds, save_dir)
        self.plot_confusion_matrix(all_labels, all_preds, save_dir)
        self.plot_roc_curves(all_labels, all_probs, save_dir)
        self.plot_per_class_accuracy(all_labels, all_preds, save_dir)
        
        print(f"\nEvaluation complete! Results saved to {save_dir}")
        
        return all_preds, all_labels, all_probs
    
    def get_predictions(self):
        """Get model predictions on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def print_classification_report(self, labels, preds, save_dir):
        """Generate and save classification report"""
        report = classification_report(
            labels, preds, 
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)
        
        # Save to file
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
    
    def plot_confusion_matrix(self, labels, preds, save_dir):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        print(f"✓ Confusion matrix saved")
    
    def plot_roc_curves(self, labels, probs, save_dir):
        """Plot ROC curves for each class"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        labels_bin = label_binarize(labels, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, 
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves.png', dpi=300)
        plt.close()
        
        print(f"✓ ROC curves saved")
    
    def plot_per_class_accuracy(self, labels, preds, save_dir):
        """Plot per-class accuracy"""
        accuracies = []
        
        for i in range(len(self.class_names)):
            mask = labels == i
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).sum() / mask.sum()
                accuracies.append(acc * 100)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_names, accuracies, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.ylim([0, 105])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'per_class_accuracy.png', dpi=300)
        plt.close()
        
        print(f"✓ Per-class accuracy plot saved")


class rPPGValidator:
    """Validate rPPG heart rate estimation"""
    
    def __init__(self, rppg_estimator):
        self.rppg = rppg_estimator
        
    def validate_on_dataset(self, video_paths, ground_truth_hrs, save_dir='results'):
        """Validate rPPG on dataset with ground truth"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("\nValidating rPPG Heart Rate Estimation...")
        
        errors = []
        predictions = []
        valid_gt = []
        
        for i, (video_path, gt_hr) in enumerate(zip(video_paths, ground_truth_hrs)):
            print(f"Processing video {i+1}/{len(video_paths)}...", end='\r')
            
            estimated_hr = self.process_video(video_path)
            
            if estimated_hr and 40 <= estimated_hr <= 200:
                error = abs(estimated_hr - gt_hr)
                errors.append(error)
                predictions.append(estimated_hr)
                valid_gt.append(gt_hr)
        
        print()
        
        if len(errors) == 0:
            print("ERROR: No valid predictions!")
            return
        
        # Calculate metrics
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        std = np.std(errors)
        
        print("\n" + "=" * 60)
        print("rPPG VALIDATION RESULTS")
        print("=" * 60)
        print(f"Valid Predictions: {len(predictions)}/{len(video_paths)}")
        print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f} BPM")
        print(f"Standard Deviation: {std:.2f} BPM")
        print(f"Correlation: {np.corrcoef(valid_gt, predictions)[0,1]:.3f}")
        
        # Save metrics
        with open(save_dir / 'rppg_metrics.txt', 'w') as f:
            f.write(f"Valid Predictions: {len(predictions)}/{len(video_paths)}\n")
            f.write(f"MAE: {mae:.2f} BPM\n")
            f.write(f"RMSE: {rmse:.2f} BPM\n")
            f.write(f"STD: {std:.2f} BPM\n")
            f.write(f"Correlation: {np.corrcoef(valid_gt, predictions)[0,1]:.3f}\n")
        
        # Plot predictions vs ground truth
        self.plot_predictions(valid_gt, predictions, save_dir)
        self.plot_bland_altman(valid_gt, predictions, save_dir)
        
        return mae, rmse
    
    def process_video(self, video_path):
        """Process video and return average HR"""
        import mediapipe as mp
        
        cap = cv2.VideoCapture(str(video_path))
        hrs = []
        
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        frame_count = 0
        max_frames = 300  # Process max 10 seconds at 30fps
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = mp_face_detection.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                face_roi = frame[max(0,y):y+h, max(0,x):x+w]
                
                if face_roi.size > 0:
                    hr = self.rppg.process_frame(face_roi)
                    if hr:
                        hrs.append(hr)
            
            frame_count += 1
        
        cap.release()
        mp_face_detection.close()
        
        return np.median(hrs) if len(hrs) > 0 else None
    
    def plot_predictions(self, ground_truth, predictions, save_dir):
        """Plot predictions vs ground truth"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(ground_truth, predictions, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(min(ground_truth), min(predictions))
        max_val = max(max(ground_truth), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Ground Truth HR (BPM)', fontsize=12)
        plt.ylabel('Predicted HR (BPM)', fontsize=12)
        plt.title('rPPG Heart Rate Predictions', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'rppg_predictions.png', dpi=300)
        plt.close()
        
        print(f"✓ Prediction plot saved")
    
    def plot_bland_altman(self, ground_truth, predictions, save_dir):
        """Plot Bland-Altman plot"""
        ground_truth = np.array(ground_truth)
        predictions = np.array(predictions)
        
        mean = (ground_truth + predictions) / 2
        diff = predictions - ground_truth
        
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(mean, diff, alpha=0.6, s=50)
        
        plt.axhline(mean_diff, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_diff:.2f}')
        plt.axhline(mean_diff + 1.96*std_diff, color='gray', 
                   linestyle='--', linewidth=2, 
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
        plt.axhline(mean_diff - 1.96*std_diff, color='gray', 
                   linestyle='--', linewidth=2,
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
        
        plt.xlabel('Mean HR (BPM)', fontsize=12)
        plt.ylabel('Difference (Predicted - Ground Truth)', fontsize=12)
        plt.title('Bland-Altman Plot', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'bland_altman.png', dpi=300)
        plt.close()
        
        print(f"✓ Bland-Altman plot saved")