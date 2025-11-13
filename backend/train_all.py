"""
End-to-End Training Script
Trains both rPPG and MHAVH models automatically
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_structure(data_dir):
    """Validate data directory structure"""
    data_dir = Path(data_dir)
    
    errors = []
    
    # Check rPPG data
    rppg_dir = data_dir / 'rppg'
    if rppg_dir.exists():
        annotations = rppg_dir / 'annotations.csv'
        if not annotations.exists():
            errors.append("❌ rppg/annotations.csv not found")
        else:
            logger.info("✅ rPPG data structure valid")
    else:
        logger.info("ℹ️ rPPG data not found (optional)")
    
    # Check posture data
    posture_dir = data_dir / 'posture'
    if posture_dir.exists():
        classes = ['standing', 'sitting', 'lying_down', 'unusual_posture']
        missing_classes = []
        
        for cls in classes:
            cls_dir = posture_dir / cls
            if not cls_dir.exists():
                missing_classes.append(cls)
            else:
                num_images = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')))
                logger.info(f"  {cls}: {num_images} images")
        
        if missing_classes:
            errors.append(f"❌ Missing posture classes: {', '.join(missing_classes)}")
        else:
            logger.info("✅ Posture data structure valid")
    else:
        logger.info("ℹ️ Posture data not found (optional)")
    
    if errors:
        logger.error("\nData validation errors:")
        for error in errors:
            logger.error(error)
        return False
    
    return True


def train_rppg(data_dir, output_dir, args):
    """Train rPPG model"""
    rppg_dir = Path(data_dir) / 'rppg'
    
    if not rppg_dir.exists():
        logger.info("⏭️ Skipping rPPG training (data not found)")
        return True
    
    logger.info("\n" + "="*60)
    logger.info("🔴 TRAINING rPPG (HEART RATE) MODEL")
    logger.info("="*60)
    
    cmd = [
        sys.executable, 'train_rppg.py',
        '--data-dir', str(rppg_dir),
        '--output-dir', str(output_dir),
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning-rate', str(args.learning_rate),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ rPPG training complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ rPPG training failed: {e}")
        return False


def train_posture(data_dir, output_dir, args):
    """Train posture model"""
    posture_dir = Path(data_dir) / 'posture'
    
    if not posture_dir.exists():
        logger.info("⏭️ Skipping posture training (data not found)")
        return True
    
    logger.info("\n" + "="*60)
    logger.info("💪 TRAINING POSTURE (MHAVH) MODEL")
    logger.info("="*60)
    
    cmd = [
        sys.executable, 'train_posture.py',
        '--data-dir', str(posture_dir),
        '--output-dir', str(output_dir),
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--learning-rate', str(args.learning_rate),
        '--image-size', str(args.image_size),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("✅ Posture training complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Posture training failed: {e}")
        return False


def generate_report(output_dir):
    """Generate training report"""
    output_dir = Path(output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("📊 TRAINING REPORT")
    logger.info("="*60)
    
    report_lines = [
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Output Directory: {output_dir}",
        "",
        "Generated Files:",
    ]
    
    # Check generated files
    files_to_check = [
        ('rppg_best.pth', 'rPPG Model Weights'),
        ('rppg_training_results.json', 'rPPG Training Results'),
        ('mhavh_best.pth', 'Posture Model Weights'),
        ('posture_training_results.json', 'Posture Training Results'),
    ]
    
    for filename, description in files_to_check:
        filepath = output_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            report_lines.append(f"  ✅ {filename} ({size_mb:.1f} MB) - {description}")
        else:
            report_lines.append(f"  ❌ {filename} - {description}")
    
    report_lines.extend([
        "",
        "Next Steps:",
        "1. Update backend/config.py with new model paths",
        "2. Restart the backend server",
        "3. Test with live video feed",
        "4. Monitor performance metrics",
    ])
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # Save report
    with open(output_dir / 'TRAINING_REPORT.txt', 'w') as f:
        f.write(report_text)
    
    logger.info(f"\n✅ Report saved to {output_dir / 'TRAINING_REPORT.txt'}")


def main():
    parser = argparse.ArgumentParser(description='End-to-end training')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Image size for posture')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("🚀 HEART ATTACK DETECTION SYSTEM - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Validate data
    logger.info("\n📂 Validating data structure...")
    if not validate_data_structure(data_dir):
        logger.error("❌ Data validation failed!")
        return 1
    
    # Train models
    success = True
    
    success = train_rppg(data_dir, output_dir, args) and success
    success = train_posture(data_dir, output_dir, args) and success
    
    # Generate report
    if success:
        generate_report(output_dir)
        logger.info("\n✅ Training pipeline complete!")
        return 0
    else:
        logger.error("\n❌ Training pipeline failed!")
        return 1


if __name__ == '__main__':
    exit(main())
