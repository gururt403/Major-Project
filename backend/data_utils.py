"""
Data Preparation Utilities
Helper scripts for preparing training data
"""

import argparse
import logging
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, output_dir, interval=5):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        interval: Extract every nth frame
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            filename = output_dir / f"{video_path.stem}_{extracted_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"✅ Extracted {extracted_count} frames from {video_path.name}")
    return extracted_count


def batch_extract_frames(video_dir, output_dir, interval=5):
    """Extract frames from all videos in directory"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.flv', '.mkv'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    logger.info(f"Found {len(videos)} videos")
    
    total_frames = 0
    for video_file in tqdm(videos, desc="Extracting frames"):
        frames = extract_frames_from_video(
            video_file,
            output_dir / video_file.stem,
            interval=interval
        )
        total_frames += frames
    
    logger.info(f"✅ Total frames extracted: {total_frames}")


def create_posture_annotations(image_dir, output_file):
    """
    Create CSV annotations for posture data
    
    Usage: Manually label images into directories:
    data/raw/posture/
    ├── standing/
    ├── sitting/
    ├── lying_down/
    └── unusual_posture/
    
    Then run:
    python -m data_utils create-posture-annotations --image-dir data/raw/posture --output annotations.csv
    """
    image_dir = Path(image_dir)
    
    classes = ['standing', 'sitting', 'lying_down', 'unusual_posture']
    annotations = []
    
    for class_name in classes:
        class_dir = image_dir / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        for img_file in image_files:
            annotations.append({
                'image_file': str(img_file.relative_to(image_dir)),
                'posture_class': class_name,
                'class_idx': classes.index(class_name)
            })
    
    df = pd.DataFrame(annotations)
    df.to_csv(output_file, index=False)
    logger.info(f"✅ Created annotations file: {output_file} ({len(df)} images)")


def validate_dataset(data_dir):
    """Validate dataset structure and report statistics"""
    data_dir = Path(data_dir)
    
    logger.info("🔍 Validating dataset...")
    logger.info("")
    
    # Check rPPG data
    rppg_dir = data_dir / 'rppg'
    if rppg_dir.exists():
        annotations_file = rppg_dir / 'annotations.csv'
        if annotations_file.exists():
            df = pd.read_csv(annotations_file)
            logger.info("📹 rPPG Dataset:")
            logger.info(f"  Videos: {len(df)}")
            logger.info(f"  HR Range: {df['ground_truth_hr'].min():.0f} - {df['ground_truth_hr'].max():.0f} BPM")
            logger.info(f"  Average Duration: {df['duration_seconds'].mean():.1f}s")
        else:
            logger.warning("⚠️ rPPG annotations.csv not found")
    
    # Check posture data
    posture_dir = data_dir / 'posture'
    if posture_dir.exists():
        classes = ['standing', 'sitting', 'lying_down', 'unusual_posture']
        logger.info("🧍 Posture Dataset:")
        
        total_images = 0
        for class_name in classes:
            class_dir = posture_dir / class_name
            if class_dir.exists():
                num_images = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
                logger.info(f"  {class_name}: {num_images} images")
                total_images += num_images
        
        logger.info(f"  Total: {total_images} images")
    
    logger.info("")
    logger.info("✅ Validation complete!")


def main():
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract-frames', help='Extract frames from videos')
    extract_parser.add_argument('--video-dir', type=str, required=True, help='Video directory')
    extract_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    extract_parser.add_argument('--interval', type=int, default=5, help='Frame extraction interval')
    
    # Create annotations command
    annotate_parser = subparsers.add_parser('create-annotations', help='Create posture annotations')
    annotate_parser.add_argument('--image-dir', type=str, required=True, help='Image directory')
    annotate_parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    
    # Validate dataset command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    
    args = parser.parse_args()
    
    if args.command == 'extract-frames':
        batch_extract_frames(args.video_dir, args.output_dir, args.interval)
    elif args.command == 'create-annotations':
        create_posture_annotations(args.image_dir, args.output)
    elif args.command == 'validate':
        validate_dataset(args.data_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
