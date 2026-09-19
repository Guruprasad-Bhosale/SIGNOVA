#!/usr/bin/env python
"""
Keypoint Extraction Script.

Extracts MediaPipe keypoints from ISL videos.

Usage:
    python scripts/extract_keypoints.py --video_dir data/raw/isl_csltr --output_dir data/processed
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import pandas as pd
import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.keypoint_extractor import KeypointExtractor, batch_extract_keypoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_annotations(csv_path):
    """Loads gloss annotations from the ISL-CSLTR CSV."""
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    
    # Try common column names
    sent_col = 'Sentence' if 'Sentence' in df.columns else df.columns[0]
    gloss_col = 'SIGN GLOSSES' if 'SIGN GLOSSES' in df.columns else df.columns[1]
    
    annotations = {}
    for _, row in df.iterrows():
        sentence = str(row[sent_col]).strip().lower()
        # Some glosses might have quotes, clean them up
        glosses = str(row[gloss_col]).strip().replace('"', '')
        # Split by space to get individual glosses
        gloss_list = [g for g in glosses.split() if g]
        annotations[sentence] = {
            'glosses': gloss_list,
            'text_english': sentence,
            'text_hindi': '' # We don't have Hindi text in this CSV, IndicTrans2 will translate
        }
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Extract keypoints from ISL videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing ISL-CSLTR Videos_Sentence_Level")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for keypoints")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to 'ISL Corpus sign glosses.csv'")
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                       help="Train/val/test split ratio")
    parser.add_argument("--skip_frames", type=int, default=1, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    csv_file = Path(args.csv_file) if args.csv_file else None
    
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return
        
    sentence_dict = {}
    if csv_file and csv_file.exists():
        logger.info(f"Loading annotations from {csv_file}")
        sentence_dict = load_annotations(csv_file)
    else:
        logger.warning("No CSV file provided or found. Annotations will be blank.")
    
    # Find all videos. In ISL-CSLTR, videos are inside folders named after the sentence.
    extensions = (".mp4", ".avi", ".mov", ".mkv", ".MP4")
    videos = []
    
    # Iterate through sentence folders
    for sentence_folder in video_dir.iterdir():
        if sentence_folder.is_dir():
            sentence_name = sentence_folder.name.strip().lower()
            for ext in extensions:
                for video_path in sentence_folder.glob(f"*{ext}"):
                    # We store the tuple of (path, sentence_name)
                    videos.append((video_path, sentence_name))
    
    logger.info(f"Found {len(videos)} videos across {len(set(v[1] for v in videos))} sentences.")
    
    if len(videos) == 0:
        logger.error("No videos found!")
        return
    
    # Extract keypoints
    logger.info("Extracting keypoints...")
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = KeypointExtractor()
    
    # Keep track of successfully extracted videos
    successful_videos = []
    
    for video_path, sentence_name in tqdm(videos, desc="Extracting"):
        video_stem = video_path.stem
        # Clean up the ID to avoid issues with slashes or weird characters
        clean_sentence = sentence_name.replace(" ", "_").replace("/", "").replace("\\", "")
        clean_stem = video_stem.replace(" ", "_")
        video_id = f"{clean_sentence}___{clean_stem}"
        
        output_file = output_dir / f"{video_id}.npz"
        
        if output_file.exists():
            successful_videos.append((video_id, sentence_name, output_file))
            continue
            
        try:
            keypoints, metadata = extractor.extract_video(str(video_path), skip_frames=args.skip_frames, show_progress=False)
            if keypoints is not None:
                extractor.save_keypoints(keypoints, str(output_file))
                successful_videos.append((video_id, sentence_name, output_file))
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
    
    # Create train/val/test splits
    logger.info("Creating data splits...")
    
    random.seed(42) # For reproducibility
    random.shuffle(successful_videos)
    
    n = len(successful_videos)
    train_end = int(n * args.split_ratio[0])
    val_end = train_end + int(n * args.split_ratio[1])
    
    splits = {
        "train": successful_videos[:train_end],
        "val": successful_videos[train_end:val_end],
        "test": successful_videos[val_end:],
    }
    
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        annotations = []
        
        # Move keypoint files and build annotations
        for video_id, sentence_name, kp_path in split_data:
            target_path = split_dir / f"{video_id}.npz"
            
            # Move file if it's not already there
            if kp_path.exists() and kp_path.parent != split_dir:
                # Instead of renaming (moving), we'll do a robust move/copy
                import shutil
                shutil.move(str(kp_path), str(target_path))
                
            # Build annotation
            ann = {
                "video_id": video_id,
                "glosses": sentence_dict.get(sentence_name, {}).get("glosses", ["UNKNOWN"]),
                "text_english": sentence_dict.get(sentence_name, {}).get("text_english", sentence_name),
                "text_hindi": ""
            }
            annotations.append(ann)
            
        logger.info(f"{split_name}: {len(split_data)} videos")
        
        # Save annotation file
        ann_path = output_dir / f"{split_name}_annotations.json"
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created {ann_path}")
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
