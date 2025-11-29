#!/usr/bin/env python3
"""
Create Balanced Dataset for Tear Detection

Takes an existing combined YOLO dataset (created with batch_convert_annotations.py --append)
and creates a balanced dataset by:
1. Pooling all slices from existing train/val/test splits
2. Identifying slices with tears (non-empty labels) vs without tears (empty labels)
3. Randomly sampling from no-tear slices to match tear slice count
4. Creating new train/val/test splits (70/15/15) at the slice level
5. Organizing into YOLO format

Usage:
    python create_balanced_dataset.py \
        --input /path/to/combined_yolo_dataset \
        --output /path/to/balanced_dataset \
        --seed 42
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def is_label_empty(label_path):
    """Check if a label file is empty (no annotations)"""
    if not label_path.exists():
        return True

    with open(label_path, 'r') as f:
        content = f.read().strip()
        return len(content) == 0


def scan_combined_dataset(dataset_root):
    """
    Scan a combined YOLO dataset and categorize ALL slices by tear presence.

    Pools slices from all existing splits (train/val/test) to re-balance and re-split.

    Args:
        dataset_root: Path to combined YOLO dataset with images/ and labels/ directories

    Returns:
        dict with 'tear_slices' and 'no_tear_slices' lists
        Each entry is {'image': path, 'label': path, 'original_split': split_name}
    """
    dataset_root = Path(dataset_root)

    tear_slices = []
    no_tear_slices = []

    print(f"\nScanning combined dataset: {dataset_root}")
    print("Pooling slices from all existing splits (train/val/test)...")

    # Scan all splits - we'll pool everything and re-split later
    for split in ['train', 'val', 'test']:
        images_dir = dataset_root / 'images' / split
        labels_dir = dataset_root / 'labels' / split

        if not images_dir.exists():
            print(f"  Warning: {split}/ not found, skipping")
            continue

        # Count slices in this split
        split_tear = 0
        split_no_tear = 0

        # Process all image files
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ['.dcm', '.png', '.jpg', '.jpeg']:
                continue

            # Find corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')

            slice_info = {
                'image': img_path,
                'label': label_path,
                'original_split': split
            }

            # Categorize by tear presence
            if is_label_empty(label_path):
                no_tear_slices.append(slice_info)
                split_no_tear += 1
            else:
                tear_slices.append(slice_info)
                split_tear += 1

        print(f"  {split:5s}: {split_tear:6d} tear, {split_no_tear:6d} no-tear")

    return {
        'tear_slices': tear_slices,
        'no_tear_slices': no_tear_slices
    }


def sample_and_split(tear_slices, no_tear_slices, train_ratio=0.7, val_ratio=0.15,
                     test_ratio=0.15, seed=42):
    """
    Sample no-tear slices to match tear count and split into train/val/test

    Args:
        tear_slices: List of slices with tears
        no_tear_slices: List of slices without tears
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed

    Returns:
        dict with 'train', 'val', 'test' keys, each containing balanced slices
    """
    random.seed(seed)

    n_tear = len(tear_slices)
    n_no_tear = len(no_tear_slices)

    print(f"\n{'='*60}")
    print("BALANCING DATASET")
    print(f"{'='*60}")
    print(f"Original class distribution (pooled from all splits):")
    print(f"  Tear slices: {n_tear:,}")
    print(f"  No-tear slices: {n_no_tear:,}")
    print(f"  Imbalance ratio: {n_no_tear/n_tear:.1f}:1 (no-tear:tear)")

    # Sample no-tear slices to match tear count
    if n_no_tear < n_tear:
        print(f"\n⚠️  Warning: Not enough no-tear slices ({n_no_tear:,}) to match tears ({n_tear:,})")
        print(f"  Using all {n_no_tear:,} no-tear slices")
        sampled_no_tear = no_tear_slices.copy()
    else:
        print(f"\n✓ Randomly sampling {n_tear:,} no-tear slices from {n_no_tear:,} available")
        sampled_no_tear = random.sample(no_tear_slices, n_tear)

    # Combine and shuffle
    all_slices = tear_slices + sampled_no_tear
    random.shuffle(all_slices)

    print(f"\nBalanced dataset size: {len(all_slices):,} slices (50% tear, 50% no-tear)")

    # Split into train/val/test
    n_total = len(all_slices)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)

    splits = {
        'train': all_slices[:train_end],
        'val': all_slices[train_end:val_end],
        'test': all_slices[val_end:]
    }

    # Print split statistics
    print(f"\n{'='*60}")
    print(f"NEW SPLITS (at slice level, not volume level)")
    print(f"{'='*60}")
    print(f"Split ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}\n")

    for split_name, slices in splits.items():
        n_tear_in_split = sum(1 for s in slices if not is_label_empty(s['label']))
        n_no_tear_in_split = len(slices) - n_tear_in_split
        tear_pct = 100 * n_tear_in_split / len(slices) if slices else 0
        print(f"  {split_name:5s}: {len(slices):6,d} slices "
              f"({n_tear_in_split:6,d} tear [{tear_pct:.1f}%], "
              f"{n_no_tear_in_split:6,d} no-tear)")

    return splits


def copy_dataset(splits, output_root):
    """
    Copy slices to organized YOLO dataset structure

    Args:
        splits: Dict with 'train', 'val', 'test' keys
        output_root: Output directory
    """
    output_root = Path(output_root)

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = defaultdict(lambda: {'total': 0, 'tear': 0, 'no_tear': 0})

    print(f"\n{'='*60}")
    print("COPYING FILES TO BALANCED DATASET")
    print(f"{'='*60}")

    # Copy files
    for split_name, slices in splits.items():
        print(f"\nCopying {split_name} set ({len(slices):,} slices)...")

        for slice_info in tqdm(slices, desc=f"  {split_name}"):
            img_src = slice_info['image']
            label_src = slice_info['label']

            # Generate destination filenames (preserve original names)
            img_dest = output_root / 'images' / split_name / img_src.name
            label_dest = output_root / 'labels' / split_name / (img_src.stem + '.txt')

            # Copy image
            shutil.copy2(img_src, img_dest)

            # Copy or create label
            if label_src.exists():
                shutil.copy2(label_src, label_dest)
            else:
                # Create empty label file
                label_dest.touch()

            # Update stats
            stats[split_name]['total'] += 1
            if is_label_empty(label_dest):
                stats[split_name]['no_tear'] += 1
            else:
                stats[split_name]['tear'] += 1

    return dict(stats)


def create_dataset_yaml(output_root, stats):
    """Create YAML configuration file for YOLOv5"""
    total_images = sum(s['total'] for s in stats.values())
    total_tears = sum(s['tear'] for s in stats.values())
    total_no_tears = sum(s['no_tear'] for s in stats.values())

    yaml_content = f"""# YOLOv5 Dataset Configuration - Balanced Tear Detection
# Auto-generated by create_balanced_dataset.py
# Source: Re-balanced and re-split from combined dataset

# Dataset paths
path: {output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Classes (single-class detection)
nc: 1
names:
  0: tear  # Meniscus tear detection

# Dataset Statistics
# Total images: {total_images:,}
# - Train: {stats['train']['total']:,} ({stats['train']['tear']:,} tear, {stats['train']['no_tear']:,} no-tear)
# - Val: {stats['val']['total']:,} ({stats['val']['tear']:,} tear, {stats['val']['no_tear']:,} no-tear)
# - Test: {stats['test']['total']:,} ({stats['test']['tear']:,} tear, {stats['test']['no_tear']:,} no-tear)
#
# Class balance: {total_tears:,} tear : {total_no_tears:,} no-tear (50:50 ratio)

# Image properties:
# - Size: 512x512 pixels
# - Format: DICOM (.dcm)
# - Modality: MRI

# Note: This dataset was created by:
# 1. Pooling all slices from original train/val/test splits
# 2. Balancing by sampling no-tear slices to match tear count
# 3. Re-splitting at slice level (not volume level)
"""

    yaml_path = output_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Dataset config saved to: {yaml_path}")
    return yaml_path


def save_metadata(output_root, splits, stats, args, original_stats):
    """Save detailed metadata about dataset creation"""
    metadata = {
        'source': {
            'input_dataset': str(args.input),
            'original_tear_count': original_stats['tear'],
            'original_no_tear_count': original_stats['no_tear'],
            'original_ratio': f"{original_stats['no_tear']/original_stats['tear']:.2f}:1"
        },
        'parameters': {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'seed': args.seed,
            'sampling_method': 'random'
        },
        'statistics': {
            'splits': stats,
            'total_images': sum(s['total'] for s in stats.values()),
            'total_tear': sum(s['tear'] for s in stats.values()),
            'total_no_tear': sum(s['no_tear'] for s in stats.values()),
            'balance_ratio': '1:1'
        },
        'slice_details': {
            split_name: [
                {
                    'image': s['image'].name,
                    'has_tear': not is_label_empty(s['label']),
                    'original_split': s['original_split']
                }
                for s in slices
            ]
            for split_name, slices in splits.items()
        }
    }

    metadata_path = output_root / 'balanced_dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced dataset from combined YOLO dataset by sampling no-tear slices'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input combined YOLO dataset directory (created with batch_convert_annotations.py --append)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for balanced dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proportion for training (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Proportion for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Proportion for testing (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return

    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return

    # Check for expected structure
    if not (input_path / 'images').exists() or not (input_path / 'labels').exists():
        print(f"Error: Input directory must contain 'images/' and 'labels/' subdirectories")
        print(f"Expected structure created by batch_convert_annotations.py")
        return

    print("="*60)
    print("Creating Balanced Dataset for Tear Detection")
    print("="*60)
    print(f"Input dataset: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"Random seed: {args.seed}")

    # Step 1: Scan combined dataset
    print("\n" + "="*60)
    print("STEP 1: Scanning combined dataset")
    print("="*60)

    results = scan_combined_dataset(input_path)
    all_tear_slices = results['tear_slices']
    all_no_tear_slices = results['no_tear_slices']

    print(f"\n✓ Pooled totals:")
    print(f"  Tear slices: {len(all_tear_slices):,}")
    print(f"  No-tear slices: {len(all_no_tear_slices):,}")

    original_stats = {
        'tear': len(all_tear_slices),
        'no_tear': len(all_no_tear_slices)
    }

    # Step 2: Sample and split
    print("\n" + "="*60)
    print("STEP 2: Balancing and splitting dataset")
    print("="*60)

    splits = sample_and_split(
        all_tear_slices,
        all_no_tear_slices,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Step 3: Copy files
    print("\n" + "="*60)
    print("STEP 3: Copying files to output directory")
    print("="*60)

    output_path = Path(args.output)
    stats = copy_dataset(splits, output_path)

    # Step 4: Create YAML and metadata
    print("\n" + "="*60)
    print("STEP 4: Creating configuration files")
    print("="*60)

    create_dataset_yaml(output_path, stats)
    save_metadata(output_path, splits, stats, args, original_stats)

    # Final summary
    print("\n" + "="*60)
    print("✓ BALANCED DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset location: {output_path.absolute()}")
    print(f"\nFinal statistics:")
    print(f"  Total images: {sum(s['total'] for s in stats.values()):,}")
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        print(f"  {split_name.capitalize():5s}: {s['total']:6,d} ({s['tear']:6,d} tear, {s['no_tear']:6,d} no-tear)")

    total_tear = sum(s['tear'] for s in stats.values())
    total_no_tear = sum(s['no_tear'] for s in stats.values())
    print(f"\n  Class balance: {total_tear:,} tear : {total_no_tear:,} no-tear (perfect 1:1)")

    print(f"\nReduction from original:")
    print(f"  Before: {original_stats['tear']:,} tear + {original_stats['no_tear']:,} no-tear = {original_stats['tear'] + original_stats['no_tear']:,} total")
    print(f"  After:  {total_tear:,} tear + {total_no_tear:,} no-tear = {total_tear + total_no_tear:,} total")
    print(f"  Removed: {original_stats['no_tear'] - total_no_tear:,} excess no-tear slices")

    print(f"\nReady to train with:")
    print(f"  python train.py --data {output_path / 'dataset.yaml'} ...")


if __name__ == '__main__':
    main()
