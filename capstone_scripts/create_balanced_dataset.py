#!/usr/bin/env python3
"""
Create Balanced Dataset for Tear Detection

Takes an existing combined YOLO dataset (created with batch_convert_annotations.py --append)
and creates a dataset with balanced training set by:
1. Pooling all slices from existing train/val/test splits
2. Grouping slices by VOLUME ID to prevent data leakage
3. Identifying volumes with tears vs without tears
4. Splitting VOLUMES (not slices) into train/val/test (70/15/15) - NO DATA LEAKAGE
5. Undersampling ONLY the train set slices to achieve 50/50 balance
6. Preserving original class imbalance in val/test sets for realistic evaluation
7. Verifying no volume appears in multiple splits
8. Organizing into YOLO format

IMPORTANT: This prevents data leakage by ensuring all slices from the same MRI volume
stay in the same split (train/val/test).

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


def extract_volume_id(filename):
    """
    Extract volume ID from filename.

    Filenames are formatted as: IM1-01-02-00011.dcm
    Volume ID is the suffix after the base name: 01-02-00011

    Args:
        filename: Path object or string filename

    Returns:
        Volume ID string (e.g., "01-02-00011")
    """
    stem = Path(filename).stem  # Remove extension
    # Split by dash and take everything after the first part (IM1, IM2, etc.)
    parts = stem.split('-', 1)
    if len(parts) > 1:
        return parts[1]  # Return volume suffix
    else:
        # Fallback: if no dash, return the whole stem
        return stem


def is_label_empty(label_path):
    """Check if a label file is empty (no annotations)"""
    if not label_path.exists():
        return True

    with open(label_path, 'r') as f:
        content = f.read().strip()
        return len(content) == 0


def scan_combined_dataset(dataset_root):
    """
    Scan a combined YOLO dataset and group slices by volume.

    Pools slices from all existing splits and groups them by volume ID to prevent data leakage.

    Args:
        dataset_root: Path to combined YOLO dataset with images/ and labels/ directories

    Returns:
        dict with 'tear_volumes' and 'no_tear_volumes' keys
        Each entry is a dict: {volume_id: [list of slice_info dicts]}
        A volume is "tear" if ANY of its slices have tears
    """
    dataset_root = Path(dataset_root)

    # Group slices by volume
    volumes = defaultdict(list)

    print(f"\nScanning combined dataset: {dataset_root}")
    print("Pooling slices from all existing splits and grouping by volume...")

    total_tear_slices = 0
    total_no_tear_slices = 0

    # Scan all splits - we'll pool everything and re-split later
    for split in ['train', 'val', 'test']:
        images_dir = dataset_root / 'images' / split
        labels_dir = dataset_root / 'labels' / split

        if not images_dir.exists():
            print(f"  Warning: {split}/ not found, skipping")
            continue

        # Process all image files
        for img_path in images_dir.glob('*'):
            if img_path.suffix.lower() not in ['.dcm', '.png', '.jpg', '.jpeg']:
                continue

            # Find corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')

            # Extract volume ID
            volume_id = extract_volume_id(img_path.name)

            slice_info = {
                'image': img_path,
                'label': label_path,
                'original_split': split,
                'has_tear': not is_label_empty(label_path)
            }

            volumes[volume_id].append(slice_info)

            # Track slice counts
            if slice_info['has_tear']:
                total_tear_slices += 1
            else:
                total_no_tear_slices += 1

    # Categorize volumes by tear presence
    # A volume has tears if ANY slice has tears
    tear_volumes = {}
    no_tear_volumes = {}

    for volume_id, slices in volumes.items():
        has_any_tear = any(s['has_tear'] for s in slices)
        if has_any_tear:
            tear_volumes[volume_id] = slices
        else:
            no_tear_volumes[volume_id] = slices

    print(f"\n  Total slices: {total_tear_slices:,} tear, {total_no_tear_slices:,} no-tear")
    print(f"  Total volumes: {len(tear_volumes):,} with tears, {len(no_tear_volumes):,} without tears")

    # Print volume statistics
    tear_slices_in_tear_vols = sum(len(slices) for slices in tear_volumes.values())
    no_tear_slices_in_no_tear_vols = sum(len(slices) for slices in no_tear_volumes.values())
    print(f"  Slices per category: {tear_slices_in_tear_vols:,} in tear volumes, {no_tear_slices_in_no_tear_vols:,} in no-tear volumes")

    return {
        'tear_volumes': tear_volumes,
        'no_tear_volumes': no_tear_volumes
    }


def sample_and_split(tear_volumes, no_tear_volumes, train_ratio=0.7, val_ratio=0.15,
                     test_ratio=0.15, seed=42):
    """
    Split volumes into train/val/test, then undersample ONLY train set for balance

    This approach PREVENTS DATA LEAKAGE by:
    1. First splits VOLUMES (not slices) into train/val/test using specified ratios
    2. Extracts all slices from each volume group
    3. Undersamples only the train set slices to achieve 50/50 balance
    4. Preserves original class imbalance in val/test for realistic evaluation

    Args:
        tear_volumes: Dict of {volume_id: [list of slices]} for volumes with tears
        no_tear_volumes: Dict of {volume_id: [list of slices]} for volumes without tears
        train_ratio, val_ratio, test_ratio: Split ratios
        seed: Random seed

    Returns:
        dict with 'train', 'val', 'test' keys, each containing slices
        train is balanced 50/50, val/test preserve original imbalance
    """
    random.seed(seed)

    n_tear_vols = len(tear_volumes)
    n_no_tear_vols = len(no_tear_volumes)

    # Count total slices
    total_tear_slices = sum(len(slices) for slices in tear_volumes.values())
    total_no_tear_slices = sum(len(slices) for slices in no_tear_volumes.values())

    print(f"\n{'='*60}")
    print("SPLITTING VOLUMES (PREVENTS DATA LEAKAGE)")
    print(f"{'='*60}")
    print(f"Original distribution:")
    print(f"  Volumes: {n_tear_vols:,} with tears, {n_no_tear_vols:,} without tears")
    print(f"  Slices: {total_tear_slices:,} in tear volumes, {total_no_tear_slices:,} in no-tear volumes")

    # Step 1: Split VOLUMES into train/val/test
    tear_vol_ids = list(tear_volumes.keys())
    no_tear_vol_ids = list(no_tear_volumes.keys())
    random.shuffle(tear_vol_ids)
    random.shuffle(no_tear_vol_ids)

    # Split tear volumes
    n_tear_vols = len(tear_vol_ids)
    tear_train_end = int(n_tear_vols * train_ratio)
    tear_val_end = tear_train_end + int(n_tear_vols * val_ratio)

    tear_train_vols = tear_vol_ids[:tear_train_end]
    tear_val_vols = tear_vol_ids[tear_train_end:tear_val_end]
    tear_test_vols = tear_vol_ids[tear_val_end:]

    # Split no-tear volumes
    n_no_tear_vols = len(no_tear_vol_ids)
    no_tear_train_end = int(n_no_tear_vols * train_ratio)
    no_tear_val_end = no_tear_train_end + int(n_no_tear_vols * val_ratio)

    no_tear_train_vols = no_tear_vol_ids[:no_tear_train_end]
    no_tear_val_vols = no_tear_vol_ids[no_tear_train_end:no_tear_val_end]
    no_tear_test_vols = no_tear_vol_ids[no_tear_val_end:]

    print(f"\n{'='*60}")
    print("STEP 1: Split volumes (70/15/15) - NO DATA LEAKAGE")
    print(f"{'='*60}")
    print(f"  Train: {len(tear_train_vols):,} tear volumes + {len(no_tear_train_vols):,} no-tear volumes")
    print(f"  Val:   {len(tear_val_vols):,} tear volumes + {len(no_tear_val_vols):,} no-tear volumes")
    print(f"  Test:  {len(tear_test_vols):,} tear volumes + {len(no_tear_test_vols):,} no-tear volumes")

    # Step 2: Extract slices from each volume group
    def extract_slices_from_volumes(volume_ids, volume_dict):
        """Extract all slices from given volume IDs"""
        slices = []
        for vol_id in volume_ids:
            slices.extend(volume_dict[vol_id])
        return slices

    tear_train_slices = extract_slices_from_volumes(tear_train_vols, tear_volumes)
    tear_val_slices = extract_slices_from_volumes(tear_val_vols, tear_volumes)
    tear_test_slices = extract_slices_from_volumes(tear_test_vols, tear_volumes)

    no_tear_train_slices = extract_slices_from_volumes(no_tear_train_vols, no_tear_volumes)
    no_tear_val_slices = extract_slices_from_volumes(no_tear_val_vols, no_tear_volumes)
    no_tear_test_slices = extract_slices_from_volumes(no_tear_test_vols, no_tear_volumes)

    print(f"\n{'='*60}")
    print("STEP 2: Extract slices from volumes")
    print(f"{'='*60}")
    print(f"  Train: {len(tear_train_slices):,} tear slices + {len(no_tear_train_slices):,} no-tear slices")
    print(f"  Val:   {len(tear_val_slices):,} tear slices + {len(no_tear_val_slices):,} no-tear slices")
    print(f"  Test:  {len(tear_test_slices):,} tear slices + {len(no_tear_test_slices):,} no-tear slices")

    # Step 3: Undersample ONLY the train set to achieve 50/50 balance
    n_tear_train = len(tear_train_slices)
    n_no_tear_train = len(no_tear_train_slices)

    print(f"\n{'='*60}")
    print("STEP 3: Undersample train set for 50/50 balance")
    print(f"{'='*60}")
    print(f"Train set before undersampling:")
    print(f"  Tear slices: {n_tear_train:,}")
    print(f"  No-tear slices: {n_no_tear_train:,}")
    if n_tear_train > 0:
        print(f"  Ratio: {n_no_tear_train/n_tear_train:.1f}:1 (no-tear:tear)")

    if n_no_tear_train < n_tear_train:
        print(f"\n⚠️  Warning: Train has fewer no-tear ({n_no_tear_train:,}) than tear ({n_tear_train:,})")
        print(f"  This is unusual - keeping all train slices")
        no_tear_train_balanced = no_tear_train_slices
    else:
        print(f"\n✓ Undersampling {n_tear_train:,} no-tear slices from {n_no_tear_train:,} available in train")
        no_tear_train_balanced = random.sample(no_tear_train_slices, n_tear_train)
        print(f"  Removed {n_no_tear_train - n_tear_train:,} no-tear slices from train set")

    # Combine train set and shuffle
    train_combined = tear_train_slices + no_tear_train_balanced
    random.shuffle(train_combined)

    # Val and test keep original imbalance - just combine and shuffle
    val_combined = tear_val_slices + no_tear_val_slices
    test_combined = tear_test_slices + no_tear_test_slices
    random.shuffle(val_combined)
    random.shuffle(test_combined)

    splits = {
        'train': train_combined,
        'val': val_combined,
        'test': test_combined
    }

    # Print final split statistics
    print(f"\n{'='*60}")
    print(f"FINAL SPLITS (train balanced, val/test preserve imbalance)")
    print(f"{'='*60}")

    for split_name, slices in splits.items():
        n_tear_in_split = sum(1 for s in slices if s['has_tear'])
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
            if slice_info['has_tear']:
                stats[split_name]['tear'] += 1
            else:
                stats[split_name]['no_tear'] += 1

    return dict(stats)


def create_dataset_yaml(output_root, stats):
    """Create YAML configuration file for YOLOv5"""
    total_images = sum(s['total'] for s in stats.values())
    total_tears = sum(s['tear'] for s in stats.values())
    total_no_tears = sum(s['no_tear'] for s in stats.values())

    train_tear_pct = 100 * stats['train']['tear'] / stats['train']['total'] if stats['train']['total'] > 0 else 0
    val_tear_pct = 100 * stats['val']['tear'] / stats['val']['total'] if stats['val']['total'] > 0 else 0
    test_tear_pct = 100 * stats['test']['tear'] / stats['test']['total'] if stats['test']['total'] > 0 else 0

    yaml_content = f"""# YOLOv5 Dataset Configuration - Balanced Train, Realistic Val/Test
# Auto-generated by create_balanced_dataset.py
# Source: Re-split and train-balanced from combined dataset

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
# - Train: {stats['train']['total']:,} ({stats['train']['tear']:,} tear [{train_tear_pct:.1f}%], {stats['train']['no_tear']:,} no-tear) - BALANCED
# - Val:   {stats['val']['total']:,} ({stats['val']['tear']:,} tear [{val_tear_pct:.1f}%], {stats['val']['no_tear']:,} no-tear) - Original imbalance
# - Test:  {stats['test']['total']:,} ({stats['test']['tear']:,} tear [{test_tear_pct:.1f}%], {stats['test']['no_tear']:,} no-tear) - Original imbalance
#
# Overall: {total_tears:,} tear : {total_no_tears:,} no-tear

# Image properties:
# - Size: 512x512 pixels
# - Format: DICOM (.dcm)
# - Modality: MRI

# Note: This dataset was created by:
# 1. Pooling all slices from original train/val/test splits
# 2. Splitting all slices into new train/val/test (70/15/15)
# 3. Undersampling ONLY train set to achieve 50/50 balance
# 4. Preserving original imbalance in val/test for realistic evaluation
"""

    yaml_path = output_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Dataset config saved to: {yaml_path}")
    return yaml_path


def verify_no_data_leakage(splits):
    """
    Verify that no volume appears in multiple splits.

    Args:
        splits: Dict with 'train', 'val', 'test' keys containing slices

    Returns:
        tuple: (is_valid, message)
    """
    # Collect volume IDs from each split
    train_volumes = set(extract_volume_id(s['image'].name) for s in splits['train'])
    val_volumes = set(extract_volume_id(s['image'].name) for s in splits['val'])
    test_volumes = set(extract_volume_id(s['image'].name) for s in splits['test'])

    # Check for overlaps
    train_val_overlap = train_volumes & val_volumes
    train_test_overlap = train_volumes & test_volumes
    val_test_overlap = val_volumes & test_volumes

    if train_val_overlap or train_test_overlap or val_test_overlap:
        message = "❌ DATA LEAKAGE DETECTED!\n"
        if train_val_overlap:
            message += f"  Train/Val overlap: {len(train_val_overlap)} volumes\n"
        if train_test_overlap:
            message += f"  Train/Test overlap: {len(train_test_overlap)} volumes\n"
        if val_test_overlap:
            message += f"  Val/Test overlap: {len(val_test_overlap)} volumes\n"
        return False, message

    message = f"✓ NO DATA LEAKAGE: All volumes are in exactly one split\n"
    message += f"  Train: {len(train_volumes)} unique volumes\n"
    message += f"  Val:   {len(val_volumes)} unique volumes\n"
    message += f"  Test:  {len(test_volumes)} unique volumes"
    return True, message


def save_metadata(output_root, splits, stats, args, original_stats):
    """Save detailed metadata about dataset creation"""
    train_balance_ratio = f"{stats['train']['tear']}:{stats['train']['no_tear']}"
    val_ratio = f"{stats['val']['tear']}:{stats['val']['no_tear']}"
    test_ratio = f"{stats['test']['tear']}:{stats['test']['no_tear']}"

    # Verify no data leakage
    is_valid, leakage_message = verify_no_data_leakage(splits)
    print(f"\n{'='*60}")
    print("DATA LEAKAGE VERIFICATION")
    print(f"{'='*60}")
    print(leakage_message)

    if not is_valid:
        raise RuntimeError("Data leakage detected! Aborting.")

    # Count unique volumes per split
    train_volumes = set(extract_volume_id(s['image'].name) for s in splits['train'])
    val_volumes = set(extract_volume_id(s['image'].name) for s in splits['val'])
    test_volumes = set(extract_volume_id(s['image'].name) for s in splits['test'])

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
            'sampling_method': 'volume_split_then_undersample_train',
            'approach': 'Split VOLUMES (not slices) 70/15/15 to prevent data leakage, then undersample only train set to 50/50',
            'data_leakage_prevention': True
        },
        'volume_statistics': {
            'train_volumes': len(train_volumes),
            'val_volumes': len(val_volumes),
            'test_volumes': len(test_volumes),
            'total_volumes': len(train_volumes) + len(val_volumes) + len(test_volumes),
            'no_overlap_verified': True
        },
        'statistics': {
            'splits': stats,
            'total_images': sum(s['total'] for s in stats.values()),
            'total_tear': sum(s['tear'] for s in stats.values()),
            'total_no_tear': sum(s['no_tear'] for s in stats.values()),
            'train_balance_ratio': train_balance_ratio,
            'val_balance_ratio': val_ratio,
            'test_balance_ratio': test_ratio
        },
        'slice_details': {
            split_name: [
                {
                    'image': s['image'].name,
                    'volume_id': extract_volume_id(s['image'].name),
                    'has_tear': s['has_tear'],
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
        description='Create dataset with balanced train set (undersampled) and realistic val/test sets'
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
    tear_volumes = results['tear_volumes']
    no_tear_volumes = results['no_tear_volumes']

    # Count total slices for original stats
    total_tear_slices = sum(len(slices) for slices in tear_volumes.values())
    total_no_tear_slices = sum(len(slices) for slices in no_tear_volumes.values())

    print(f"\n✓ Pooled totals:")
    print(f"  Tear volumes: {len(tear_volumes):,} ({total_tear_slices:,} slices)")
    print(f"  No-tear volumes: {len(no_tear_volumes):,} ({total_no_tear_slices:,} slices)")

    original_stats = {
        'tear': total_tear_slices,
        'no_tear': total_no_tear_slices
    }

    # Step 2: Split and undersample train
    print("\n" + "="*60)
    print("STEP 2: Splitting volumes and undersampling train set")
    print("="*60)

    splits = sample_and_split(
        tear_volumes,
        no_tear_volumes,
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
    print("✓ DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset location: {output_path.absolute()}")
    print(f"\nFinal statistics:")
    print(f"  Total images: {sum(s['total'] for s in stats.values()):,}")
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        tear_pct = 100 * s['tear'] / s['total'] if s['total'] > 0 else 0
        balance_note = "BALANCED 50/50" if split_name == 'train' else "original imbalance"
        print(f"  {split_name.capitalize():5s}: {s['total']:6,d} ({s['tear']:6,d} tear [{tear_pct:.1f}%], {s['no_tear']:6,d} no-tear) - {balance_note}")

    total_tear = sum(s['tear'] for s in stats.values())
    total_no_tear = sum(s['no_tear'] for s in stats.values())

    print(f"\n  Overall: {total_tear:,} tear : {total_no_tear:,} no-tear")
    print(f"  Train set: {stats['train']['tear']:,} tear : {stats['train']['no_tear']:,} no-tear (balanced 1:1)")
    print(f"  Val/Test: Preserve original class imbalance for realistic evaluation")

    print(f"\nReduction from original:")
    print(f"  Before: {original_stats['tear']:,} tear + {original_stats['no_tear']:,} no-tear = {original_stats['tear'] + original_stats['no_tear']:,} total")
    print(f"  After:  {total_tear:,} tear + {total_no_tear:,} no-tear = {total_tear + total_no_tear:,} total")
    print(f"  Removed: {original_stats['no_tear'] - total_no_tear:,} no-tear slices (from train set only)")

    print(f"\nReady to train with:")
    print(f"  python train.py --data {output_path / 'dataset.yaml'} ...")


if __name__ == '__main__':
    main()
