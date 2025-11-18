#!/usr/bin/env python3
"""
Batch Annotation Converter for Knee MRI Meniscus Tear Detection
Converts MD.ai CSV annotations to YOLO format for binary classification

Usage:
    python batch_convert_annotations.py --dicom_root /path/to/dicoms --csv_dir /path/to/csvs --output /path/to/yolo_dataset

Features:
- Binary classification: 0 (no tear) vs 1 (tear present)
- Processes all volumes in TBRecon_ID_key_meniscus.csv
- Maps anonymized UIDs to DICOM filenames
- Converts all meniscus region annotations to class 1 (tear)
- Creates proper train/val/test splits at volume level
- Handles missing labels gracefully (empty files = no tear)
"""

import argparse
import ast
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import configs
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import (
    DICOM_NORMAL,
    DICOM_MENISCAL,
    MENISCUS_ANNOTATIONS_DIR,
    NORMAL_ANNOTATIONS_DIR,
    USE_HPC
)


# Class mapping for meniscus tear detection (binary classification)
# All meniscus regions map to class 1 (tear present)
# Empty labels indicate class 0 (no tear)
CLASS_MAPPING = {
    'LatAntHorn': 1,  # Lateral Anterior Horn -> tear
    'LatMenBody': 1,  # Lateral Meniscus Body -> tear
    'LatPosHorn': 1,  # Lateral Posterior Horn -> tear
    'MedAntHorn': 1,  # Medial Anterior Horn -> tear
    'MedMenBody': 1,  # Medial Meniscus Body -> tear
    'MedPosHorn': 1,  # Medial Posterior Horn -> tear
}

# Image dimensions (assuming all are same)
IMG_WIDTH = 512
IMG_HEIGHT = 512


def convert_bbox_to_yolo(bbox_dict, img_width, img_height):
    """
    Convert bounding box from pixel coordinates to YOLO format

    Args:
        bbox_dict: {'x': x_topleft, 'y': y_topleft, 'width': w, 'height': h}
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        (x_center_norm, y_center_norm, width_norm, height_norm)
    """
    x = bbox_dict['x']
    y = bbox_dict['y']
    w = bbox_dict['width']
    h = bbox_dict['height']

    # Calculate center
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    # Normalize to 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    # Clip to valid range
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return x_center_norm, y_center_norm, width_norm, height_norm


def load_annotations_and_mapping(csv_dir, data_source):
    """
    Load annotation and mapping CSV files

    Args:
        csv_dir: Directory containing CSV files

    Returns:
        (annotations_df, mapping_df)
    """
    csv_dir = Path(csv_dir)

    # Load annotation file
    if data_source == 'meniscus_tear':
        annot_file = csv_dir / 'TBRecon_anomaly_meniscus_MDai_df.csv'
    else: 
        annot_file = csv_dir / 'TBRecon_anomaly_normal_MDai_df.csv'
    if not annot_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_file}")
    df_annot = pd.read_csv(annot_file)

    # Load mapping file
    if data_source == 'meniscus_tear':
        map_file = csv_dir / 'TBRecon_ID_key_meniscus.csv'
    else:
        map_file = csv_dir / 'TBRecon_ID_key_normal.csv'
    if not map_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {map_file}")
    df_map = pd.read_csv(map_file)

    return df_annot, df_map


def get_volume_list(df_map):
    """Get list of unique volume IDs"""
    volumes = df_map['tbrecon_id'].unique().tolist()
    return sorted(volumes)


def process_volume(volume_id, dicom_dir, df_annot, df_map, output_labels_dir,
                   create_empty_labels=True):
    """
    Process a single volume: map annotations and create YOLO label files

    Args:
        volume_id: TBrecon volume ID (e.g., 'TBrecon-01-02-00011')
        dicom_dir: Directory containing DICOM files for this volume
        df_annot: Full annotations dataframe
        df_map: Full mapping dataframe
        output_labels_dir: Directory to save label files
        create_empty_labels: If True, create empty .txt files for slices without annotations

    Returns:
        dict with statistics
    """
    dicom_dir = Path(dicom_dir)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Filter for this volume
    df_map_vol = df_map[df_map['tbrecon_id'] == volume_id]
    df_annot_vol = df_annot[df_annot['TBreconID'] == volume_id]
    df_annot_bbox = df_annot_vol[df_annot_vol['data'].notna() & (df_annot_vol['data'] != '')]

    if len(df_map_vol) == 0:
        return {
            'volume_id': volume_id,
            'status': 'skipped',
            'reason': 'No mapping found',
            'total_slices': 0,
            'annotated_slices': 0,
            'total_annotations': 0
        }

    # Create UID mapping
    anon_to_instance = dict(zip(df_map_vol['SOPInstanceUID_anon'],
                               df_map_vol['InstanceNumber']))

    # Collect annotations by filename
    annotations_by_file = defaultdict(list)

    for idx, row in df_annot_bbox.iterrows():
        sop_anon = row['SOPInstanceUID_anon']

        if sop_anon not in anon_to_instance:
            continue

        instance_num = anon_to_instance[sop_anon]
        filename = f"IM{instance_num}.dcm"
        label_name = row['labelName']
        bbox_data = row['data']

        # Skip non-bbox labels
        if label_name not in CLASS_MAPPING:
            continue

        try:
            bbox_dict = ast.literal_eval(bbox_data)
            annotations_by_file[filename].append({
                'class_id': CLASS_MAPPING[label_name],
                'class_name': label_name,
                'bbox': bbox_dict
            })
        except Exception as e:
            print(f"Warning: Could not parse bbox for {volume_id}/{filename}: {e}")

    # Get all DICOM files in directory
    dicom_files = sorted(dicom_dir.glob('*.dcm'))

    # Create label files
    created_count = 0
    empty_count = 0

    for dcm_file in dicom_files:
        filename = dcm_file.name
        label_filename = filename.replace('.dcm', '.txt')
        label_path = output_labels_dir / label_filename

        if filename in annotations_by_file:
            # Write annotations
            with open(label_path, 'w') as f:
                for annot in annotations_by_file[filename]:
                    class_id = annot['class_id']
                    bbox = annot['bbox']

                    # Convert to YOLO format
                    x_c, y_c, w, h = convert_bbox_to_yolo(bbox, IMG_WIDTH, IMG_HEIGHT)

                    # Write YOLO format: class x_center y_center width height
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            created_count += 1
        elif create_empty_labels:
            # Create empty label file (negative sample)
            label_path.touch()
            empty_count += 1

    # Calculate statistics
    total_annotations = sum(len(annots) for annots in annotations_by_file.values())

    return {
        'volume_id': volume_id,
        'status': 'success',
        'total_slices': len(dicom_files),
        'annotated_slices': len(annotations_by_file),
        'empty_slices': empty_count,
        'total_annotations': total_annotations,
        'class_distribution': get_class_distribution(annotations_by_file)
    }


def get_class_distribution(annotations_by_file):
    """Calculate class distribution from annotations"""
    class_counts = defaultdict(int)
    for annots in annotations_by_file.values():
        for annot in annots:
            class_counts[annot['class_name']] += 1
    return dict(class_counts)


def split_volumes(volumes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split volumes into train/val/test sets

    Args:
        volumes: List of volume IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility

    Returns:
        (train_volumes, val_volumes, test_volumes)
    """
    import random
    random.seed(seed)

    volumes = list(volumes)
    random.shuffle(volumes)

    n = len(volumes)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_vols = volumes[:train_end]
    val_vols = volumes[train_end:val_end]
    test_vols = volumes[val_end:]

    return train_vols, val_vols, test_vols


def organize_yolo_dataset(dicom_root, output_root, df_annot, df_map,
                         train_vols, val_vols, test_vols,
                         create_empty_labels=True):
    """
    Organize dataset into YOLO-compatible structure with train/val/test splits

    Args:
        dicom_root: Root directory containing all volume folders
        output_root: Output directory for organized dataset
        df_annot: Annotations dataframe
        df_map: Mapping dataframe
        train_vols, val_vols, test_vols: Lists of volume IDs for each split
        create_empty_labels: Whether to create empty label files

    Returns:
        dict with statistics for all volumes
    """
    dicom_root = Path(dicom_root)
    output_root = Path(output_root)

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    all_stats = {}

    # Process each split
    splits = {
        'train': train_vols,
        'val': val_vols,
        'test': test_vols
    }

    for split_name, volume_list in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} set ({len(volume_list)} volumes)")
        print(f"{'='*60}")

        for volume_id in tqdm(volume_list, desc=f"{split_name}"):
            dicom_dir = dicom_root / volume_id

            if not dicom_dir.exists():
                print(f"Warning: DICOM directory not found: {dicom_dir}")
                all_stats[volume_id] = {
                    'volume_id': volume_id,
                    'status': 'skipped',
                    'reason': 'Directory not found'
                }
                continue

            # Process volume
            temp_labels_dir = output_root / 'temp_labels' / volume_id
            stats = process_volume(
                volume_id,
                dicom_dir,
                df_annot,
                df_map,
                temp_labels_dir,
                create_empty_labels=create_empty_labels
            )

            all_stats[volume_id] = stats
            stats['split'] = split_name

            # Copy/symlink files to split directories
            if stats['status'] == 'success':
                # Extract volume suffix (e.g., "01-02-00011" from "TBrecon-01-02-00011")
                volume_suffix = volume_id.split('-', 1)[1] if '-' in volume_id else volume_id

                # Copy DICOM files with volume suffix
                for dcm_file in dicom_dir.glob('*.dcm'):
                    # Rename: IM1.dcm -> IM1-01-02-00011.dcm
                    base_name = dcm_file.stem  # e.g., "IM1"
                    new_name = f"{base_name}-{volume_suffix}.dcm"
                    dest = output_root / 'images' / split_name / new_name
                    shutil.copy2(dcm_file, dest)

                # Copy label files with volume suffix
                for label_file in temp_labels_dir.glob('*.txt'):
                    # Rename: IM1.txt -> IM1-01-02-00011.txt
                    base_name = label_file.stem  # e.g., "IM1"
                    new_name = f"{base_name}-{volume_suffix}.txt"
                    dest = output_root / 'labels' / split_name / new_name
                    shutil.copy2(label_file, dest)

        # Clean up temp directory
        temp_dir = output_root / 'temp_labels'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return all_stats


def create_dataset_yaml(output_root, stats_summary):
    """Create YAML configuration file for YOLOv5"""
    yaml_content = f"""# YOLOv5 Dataset Configuration for Knee Meniscus Tear Detection (Binary Classification)
# Auto-generated by batch_convert_annotations.py

# Dataset paths
path: {output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Classes (binary classification)
names:
  0: no_tear     # No meniscus tear
  1: tear        # Meniscus tear present

# Dataset Statistics
# Total volumes: {stats_summary['total_volumes']}
# Train volumes: {stats_summary['train_volumes']}
# Val volumes: {stats_summary['val_volumes']}
# Test volumes: {stats_summary['test_volumes']}
# Total images: {stats_summary['total_images']}
# Total annotations: {stats_summary['total_annotations']}

# Class Distribution (total):
{format_class_distribution(stats_summary.get('class_distribution', {}))}

# Image properties:
# - Size: 512x512 pixels
# - Format: DICOM (.dcm)
# - Modality: MRI (Sagittal PD with Fat Saturation)
"""

    yaml_path = output_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path


def format_class_distribution(class_dist):
    """Format class distribution for YAML"""
    # For binary classification, aggregate all tear types
    total_tears = sum(count for class_name, count in class_dist.items()
                     if class_name in CLASS_MAPPING)
    if total_tears > 0:
        return f"#   1: tear - {total_tears} annotations"
    return "#   No annotations"


def generate_summary_report(all_stats, splits, output_root, append=False):
    """Generate detailed summary report

    Args:
        all_stats: Stats for newly processed volumes only
        splits: Full splits dict with all volumes (existing + new)
        output_root: Output directory
        append: Whether this is an append operation
    """
    report_path = output_root / 'conversion_report.json'

    # If appending, load existing stats and merge
    existing_stats = {}
    if append and report_path.exists():
        try:
            with open(report_path, 'r') as f:
                existing_report = json.load(f)
                existing_stats = existing_report.get('volumes', {})
        except Exception as e:
            print(f"Warning: Could not load existing stats: {e}")

    # Merge stats (new stats override old if same volume ID)
    merged_stats = {**existing_stats, **all_stats}

    # Calculate overall statistics from merged data
    total_volumes = len(merged_stats)
    successful_volumes = sum(1 for s in merged_stats.values() if s['status'] == 'success')
    total_images = sum(s.get('total_slices', 0) for s in merged_stats.values())
    total_annotations = sum(s.get('total_annotations', 0) for s in merged_stats.values())

    # Aggregate class distribution
    overall_class_dist = defaultdict(int)
    for stats in merged_stats.values():
        for class_name, count in stats.get('class_distribution', {}).items():
            overall_class_dist[class_name] += count

    stats_summary = {
        'total_volumes': total_volumes,
        'successful_volumes': successful_volumes,
        'train_volumes': len(splits['train']),
        'val_volumes': len(splits['val']),
        'test_volumes': len(splits['test']),
        'total_images': total_images,
        'total_annotations': total_annotations,
        'class_distribution': dict(overall_class_dist)
    }

    # Print summary
    print("\n" + "="*60)
    if append:
        print("MERGED DATASET SUMMARY")
    else:
        print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total volumes: {total_volumes}")
    if append:
        print(f"  Previously existing: {len(existing_stats)}")
        print(f"  Newly added: {len(all_stats)}")
    print(f"Successful: {successful_volumes}")
    print(f"Failed: {total_volumes - successful_volumes}")
    print(f"\nDataset split:")
    print(f"  Train: {len(splits['train'])} volumes")
    print(f"  Val: {len(splits['val'])} volumes")
    print(f"  Test: {len(splits['test'])} volumes")
    print(f"\nTotal images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"\nClass distribution (binary):")
    print(f"  0: no_tear       - (indicated by empty label files)")
    print(f"  1: tear          - {total_annotations:4d} annotations")

    # Save detailed report
    report = {
        'summary': stats_summary,
        'splits': {
            'train': splits['train'],
            'val': splits['val'],
            'test': splits['test']
        },
        'volumes': merged_stats
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Detailed report saved to: {report_path}")

    return stats_summary


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert MD.ai annotations to YOLO format for binary meniscus tear detection (0=no tear, 1=tear)'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        choices=['meniscus_tear', 'normal'],
        default='meniscus_tear',
        help='Which dataset to process: meniscus_tear (default) or normal. Paths configured in configs.py'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for organized YOLO dataset'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proportion of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Proportion of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Proportion of data for testing (default: 0.15)'
    )
    parser.add_argument(
        '--no_empty_labels',
        action='store_true',
        help='Do not create empty label files for unannotated slices'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val/test split (default: 42)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing dataset instead of creating new one. Merges stats and preserves existing files.'
    )

    args = parser.parse_args()

    print("="*60)
    print("Knee Meniscus Tear Detection - Binary Classification")
    print("="*60)
    print(f"Environment: {'HPC' if USE_HPC else 'Local'}")
    print(f"Data source: {args.data_source}")

    # Select data source from configs
    if args.data_source == 'meniscus_tear':
        dicom_root = DICOM_MENISCAL
        csv_dir = MENISCUS_ANNOTATIONS_DIR
    else:  # normal
        dicom_root = DICOM_NORMAL
        csv_dir = NORMAL_ANNOTATIONS_DIR

    print(f"DICOM root: {dicom_root}")
    print(f"CSV dir: {csv_dir}")
    print("="*60)

    # Load CSV files
    print("\nLoading annotation and mapping files...")
    df_annot, df_map = load_annotations_and_mapping(csv_dir, data_source=args.data_source)
    print(f"✓ Loaded {len(df_annot)} annotations")
    print(f"✓ Loaded {len(df_map)} UID mappings")

    # Get volume list
    volumes = get_volume_list(df_map)
    print(f"\n✓ Found {len(volumes)} unique volumes to process")

    # Load existing data if appending
    output_path = Path(args.output)
    existing_splits = {'train': [], 'val': [], 'test': []}
    existing_volumes = set()

    if args.append:
        report_path = output_path / 'conversion_report.json'
        if report_path.exists():
            print("\n" + "="*60)
            print("LOADING EXISTING DATASET FOR APPEND")
            print("="*60)
            try:
                with open(report_path, 'r') as f:
                    existing_report = json.load(f)
                    existing_splits = existing_report.get('splits', {'train': [], 'val': [], 'test': []})
                    existing_volumes = set(existing_report.get('volumes', {}).keys())
                    print(f"✓ Found existing dataset with {len(existing_volumes)} volumes")
                    print(f"  - Train: {len(existing_splits['train'])} volumes")
                    print(f"  - Val: {len(existing_splits['val'])} volumes")
                    print(f"  - Test: {len(existing_splits['test'])} volumes")
            except Exception as e:
                print(f"Warning: Could not load existing report: {e}")
                print("Proceeding without appending...")
                args.append = False
        else:
            print(f"\nNote: --append specified but no existing dataset found at {report_path}")
            print("Creating new dataset...")
            args.append = False

    # Filter out volumes that already exist (when appending)
    if args.append:
        new_volumes = [v for v in volumes if v not in existing_volumes]
        if not new_volumes:
            print("\n✓ All volumes already exist in dataset. Nothing to add.")
            return
        print(f"\n✓ {len(new_volumes)} new volumes to add (skipping {len(volumes) - len(new_volumes)} existing)")
        volumes = new_volumes

    # Split NEW volumes
    print(f"\nSplitting NEW volumes (train:{args.train_ratio}, val:{args.val_ratio}, test:{args.test_ratio})...")
    train_vols, val_vols, test_vols = split_volumes(
        volumes,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    print(f"✓ New Train: {len(train_vols)} volumes")
    print(f"✓ New Val: {len(val_vols)} volumes")
    print(f"✓ New Test: {len(test_vols)} volumes")

    # Create splits dict with NEW volumes only for processing
    splits_to_process = {
        'train': train_vols,
        'val': val_vols,
        'test': test_vols
    }

    # Create full splits dict for reporting (merging existing + new)
    full_splits = {
        'train': existing_splits['train'] + train_vols,
        'val': existing_splits['val'] + val_vols,
        'test': existing_splits['test'] + test_vols
    }

    if args.append:
        print(f"\n✓ Total after append - Train: {len(full_splits['train'])} volumes")
        print(f"✓ Total after append - Val: {len(full_splits['val'])} volumes")
        print(f"✓ Total after append - Test: {len(full_splits['test'])} volumes")

    # Process NEW volumes only
    print(f"\nProcessing NEW volumes and appending to dataset...")
    all_stats = organize_yolo_dataset(
        dicom_root,
        output_path,
        df_annot,
        df_map,
        train_vols,
        val_vols,
        test_vols,
        create_empty_labels=not args.no_empty_labels
    )

    # Generate summary report with full splits
    stats_summary = generate_summary_report(all_stats, full_splits, output_path, append=args.append)

    # Create dataset YAML
    yaml_path = create_dataset_yaml(Path(args.output), stats_summary)
    print(f"\n✓ Dataset config saved to: {yaml_path}")

    print("\n" + "="*60)
    print("✓ CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nDataset ready at: {Path(args.output).absolute()}")


if __name__ == '__main__':
    main()
