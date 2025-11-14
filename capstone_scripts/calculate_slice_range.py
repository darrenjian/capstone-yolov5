#!/usr/bin/env python3
"""
Calculate Global Slice Range for Meniscus Tear Detection

This script analyzes all label files in the YOLO dataset to find the global
min/max slice indices where tears exist. Slices outside this range can be
filtered out during training to improve efficiency.

Usage:
    python calculate_slice_range.py --dataset_root /path/to/yolo_dataset --output slice_range.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def extract_slice_and_volume(filename):
    """
    Extract slice index and volume ID from filename.

    Expected format: IM<slice_num>-<volume_id>.txt
    Example: IM5-01-02-00011.txt -> (5, "01-02-00011")

    Args:
        filename: Label filename (e.g., "IM5-01-02-00011.txt")

    Returns:
        (slice_index, volume_id) tuple or (None, None) if parsing fails
    """
    # Pattern: IM followed by digits, then hyphen, then volume ID
    pattern = r'IM(\d+)-(.+)\.txt'
    match = re.match(pattern, filename)

    if match:
        slice_idx = int(match.group(1))
        volume_id = match.group(2)
        return slice_idx, volume_id

    return None, None


def has_tear_annotation(label_file):
    """
    Check if a label file contains any tear annotations.

    Args:
        label_file: Path to label file

    Returns:
        True if file has annotations (non-empty), False otherwise
    """
    try:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            return len(content) > 0
    except Exception as e:
        print(f"Warning: Could not read {label_file}: {e}")
        return False


def analyze_dataset(dataset_root, splits=['train', 'val', 'test']):
    """
    Analyze all label files to find slice ranges with tears.

    Args:
        dataset_root: Root directory of YOLO dataset
        splits: List of splits to analyze (default: ['train', 'val', 'test'])

    Returns:
        dict with analysis results
    """
    dataset_root = Path(dataset_root)

    # Statistics per volume
    volume_stats = defaultdict(lambda: {
        'min_slice': float('inf'),
        'max_slice': float('-inf'),
        'tear_slices': [],
        'total_slices': 0
    })

    # Global statistics across all volumes
    global_min_slice = float('inf')
    global_max_slice = float('-inf')
    all_tear_slices = []

    total_files = 0
    total_with_tears = 0

    # Process each split
    for split in splits:
        labels_dir = dataset_root / 'labels' / split

        if not labels_dir.exists():
            print(f"Warning: Labels directory not found: {labels_dir}")
            continue

        label_files = sorted(labels_dir.glob('*.txt'))
        print(f"\nAnalyzing {split} split ({len(label_files)} files)...")

        for label_file in tqdm(label_files, desc=f"Processing {split}"):
            filename = label_file.name
            slice_idx, volume_id = extract_slice_and_volume(filename)

            if slice_idx is None:
                print(f"Warning: Could not parse filename: {filename}")
                continue

            total_files += 1
            volume_stats[volume_id]['total_slices'] += 1

            # Check if this slice has tear annotations
            if has_tear_annotation(label_file):
                total_with_tears += 1

                # Update volume statistics
                volume_stats[volume_id]['min_slice'] = min(
                    volume_stats[volume_id]['min_slice'], slice_idx
                )
                volume_stats[volume_id]['max_slice'] = max(
                    volume_stats[volume_id]['max_slice'], slice_idx
                )
                volume_stats[volume_id]['tear_slices'].append(slice_idx)

                # Update global statistics
                global_min_slice = min(global_min_slice, slice_idx)
                global_max_slice = max(global_max_slice, slice_idx)
                all_tear_slices.append(slice_idx)

    # Calculate statistics
    volumes_with_tears = {
        vol_id: stats for vol_id, stats in volume_stats.items()
        if stats['min_slice'] != float('inf')
    }

    # Calculate percentiles for robustness
    tear_slices_array = np.array(all_tear_slices)
    percentiles = {}
    if len(tear_slices_array) > 0:
        percentiles = {
            'p01': float(np.percentile(tear_slices_array, 1)),
            'p05': float(np.percentile(tear_slices_array, 5)),
            'p10': float(np.percentile(tear_slices_array, 10)),
            'p25': float(np.percentile(tear_slices_array, 25)),
            'p50': float(np.percentile(tear_slices_array, 50)),  # median
            'p75': float(np.percentile(tear_slices_array, 75)),
            'p90': float(np.percentile(tear_slices_array, 90)),
            'p95': float(np.percentile(tear_slices_array, 95)),
            'p99': float(np.percentile(tear_slices_array, 99)),
        }

    # Compile results
    results = {
        'global_statistics': {
            'min_slice_with_tear': int(global_min_slice) if global_min_slice != float('inf') else None,
            'max_slice_with_tear': int(global_max_slice) if global_max_slice != float('-inf') else None,
            'percentiles': percentiles,
            'total_volumes': len(volume_stats),
            'volumes_with_tears': len(volumes_with_tears),
            'total_slices_analyzed': total_files,
            'slices_with_tears': total_with_tears,
            'slices_without_tears': total_files - total_with_tears,
        },
        'per_volume_statistics': {}
    }

    # Add per-volume statistics
    for vol_id, stats in sorted(volumes_with_tears.items()):
        results['per_volume_statistics'][vol_id] = {
            'min_slice': int(stats['min_slice']),
            'max_slice': int(stats['max_slice']),
            'num_slices_with_tears': len(stats['tear_slices']),
            'total_slices': stats['total_slices'],
        }

    return results


def print_summary(results):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("SLICE RANGE ANALYSIS SUMMARY")
    print("="*60)

    global_stats = results['global_statistics']

    print(f"\nGlobal Statistics:")
    print(f"  Total volumes analyzed: {global_stats['total_volumes']}")
    print(f"  Volumes with tears: {global_stats['volumes_with_tears']}")
    print(f"  Total slices analyzed: {global_stats['total_slices_analyzed']}")
    print(f"  Slices with tears: {global_stats['slices_with_tears']}")
    print(f"  Slices without tears: {global_stats['slices_without_tears']}")

    if global_stats['min_slice_with_tear'] is not None:
        print(f"\nGlobal Slice Range (with tears):")
        print(f"  Min slice index: {global_stats['min_slice_with_tear']}")
        print(f"  Max slice index: {global_stats['max_slice_with_tear']}")
        print(f"  Range: {global_stats['max_slice_with_tear'] - global_stats['min_slice_with_tear'] + 1} slices")

        if global_stats['percentiles']:
            print(f"\nPercentiles of slice indices with tears:")
            for p, val in global_stats['percentiles'].items():
                print(f"  {p.upper()}: {val:.1f}")

        # Calculate potential savings
        total_slices = global_stats['total_slices_analyzed']
        min_idx = global_stats['min_slice_with_tear']
        max_idx = global_stats['max_slice_with_tear']

        # Estimate slices that would be filtered out
        # Assuming slice indices typically range from 1 to ~30-40 for knee MRI
        # This is a rough estimate
        print(f"\nRecommended filtering strategy:")
        print(f"  Keep only slices in range [{min_idx}, {max_idx}]")
        print(f"  This ensures all tear annotations are included")

        # Show percentile-based options for more aggressive filtering
        if global_stats['percentiles']:
            p05 = int(global_stats['percentiles']['p05'])
            p95 = int(global_stats['percentiles']['p95'])
            print(f"\nAlternative (95% coverage, more aggressive):")
            print(f"  Keep slices in range [{p05}, {p95}]")
            print(f"  This covers 95% of tear cases, may miss some edge cases")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate global slice range for meniscus tear detection dataset'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root directory of YOLO dataset (contains images/ and labels/ folders)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='slice_range.json',
        help='Output JSON file for slice range statistics (default: slice_range.json)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to analyze (default: train val test)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Slice Range Analysis for Meniscus Tear Detection")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Splits to analyze: {args.splits}")
    print(f"Output file: {args.output}")

    # Run analysis
    results = analyze_dataset(args.dataset_root, args.splits)

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nNext steps:")
    print("1. Review the slice range statistics above")
    print("2. Use the global min/max values to filter your dataloader")
    print("3. Update your training configuration to use filtered data")


if __name__ == '__main__':
    main()