#!/usr/bin/env python3
"""
Validation script for slice range filtering

This script validates that the slice filtering is working correctly by:
1. Loading a sample of images from the dataset with and without filtering
2. Comparing the number of images loaded
3. Verifying that filtered images are within the expected slice range

Usage:
    python validate_slice_filtering.py --dataset_root /path/to/yolo_dataset --slice_range slice_range.json
    python validate_slice_filtering.py --dataset_root /path/to/yolo_dataset --slice_range 5,25
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataloaders import LoadImagesAndLabels, extract_slice_index


def parse_slice_range(slice_range_arg):
    """Parse slice_range argument into (min, max) tuple."""
    if slice_range_arg is None:
        return None

    if ',' in str(slice_range_arg):
        # Parse "min,max" format
        min_s, max_s = map(int, slice_range_arg.split(','))
        return (min_s, max_s)
    else:
        # Load from JSON file
        with open(slice_range_arg, 'r') as f:
            data = json.load(f)
        min_s = data['global_statistics']['min_slice_with_tear']
        max_s = data['global_statistics']['max_slice_with_tear']
        return (min_s, max_s)


def validate_filtering(dataset_root, slice_range_arg, split='train', img_size=640):
    """
    Validate that slice filtering is working correctly.

    Args:
        dataset_root: Root directory of YOLO dataset
        slice_range_arg: Slice range specification (JSON path or "min,max" string)
        split: Dataset split to validate ('train', 'val', or 'test')
        img_size: Image size for dataset loader

    Returns:
        dict with validation results
    """
    dataset_root = Path(dataset_root)
    images_path = dataset_root / 'images' / split

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    print("="*60)
    print("SLICE FILTERING VALIDATION")
    print("="*60)
    print(f"Dataset: {dataset_root}")
    print(f"Split: {split}")
    print(f"Slice range: {slice_range_arg}")

    # Parse slice range
    slice_range = parse_slice_range(slice_range_arg)
    if slice_range:
        min_slice, max_slice = slice_range
        print(f"Expected range: [{min_slice}, {max_slice}]")

    # Load dataset WITHOUT filtering
    print("\n" + "-"*60)
    print("Loading dataset WITHOUT slice filtering...")
    print("-"*60)
    try:
        dataset_unfiltered = LoadImagesAndLabels(
            path=str(images_path),
            img_size=img_size,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f"{split} (unfiltered): ",
            slice_range=None,  # No filtering
        )
        n_unfiltered = len(dataset_unfiltered)
        print(f"✓ Loaded {n_unfiltered} images without filtering")
    except Exception as e:
        print(f"✗ Error loading unfiltered dataset: {e}")
        return {'success': False, 'error': str(e)}

    # Load dataset WITH filtering
    print("\n" + "-"*60)
    print("Loading dataset WITH slice filtering...")
    print("-"*60)
    try:
        dataset_filtered = LoadImagesAndLabels(
            path=str(images_path),
            img_size=img_size,
            batch_size=16,
            augment=False,
            hyp=None,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f"{split} (filtered): ",
            slice_range=slice_range_arg,  # Apply filtering
        )
        n_filtered = len(dataset_filtered)
        print(f"✓ Loaded {n_filtered} images with filtering")
    except Exception as e:
        print(f"✗ Error loading filtered dataset: {e}")
        return {'success': False, 'error': str(e)}

    # Analyze filtered dataset
    print("\n" + "-"*60)
    print("ANALYSIS")
    print("-"*60)

    n_removed = n_unfiltered - n_filtered
    pct_removed = 100 * n_removed / n_unfiltered if n_unfiltered > 0 else 0

    print(f"\nFiltering Statistics:")
    print(f"  Images before filtering: {n_unfiltered}")
    print(f"  Images after filtering:  {n_filtered}")
    print(f"  Images removed:          {n_removed} ({pct_removed:.1f}%)")

    if slice_range:
        # Check that all filtered images are within range
        slice_indices = []
        out_of_range = []

        for img_file in dataset_filtered.im_files[:100]:  # Sample first 100 files
            slice_idx = extract_slice_index(img_file)
            if slice_idx is not None:
                slice_indices.append(slice_idx)
                if not (min_slice <= slice_idx <= max_slice):
                    out_of_range.append((img_file, slice_idx))

        if slice_indices:
            actual_min = min(slice_indices)
            actual_max = max(slice_indices)

            print(f"\nSlice Range Validation (sampled {len(slice_indices)} images):")
            print(f"  Expected range: [{min_slice}, {max_slice}]")
            print(f"  Actual range:   [{actual_min}, {actual_max}]")

            if out_of_range:
                print(f"\n✗ WARNING: Found {len(out_of_range)} images outside expected range!")
                for img_file, idx in out_of_range[:5]:
                    print(f"    - {Path(img_file).name}: slice {idx}")
                validation_passed = False
            else:
                print(f"  ✓ All sampled images are within expected range")
                validation_passed = True

            # Additional statistics
            print(f"\nSlice Distribution (sampled):")
            print(f"  Mean: {np.mean(slice_indices):.1f}")
            print(f"  Median: {np.median(slice_indices):.1f}")
            print(f"  Std: {np.std(slice_indices):.1f}")
        else:
            print("\nWarning: Could not extract slice indices from filenames")
            validation_passed = None
    else:
        validation_passed = None
        print("\nNo slice range specified, skipping range validation")

    # Summary
    print("\n" + "="*60)
    if validation_passed:
        print("✓ VALIDATION PASSED")
    elif validation_passed is False:
        print("✗ VALIDATION FAILED - Some images outside expected range")
    else:
        print("⚠ VALIDATION INCOMPLETE - Could not verify slice ranges")
    print("="*60)

    results = {
        'success': True,
        'validation_passed': validation_passed,
        'n_unfiltered': n_unfiltered,
        'n_filtered': n_filtered,
        'n_removed': n_removed,
        'pct_removed': pct_removed,
        'slice_range': slice_range,
    }

    if slice_indices:
        results['actual_range'] = (actual_min, actual_max)
        results['slice_stats'] = {
            'mean': float(np.mean(slice_indices)),
            'median': float(np.median(slice_indices)),
            'std': float(np.std(slice_indices)),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate slice range filtering for meniscus tear detection dataset'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Root directory of YOLO dataset'
    )
    parser.add_argument(
        '--slice_range',
        type=str,
        required=True,
        help='Slice range: JSON file path or "min,max" format'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to validate (default: train)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Image size for dataset loader (default: 640)'
    )

    args = parser.parse_args()

    try:
        results = validate_filtering(
            args.dataset_root,
            args.slice_range,
            args.split,
            args.img_size
        )

        if results['success'] and results.get('validation_passed'):
            print("\n✓ Slice filtering is working correctly!")
            sys.exit(0)
        elif results['success'] and results.get('validation_passed') is False:
            print("\n✗ Slice filtering validation failed!")
            sys.exit(1)
        else:
            print("\n⚠ Validation completed with warnings")
            sys.exit(0)

    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()