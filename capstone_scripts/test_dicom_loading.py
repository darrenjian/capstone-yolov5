#!/usr/bin/env python3
"""
Test script to verify DICOM loading functionality in modified YOLOv5 dataloader
"""

import sys
from pathlib import Path
import numpy as np

# Add yolov5 to path
yolov5_path = Path(__file__).parent.parent
sys.path.insert(0, str(yolov5_path))

# Import DICOM functions from modified dataloaders
from utils.dataloaders import load_dicom_image, verify_dicom_image

def test_single_dicom_file():
    """Test loading a single DICOM file"""

    # Test file path
    test_file = Path(__file__).parent.parent / 'yolo_dataset' / 'images' / 'test' / 'IM1-01-02-00015.dcm'

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return False

    print(f"Testing DICOM file: {test_file}")
    print("="*70)

    # Test 1: Verify DICOM image
    print("\n1. Testing verify_dicom_image()...")
    shape = verify_dicom_image(test_file)

    if shape is None:
        print("verify_dicom_image() failed - returned None")
        return False

    print(f"✓ DICOM verified successfully")
    print(f"  Shape: {shape}")

    # Test 2: Load DICOM image
    print("\n2. Testing load_dicom_image()...")
    try:
        img = load_dicom_image(test_file)
        print(f"✓ DICOM loaded successfully")
    except Exception as e:
        print(f"load_dicom_image() failed: {e}")
        return False

    # Test 3: Verify image properties
    print("\n3. Verifying image properties...")

    # Check type
    if not isinstance(img, np.ndarray):
        print(f"Image is not numpy array, got: {type(img)}")
        return False
    print(f"✓ Image type: {type(img)}")

    # Check shape (should be HxWx3 for BGR)
    if len(img.shape) != 3:
        print(f"Image should be 3D (HxWx3), got shape: {img.shape}")
        return False
    print(f"✓ Image shape: {img.shape}")

    if img.shape[2] != 3:
        print(f"Image should have 3 channels (BGR), got: {img.shape[2]}")
        return False
    print(f"✓ Channels: {img.shape[2]} (BGR format)")

    # Check dtype (should be uint8)
    if img.dtype != np.uint8:
        print(f"Image dtype should be uint8, got: {img.dtype}")
        return False
    print(f"✓ Data type: {img.dtype}")

    # Check value range (should be 0-255)
    min_val, max_val = img.min(), img.max()
    if min_val < 0 or max_val > 255:
        print(f"Image values out of range [0, 255]: [{min_val}, {max_val}]")
        return False
    print(f"✓ Value range: [{min_val}, {max_val}]")

    # Check dimensions (should be 512x512 for this dataset)
    if img.shape[0] != 512 or img.shape[1] != 512:
        print(f"Warning: Expected 512x512, got {img.shape[0]}x{img.shape[1]}")
    else:
        print(f"✓ Dimensions: {img.shape[0]}x{img.shape[1]}")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

    return True

def test_multiple_slices():
    """Test loading multiple DICOM slices from the volume"""

    volume_dir = Path(__file__).parent.parent / 'data' / 'dicoms' / 'TBrecon-01-02-00011'

    if not volume_dir.exists():
        print(f"Volume directory not found: {volume_dir}")
        return False

    print(f"\nTesting multiple DICOM slices from: {volume_dir.name}")
    print("="*70)

    # Get first 5 DICOM files
    dcm_files = sorted(volume_dir.glob('IM*.dcm'))[:5]

    if not dcm_files:
        print("No DICOM files found")
        return False

    print(f"Testing {len(dcm_files)} files...")

    for i, dcm_file in enumerate(dcm_files, 1):
        try:
            img = load_dicom_image(dcm_file)
            print(f"  {i}. {dcm_file.name}: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}] ✓")
        except Exception as e:
            print(f"  {i}. {dcm_file.name}: FAILED - {e}")
            return False

    print("\nMultiple slice test passed!")
    print("="*70)

    return True

def main():
    print("\n" + "="*70)
    print("DICOM LOADING TEST SUITE")
    print("="*70)

    # Test 1: Single file
    if not test_single_dicom_file():
        print("\nSingle file test failed")
        return 1

    # Test 2: Multiple files
    if not test_multiple_slices():
        print("\nMultiple slice test failed")
        return 1

    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python test_dataloader_batch.py")
    print("  2. Train with DICOM: cd yolov5 && python train.py --data ../yolo_dataset/dataset.yaml")
    print()

    return 0

if __name__ == '__main__':
    sys.exit(main())
