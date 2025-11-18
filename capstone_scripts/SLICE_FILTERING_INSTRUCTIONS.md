# Efficient Slice Filtering for Volumetric Medical Imaging

This feature implements efficient data loading by filtering out irrelevant slices from volumetric MRI data based on global tear annotation statistics.

## Overview

In volumetric knee MRI scans, meniscus tears typically only occur in a subset of slices in the middle of the volume. The slices at the very beginning and end of each volume rarely contain any tears. By identifying the global min/max slice indices where tears occur across ALL volumes, we can safely filter out these irrelevant slices during training, significantly reducing:

- Training time
- Memory usage
- I/O overhead
- Dataset size on disk

## How It Works

1. **Analysis Phase**: Analyze all label files to find the global min/max slice indices where tears exist across all volumes
2. **Filtering Phase**: During data loading, automatically skip any slices outside this global range
3. **Training Phase**: Train on only the relevant slices that can potentially contain tears

## Quick Start

### Step 0: Find Your YOLO Dataset Path

First, locate your YOLO-formatted dataset (created by `batch_convert_annotations.py`):

```bash
# Find your dataset - look for conversion_report.json
find /gpfs -name "conversion_report.json" 2>/dev/null

# Or look for directories with both images/ and labels/
find /gpfs -type d -name "labels" 2>/dev/null | grep -v ".git"

# The directory should have this structure:
# your_dataset/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# ├── labels/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── dataset.yaml
```

**Example path:** `/gpfs/home/username/capstone-yolov5/yolo_dataset`

**Use YOUR actual path** in the commands below.

### Step 1: Calculate Global Slice Range

First, analyze your dataset to find the optimal slice range:

```bash
# Replace <YOUR_DATASET_PATH> with your actual path
python capstone_scripts/calculate_slice_range.py \
    --dataset_root <YOUR_DATASET_PATH> \
    --output slice_range.json \
    --splits train val test

# Real example:
python capstone_scripts/calculate_slice_range.py \
    --dataset_root /gpfs/home/ic2664/capstone-yolov5/yolo_dataset \
    --output slice_range.json \
    --splits train val test
```

This will output statistics like:
```
Global Slice Range (with tears):
  Min slice index: 5
  Max slice index: 25
  Range: 21 slices

Recommended filtering strategy:
  Keep only slices in range [5, 25]
```

### Step 2: Validate the Filtering

Before training, validate that the filtering works correctly:

```bash
# Replace <YOUR_DATASET_PATH> with your actual path
python capstone_scripts/validate_slice_filtering.py \
    --dataset_root <YOUR_DATASET_PATH> \
    --slice_range slice_range.json \
    --split train

# Real example:
python capstone_scripts/validate_slice_filtering.py \
    --dataset_root /gpfs/home/ic2664/capstone-yolov5/yolo_dataset \
    --slice_range slice_range.json \
    --split train
```

This should show output like:
```
Filtering Statistics:
  Images before filtering: 1500
  Images after filtering:  1050
  Images removed:          450 (30.0%)
```

### Step 3: Train with Slice Filtering

Use the `--slice-range` parameter when training.

**Important:** The `--data` argument should point to your `dataset.yaml` file (inside your dataset directory).

```bash
# Option 1: Use JSON file (recommended)
python train.py \
    --data <YOUR_DATASET_PATH>/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 100 \
    --slice-range slice_range.json

# Option 2: Specify min,max directly
python train.py \
    --data <YOUR_DATASET_PATH>/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 100 \
    --slice-range 5,25

# Real example:
python train.py \
    --data /gpfs/home/ic2664/capstone-yolov5/yolo_dataset/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 100 \
    --slice-range slice_range.json
```

## Detailed Usage

### calculate_slice_range.py

**Purpose**: Analyze all label files to find global min/max slice indices where tears exist.

**Arguments**:
- `--dataset_root`: Root directory of your YOLO dataset (required)
- `--output`: Output JSON file for statistics (default: `slice_range.json`)
- `--splits`: Dataset splits to analyze (default: `train val test`)

**Output**: Creates a JSON file with:
- Global min/max slice indices
- Per-volume statistics
- Percentile distributions
- Recommendations for filtering

**Example Output**:
```json
{
  "global_statistics": {
    "min_slice_with_tear": 5,
    "max_slice_with_tear": 25,
    "total_volumes": 50,
    "volumes_with_tears": 45,
    "total_slices_analyzed": 1500,
    "slices_with_tears": 800,
    "percentiles": {
      "p05": 6.0,
      "p50": 15.0,
      "p95": 24.0
    }
  }
}
```

### validate_slice_filtering.py

**Purpose**: Validate that slice filtering is working correctly before training.

**Arguments**:
- `--dataset_root`: Root directory of YOLO dataset (required)
- `--slice_range`: Slice range specification - JSON path or "min,max" (required)
- `--split`: Dataset split to validate (default: `train`)
- `--img_size`: Image size for loader (default: 640)

**What it checks**:
- ✓ Dataset loads successfully with and without filtering
- ✓ Correct number of images are filtered
- ✓ All filtered images are within expected slice range
- ✓ Provides statistics on filtering efficiency

### Training with --slice-range

**Purpose**: Enable slice filtering during training.

**Format Options**:

1. **JSON file path**: Points to output from `calculate_slice_range.py`
   ```bash
   --slice-range /path/to/slice_range.json
   ```

2. **Min,max tuple**: Directly specify the range
   ```bash
   --slice-range 5,25
   ```

3. **No filtering**: Omit the parameter or set to None
   ```bash
   # No --slice-range parameter = no filtering
   ```

## Understanding Your Data

### File Naming Convention

This feature assumes your DICOM files follow the naming pattern:
```
IM<slice_number>-<volume_id>.dcm
```

Examples:
- `IM5-01-02-00011.dcm` → Slice 5 of volume "01-02-00011"
- `IM15-01-02-00011.dcm` → Slice 15 of volume "01-02-00011"
- `IM25-01-02-00011.dcm` → Slice 25 of volume "01-02-00011"

### Slice Index Extraction

The system automatically extracts slice indices from filenames using the pattern `IM(\d+)-`. If your files use a different naming convention, you may need to modify the `extract_slice_index()` function in `utils/dataloaders.py`.

## Expected Benefits

Based on typical knee MRI datasets:

- **~30-40% reduction in training data**: If your volumes have 30-40 slices but tears only occur in slices 5-25, you save ~30-40% of slices
- **Faster training**: Proportional to reduction in data (30-40% fewer slices = ~30-40% faster epochs)
- **Reduced memory usage**: Less data cached in RAM
- **Faster I/O**: Fewer files to read from disk

## Advanced Options

### Using Percentiles for More Aggressive Filtering

The `calculate_slice_range.py` script also calculates percentiles. For more aggressive filtering at the cost of potentially missing some edge cases:

```bash
# Use p05 and p95 instead of absolute min/max
# This covers 90% of cases while filtering more aggressively
python train.py \
    --slice-range 6,24 \
    ...
```

Check the percentiles in `slice_range.json` under `global_statistics.percentiles`.

### Per-Split Analysis

You can analyze specific splits separately:

```bash
# Analyze only training split
python capstone_scripts/calculate_slice_range.py \
    --dataset_root /path/to/dataset \
    --output train_slice_range.json \
    --splits train
```

## Testing with SLURM

For quick testing on HPC with GPU:

### Option 1: Quick One-Liner

```bash
sbatch --partition=a100_short --time=01:00:00 --gres=gpu:1 --mem=16G \
  --wrap="cd /gpfs/home/ic2664/capstone-yolov5 && python train.py \
  --data /gpfs/home/ic2664/capstone-yolov5/yolo_dataset/dataset.yaml \
  --weights yolov5s.pt --epochs 1 --batch-size 16 \
  --slice-range slice_range.json --device 0"
```

### Option 2: Use the Provided Script

```bash
# Submit the test job
sbatch capstone_scripts/test_slice_filter_train.sbatch

# Monitor progress
squeue -u $USER
tail -f logs/test_slice_filter_<jobid>.out
```

**Note:** Adjust `#SBATCH --partition=` in the script to match your HPC's partition names (e.g., `a100_short`, `gpu`, `gpu_short`, etc.).

The test job runs just 1 epoch to verify everything works before committing to a full training run.

## Troubleshooting

### Issue: "Labels directory not found"

**Cause**: You're pointing to the wrong directory. The script needs the YOLO-formatted dataset, not the raw DICOM directory.

**Solution**:
1. Find your YOLO dataset:
   ```bash
   find /gpfs -name "conversion_report.json" 2>/dev/null
   ```
2. The directory should contain `images/` and `labels/` folders
3. Use that path with `--dataset_root`

**Example:**
- ✅ Correct: `/gpfs/home/ic2664/capstone-yolov5/yolo_dataset`
- ❌ Wrong: `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Dicom_Meniscal_Tear`

### Issue: "Could not parse filename"

**Cause**: Your files don't follow the expected naming pattern `IM<number>-<volume_id>.dcm`

**Solution**: Modify the `extract_slice_index()` function in `utils/dataloaders.py` to match your naming convention.

### Issue: "No slices filtered"

**Possible causes**:
1. All your slices contain tears (unusual but possible)
2. Filename pattern doesn't match
3. Slice range is too wide

**Solution**: Check the output of `calculate_slice_range.py` to see what range was detected.

### Issue: "Validation failed - images outside range"

**Cause**: The filtering logic may have a bug or the slice range is incorrect.

**Solution**:
1. Re-run `calculate_slice_range.py` to verify the range
2. Check a few filenames manually to ensure they match the pattern
3. Report the issue with example filenames

## Integration with Existing Workflows

### With Batch Scripts

Add the `--slice-range` parameter to your existing training scripts:

```bash
#!/bin/bash
#SBATCH ...

python train.py \
    --data data/meniscus.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 32 \
    --epochs 300 \
    --slice-range slice_range.json \  # Add this line
    --device 0,1,2,3
```

### With Hyperparameter Tuning

The slice filtering is applied before data augmentation and hyperparameter optimization, so it works seamlessly with evolve and other tuning methods.

### With Multi-GPU Training

Slice filtering works with DDP (DistributedDataParallel) training automatically.

## Files Modified

1. **utils/dataloaders.py**:
   - Added `extract_slice_index()` function
   - Added `slice_range` parameter to `LoadImagesAndLabels`
   - Added `slice_range` parameter to `create_dataloader()`
   - Added filtering logic after min_items filter

2. **train.py**:
   - Added `--slice-range` argument to `parse_opt()`
   - Added slice range parsing logic
   - Passed `slice_range` to `create_dataloader()` calls

3. **capstone_scripts/calculate_slice_range.py** (new):
   - Script to analyze dataset and find optimal slice range

4. **capstone_scripts/validate_slice_filtering.py** (new):
   - Script to validate filtering is working correctly

## Performance Metrics

To measure the impact of slice filtering on your training:

1. **Without filtering**:
```bash
python train.py --data data/meniscus.yaml --epochs 10 --batch-size 16
# Note the epoch time and total images
```

2. **With filtering**:
```bash
python train.py --data data/meniscus.yaml --epochs 10 --batch-size 16 --slice-range 5,25
# Compare epoch time and total images
```

Expected improvements:
- Epoch time: ~30-40% faster
- Total images: ~30-40% fewer
- mAP: Should be similar or slightly better (less noise from irrelevant slices)

## Future Enhancements

Potential improvements to this feature:

1. **Per-volume filtering**: Instead of global min/max, use per-volume statistics
2. **Adaptive filtering**: Learn optimal range during training
3. **Class-specific filtering**: Different ranges for different tear types
4. **Slice importance weighting**: Give higher weight to slices near the optimal range

## Questions?

For issues or questions about this feature:
1. Check the validation script output
2. Review the slice_range.json statistics
3. Verify your filename pattern matches expected format
4. Check training logs for filtering messages

## Summary

Complete workflow from start to finish:

```bash
# 0. Find your YOLO dataset path
find /gpfs -name "conversion_report.json" 2>/dev/null
# Example output: /gpfs/home/ic2664/capstone-yolov5/yolo_dataset/conversion_report.json
# Your dataset path is: /gpfs/home/ic2664/capstone-yolov5/yolo_dataset

# Navigate to project directory
cd /gpfs/home/ic2664/capstone-yolov5

# 1. Calculate optimal slice range
python capstone_scripts/calculate_slice_range.py \
    --dataset_root /gpfs/home/ic2664/capstone-yolov5/yolo_dataset \
    --output slice_range.json \
    --splits train val test

# 2. Validate filtering works (takes < 1 minute)
python capstone_scripts/validate_slice_filtering.py \
    --dataset_root /gpfs/home/ic2664/capstone-yolov5/yolo_dataset \
    --slice_range slice_range.json \
    --split train

# 3a. Test with 1 epoch (optional but recommended)
sbatch capstone_scripts/test_slice_filter_train.sbatch

# 3b. Full training with filtering
python train.py \
    --data /gpfs/home/ic2664/capstone-yolov5/yolo_dataset/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 100 \
    --slice-range slice_range.json \
    --device 0
```

**Key Points:**
- Replace paths with YOUR actual dataset location
- `--dataset_root` points to the directory with `images/` and `labels/` folders
- `--data` points to the `dataset.yaml` file inside your dataset
- Validation step (step 2) is fast and confirms everything works

This should give you significant training speedups while maintaining or improving model performance!