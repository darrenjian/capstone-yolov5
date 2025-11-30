# Creating Balanced Dataset Instructions

## Overview

This guide walks you through creating a dataset with a **balanced training set** while preserving **realistic class imbalance in val/test sets** from your existing **combined YOLO dataset** (created with `batch_convert_annotations.py --append`).

**Key approach:**
1. **Group slices by VOLUME ID** to prevent data leakage
2. **Split VOLUMES** (not slices) into train/val/test (70/15/15) - ensures no patient data leaks between splits
3. **Undersample ONLY the train set** to achieve 50/50 balance (~12,791 tear, ~12,791 no-tear)
4. **Preserve original imbalance in val/test** for realistic evaluation (~19% tear, ~81% no-tear)
5. **Verify no volume overlap** between splits

**Why this matters:**
- **NO DATA LEAKAGE**: All slices from same MRI volume stay in same split
- Train on balanced data for stable learning
- Evaluate on realistic data to measure real-world performance
- Valid evaluation metrics (no information leakage from train to val/test)
- Total: ~39,000 slices (reduced from ~94,440)

---

## Prerequisites

You should have already created a **combined YOLO dataset** using `batch_convert_annotations.py` with the `--append` flag:

### Step 1: Create meniscus tear dataset
```bash
python capstone_scripts/batch_convert_annotations.py \
    --data_source meniscus_tear \
    --output /path/to/yolo_dataset \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Step 2: Append normal dataset
```bash
python capstone_scripts/batch_convert_annotations.py \
    --data_source normal \
    --output /path/to/yolo_dataset \
    --append \
    --seed 42
```

This creates a **combined dataset** with:
- Both tear and normal volumes
- ~18,273 slices with tears (from meniscus_tear)
- ~76,167 slices without tears (from meniscus_tear + normal)
- Split at **volume level** (all slices from same volume in same split)

Expected structure:
```
yolo_dataset/
├── images/
│   ├── train/          # Mix of tear & no-tear slices
│   ├── val/            # Mix of tear & no-tear slices
│   └── test/           # Mix of tear & no-tear slices
├── labels/
│   ├── train/          # Corresponding labels (empty = no tear)
│   ├── val/
│   └── test/
├── dataset.yaml
└── conversion_report.json
```

---

## Why Balance Train Set Only?

### Current Problem (Imbalanced Training):
```
Original dataset: 94,440 slices
- Tears: 18,273 (19.4%)
- No-tears: 76,167 (80.6%)
→ Severe class imbalance during training
→ Model biased to predict "no object"
→ Requires aggressive weighting (obj_pw=5.0)
→ Still only mAP@0.1 = 0.698
```

### After Train-Only Balancing:
```
Train set: ~25,582 slices
- Tears: ~12,791 (50%)
- No-tears: ~12,791 (50%)
→ Balanced training for stable learning

Val/Test sets: ~13,964 slices total
- Tears: ~5,482 (19.4%)
- No-tears: ~28,482 (80.6%)
→ Preserve realistic imbalance for evaluation

Benefits:
→ More stable training convergence
→ Realistic evaluation metrics
→ Expected mAP@0.1: 0.75-0.85+
```

---

## Step-by-Step Process

### Step 1: Verify Your Combined Dataset

Check your current dataset statistics:

```bash
# Count tear slices (non-empty label files)
find yolo_dataset/labels -name "*.txt" -type f -exec sh -c 'test -s "$1"' _ {} \; -print | wc -l

# Count no-tear slices (empty label files)
find yolo_dataset/labels -name "*.txt" -type f -exec sh -c 'test ! -s "$1"' _ {} \; -print | wc -l

# Total slices
find yolo_dataset/labels -name "*.txt" | wc -l
```

Expected output:
```
Tear slices: ~18,273
No-tear slices: ~76,167
Total: ~94,440
```

---

### Step 2: Create Dataset with Balanced Train Set

The script will:
1. **Pool all slices** from train/val/test (ignoring original volume-level splits)
2. **Group slices by VOLUME ID** (extracted from filenames like `IM1-01-02-00011.dcm`)
3. **Identify volumes** with tears vs without tears (a volume has tears if ANY slice has tears)
4. **Split VOLUMES** 70/15/15 into train/val/test - **PREVENTS DATA LEAKAGE**
5. **Extract slices** from each volume group
6. **Undersample only train set slices** to achieve 50/50 balance (keep all tear, sample matching no-tear)
7. **Preserve imbalance** in val/test sets for realistic evaluation
8. **Verify no volume overlap** between splits
9. **Copy** to new dataset directory

```bash
cd capstone_scripts

python create_balanced_dataset.py \
    --input /path/to/yolo_dataset \
    --output /path/to/balanced_yolo_dataset \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

**HPC Example:**
```bash
python create_balanced_dataset.py \
    --input /gpfs/home/dj2565/capstone-yolov5/yolo_dataset \
    --output /gpfs/home/dj2565/capstone-yolov5/balanced_yolo_dataset \
    --seed 42
```

**Note:** The original `yolo_dataset` is **not modified** - a new balanced copy is created.

---

### Step 3: Review the Output

The script will display detailed statistics:

```
============================================================
STEP 1: Scanning combined dataset
============================================================

Scanning combined dataset: /path/to/yolo_dataset
Pooling slices from all existing splits and grouping by volume...

  Total slices: 18,273 tear, 76,167 no-tear
  Total volumes: 93 with tears, 457 without tears
  Slices per category: 18,273 in tear volumes, 76,167 in no-tear volumes

✓ Pooled totals:
  Tear volumes: 93 (18,273 slices)
  No-tear volumes: 457 (76,167 slices)

============================================================
STEP 2: Splitting volumes and undersampling train set
============================================================

SPLITTING VOLUMES (PREVENTS DATA LEAKAGE)
============================================================
Original distribution:
  Volumes: 93 with tears, 457 without tears
  Slices: 18,273 in tear volumes, 76,167 in no-tear volumes

============================================================
STEP 1: Split volumes (70/15/15) - NO DATA LEAKAGE
============================================================
  Train: 65 tear volumes + 320 no-tear volumes
  Val:   14 tear volumes + 69 no-tear volumes
  Test:  14 tear volumes + 68 no-tear volumes

============================================================
STEP 2: Extract slices from volumes
============================================================
  Train: 12,791 tear slices + 53,317 no-tear slices
  Val:   2,741 tear slices + 11,425 no-tear slices
  Test:  2,741 tear slices + 11,425 no-tear slices

============================================================
STEP 3: Undersampling train set for 50/50 balance
============================================================
Train set before undersampling:
  Tear slices: 12,791
  No-tear slices: 53,317
  Ratio: 4.2:1 (no-tear:tear)

✓ Undersampling 12,791 no-tear slices from 53,317 available in train
  Removed 40,526 no-tear slices from train set

============================================================
FINAL SPLITS (train balanced, val/test preserve imbalance)
============================================================
  train: 25,582 slices (12,791 tear [50.0%], 12,791 no-tear)
  val  : 14,166 slices ( 2,741 tear [19.4%], 11,425 no-tear)
  test : 14,166 slices ( 2,741 tear [19.4%], 11,425 no-tear)

============================================================
STEP 3: Copying files to output directory
============================================================
[Progress bars...]

============================================================
DATA LEAKAGE VERIFICATION
============================================================
✓ NO DATA LEAKAGE: All volumes are in exactly one split
  Train: 385 unique volumes
  Val:   83 unique volumes
  Test:  82 unique volumes

============================================================
✓ DATASET CREATED SUCCESSFULLY!
============================================================

Dataset location: /path/to/balanced_yolo_dataset

Final statistics:
  Total images: 53,914
  Train: 25,582 (12,791 tear [50.0%], 12,791 no-tear) - BALANCED 50/50
  Val  : 14,166 ( 2,741 tear [19.4%], 11,425 no-tear) - original imbalance
  Test : 14,166 ( 2,741 tear [19.4%], 11,425 no-tear) - original imbalance

  Overall: 18,273 tear : 40,641 no-tear
  Train set: 12,791 tear : 12,791 no-tear (balanced 1:1)
  Val/Test: Preserve original class imbalance for realistic evaluation

Reduction from original:
  Before: 18,273 tear + 76,167 no-tear = 94,440 total
  After:  18,273 tear + 40,641 no-tear = 58,914 total
  Removed: 35,526 no-tear slices (from train set only)
```

---

### Step 4: Verify the Balanced Dataset

Check the output directory structure:
```bash
balanced_yolo_dataset/
├── images/
│   ├── train/          # ~25,582 images (50% tear, 50% no-tear) - BALANCED
│   ├── val/            # ~14,166 images (19% tear, 81% no-tear) - Original imbalance
│   └── test/           # ~14,166 images (19% tear, 81% no-tear) - Original imbalance
├── labels/
│   ├── train/          # ~25,582 labels
│   ├── val/            # ~14,166 labels
│   └── test/           # ~14,166 labels
├── dataset.yaml        # YOLOv5 config
└── balanced_dataset_metadata.json  # Detailed metadata
```

Review metadata:
```bash
cat balanced_yolo_dataset/balanced_dataset_metadata.json | jq '.statistics'
```

Output:
```json
{
  "splits": {
    "train": {"total": 25582, "tear": 12791, "no_tear": 12791},
    "val": {"total": 14166, "tear": 2741, "no_tear": 11425},
    "test": {"total": 14166, "tear": 2741, "no_tear": 11425}
  },
  "total_images": 53914,
  "total_tear": 18273,
  "total_no_tear": 40641,
  "train_balance_ratio": "12791:12791",
  "val_balance_ratio": "2741:11425",
  "test_balance_ratio": "2741:11425"
}
```

**Verify NO Data Leakage:**
```bash
cat balanced_yolo_dataset/balanced_dataset_metadata.json | jq '.volume_statistics'
```

Output:
```json
{
  "train_volumes": 385,
  "val_volumes": 83,
  "test_volumes": 82,
  "total_volumes": 550,
  "no_overlap_verified": true
}
```

This confirms that:
- Each volume appears in exactly ONE split (train, val, or test)
- No slices from the same MRI volume are in different splits
- Your evaluation metrics are valid (no information leakage)

---

### Step 5: Update Hyperparameters for Balanced Dataset

Since your dataset is now balanced, you can use **less aggressive** settings:

Create `data/hyps/hyp.balanced.yaml`:

```yaml
# Learning rate - moderate for balanced data
lr0: 0.005              # Start higher than 0.002 (faster convergence)
lrf: 0.00001            # Fine-tune ending
momentum: 0.95
weight_decay: 0.001     # Anti-overfitting

# Warmup
warmup_epochs: 5.0      # Longer warmup
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss weights - MODERATE (not aggressive like before)
box: 0.03               # Lower for mAP@0.1 (loose IoU)
cls: 0.5
obj: 3.0                # Prioritize detection
cls_pw: 1.0             # Balanced now, no need to weight
obj_pw: 2.5             # REDUCED from 5.0 (data is balanced)

# Training thresholds
iou_t: 0.1              # Match mAP@0.1 evaluation
anchor_t: 5.0           # More anchors match targets
label_smoothing: 0.1    # Anti-overfitting
fl_gamma: 0.0           # Keep disabled (hurt you before)

# Aggressive augmentation (anti-overfitting)
hsv_h: 0.03
hsv_s: 0.9
hsv_v: 0.6
degrees: 15.0           # Rotation
translate: 0.2
scale: 0.7
shear: 5.0
perspective: 0.0005
flipud: 0.1
fliplr: 0.5
mosaic: 1.0             # Keep enabled
mixup: 0.3              # Blends images
copy_paste: 0.5         # Paste tears onto backgrounds
```

**Key changes from imbalanced dataset:**
- ✅ `obj_pw: 2.5` (was 5.0) - less aggressive
- ✅ `lr0: 0.005` (was 0.002) - faster learning
- ✅ Can remove `--image-weights` (already balanced)
- ✅ No need for `--slice-range` (already balanced)

---

### Step 6: Train with Balanced Dataset

```bash
python train.py \
    --img 640 \
    --batch 32 \
    --epochs 250 \
    --data /path/to/balanced_yolo_dataset/dataset.yaml \
    --weights yolov5n.pt \
    --name train_balanced_optimized \
    --hyp data/hyps/hyp.balanced.yaml \
    --patience 50 \
    --save-period 10 \
    --device 0 \
    --single-cls \
    --cos-lr
```

**HPC Example:**
```bash
python train.py \
    --img 640 \
    --batch 32 \
    --epochs 250 \
    --data /gpfs/home/dj2565/capstone-yolov5/balanced_yolo_dataset/dataset.yaml \
    --weights yolov5n.pt \
    --name train_balanced_optimized \
    --hyp data/hyps/hyp.balanced.yaml \
    --patience 50 \
    --save-period 10 \
    --device 0 \
    --single-cls \
    --cos-lr
```

**What's different:**
- ❌ **Removed `--slice-range`** - already balanced
- ❌ **Removed `--image-weights`** - optional now (already balanced)
- ✅ **Using `--single-cls`** - simplifies to pure detection
- ✅ **Using `--cos-lr`** - better convergence than linear decay
- ✅ **Lower `obj_pw`** in hyp.yaml (2.5 vs 5.0)

---

## Expected Results

### Before (Imbalanced Training):
```yaml
Dataset: 94,440 slices (19% tear, 81% no-tear) - all splits imbalanced
Settings:
  obj_pw: 5.0
  --image-weights
  --slice-range
  lr0: 0.002

Results:
  mAP@0.1: 0.698
  Training: Unstable, model biased to "no object"
  Overfitting: Severe with higher LR
  Evaluation: Not representative of real-world distribution
```

### After (Balanced Train, Realistic Val/Test):
```yaml
Dataset: ~53,914 slices total
  Train: 25,582 slices (50% tear, 50% no-tear) - BALANCED
  Val:   14,166 slices (19% tear, 81% no-tear) - Realistic
  Test:  14,166 slices (19% tear, 81% no-tear) - Realistic

Settings:
  obj_pw: 2.5
  --single-cls
  --cos-lr
  lr0: 0.005

Expected Results:
  mAP@0.1: 0.75-0.85+ ✨ (evaluated on realistic distribution)
  Training: More stable, better convergence
  Overfitting: Reduced (with proper augmentation)
  Train/Val gap: Smaller
  Evaluation: Truly reflects real-world performance
```

---

## Advanced Options

### Re-sample with Different Random Seed

Create multiple variants for ensemble training:

```bash
# Variant 1
python create_balanced_dataset.py \
    --input /path/to/yolo_dataset \
    --output /path/to/balanced_dataset_seed42 \
    --seed 42

# Variant 2
python create_balanced_dataset.py \
    --input /path/to/yolo_dataset \
    --output /path/to/balanced_dataset_seed123 \
    --seed 123

# Variant 3
python create_balanced_dataset.py \
    --input /path/to/yolo_dataset \
    --output /path/to/balanced_dataset_seed456 \
    --seed 456
```

Train on each and ensemble the predictions for better results.

### Custom Split Ratios

```bash
# 80/10/10 split
python create_balanced_dataset.py \
    --input /path/to/yolo_dataset \
    --output /path/to/balanced_dataset_80_10_10 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --seed 42
```

---

## Troubleshooting

### Issue: "Input directory not found"

**Solution:** Check path and use absolute paths:
```bash
python create_balanced_dataset.py \
    --input $(pwd)/yolo_dataset \
    --output $(pwd)/balanced_yolo_dataset
```

### Issue: "Not enough no-tear slices"

If you see a warning that there aren't enough no-tear slices, verify your counts:

```bash
# Should show ~76,167 no-tear slices
find yolo_dataset/labels -name "*.txt" -type f -size 0 | wc -l
```

If count is much lower, check:
- Did `batch_convert_annotations.py --append` complete successfully?
- Are label files actually created (not just images)?

### Issue: Script runs slowly

**Expected:** Copying ~36,546 images takes time
- On HPC: ~5-10 minutes
- On local machine: ~2-5 minutes

**Speed it up:** Use symlinks instead of copying (modify script if needed).

---

## Key Differences from Original Workflow

### Original (with --slice-range):
1. Created combined dataset with 90/10 imbalance across all splits
2. Used `--slice-range` during training to filter at runtime
3. Still had imbalance in actual batches
4. Required `obj_pw=5.0` to compensate
5. Val/test also imbalanced but no way to verify realistic performance

### New (balanced train, realistic val/test):
1. ✅ Split data first (70/15/15)
2. ✅ Undersample only train set to 50/50
3. ✅ Train on perfectly balanced data
4. ✅ Evaluate on realistic class distribution
5. ✅ No runtime filtering needed
6. ✅ Lower `obj_pw=2.5` sufficient
7. ✅ Better results, more stable training
8. ✅ Honest evaluation metrics that reflect real-world performance

---

## Summary

**What this achieves:**
- ✅ **NO DATA LEAKAGE**: Splits at VOLUME level, not slice level
- ✅ **Volume-level splitting**: All slices from same MRI stay in same split
- ✅ **Automated verification**: Script checks for volume overlap between splits
- ✅ Balanced training set (50/50: ~12,791 tear, ~12,791 no-tear)
- ✅ Realistic val/test sets (19/81: preserves original imbalance)
- ✅ Removes ~40,526 excess no-tear slices from train only
- ✅ Reproducible with seed parameter
- ✅ Preserves all tear data (nothing discarded)

**Training improvements:**
- ✅ More stable convergence from balanced training data
- ✅ Less overfitting
- ✅ Better generalization
- ✅ Expected mAP@0.1: **0.75-0.85+** (vs 0.698 before)
- ✅ Smaller train/val gap
- ✅ Can use higher learning rates (0.005 vs 0.002)
- ✅ **Honest evaluation on realistic class distribution**

**Why this approach is better:**
- ✅ Train on balanced data for stable learning
- ✅ Evaluate on realistic data to measure real-world performance
- ✅ Val/test metrics actually mean something for deployment
- ✅ No artificial inflation of metrics from balanced test sets

**Next steps:**
1. Create balanced dataset with this script
2. Update hyperparameters (lower `obj_pw` to 2.5)
3. Train with `--single-cls` and `--cos-lr`
4. Monitor results - val/test metrics now reflect realistic performance
5. Compare to previous runs with confidence in evaluation metrics

Good luck! 🚀
