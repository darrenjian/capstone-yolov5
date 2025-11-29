# Creating Balanced Dataset Instructions

## Overview

This guide walks you through creating a balanced dataset from your existing **combined YOLO dataset** (created with `batch_convert_annotations.py --append`) where:
- **18,273 slices WITH tears** (all available)
- **18,273 slices WITHOUT tears** (randomly sampled from 76,167 available)
- **Total: 36,546 slices** with perfect 50/50 class balance
- Re-split into train/val/test (70/15/15) at the **slice level** (not volume level)

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

## Why Re-balance?

### Current Problem (90/10 Imbalance):
```
Total dataset: 94,440 slices
- Tears: 18,273 (19.4%)
- No-tears: 76,167 (80.6%)
→ Severe class imbalance
→ Model biased to predict "no object"
→ Requires aggressive weighting (obj_pw=5.0)
→ Still only mAP@0.1 = 0.698
```

### After Balancing (50/50):
```
Total dataset: 36,546 slices
- Tears: 18,273 (50%)
- No-tears: 18,273 (50%)
→ Perfect class balance
→ More stable training
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

### Step 2: Create Balanced Dataset

The script will:
1. **Pool all slices** from train/val/test (ignoring original volume-level splits)
2. **Identify** tear vs no-tear by checking if label files are empty
3. **Sample** 18,273 random no-tear slices from the 76,167 available
4. **Combine** with all 18,273 tear slices
5. **Re-split** 70/15/15 at **slice level** (not volume level)
6. **Copy** to new balanced dataset directory

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
Pooling slices from all existing splits (train/val/test)...
  train: 12,791 tear, 53,317 no-tear
  val  :  2,741 tear, 11,425 no-tear
  test :  2,741 tear, 11,425 no-tear

✓ Pooled totals:
  Tear slices: 18,273
  No-tear slices: 76,167

============================================================
STEP 2: Balancing and splitting dataset
============================================================

BALANCING DATASET
============================================================
Original class distribution (pooled from all splits):
  Tear slices: 18,273
  No-tear slices: 76,167
  Imbalance ratio: 4.2:1 (no-tear:tear)

✓ Randomly sampling 18,273 no-tear slices from 76,167 available

Balanced dataset size: 36,546 slices (50% tear, 50% no-tear)

============================================================
NEW SPLITS (at slice level, not volume level)
============================================================
Split ratios: 70% / 15% / 15%

  train: 25,582 slices (12,791 tear [50.0%], 12,791 no-tear)
  val  :  5,482 slices ( 2,741 tear [50.0%],  2,741 no-tear)
  test :  5,482 slices ( 2,741 tear [50.0%],  2,741 no-tear)

============================================================
STEP 3: Copying files to output directory
============================================================
[Progress bars...]

============================================================
✓ BALANCED DATASET CREATED SUCCESSFULLY!
============================================================

Dataset location: /path/to/balanced_yolo_dataset

Final statistics:
  Total images: 36,546
  Train:     25,582 ( 12,791 tear,  12,791 no-tear)
  Val  :      5,482 (  2,741 tear,   2,741 no-tear)
  Test :      5,482 (  2,741 tear,   2,741 no-tear)

  Class balance: 18,273 tear : 18,273 no-tear (perfect 1:1)

Reduction from original:
  Before: 18,273 tear + 76,167 no-tear = 94,440 total
  After:  18,273 tear + 18,273 no-tear = 36,546 total
  Removed: 57,894 excess no-tear slices
```

---

### Step 4: Verify the Balanced Dataset

Check the output directory structure:
```bash
balanced_yolo_dataset/
├── images/
│   ├── train/          # 25,582 images (50% tear, 50% no-tear)
│   ├── val/            # 5,482 images (50% tear, 50% no-tear)
│   └── test/           # 5,482 images (50% tear, 50% no-tear)
├── labels/
│   ├── train/          # 25,582 labels
│   ├── val/            # 5,482 labels
│   └── test/           # 5,482 labels
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
    "val": {"total": 5482, "tear": 2741, "no_tear": 2741},
    "test": {"total": 5482, "tear": 2741, "no_tear": 2741}
  },
  "total_images": 36546,
  "total_tear": 18273,
  "total_no_tear": 18273,
  "balance_ratio": "1:1"
}
```

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

### Before (90/10 Imbalance):
```yaml
Dataset: 94,440 slices (19% tear, 81% no-tear)
Settings:
  obj_pw: 5.0
  --image-weights
  --slice-range
  lr0: 0.002

Results:
  mAP@0.1: 0.698
  Training: Unstable, model biased to "no object"
  Overfitting: Severe with higher LR
```

### After (50/50 Balance):
```yaml
Dataset: 36,546 slices (50% tear, 50% no-tear)
Settings:
  obj_pw: 2.5
  --single-cls
  --cos-lr
  lr0: 0.005

Expected Results:
  mAP@0.1: 0.75-0.85+ ✨
  Training: More stable, better convergence
  Overfitting: Reduced (with proper augmentation)
  Train/Val gap: Smaller
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
1. Created combined dataset with 90/10 imbalance
2. Used `--slice-range` during training to filter at runtime
3. Still had imbalance in actual batches
4. Required `obj_pw=5.0` to compensate

### New (with balanced dataset):
1. ✅ Pre-filter and balance dataset once
2. ✅ Training uses perfectly balanced data
3. ✅ No runtime filtering needed
4. ✅ Lower `obj_pw=2.5` sufficient
5. ✅ Better results, more stable training

---

## Summary

**What this achieves:**
- ✅ Perfect 50/50 class balance (18,273:18,273)
- ✅ Removes 57,894 excess no-tear slices
- ✅ Re-splits at slice level (not volume level)
- ✅ Reproducible with seed parameter
- ✅ Preserves all tear data (nothing discarded)

**Training improvements:**
- ✅ More stable convergence
- ✅ Less overfitting
- ✅ Better generalization
- ✅ Expected mAP@0.1: **0.75-0.85+** (vs 0.698 before)
- ✅ Smaller train/val gap
- ✅ Can use higher learning rates (0.005 vs 0.002)

**Next steps:**
1. Create balanced dataset
2. Update hyperparameters (lower `obj_pw`)
3. Train with `--single-cls` and `--cos-lr`
4. Monitor results and compare to previous runs
5. Celebrate better mAP@0.1! 🎉

Good luck! 🚀
