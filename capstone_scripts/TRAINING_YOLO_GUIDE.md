# Complete Usage Guide: Knee Meniscus Detection with YOLOv5

## Overview

This guide covers the complete pipeline from DICOM files with MD.ai annotations to a trained YOLOv5 model.

**✨ NEW: YOLOv5 now natively supports direct DICOM loading!** No PNG conversion required. See `dataloader_modifications.md` for technical details.

---

## Prerequisites

```bash
# Install required packages
pip install pandas numpy opencv-python pydicom tqdm

# YOLOv5 is already in yolov5/ directory with DICOM support
cd yolov5
pip install -r requirements.txt  # Includes pydicom>=2.3.0 for DICOM loading
```

---

## Configuration

All HPC paths are stored in `config/hpc_paths.yaml`. This file contains:
- Meniscus tear data paths (positive samples)
- Normal/no-tear data paths (negative samples)

## Usage Examples

### Process Meniscus Tear Data (Positive Samples)

```bash
python capstone_scripts/batch_convert_annotations.py \
    --use_hpc \
    --data_source meniscus_tear \
    --output yolo_dataset_tears \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Process Normal Data (Negative Samples)

```bash
python capstone_scripts/batch_convert_annotations.py \
    --use_hpc \
    --data_source normal \
    --output yolo_dataset_normal \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Local Development (Without HPC Config)

```bash
python capstone_scripts/batch_convert_annotations.py \
    --dicom_root data/dicoms \
    --csv_dir data/dicoms \
    --output yolo_dataset \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --seed 42
```

## HPC Data Structure

### Meniscus Tear Data
- **Annotations:**
  - `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Annotations/meniscus_specific/TBRecon_anomaly_meniscus_MDai_df.csv`
  - `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Annotations/meniscus_specific/TBRecon_ID_key_meniscus.csv`
- **DICOMs:** `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Dicom_Meniscal_Tear/TBrecon-XX-XX-XXXXX/`

### Normal Data (No Tear)
- **Annotations:**
  - `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Annotations/normal_no_anomalies/TBRecon_anomaly_normal_MDai_df.csv`
  - `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Annotations/normal_no_anomalies/TBRecon_ID_key_normal.csv`
- **DICOMs:** `/gpfs/data/lattanzilab/Ilias/NYU_UCSF_Collab/Dicom_Normal/TBrecon-XX-XX-XXXXX/`

## Output

The script creates a YOLO-format dataset with:
- Binary classification: 0 (no tear) vs 1 (tear)
- Train/val/test splits at volume level
- Files renamed with volume ID suffix (e.g., `IM1-01-02-00011.dcm`)
- YAML config file for YOLOv5 training

---

## Train YOLOv5

### Basic Training

```bash
cd yolov5/

python train.py \
  --data capstone-yolov5/yolo_dataset/dataset.yaml \
  --img 512 \
  --batch 8 \
  --epochs 100 \
  --weights yolov5s.pt \
  --cache ram \
  --name meniscus_detection
```

### Recommended Settings for Medical Imaging

```bash
python train.py \
  --data capstone-yolov5/yolo_dataset/dataset.yaml \
  --img 512 \
  --batch 4 \
  --epochs 300 \
  --weights yolov5x.pt \
  --hyp hyp.scratch-low.yaml \
  --cache disk \
  --patience 50 \
  --save-period 10 \
  --name meniscus_detection_v1 \
  --exist-ok
```

**Parameter explanation:**
- `--img 512`: Image size (matches MRI resolution)
- `--batch 4`: Small batch for high-res images (adjust based on GPU)
- `--epochs 300`: More epochs for medical imaging (fewer samples)
- `--weights yolov5x.pt`: Largest model for best accuracy
- `--hyp hyp.scratch-low.yaml`: Lower augmentation for medical data
- `--cache disk`: Cache preprocessed images for faster training
- `--patience 50`: Early stopping patience
- `--save-period 10`: Save checkpoint every 10 epochs

### Multi-GPU Training

```bash
python -m torch.distributed.run --nproc_per_node 4 train.py \
  --data ../yolo_dataset/dataset.yaml \
  --img 512 \
  --batch 16 \
  --epochs 300 \
  --weights yolov5x.pt \
  --device 0,1,2,3 \
  --name meniscus_detection_multi
```

---

## Step 6: Validate Model

```bash
python val.py \
  --data ../yolo_dataset/dataset.yaml \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --img 512 \
  --batch 4 \
  --task test \
  --save-txt \
  --save-conf
```

---

## Step 7: Run Inference

### On Single DICOM File

```bash
python detect.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --source ../yolo_dataset/images/test/IM120.dcm \
  --img 512 \
  --conf 0.25 \
  --save-txt \
  --save-conf \
  --name test_inference
```

### On Test Set

```bash
python detect.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --source ../yolo_dataset/images/test/ \
  --img 512 \
  --conf 0.25 \
  --save-txt \
  --save-conf \
  --name test_set_inference
```

### On New DICOM Volume

```bash
# No conversion needed - run inference directly on DICOM files!
python detect.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --source ../new_patient/dicoms/ \
  --img 512 \
  --conf 0.25 \
  --save-txt \
  --name new_patient_inference
```

### On PNG Files (if you converted)

```bash
python detect.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --source ../yolo_dataset/images_png/test/IM120.png \
  --img 512 \
  --conf 0.25 \
  --name test_inference_png
```

---

## Step 8: Export Model

### Export to ONNX (for deployment)

```bash
python export.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --img 512 512 \
  --include onnx \
  --simplify
```

### Export to TorchScript (for C++ deployment)

```bash
python export.py \
  --weights runs/train/meniscus_detection/weights/best.pt \
  --img 512 512 \
  --include torchscript
```

---

## Troubleshooting

### Issue: "No labels found"
**Solution:** Check that label files exist and match image names:
```bash
ls yolo_dataset/labels/train/ | head
ls yolo_dataset/images/train/ | head  # For DICOM
# ls yolo_dataset/images_png/train/ | head  # For PNG
```

### Issue: "Out of memory"
**Solutions:**
1. Reduce batch size: `--batch 2` or `--batch 1`
2. Use smaller model: `--weights yolov5s.pt`
3. Reduce image size: `--img 416`

### Issue: "Dataset not found"
**Solution:** Use absolute paths in dataset.yaml:
```yaml
path: /absolute/path/to/yolo_dataset
```

### Issue: "No module named 'pydicom'"
**Solution:** Install pydicom for DICOM support:
```bash
pip install pydicom>=2.3.0
# Or reinstall YOLOv5 requirements
cd yolov5 && pip install -r requirements.txt
```

### Issue: Poor performance / not learning
**Possible causes:**
1. **Class imbalance**: Some classes have very few samples
   - Solution: Use weighted loss or augment minority classes
2. **Insufficient data**: Need more annotated volumes
3. **Wrong hyperparameters**: Try `hyp.scratch-low.yaml`
4. **Images not normalized correctly**: Check window/level settings (DICOM uses 600/1200)

---

## Advanced: Custom Hyperparameters for Medical Imaging

Create `hyp.medical.yaml`:

```yaml
lr0: 0.001  # Lower initial learning rate
lrf: 0.01   # Lower final learning rate
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Reduce augmentation for medical images
hsv_h: 0.0      # No hue shift (grayscale MRI)
hsv_s: 0.0      # No saturation shift
hsv_v: 0.3      # Mild brightness variation
degrees: 5.0    # Mild rotation (anatomical constraints)
translate: 0.05 # Mild translation
scale: 0.3      # Mild scaling
shear: 2.0      # Mild shear
perspective: 0.0  # No perspective (2D slices)
flipud: 0.0     # No vertical flip (anatomy orientation)
fliplr: 0.5     # Horizontal flip OK (left/right knee)
mosaic: 0.5     # Reduce mosaic augmentation
mixup: 0.0      # No mixup for medical

# Class weights (if needed for imbalanced classes)
cls_pw: 1.0
obj_pw: 1.0
```

Train with:
```bash
python train.py --hyp hyp.medical.yaml ...
```

---

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/train
# Open browser to http://localhost:6006
```

### Weights & Biases (wandb)

```bash
pip install wandb
wandb login

python train.py --project meniscus-detection ...
```

---

## Quick Reference

### File Structure
```
project/
├── batch_convert_annotations.py   # Step 2
├── convert_dicom_to_png.py        # Step 3 (optional)
├── yolo_dataset/
│   ├── images/                    # DICOM files (.dcm)
│   ├── labels/                    # YOLO labels (.txt)
│   └── dataset.yaml               # Config
└── yolov5/
    ├── train.py                   # Step 5
    ├── val.py                     # Step 6
    ├── detect.py                  # Step 7
    └── export.py                  # Step 8
```

### Class IDs
```
0: LatAntHorn  - Lateral Anterior Horn
1: LatMenBody  - Lateral Meniscus Body
2: LatPosHorn  - Lateral Posterior Horn
3: MedAntHorn  - Medial Anterior Horn
4: MedMenBody  - Medial Meniscus Body
5: MedPosHorn  - Medial Posterior Horn
```

### Common Commands

```bash
# Full pipeline (DICOM - no conversion needed!)
python batch_convert_annotations.py --dicom_root dicoms --csv_dir dicoms --output yolo_dataset
cd yolov5 && python train.py --data ../yolo_dataset/dataset.yaml --img 512 --batch 4 --epochs 100 --weights yolov5x.pt

# Full pipeline (with PNG conversion - optional)
python batch_convert_annotations.py --dicom_root dicoms --csv_dir dicoms --output yolo_dataset
python convert_dicom_to_png.py --input yolo_dataset/images --output yolo_dataset/images_png
# Update dataset.yaml to use images_png/ instead of images/
cd yolov5 && python train.py --data ../yolo_dataset/dataset.yaml --img 512 --batch 4 --epochs 100 --weights yolov5x.pt

# Quick test (DICOM)
python detect.py --weights runs/train/exp/weights/best.pt --source ../yolo_dataset/images/test/ --img 512

# Validate
python val.py --data ../yolo_dataset/dataset.yaml --weights runs/train/exp/weights/best.pt --img 512
```

---

## Next Steps

1. ✅ Run batch conversion on all volumes
2. ✅ Train initial model directly on DICOM (100 epochs)
3. 📊 Analyze results, adjust hyperparameters
4. 🔄 Retrain with optimized settings
5. 🧪 Validate on test set
6. 🚀 Deploy for inference

**Optional:** Convert to PNG if needed for compatibility with other tools

---

## Support

For issues:
1. Check conversion_report.json for data statistics
2. Verify dataset structure matches expected format
3. Review YOLOv5 logs in runs/train/
4. Check DICOM windowing if images look wrong
