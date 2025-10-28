# Usage Guide

## Configuration

Edit `configs.py` in the root directory:

```python
# Set to True when running on HPC, False for local development
USE_HPC = False  # Change to True for HPC
```

All paths are configured in `configs.py`. The script automatically uses the correct paths based on the `USE_HPC` flag.

### Using Configs in Your Code

The paths from `configs.py` are automatically imported in `utils/dataloaders.py`:

```python
from configs import DATA_DIR, DICOM_NORMAL, DICOM_MENISCAL

# Or use the helper function
from utils.dataloaders import get_dicom_data_path

meniscal_path = get_dicom_data_path('meniscus_tear')
normal_path = get_dicom_data_path('normal')
```

## Running Locally

```bash
python capstone_scripts/batch_convert_annotations.py \
    --data_source meniscus_tear \
    --output yolo_dataset \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --seed 42
```

## Running on HPC

1. **Edit configs.py:**
   ```python
   USE_HPC = True  # Set to True
   ```

2. **Process meniscus tear data:**
   ```bash
   python capstone_scripts/batch_convert_annotations.py \
       --data_source meniscus_tear \
       --output yolo_dataset_tears \
       --train_ratio 0.6 \
       --val_ratio 0.2 \
       --test_ratio 0.2 \
       --seed 42
   ```

3. **Process normal data:**
   ```bash
   python capstone_scripts/batch_convert_annotations.py \
       --data_source normal \
       --output yolo_dataset_normal \
       --train_ratio 0.6 \
       --val_ratio 0.2 \
       --test_ratio 0.2 \
       --seed 42
   ```

## SLURM Job Example

```bash
#!/bin/bash
#SBATCH --job-name=yolo_prep
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load python/3.9

cd $HOME/capstone-yolov5

# Make sure USE_HPC=True in configs.py before running

# Process tear data
python capstone_scripts/batch_convert_annotations.py \
    --data_source meniscus_tear \
    --output yolo_dataset_tears \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --seed 42

# Process normal data
python capstone_scripts/batch_convert_annotations.py \
    --data_source normal \
    --output yolo_dataset_normal \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2 \
    --seed 42
```

## Arguments

- `--data_source`: Choose `meniscus_tear` or `normal`
- `--output`: Output directory for YOLO dataset
- `--train_ratio`: Proportion for training (default: 0.7)
- `--val_ratio`: Proportion for validation (default: 0.15)
- `--test_ratio`: Proportion for testing (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)
- `--no_empty_labels`: Don't create empty label files for unannotated slices

## Output

The script creates:
- `yolo_dataset/images/{train,val,test}/` - DICOM files with volume suffix
- `yolo_dataset/labels/{train,val,test}/` - YOLO format labels
- `yolo_dataset/dataset.yaml` - YOLOv5 config file
- `yolo_dataset/conversion_report.json` - Detailed statistics
