#!/bin/bash
#SBATCH --job-name=feature_maps
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/feature_maps_%j.out
#SBATCH --error=slurm_out/feature_maps_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any modules you need (adjust based on your cluster)
module load python/gpu/3.10.6-cuda12.9

# Go to repo root
cd /gpfs/home/rrr9340/capstone-yolov5

# Activate your conda/venv if needed
source venv/bin/activate

# Run the feature map checking script
python capstone_scripts/feature_maps/check_best_feature_maps.py \
    --image /gpfs/home/rrr9340/capstone-yolov5/yolo_dataset/images/val/IM65-01-02-00483.dcm \
    --yolov5-path /gpfs/home/rrr9340/capstone-yolov5 \
    --weights /gpfs/home/rrr9340/capstone-yolov5/runs/train/exp7/weights/best.pt \
    --brightness 1.5 \
    --conf-threshold 0.1 \
    --target-layer 21 \
    --output-dir capstone_scripts/feature_maps/outputs

echo "End time: $(date)"
echo "Done!"