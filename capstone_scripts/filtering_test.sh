#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --job-name=test_slice_filter
#SBATCH --output=logs/test_slice_filter_%j.out
#SBATCH --error=logs/test_slice_filter_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load modules (adjust based on your HPC setup)
# module load anaconda3
# module load cuda/11.8

# Activate your environment if needed
# source activate yolov5

# Navigate to project directory
cd /gpfs/home/ic2664/capstone-yolov5

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Testing Slice Filtering - 1 Epoch"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run 1 epoch training with slice filtering
python train.py \
    --data /gpfs/home/ic2664/capstone-yolov5/yolo_dataset/dataset.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 1 \
    --slice-range slice_range.json \
    --project test_output/training \
    --name slice_filter_test_${SLURM_JOB_ID} \
    --device 0 \
    --workers 4

echo ""
echo "=========================================="
echo "Test Complete!"
echo "End time: $(date)"
echo "Results saved to: test_output/training/slice_filter_test_${SLURM_JOB_ID}"
echo "=========================================="