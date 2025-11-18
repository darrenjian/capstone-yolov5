#!/bin/bash
#SBATCH --partition=gpu8_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --output=n_cls_05_meniscus.out

# fail on error
set -euo pipefail

# go to repo root
cd /gpfs/home/dj2565/capstone-yolov5

# load python module used interactively
module load python/gpu/3.10.6-cuda12.9

# create or activate virtualenv
if [ -d "venv" ]; then
    echo "Activating existing virtual environment..."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Multi-GPU training with DDP (Distributed Data Parallel)
# Using 8 GPUs, batch can be increased proportionally
python -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_port 29500 \
    train.py \
    --img 640 \
    --batch 32 \
    --epochs 100 \
    --data /gpfs/home/dj2565/capstone-yolov5/yolo_dataset/dataset.yaml \
    --weights yolov5n.pt \
    --device 0,1,2,3,4,5,6,7 \
    --name n_cls_05_meniscus \
    --hyp ./data/hyps/hyp.scratch-low-custom.yaml \
    --patience 50 \
    --save-period 10