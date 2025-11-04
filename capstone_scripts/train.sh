#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02-00:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_validate_%j.out

# fail on error
set -euo pipefail

# go to repo root
cd /gpfs/home/pb3060/capstone-yolov5

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

# run training (train.py is in the root directory)
python train.py \
    --img 640 \
    --batch 4 \
    --epochs 5 \
    --data /gpfs/home/pb3060/capstone-yolov5/yolo_dataset/dataset.yaml \
    --weights yolov5n.pt \
    --name meniscus_yolov5n_test \
    --hyp ./data/hyps/hyp.scratch-low.yaml