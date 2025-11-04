#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02-00:00:0
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_validate_%j.out

cd yolov5/ && \
python val.py \
--data ./data/FastMRI_plus_cartilage_defects_meniscus_tears_andVNx4.yaml \
--weights runs/train/yolo5x_cart_def_men_tea_gt_ssim4xvn_detection_mri/weights/best.pt \
--batch-size 1 \
--imgsz 640 \
--iou-thres 0.01 \
