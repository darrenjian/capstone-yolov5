#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02-00:00:0
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm2_%j.out

cd yolov5/ && \
python3 train.py \
--img 640 \
--batch 4 \
--epochs 100 \
--data ./data/FastMRI_plus_cartilage_defects_meniscus_tears_andVNx4.yaml \
--cfg ./models/yolov5x.yaml \
--weights yolov5x.pt \
--name yolo5x_cart_def_men_tea_gt_ssim4xvn_detection_mri \
--cache \
--hyp hyp.scratch-low-custom.yaml
