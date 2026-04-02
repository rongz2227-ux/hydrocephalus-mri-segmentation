#!/bin/bash
#SBATCH --job-name=hydro_swin
#SBATCH --output=swin_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# 激活环境
source activate hydro_seg 

python src/train_large.py