#!/bin/bash
#SBATCH --job-name=boundary_ft
#SBATCH --output=logs/boundary_ft_a01_%j.log
#SBATCH --error=logs/boundary_ft_a01_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate hydro_seg
cd ~/projects/Hydro_Seg_Project
python src/train_boundary_finetune.py
