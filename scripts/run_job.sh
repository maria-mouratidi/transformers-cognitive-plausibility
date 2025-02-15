#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpua16

source /scratch/7982399/conda/bin/activate thesis_env

# Run Python script with timestamped logs
python -u /scratch/7982399/thesis/scripts/probing/raw.py > "/scratch/7982399/thesis/logs/output_$(date +"%Y-%m-%d_%H-%M-%S").log" 2> "/scratch/7982399/thesis/logs/error_$(date +"%Y-%m-%d_%H-%M-%S").log"