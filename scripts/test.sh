#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --partition=gpua100
#SBATCH --output=/scratch/7982399/thesis/logs/slurm_test_output.log
#SBATCH --error=/scratch/7982399/thesis/logs/slurm_test_error.log

echo "Test script running" > /scratch/7982399/thesis/logs/test.log
