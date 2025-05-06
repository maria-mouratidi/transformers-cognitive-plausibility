#!/bin/bash
#SBATCH --job-name=probing
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=30
#SBATCH --partition=gpua100
#SBATCH --output=/dev/null       # Disables SLURM standard output file
#SBATCH --error=/dev/null        # Disables SLURM error output file
source /scratch/7982399/conda/bin/activate thesis_env

# Create month directory and subdirectories for output and error logs
MONTH_DIR="/scratch/7982399/thesis/logs/$(date +"%Y-%b")"
OUTPUT_DIR="$MONTH_DIR/output"
ERROR_DIR="$MONTH_DIR/error"
mkdir -p "$OUTPUT_DIR" "$ERROR_DIR"

# Run Python script with logs in categorized directories
OUTPUT_LOG="$OUTPUT_DIR/output_$(date +"%d-%H:%M:%S").log"
ERROR_LOG="$ERROR_DIR/error_$(date +"%d-%H:%M:%S").log"

cd /scratch/7982399/thesis
export PYTHONPATH=$(pwd)
python -m "scripts.probing.attn_flow" > "$OUTPUT_LOG" 2> "$ERROR_LOG"

# Remove log files if they are empty
[ ! -s "$OUTPUT_LOG" ] && rm "$OUTPUT_LOG"
[ ! -s "$ERROR_LOG" ] && rm "$ERROR_LOG"