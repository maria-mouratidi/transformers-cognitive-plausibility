# MsC Artificial Intelligence

Thesis directory:
cd /scratch/7982399/thesis

Conda environment:
source /scratch/7982399/conda/bin/activate thesis_env

Run scripts
- On two separate terminals:
    - sbatch scripts/run_job_cs02.sh
    - tail -n +1 output.out -f & tail -n +1 error.out -f
- Or locally:
    - python/scripts/run_job.sh

Check free memory on server:
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

To add speed in slurm job add:
#SBATCH --gres=gpu:1
