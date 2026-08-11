#!/bin/bash
#SBATCH --job-name=YOLO-GunRecog-V0
#SBATCH --partition=kimq
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=YOLO-GunRecog-%j.out
#SBATCH --error=YOLO-GunRecog-%j.err

source ~/miniconda3/bin/activate
conda activate mujoco-env

# One BLAS/OpenMP thread per process. Each SubprocVecEnv worker runs torch,
# which otherwise spawns threads for every core on the node and the 8 workers
# thrash the 8-CPU allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ~/YOLO-GunRecognizer-v0

echo "Running on $(hostname)"
echo "current directory: $(pwd)"
python --version

nvidia-smi

python train.py

WEBHOOK_URL=$(cat ~/.discord_webhook)

curl -sS \
    -H "Content-Type: application/json" \
    -d "{\"content\":\"Job ${SLURM_JOB_ID} finished successfully\"}" \
    "$WEBHOOK_URL"