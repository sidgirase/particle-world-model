#!/bin/bash
#SBATCH --job-name=particle_world_model
#SBATCH --account=gts-your_pi_account  # Replace with actual PI account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1                   # Request 1 GPU (e.g., A100 or V100)
#SBATCH --time=12:00:00                # 12 hours time limit
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Starting training job on PACE ICE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Load modules (Adjust based on PACE ICE specific module names)
module load anaconda3
module load cuda/11.8

# Activate conda environment
# conda activate particle_env

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Run training
python src/train.py --epochs 100 --batch_size 32 --hidden_dim 128

echo "Training job completed."
