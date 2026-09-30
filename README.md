# Particle World Model for Dexterous Manipulation

This repository implements a particle-based world model for dexterous manipulation based on the paper: *Scaling Cross-Embodiment World Models for Dexterous Manipulation*. It predicts object particle dynamics given hand particle movements, allowing for cross-embodiment generalization.

## Workspace Structure
- `src/model/`: Contains the DPI-Net (Dynamic Particle Interaction Network) implementation using PyTorch Geometric.
- `src/data/`: Data loading and point cloud processing (FPS downsampling, RGB-D to PC).
- `src/train.py`: Main training loop.
- `cluster/`: Scripts for running jobs on the Georgia Tech PACE ICE cluster.

## Setup Instructions (Local / Dev)
1. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Note: PyTorch Geometric and PyTorch3D may require specific installation commands based on your CUDA version).*

## PACE ICE Cluster Workflow

Because model training on 3D particle graphs is computationally intensive, we use the PACE ICE cluster.

### 1. Syncing Data to PACE
Use the `cluster/sync_data.sh` script to push your local dataset to the cluster, and pull trained checkpoints back.
```bash
# Push data to cluster
bash cluster/sync_data.sh push

# Pull models/results back from cluster
bash cluster/sync_data.sh pull
```

### 2. Running a Training Job
Submit the SLURM job to the ICE cluster queue:
```bash
sbatch cluster/train_job.slurm
```
You can monitor your job using `squeue -u $USER`. Logs will be written to `cluster/logs/`.
