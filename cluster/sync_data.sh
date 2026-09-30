#!/bin/bash
# Script to sync data and models between local Windows machine and PACE ICE cluster

# PACE credentials
USERNAME="sgirase3"
PACE_HOST="login-ice.pace.gatech.edu"
REMOTE_DIR="/storage/home/hcoda1/6/$USERNAME/p-your_pi_account-0/particle-world-model"
LOCAL_DIR=".."

ACTION=$1

if [ -z "$ACTION" ]; then
    echo "Usage: ./sync_data.sh [push|pull]"
    echo "  push: Syncs local dataset/code TO PACE ICE"
    echo "  pull: Syncs trained models/logs FROM PACE ICE"
    exit 1
fi

if [ "$ACTION" == "push" ]; then
    echo "Pushing code and data to PACE ICE..."
    # Create remote directory if it doesn't exist
    ssh ${USERNAME}@${PACE_HOST} "mkdir -p ${REMOTE_DIR}"
    
    # Sync code and data (excluding heavy environments or local logs)
    rsync -avz --progress --exclude '.git' --exclude '__pycache__' \
        --exclude 'models/' --exclude 'cluster/logs/' \
        ${LOCAL_DIR}/ ${USERNAME}@${PACE_HOST}:${REMOTE_DIR}/
    
    echo "Push complete."
    
elif [ "$ACTION" == "pull" ]; then
    echo "Pulling models and logs from PACE ICE..."
    
    # Sync models and logs back to local machine
    rsync -avz --progress \
        ${USERNAME}@${PACE_HOST}:${REMOTE_DIR}/models/ ${LOCAL_DIR}/models/
    rsync -avz --progress \
        ${USERNAME}@${PACE_HOST}:${REMOTE_DIR}/cluster/logs/ ${LOCAL_DIR}/cluster/logs/
        
    echo "Pull complete."
else
    echo "Invalid action. Use 'push' or 'pull'."
fi
