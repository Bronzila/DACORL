#!/bin/bash

#SBATCH --partition mldlc2_gpu-l40s
#SBATCH --job-name DACORL-HPO-exponential
#SBATCH --time=1-00:00:00
#SBATCH --output %x-%A.out
#SBATCH --error %x-%A.err
#SBATCH --mem 32GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on $SLURM_JOB_PARTITION"

# Change to project root directory FIRST
cd /home/hasana/cloud_dacorl
echo "Changed to project root: $PWD"


source ~/miniconda3/bin/activate
conda activate DACORL


echo "Conda environment activated. Running DACORL Teacher HPO (Exponential Decay)..."
python -u compare_teachers_agents.py

echo "=========================================="
echo "Job execution complete at $(date)"
echo "Results saved in: data/nanoGPT/exponential_decay/"
echo "=========================================="
