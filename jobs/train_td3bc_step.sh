#!/bin/bash
#SBATCH --job-name=TD3BC-step
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=DACORL-TRAIN-TD3BC-step-%j.out
#SBATCH --error=DACORL-TRAIN-TD3BC-step-%j.err

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on $SLURM_NODELIST"

# Change to project root
cd /home/hasana/cloud_dacorl || exit 1
echo "Changed to project root: $(pwd)"

# Activate conda environment
source /home/hasana/miniconda3/bin/activate DACORL
echo "Conda environment activated. Running DACORL TD3+BC Training (Step Decay)..."

# Run training with TD3+BC on step_decay teacher data
python main.py \
  env=LayerwiseNanoGPT/Shakespeare_cluster \
  teacher=step_decay \
  agent_type=td3_bc \
  combination=single \
  mode=train \
  results_dir=data/nanoGPT_shakespeare/step_decay \
  num_train_iter=60000 \
  val_freq=70000 \
  seed=0 \
  data_exists=true \
  device=cuda

echo "=========================================="
echo "Job execution complete at $(date)"
echo "Trained agent saved in: trained_agents/td3bc/step_decay/"
echo "=========================================="
