#!/bin/bash
#SBATCH --job-name=EVAL-TD3BC-exponential
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=DACORL-EVAL-TD3BC-exponential-%j.out
#SBATCH --error=DACORL-EVAL-TD3BC-exponential-%j.err

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on $SLURM_NODELIST"

# Change to project root
cd /home/hasana/cloud_dacorl || exit 1
echo "Changed to project root: $(pwd)"

# Activate conda environment
source /home/hasana/miniconda3/bin/activate DACORL
echo "Conda environment activated. Running DACORL TD3+BC Evaluation (Exponential Decay)..."

# Evaluate trained TD3+BC agent on exponential_decay teacher dataset
python main.py \
  env=LayerwiseNanoGPT/Shakespeare_cluster \
  teacher=exponential_decay \
  agent_type=td3_bc \
  combination=single \
  mode=eval \
  results_dir=data/nanoGPT_shakespeare/exponential_decay \
  num_train_iter=60000 \
  seed=0 \
  eval_seed=42 \
  device=cuda

echo "=========================================="
echo "Job execution complete at $(date)"
echo "Evaluation results saved in: data/nanoGPT_shakespeare/exponential_decay/LayerwiseNanoGPT/exponential_decay/0/results/td3_bc/"
echo "=========================================="
