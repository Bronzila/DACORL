#!/bin/bash

#SBATCH --partition mldlc2_gpu-l40s
#SBATCH --job-name DACORL-HPO-sgdr
#SBATCH --time=1-00:00:00
#SBATCH --output %x-%A.out
#SBATCH --error %x-%A.err
#SBATCH --mem 64GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on $SLURM_JOB_PARTITION"

# Change to project root directory FIRST
cd /home/hasana/cloud_dacorl
echo "Changed to project root: $PWD"

# Create log directories (now relative to project root)
mkdir -p jobs/logs/sgdr_hpo jobs/data/nanoGPT/sgdr

source ~/miniconda3/bin/activate
conda activate DACORL

# Thread settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export PYTHONUNBUFFERED=1


echo "Conda environment activated. Running DACORL Teacher HPO (SGDR)..."
python -u teacher_hpo_SGD.py \
  --config-name=nanoGPT \
  env=LayerwiseNanoGPT/Shakespeare_cluster \
  teacher=sgdr \
  results_dir=./jobs/data/nanoGPT \
  seed=42 \
  2>&1 | tee -a "jobs/logs/sgdr_hpo/run-$(date +%Y%m%d-%H%M%S).log"

echo "=========================================="
echo "Job execution complete at $(date)"
echo "Results saved in: data/nanoGPT/sgdr/"
echo "=========================================="
