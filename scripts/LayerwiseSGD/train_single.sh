#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # mlhiwidlc_gpu-rtx2080 # bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J train_single              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-34:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set -x

SEED=${1:-0}
AGENT=${2:-td3_bc}

ID=0
NUM_TRAIN_ITER=30000
RESULTS_DIR="LayerwiseSGD_data/wandb"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: exponential_decay";
    python main.py env=LayerwiseSGD/MNIST teacher=exponential_decay id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=train agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER wandb_group="single"
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: step_decay";
    python main.py env=LayerwiseSGD/MNIST teacher=step_decay id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=train agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER wandb_group="single"
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: sgdr";
    python main.py env=LayerwiseSGD/MNIST teacher=sgdr id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=train agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER wandb_group="single"
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: constant";
    python main.py env=LayerwiseSGD/MNIST teacher=constant id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=train agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER wandb_group="single"
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
