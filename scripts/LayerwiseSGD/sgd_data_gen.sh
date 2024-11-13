#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # mlhiwidlc_gpu-rtx2080 # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J SGD_data              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-48:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ID=${1:-default}
RESULTS_DIR="LayerwiseSGD_data/FMNIST"
SEED=0
DATASET=FashionMNIST
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Teacher: exponential_decay";
    python main.py env=LayerwiseSGD/$DATASET teacher=exponential_decay id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=data_gen
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: step_decay";
    python main.py env=LayerwiseSGD/$DATASET teacher=step_decay id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=data_gen
elif [ 3 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: sgdr";
    python main.py env=LayerwiseSGD/$DATASET teacher=sgdr id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=data_gen
elif [ 4 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: constant";
    python main.py env=LayerwiseSGD/$DATASET teacher=constant id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=data_gen
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
