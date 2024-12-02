#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J SGD_data              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-24:00:00
#SBATCH --mem 32GB

cd /work/dlclarge1/gieringl-DACORL_normal/DACORL
source ~/.bashrc
conda activate DACORL_normal

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ID=${1:-default}
DATASET=CIFAR10
RESULTS_DIR=LayerwiseSGD_data/$DATASET
SEED=0
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Teacher: exponential_decay";
    python teacher_hpo_SGD.py env=LayerwiseSGD/$DATASET teacher=exponential_decay results_dir=$RESULTS_DIR seed=$SEED
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: step_decay";
    python teacher_hpo_SGD.py env=LayerwiseSGD/$DATASET teacher=step_decay results_dir=$RESULTS_DIR seed=$SEED
elif [ 3 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: sgdr";
    python teacher_hpo_SGD.py env=LayerwiseSGD/$DATASET teacher=sgdr results_dir=$RESULTS_DIR seed=$SEED
elif [ 4 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Teacher: constant";
    python teacher_hpo_SGD.py env=LayerwiseSGD/$DATASET teacher=constant results_dir=$RESULTS_DIR seed=$SEED
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
