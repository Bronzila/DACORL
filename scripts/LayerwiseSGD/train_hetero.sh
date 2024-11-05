#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # mlhiwidlc_gpu-rtx2080 # bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J train_hetero              # sets the job name. 
#SBATCH -a 1-11 # array size
#SBATCH -t 0-35:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set -x

SEED=${1:-0}
AGENT=${2:-td3_bc}

NUM_TRAIN_ITER=60000
RESULTS_DIR="LayerwiseSGD_data/wandb"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-ST-SG-C";
    python main.py env=LayerwiseSGD/MNIST teacher=E-ST-SG-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-C";
    python main.py env=LayerwiseSGD/MNIST teacher=E-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-SG";
    python main.py env=LayerwiseSGD/MNIST teacher=E-SG id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-SG-C";
    python main.py env=LayerwiseSGD/MNIST teacher=E-SG-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 5 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-ST";
    python main.py env=LayerwiseSGD/MNIST teacher=E-ST id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 6 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-ST-C";
    python main.py env=LayerwiseSGD/MNIST teacher=E-ST-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 7 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: E-ST-SG";
    python main.py env=LayerwiseSGD/MNIST teacher=E-ST-SG id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 8 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: SG-C";
    python main.py env=LayerwiseSGD/MNIST teacher=SG-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 9 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: ST-C";
    python main.py env=LayerwiseSGD/MNIST teacher=ST-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 10 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: ST-SG";
    python main.py env=LayerwiseSGD/MNIST teacher=ST-SG id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
elif [ 11 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Teacher: ST-SG-C";
    python main.py env=LayerwiseSGD/MNIST teacher=ST-SG-C id=$ID results_dir=$RESULTS_DIR seed=$SEED mode=all data_exists=true agent_type=$AGENT num_train_iter=$NUM_TRAIN_ITER combination=heterogeneous wandb_group="hetero"
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
