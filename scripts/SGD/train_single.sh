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

# Random seeds available: "209652396" "398764591" "924231285" "1478610112" "441365315"
SEED=${1:-209652396}

ID=0
BS=256
NUM_TRAIN_ITER=30000
VAL_FREQ=30000
NUM_EVAL_RUNS=20
DATA_DIR="SGD_data/homo_20_cpu"
HIDDEN_DIM=256

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: exponential_decay";
    python train.py --data_dir $DATA_DIR/SGD/exponential_decay/$ID --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: step_decay";
    python train.py --data_dir $DATA_DIR/SGD/step_decay/$ID --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: sgdr";
    python train.py --data_dir $DATA_DIR/SGD/sgdr/$ID --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: constant";
    python train.py --data_dir $DATA_DIR/SGD/constant/$ID --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
