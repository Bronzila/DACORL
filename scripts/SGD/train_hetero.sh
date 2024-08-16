#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # mlhiwidlc_gpu-rtx2080 # bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J train_hetero              # sets the job name. 
#SBATCH -a 1-11 # array size
#SBATCH -t 0-48:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

set -x

# Random seeds available: "209652396" "398764591" "924231285" "1478610112" "441365315"
SEED=${1:-209652396}

BS=256
NUM_TRAIN_ITER=60000
VAL_FREQ=60000
NUM_EVAL_RUNS=20
DATA_DIR="SGD_data/hetero_20_cpu"
HIDDEN_DIM=256

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined";
    python train.py --data_dir $DATA_DIR/SGD/combined --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_sg";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_sg --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_sg_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_sg_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 5 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_st";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_st --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 6 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_st_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_st_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 7 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_e_st_sg";
    python train.py --data_dir $DATA_DIR/SGD/combined_e_st_sg --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 8 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_sg_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_sg_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 9 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_st_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_st_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 10 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_st_sg";
    python train.py --data_dir $DATA_DIR/SGD/combined_st_sg --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
elif [ 11 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    echo "Agent: combined_st_sg_c";
    python train.py --data_dir $DATA_DIR/SGD/combined_st_sg_c --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $SEED
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
