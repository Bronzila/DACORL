#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J train_single              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-24:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/gieringl-DACORL/MTORL-DAC
source ~/.bashrc
conda activate DACORL

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_DIR=${1:-SGD_data/single}
SEED=${2:-209652396}
AGENT=${3:-td3_bc}

ID=0
BS=256
NUM_TRAIN_ITER=30000
VAL_FREQ=3000
NUM_EVAL_RUNS=10
HIDDEN=256
if [ "$HIDDEN" = 256 ]; then
    CS="reduced_no_arch_dropout_256"
elif [ "$HIDDEN" = 64 ]; then
    CS="reduced_no_arch_dropout"
else
    echo "$HIDDEN is not a valid value for Hidden Layers"
    exitfunc
fi

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    python train.py --data_dir $DATA_DIR/SGD/exponential_decay/$ID --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_EVAL_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb --seed $SEED
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    python train.py --data_dir $DATA_DIR/SGD/step_decay/$ID --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb --seed $SEED
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    python train.py --data_dir $DATA_DIR/SGD/sgdr/$ID --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb --seed $SEED
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Seed: $SEED";
    python train.py --data_dir $DATA_DIR/SGD/constant/$ID --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb --seed $SEED
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";