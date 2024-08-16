#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J train_single              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-20:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=0
BS=256
NUM_TRAIN_ITER=100000
VAL_FREQ=10000
NUM_RUNS=1000
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
DATA_DIR="data_reward_test"
SEEDS="209652396 398764591 924231285 1478610112 441365315 1537364731 192771779 1491434855 1819583497 530702035"
HIDDEN_DIM=256

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    for seed in $SEEDS
    do
        echo "Seed: $seed";
        python train.py --data_dir $DATA_DIR/ToySGD/$AGENT/$ID/$FC1 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $seed
    done
elif [ 2 -eq $SLURM_ARRAY_TASK_ID ]
then
    for seed in $SEEDS
    do
        echo "Seed: $seed";
        python train.py --data_dir $DATA_DIR/ToySGD/$AGENT/$ID/$FC2 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $seed
    done
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    for seed in $SEEDS
    do
        echo "Seed: $seed";
        python train.py --data_dir $DATA_DIR/ToySGD/$AGENT/$ID/$FC3 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $seed
    done
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    for seed in $SEEDS
    do
        echo "Seed: $seed";
        python train.py --data_dir $DATA_DIR/ToySGD/$AGENT/$ID/$FC4 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS --debug --hidden_dim $HIDDEN_DIM --seed $seed
    done
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
