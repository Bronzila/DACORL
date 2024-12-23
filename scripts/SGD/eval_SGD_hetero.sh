#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Eval_Agents              # sets the job name. 
#SBATCH -a 1-11 # array size
#SBATCH -t 0-30:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

RESULTS_DIR=${1:-data}
SEED=${2:-209652396}
TRAIN_ITER=${3:-60000}
# "209652396" "398764591" "924231285" "1478610112" "441365315"
NUM_RUNS=20
GEN_SEED=123
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running combined_e_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_e_sg of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_sg/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_e_sg_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_sg_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 5 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running combined_e_st of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_st/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 6 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_e_st_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_st_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 7 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_e_st_sg of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_e_st_sg/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 8 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running combined_sg_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_sg_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 9 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_st_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_st_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 10 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_st_sg of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_st_sg/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
elif [ 11 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running combined_st_sg_c of seed $SEED on generating seed $GEN_SEED";
    python3.10 eval_sgd.py --data_dir $RESULTS_DIR/SGD/combined_st_sg_c/ --training_seed $SEED --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
