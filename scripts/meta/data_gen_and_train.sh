#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-05:00:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data_gen_and_train_single_agent # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH --mem 8GB

TEACHER=${1:-exponential_decay}
AGENT=${2:-td3_bc}
ID=${3:-0}
NUM_TRAIN_ITER=15000
VAL_FREQ=2000
NUM_RUNS=100
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
STATE_VERSION=extended_velocity
RESULTS_DIR="data/test_agents"

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env $FC1\_$STATE_VERSION --agent $TEACHER --num_runs $NUM_RUNS  --id $ID --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC1/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ 
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env $FC2\_$STATE_VERSION --agent $TEACHER --num_runs $NUM_RUNS  --id $ID --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC2/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ 
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env $FC3\_$STATE_VERSION --agent $TEACHER --num_runs $NUM_RUNS  --id $ID --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC3/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ 
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env $FC4\_$STATE_VERSION --agent $TEACHER --num_runs $NUM_RUNS  --id $ID --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC4/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ 
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
