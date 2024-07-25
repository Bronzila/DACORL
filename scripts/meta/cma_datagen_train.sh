#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-05:00:00
#SBATCH -o logs/%A[%a].%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J DAC4CMA-ES # sets the job name. 
#SBATCH -a 1-2 # array size
#SBATCH --mem 16GB

RESULTS_DIR=${1:-data}
AGENT=${2:-td3_bc}
HIDDEN=${3:-256}
ID=0
NUM_TRAIN_ITER=30000
VAL_FREQ=5000
NUM_RUNS=1000

if [ "$HIDDEN" = 256 ]; then
    CS="reduced_no_arch_dropout_256"
elif [ "$HIDDEN" = 64 ]; then
    CS="reduced_no_arch_dropout"
else
    echo "$HIDDEN is not a valid value for Hidden Layers"
    exitfunc
fi


start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env default --benchmark CMAES --agent csa --id $ID --num_runs $NUM_RUNS  --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/CMAES/csa/$ID/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python -W ignore data_gen.py --results_dir $RESULTS_DIR --env default --benchmark CMAES --agent cmaes_constant --id $ID --num_runs $NUM_RUNS  --save_run_data --save_rep_buffer
    python -W ignore train.py --data_dir $RESULTS_DIR/CMAES/cmaes_constant/$ID/ --agent_type $AGENT --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --cs_type $CS --no-wandb
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
