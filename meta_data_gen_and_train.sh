#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #gki_cpu-caskadelake # relea_gpu-rtx2080 mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-12:00:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data_gen_and_train_single_agent # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -mem 8GB

AGENT=${1:-exponential_decay}
ID=${2:-0}
BS=256
NUM_TRAIN_ITER=10000
VAL_FREQ=2000
NUM_RUNS=100
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
VERSION=default
RESULTS_DIR="data_256_no_term"

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
