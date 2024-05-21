#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-30:00:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J HPO # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH --mem 8GB

TEACHER=${1:-exponential_decay}
RESULTS_DIR=${2:-data/data_hpo_10}
RL_AGENT=${3:-td3_bc}
ID=${4:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=100
VERSION=extended_velocity

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL

echo "Running in directory ($RESULTS_DIR) using the data teacher ($TEACHER) on ($ID) and training ($RL_AGENT)"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC1 --agent_type $RL_AGENT --output_path $RESULTS_DIR/smac/smac_$TEACHER\_$FC1\_$RL_AGENT --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC2 --agent_type $RL_AGENT --output_path $RESULTS_DIR/smac/smac_$TEACHER\_$FC2\_$RL_AGENT --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC3 --agent_type $RL_AGENT --output_path $RESULTS_DIR/smac/smac_$TEACHER\_$FC3\_$RL_AGENT --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$TEACHER/$ID/$FC4 --agent_type $RL_AGENT --output_path $RESULTS_DIR/smac/smac_$TEACHER\_$FC4\_$RL_AGENT --no-tanh_scaling
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
