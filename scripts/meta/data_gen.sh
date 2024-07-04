#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # bosch_cpu-cascadelake #gki_cpu-caskadelake # relea_gpu-rtx2080 mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-10:00:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data_gen # sets the job name. 
#SBATCH --mem 128GB
#SBATCH -a 1-4 

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

RESULTS_DIR=${1:-data}
TEACHER=${2:-exponential_decay}
BENCH=${3:-SGD}
ID=${4:-0}
NUM_RUNS=50
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
VERSION=extended_velocity

if [ "$BENCH" = "SGD" ]; then
    ARGS="--env default"
fi

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Setup: $TEACHER $ID";

start=`date +%s`

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ] 
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION $ARGS --benchmark $BENCH --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ] 
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION $ARGS --benchmark $BENCH --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ] 
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION $ARGS --benchmark $BENCH --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ] 
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION $ARGS --benchmark $BENCH --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
fi

end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished after $(runtime)";