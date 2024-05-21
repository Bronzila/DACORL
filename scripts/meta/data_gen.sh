#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #gki_cpu-caskadelake # relea_gpu-rtx2080 mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-0:10:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data_gen_homogeneous # sets the job name. 
#SBATCH --mem 16GB
#SBATCH -a 1-4 

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

TEACHER=${1:-exponential_decay}
ID=${2:-0}
NUM_RUNS=100
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
VERSION=extended_velocity
RESULTS_DIR="data/data_homogeneous_10"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Setup: $TEACHER $ID";

start=`date +%s`

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
fi

end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished after $(runtime)";