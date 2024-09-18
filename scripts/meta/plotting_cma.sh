#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-00:30:00
#SBATCH -o logs/%A[%a].%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting # sets the job name. 
#SBATCH -a 1-2 # array size
#SBATCH --mem 8GB

AGENT=${1:-td3_bc}
RESULTS_DIR=${2:-data/data_hpo_10}
ID=${3:-0}
FIDELITY=${4:-30000}
TEACHERS=(cmaes_constant csa)
NUM_RUNS=1000
VERSION=extended_velocity

# SEED=1478610112

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
	python -W ignore cma_plotting.py --data_dir $RESULTS_DIR/CMAES/${TEACHERS[0]}/$ID/ --agent $AGENT --teacher_type ${TEACHERS[0]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
	python -W ignore cma_plotting.py --data_dir $RESULTS_DIR/CMAES/${TEACHERS[1]}/$ID/ --agent $AGENT --teacher_type ${TEACHERS[1]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";