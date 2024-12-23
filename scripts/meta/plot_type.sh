#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-00:30:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting # sets the job name. 
#SBATCH -a 1-16 # array size
#SBATCH --mem 8GB

AGENT=${1:-td3_bc}
FIDELITY=${2:-10000}
ID=${3:-0}
FUNCTIONS=(Ackley Rastrigin Rosenbrock Sphere)
TEACHERS=(exponential_decay step_decay sgdr constant)
NUM_RUNS=100
VERSION=extended_velocity
RESULTS_DIR="data/data_hpo_10"
PLOT_TYPE="action"

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[0]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[1]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[2]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[3]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 5 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[0]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[1]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 7 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[2]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 8 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[3]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 9 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[0]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 10 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[1]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 11 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[2]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 12 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[3]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 13 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[0]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 14 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[1]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 15 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[2]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 16 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[3]}/ --fidelity $FIDELITY --plot_type $PLOT_TYPE --teacher
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";