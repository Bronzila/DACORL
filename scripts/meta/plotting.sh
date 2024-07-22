#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-00:30:00
#SBATCH -o logs/%A[%a].%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting # sets the job name. 
#SBATCH -a 1-16 # array size
#SBATCH --mem 8GB

AGENT=${1:-td3_bc}
RESULTS_DIR=${2:-data/data_hpo_10}
ID=${3:-0}
FIDELITY=${4:-30000}
FUNCTIONS=(Ackley Rastrigin Rosenbrock Sphere)
TEACHERS=(exponential_decay step_decay sgdr constant)
NUM_RUNS=1000
VERSION=extended_velocity

# SEED=1478610112

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[0]}/ --agent $AGENT --teacher_type ${TEACHERS[0]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[1]}/ --agent $AGENT --teacher_type ${TEACHERS[0]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[2]}/ --agent $AGENT --teacher_type ${TEACHERS[0]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[0]}/$ID/${FUNCTIONS[3]}/ --agent $AGENT --teacher_type ${TEACHERS[0]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 5 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[0]}/ --agent $AGENT --teacher_type ${TEACHERS[1]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[1]}/ --agent $AGENT --teacher_type ${TEACHERS[1]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 7 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[2]}/ --agent $AGENT --teacher_type ${TEACHERS[1]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 8 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[1]}/$ID/${FUNCTIONS[3]}/ --agent $AGENT --teacher_type ${TEACHERS[1]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 9 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[0]}/ --agent $AGENT --teacher_type ${TEACHERS[2]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 10 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[1]}/ --agent $AGENT --teacher_type ${TEACHERS[2]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 11 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[2]}/ --agent $AGENT --teacher_type ${TEACHERS[2]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 12 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[2]}/$ID/${FUNCTIONS[3]}/ --agent $AGENT --teacher_type ${TEACHERS[2]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 13 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[0]}/ --agent $AGENT --teacher_type ${TEACHERS[3]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 14 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[1]}/ --agent $AGENT --teacher_type ${TEACHERS[3]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 15 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[2]}/ --agent $AGENT --teacher_type ${TEACHERS[3]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
elif [ ${SLURM_ARRAY_TASK_ID} -eq 16 ]
then
	python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/${TEACHERS[3]}/$ID/${FUNCTIONS[3]}/ --agent $AGENT --teacher_type ${TEACHERS[3]} --fidelity $FIDELITY --action --comparison --num_runs 0 --teacher
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";