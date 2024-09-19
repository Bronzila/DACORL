#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting              # sets the job name. 
#SBATCH -t 0-3:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

RESULTS_DIR=${1:-data}
NUM_RUNS=0
VERSION=extended_velocity
IDS="0 1 2 3 4 combined"

for teacher in step_decay sgdr constant exponential_decay 
do
    for function in Rastrigin Rosenbrock Sphere Ackley
    do
        python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/$teacher/ --action_teacher --single_plot --function $function
    done
done