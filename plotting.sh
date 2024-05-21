#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting              # sets the job name. 
#SBATCH -t 0-3:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

AGENT=${1:-td3_bc}
RESULTS_DIR=${2:-data}
FIDELITY=${3:-15000}
ID=${4:-0}
FUNCTIONS=(Ackley Rastrigin Rosenbrock Sphere)
TEACHERS=(exponential_decay step_decay sgdr constant)
NUM_RUNS=0
VERSION=extended_velocity
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"

for teacher in step_decay sgdr constant exponential_decay 
do
    for function in Rastrigin Rosenbrock Sphere Ackley
    do
        for seed in $SEEDS
        do
            python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/$teacher/$ID/$function/ --agent $AGENT --fidelity $FIDELITY --action --num_runs $NUM_RUNS --teacher --seed $seed  --reward
        done
    done
done