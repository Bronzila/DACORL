#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting              # sets the job name. 
#SBATCH -t 0-1:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=${1:-td3_bc}
RESULTS_DIR=${2:-data}
FIDELITY=${3:-30000}
ID=${4:-0}
FUNCTIONS=(Ackley Rastrigin Rosenbrock Sphere)
TEACHERS=(exponential_decay step_decay sgdr constant)
NUM_RUNS=0
VERSION=extended_velocity
# SEEDS="3653403230 2735729614 2195314464 1158725111 1322117303 175979944 323153948 70985653 752767290 3492969079 2789219405 3920255352"
SEEDS="209652396 398764591 924231285 1478610112 441365315 1537364731 192771779 1491434855 1819583497 530702035"

for teacher in $TEACHERS
do
    for function in $FUNCTIONS
    do
        for seed in $SEEDS
        do
            python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/$teacher/$ID/$function/ --agent $AGENT --fidelity $FIDELITY --action --num_runs $NUM_RUNS --seed $seed --reward
        done
    done
done