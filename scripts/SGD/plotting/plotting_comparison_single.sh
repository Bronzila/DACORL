#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Plotting              # sets the job name. 
#SBATCH -t 0-24:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=td3_bc
RESULTS_DIR=${1:-data}
ID=${2:-0}
FUNCTIONS="Ackley Rastrigin Rosenbrock Sphere"
TEACHERS="exponential_decay step_decay sgdr constant"
NUM_RUNS=0
# SEEDS="209652396 398764591 924231285 1478610112 441365315 1537364731 192771779 1491434855 1819583497 530702035"

for teacher in $TEACHERS
do
    case $teacher in
        "constant")
            LABEL="Constant"
            ;;
        "exponential_decay")
            LABEL="Exponential Decay"
            ;;
        "step_decay")
            LABEL="Step Decay"
            ;;
        "sgdr")
            LABEL="SGDR"
            ;;
        *)
            LABEL="Unknown"
            ;;
    esac
    # TITLE="Comparing Teacher and Agent Trajectories on $function"
    TITLE=""

    python -W ignore plotting.py --data_dir $RESULTS_DIR/SGD/$teacher/$ID/ --comparison --teacher --agent_labels "TD3+BC" --title "$TITLE" --metric "valid_acc"
done