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

AGENT=td3_bc
RESULTS_DIR=${1:-data}
FIDELITY=${2:-60000}
ID=${3:-"combined"}
FUNCTIONS="Ackley Rastrigin Rosenbrock Sphere"
TEACHERS="exponential_decay step_decay sgdr constant"
NUM_RUNS=0
# SEEDS="209652396 398764591 924231285 1478610112 441365315 1537364731 192771779 1491434855 1819583497 530702035"

for teacher in $TEACHERS
do
    for function in $FUNCTIONS
    do
        # for seed in $SEEDS
        # do

            case $teacher in
                "constant")
                    LABEL="Combined Constant"
                    ;;
                "exponential_decay")
                    LABEL="Combined Exponential Decay"
                    ;;
                "step_decay")
                    LABEL="Combined Step Decay"
                    ;;
                "sgdr")
                    LABEL="Combined SGDR"
                    ;;
                *)
                    LABEL="Unknown"
                    ;;
            esac
            # TITLE="Comparing Teacher and Agent Behavior on $function"
            TITLE=""

            python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/$teacher/$ID/$function/ --agent $AGENT --fidelity $FIDELITY --action --num_runs $NUM_RUNS --teacher --agent_labels "$LABEL" "TD3+BC" --title "$TITLE"
        # done
    done
done