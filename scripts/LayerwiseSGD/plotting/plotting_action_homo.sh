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
ID=${2:-"combined"}
TEACHERS="exponential_decay step_decay sgdr constant"
NUM_RUNS=0
FIDELITY=60000

for teacher in $TEACHERS
do
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

    python -W ignore plotting.py --data_dir $RESULTS_DIR/LayerwiseSGD/$teacher/$ID --agent $AGENT --action --fidelity $FIDELITY --num_runs $NUM_RUNS --teacher --agent_labels "$LABEL" "TD3+BC" --title "$TITLE" --metric "validation_accuracy"
done