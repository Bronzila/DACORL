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
FUNCTIONS="Ackley Rastrigin Rosenbrock Sphere"
TEACHERS="E-ST-SG-C E-C E-SG E-SG-C E-ST E-ST-C E-ST-SG SG-C ST-C ST-SG ST-SG-C"
NUM_RUNS=0
FIDELITY=60000

for teacher in $TEACHERS
do
    case $teacher in
        "E-ST-SG-C")
            LABEL="Combined all"
            ;;
        "E-C")
            LABEL="Exp + Const"
            ;;
        "E-SG")
            LABEL="Exp + SGDR"
            ;;
        "E-SG-C")
            LABEL="Exp + SGDR + Const"
            ;;
        "E-ST")
            LABEL="Exp + Step"
            ;;
        "E-ST-C")
            LABEL="Exp + Step + Const"
            ;;
        "E-ST-SG")
            LABEL="Exp + Step + SGDR"
            ;;
        "SG-C")
            LABEL="SGDR + Const"
            ;;
        "ST-C")
            LABEL="Step + Const"
            ;;
        "ST-SG")
            LABEL="Step + SGDR"
            ;;
        "ST-SG-C")
            LABEL="Step + SGDR + Const"
            ;;
        *)
            LABEL="Unknown"
            ;;
    esac
    # TITLE="Comparing Teacher and Agent Behavior on $function"
    TITLE=""

    python -W ignore plotting.py --data_dir $RESULTS_DIR/LayerwiseSGD/$teacher/ --agent $AGENT --action --fidelity $FIDELITY --num_runs $NUM_RUNS --teacher --agent_labels "$LABEL" "TD3+BC" --title "$TITLE" --heterogeneous --metric "valid_acc"
done