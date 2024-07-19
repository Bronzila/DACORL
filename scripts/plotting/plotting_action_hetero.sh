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
FUNCTIONS="Ackley Rastrigin Rosenbrock Sphere"
TEACHERS="combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c"
NUM_RUNS=0
VERSION=extended_velocity
# SEEDS="3653403230 2735729614 2195314464 1158725111 1322117303 175979944 323153948 70985653 752767290 3492969079 2789219405 3920255352"
# SEEDS="209652396 398764591 924231285 1478610112 441365315 1537364731 192771779 1491434855 1819583497 530702035"

for teacher in $TEACHERS
do
    for function in $FUNCTIONS
    do
        # for seed in $SEEDS
        # do

            case $teacher in
                "combined")
                    LABEL="Combined all"
                    ;;
                "combined_e_c")
                    LABEL="Exp + Const"
                    ;;
                "combined_e_sg")
                    LABEL="Exp + SGDR"
                    ;;
                "combined_e_sg_c")
                    LABEL="Exp + SGDR + Const"
                    ;;
                "combined_e_st")
                    LABEL="Exp + Step"
                    ;;
                "combined_e_st_c")
                    LABEL="Exp + Step + Const"
                    ;;
                "combined_e_st_sg")
                    LABEL="Exp + Step + SGDR"
                    ;;
                "combined_sg_c")
                    LABEL="SGDR + Const"
                    ;;
                "combined_st_c")
                    LABEL="Step + Const"
                    ;;
                "combined_st_sg")
                    LABEL="Step + SGDR"
                    ;;
                "combined_st_sg_c")
                    LABEL="Step + SGDR + Const"
                    ;;
                *)
                    LABEL="Unknown"
                    ;;
            esac
            # TITLE="Comparing Teacher and Agent Behavior on $function"
            TITLE=""

            python -W ignore plotting.py --data_dir $RESULTS_DIR/ToySGD/$teacher/$function/ --agent $AGENT --fidelity $FIDELITY --action --num_runs $NUM_RUNS --teacher --agent_labels "$LABEL" "TD3+BC" --title "$TITLE" --heterogeneous
        # done
    done
done