#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data              # sets the job name. 
#SBATCH -t 0-2:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

VERSION=extended_vel
ID=0
NUM_RUNS=1000
RESULTS_DIR=data_teacher_eval_66316748_mixed
COMBINED_IDS="combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c"

# Create the directory if it doesn't exist
mkdir -p $RESULTS_DIR/custom_paths
mkdir -p $RESULTS_DIR/custom_paths/$COMBINED_ID

for COMBINED_ID in $COMBINED_IDS
do
    for FCT in "Ackley" "Rastrigin" "Rosenbrock" "Sphere"
    do
        # for AGENT in "exponential_decay" "step_decay" "sgdr" "constant"    
        # do
        #     python data_gen.py --save_run_data --save_rep_buffer --env $FCT\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
        # done
        # Create the JSON file with the specified content
    #     cat <<EOL > $RESULTS_DIR/custom_paths/$COMBINED_ID/$FCT.json
    #     [
    #         "$RESULTS_DIR/ToySGD/exponential_decay/0/$FCT",
    #         "$RESULTS_DIR/ToySGD/step_decay/0/$FCT",
    #         "$RESULTS_DIR/ToySGD/sgdr/0/$FCT",
    #         "$RESULTS_DIR/ToySGD/constant/0/$FCT"
    #     ]
    # EOL

        python combine_buffers.py --custom_paths $RESULTS_DIR/custom_paths/$COMBINED_ID/$FCT.json --combined_dir $RESULTS_DIR/ToySGD/$COMBINED_ID/$FCT
    done
done