#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J MF_HPO              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-32:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=${2:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
TIMELIMIT=30
BUDGET=15000
VERSION=extended_vel_no_opt
RESULTS_DIR="data_mf_no_opt_arch"
ARCH_CS=--arch_cs
# ARCH_CS= 
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC1";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    
    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --output_path smac_$AGENT\_$FC1 --debug $ARCH_CS --budget $BUDGET

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean --results --multi_seed --num_runs 1000
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running $AGENT on $FC2";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --output_path smac_$AGENT\_$FC2 --debug $ARCH_CS --budget $BUDGET

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean --results --multi_seed --num_runs 1000
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC3";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --output_path smac_$AGENT\_$FC3 --debug $ARCH_CS --budget $BUDGET

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean --results --multi_seed --num_runs 1000
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC4";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --output_path smac_$AGENT\_$FC4 --debug $ARCH_CS --budget $BUDGET
    
    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean --results --multi_seed --num_runs 1000
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
