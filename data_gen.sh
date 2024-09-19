#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J teacher_eval              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-00:20:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=${2:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
VERSION=extended_vel
RESULTS_DIR="data_teacher_eval_66316748"
SEED=66316748
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC1";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR --seed $SEED

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running $AGENT on $FC2";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR --seed $SEED

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC3";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR --seed $SEED

    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC4";
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR --seed $SEED
    
    python generate_tables.py --path $RESULTS_DIR/ToySGD --lowest --mean
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
