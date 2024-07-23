#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J SGD_data              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-24:00:00
#SBATCH --mem 16GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ID=${1:-SGD/default}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=50
VERSION=default
INSTANCE_MODE=random_seed
RESULTS_DIR="SGD_data/single"
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --instance_mode $INSTANCE_MODE --agent exponential_decay --benchmark SGD --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --instance_mode $INSTANCE_MODE --agent step_decay --benchmark SGD --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ 3 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --instance_mode $INSTANCE_MODE --agent sgdr --benchmark SGD --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
elif [ 4 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --instance_mode $INSTANCE_MODE --agent constant --benchmark SGD --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
