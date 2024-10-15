#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Teacher_HPO              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-20:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=${1:-exponential_decay}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
BUDGET=30000
VERSION=extended_vel
RESULTS_DIR="data_teacher_hpo"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type $AGENT --output_path teach_hpo_$AGENT\_Ackley --env Ackley_$VERSION
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type $AGENT --output_path teach_hpo_$AGENT\_Rastrigin --env Rastrigin_$VERSION
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type $AGENT --output_path teach_hpo_$AGENT\_Rosenbrock --env Rosenbrock_$VERSION
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type $AGENT --output_path teach_hpo_$AGENT\_Sphere --env Sphere_$VERSION
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
