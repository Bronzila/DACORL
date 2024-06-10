#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Teacher_HPO              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-15:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
BUDGET=15000
VERSION=extended_vel
RESULTS_DIR="data_teacher_hpo"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type exponential_decay --output_path teach_hpo_exp_Ackley --env Ackley_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type exponential_decay --output_path teach_hpo_exp_Rastrigin --env Rastrigin_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type exponential_decay --output_path teach_hpo_exp_Rosenbrock --env Rosenbrock_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type exponential_decay --output_path teach_hpo_exp_Sphere --env Sphere_$VERSION
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type step_decay --output_path teach_hpo_step_Ackley --env Ackley_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type step_decay --output_path teach_hpo_step_Rastrigin --env Rastrigin_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type step_decay --output_path teach_hpo_step_Rosenbrock --env Rosenbrock_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type step_decay --output_path teach_hpo_step_Sphere --env Sphere_$VERSION
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type sgdr --output_path teach_hpo_sgdr_Ackley --env Ackley_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type sgdr --output_path teach_hpo_sgdr_Rastrigin --env Rastrigin_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type sgdr --output_path teach_hpo_sgdr_Rosenbrock --env Rosenbrock_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type sgdr --output_path teach_hpo_sgdr_Sphere --env Sphere_$VERSION
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type constant --output_path teach_hpo_const_Ackley --env Ackley_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type constant --output_path teach_hpo_const_Rastrigin --env Rastrigin_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type constant --output_path teach_hpo_const_Rosenbrock --env Rosenbrock_$VERSION
    python3.10 teacher_hpo.py --data_dir $RESULTS_DIR --agent_type constant --output_path teach_hpo_const_Sphere --env Sphere_$VERSION
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
