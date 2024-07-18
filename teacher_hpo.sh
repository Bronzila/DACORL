#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Teacher_HPO              # sets the job name. 
#SBATCH -a 1-3 # array size
#SBATCH -t 0-24:00:00
#SBATCH --mem 128GB

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

NUM_RUNS=1
RESULTS_DIR="data_teacher_hpo"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    python teacher_hpo.py --data_dir $RESULTS_DIR --agent_type exponential_decay --output_path teach_hpo_exponential_decay --env default
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    python teacher_hpo.py --data_dir $RESULTS_DIR --agent_type step_decay --output_path teach_hpo_step_decay --env default
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    python teacher_hpo.py --data_dir $RESULTS_DIR --agent_type sgdr --output_path teach_hpo_sgdr --env default
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";