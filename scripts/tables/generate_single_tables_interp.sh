#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Table_gen              # sets the job name. 
#SBATCH -t 0-2:00:00
#SBATCH -a 1-2 # array size
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Single Teacher Teacher Table";
    python generate_latex_tables.py --path data_teacher_eval_66316748/ToySGD/ --lowest --mean --auc
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Single Teacher Agent Table";
    python generate_latex_tables.py --path data_single_64/ToySGD/ --lowest --mean --auc --results --multi_seed --num_runs 1000 --interpolation
fi