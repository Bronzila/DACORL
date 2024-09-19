#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Table_gen              # sets the job name. 
#SBATCH -t 0-3:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

RESULTS_DIR=${1:-data_homo_256_60k}
TEACHER_PATH=${2:-""}

if [ -z "$TEACHER_PATH" ]; then
    TEACHER_ARG=""
else
    TEACHER_ARG="--teacher_base_path $TEACHER_PATH/ToySGD/"
fi

echo "Homo Teacher Agent Table";
python generate_latex_tables.py --path $RESULTS_DIR/ToySGD/ --lowest --mean --auc --multi_seed --num_runs 1000 --ids combined --interpolation $TEACHER_ARG
