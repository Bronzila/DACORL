#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-00:30:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Statistics # sets the job name. 
#SBATCH --mem 8GB

RESULTS_DIR=${1:-data}

ID=0
FIDELITY=30000
ENV=SGD

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL

python -W ignore generate_sgd_tables.py --path $RESULTS_DIR/$ENV/ --auc --mean --lowest --ids $ID --hpo_budget $FIDELITY --format latex --num_runs 20


# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";