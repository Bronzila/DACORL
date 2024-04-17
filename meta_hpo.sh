#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #gki_cpu-caskadelake # relea_gpu-rtx2080 mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-12:00:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J HPO # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH --mem 8GB

AGENT=${1:-exponential_decay}
RL_AGENT=${2:-td3_bc}
ID=${3:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=100
VERSION=default
RESULTS_DIR="data_hpo"

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --agent_type $RL_AGENT
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --agent_type $RL_AGENT
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --agent_type $RL_AGENT
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --agent_type $RL_AGENT
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
