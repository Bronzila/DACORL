#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # partition (queue)
#SBATCH -t 0-60:00:00
#SBATCH -o logs/%A[%a].%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J HPO # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH --mem 8GB

TEACHER=${1:-exponential_decay}
RESULTS_DIR=${2:-data/data_hpo_10}
ID=${4:-0}
NUM_RUNS=1000
VERSION=default
CS_TYPE=reduced_no_arch_dropout
BENCHMARK=CMAES
RL_AGENTS=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql" "td3")

start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL

echo "Running in directory ($RESULTS_DIR) using the data teacher ($TEACHER) on ($ID) and training all agents"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[1]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[1]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[2]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[2]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[3]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[3]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[4]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[4]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 5 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[5]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[5]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[6]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[6]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 7 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[7]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[7]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 8 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[8]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[8]} --cs_type $CS_TYPE --no-tanh_scaling
elif [ ${SLURM_ARRAY_TASK_ID} -eq 9 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $VERSION --benchmark $BENCHMARK --agent $TEACHER --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    python train_hpo.py --data_dir $RESULTS_DIR/$BENCHMARK/$TEACHER/$ID --agent_type ${RL_AGENTS[9]} --output_path $RESULTS_DIR/smac/smac_$TEACHER\_${RL_AGENTS[9]} --cs_type $CS_TYPE --no-tanh_scaling
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";
