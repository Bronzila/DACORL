#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Eval_Agents              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-3:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

RESULTS_DIR=${1:-data}
AGENT=${2:-exponential_decay}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
GEN_SEED=123
TRAIN_ITER=30000
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC1";
    # for seed in {0..11}
    # for seed in "209652396" "398764591" "924231285" "1478610112" "441365315" "1537364731" "192771779" "1491434855" "1819583497" "530702035"
    for seed in "3653403230" "2735729614" "2195314464" "1158725111" "1322117303" "175979944" "323153948" "70985653" "752767290" "3492969079" "2789219405" "3920255352"
    do
        python3.10 eval.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/0/$FC1/ --training_seed $seed --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
    done
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running $AGENT on $FC2";
    # for seed in {0..11}
    # for seed in "209652396" "398764591" "924231285" "1478610112" "441365315" "1537364731" "192771779" "1491434855" "1819583497" "530702035"
    for seed in "3653403230" "2735729614" "2195314464" "1158725111" "1322117303" "175979944" "323153948" "70985653" "752767290" "3492969079" "2789219405" "3920255352"
    do
        python3.10 eval.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/0/$FC2/ --training_seed $seed --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
    done
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC3";
    # for seed in {0..11}
    # for seed in "209652396" "398764591" "924231285" "1478610112" "441365315" "1537364731" "192771779" "1491434855" "1819583497" "530702035"
    for seed in "3653403230" "2735729614" "2195314464" "1158725111" "1322117303" "175979944" "323153948" "70985653" "752767290" "3492969079" "2789219405" "3920255352"
    do
        python3.10 eval.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/0/$FC3/ --training_seed $seed --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
    done
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC4";
    # for seed in {0..11}
    # for seed in "209652396" "398764591" "924231285" "1478610112" "441365315" "1537364731" "192771779" "1491434855" "1819583497" "530702035"
    for seed in "3653403230" "2735729614" "2195314464" "1158725111" "1322117303" "175979944" "323153948" "70985653" "752767290" "3492969079" "2789219405" "3920255352"
    do
        python3.10 eval.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/0/$FC4/ --training_seed $seed --num_runs $NUM_RUNS --num_train_iter $TRAIN_ITER --eval_protocol interpolation --gen_seed $GEN_SEED
    done
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
