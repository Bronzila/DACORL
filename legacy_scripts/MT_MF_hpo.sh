#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Multi-Teacher-HPO              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-35:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=${2:-0} # use ID combined for combined agents
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=1000
TIMELIMIT=30
VAL_FREQ=30000
BUDGET=30000
VERSION=extended_vel
RESULTS_DIR="data_hetero_MF_dropout_256"
HIDDEN_DIM=256
# ARCH_CS=--arch_cs
ARCH_CS= 
# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC1";
    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$FC1 --output_path smac_$AGENT\_$FC1 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --output_path smac_$AGENT\_$FC1 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running $AGENT on $FC2";
    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$FC2 --output_path smac_$AGENT\_$FC2 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --output_path smac_$AGENT\_$FC2 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC3";
    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$FC3 --output_path smac_$AGENT\_$FC3 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --output_path smac_$AGENT\_$FC3 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC4";
    python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$FC4 --output_path smac_$AGENT\_$FC4 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --output_path smac_$AGENT\_$FC4 $ARCH_CS --time_limit $TIMELIMIT --debug --budget $BUDGET --val_freq $VAL_FREQ --hidden_dim $HIDDEN_DIM
fi

python generate_tables.py --custom_path $RESULTS_DIR/ToySGD/$AGENT/ --lowest --mean --agents $AGENT --num_runs $NUM_RUNS --multi_seed --results
# python3.10 generate_tables.py --path $RESULTS_DIR/ToySGD/ --lowest --mean --ids $ID --num_runs $NUM_RUNS --results --multi_seed
# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
