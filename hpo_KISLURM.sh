#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J HPO              # sets the job name. 
#SBATCH -a 1-4 # array size
#SBATCH -t 0-31:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=${2:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=100
BUDGET=15000
VERSION=extended_vel
RESULTS_DIR="data_combined_hetero"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Agent: $AGENT";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC1";
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/combined/$FC1 --output_path smac_$AGENT\_$FC1
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC1 --output_path smac_$AGENT\_$FC1
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --output_path smac_$AGENT\_$FC1 #--budget $BUDGET

    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --output_path smac_$AGENT\_$FC1 --budget $BUDGET
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/combined_e_st_sgdr/$FC1 --output_path smac_$AGENT\_$FC1
    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC1 --output_path smac_$AGENT\_$FC1
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Running $AGENT on $FC2";
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/combined/$FC2 --output_path smac_$AGENT\_$FC2
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC2 --output_path smac_$AGENT\_$FC2
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --output_path smac_$AGENT\_$FC2 #--budget $BUDGET

    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --output_path smac_$AGENT\_$FC2 --budget $BUDGET
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/combined_e_st_sgdr/$FC2 --output_path smac_$AGENT\_$FC2
    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC2 --output_path smac_$AGENT\_$FC2
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC3";
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/combined/$FC3 --output_path smac_$AGENT\_$FC3
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC3 --output_path smac_$AGENT\_$FC3
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --output_path smac_$AGENT\_$FC3 #--budget $BUDGET

    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --output_path smac_$AGENT\_$FC3 --budget $BUDGET
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/combined_e_st_sgdr/$FC3 --output_path smac_$AGENT\_$FC3
    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC3 --output_path smac_$AGENT\_$FC3
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Running $AGENT on $FC4";
    # python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR

    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/combined/$FC4 --output_path smac_$AGENT\_$FC4
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC4 --output_path smac_$AGENT\_$FC4
    # python train_hpo_MF.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --output_path smac_$AGENT\_$FC4 #--budget $BUDGET

    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --output_path smac_$AGENT\_$FC4 --budget $BUDGET
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/combined_e_st_sgdr/$FC4 --output_path smac_$AGENT\_$FC4
    # python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/combined/$FC4 --output_path smac_$AGENT\_$FC4
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
