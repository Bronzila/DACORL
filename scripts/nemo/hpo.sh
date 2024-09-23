#MSUB -N HPO
#MSUB -e logs/${MOAB_JOBID}.e
#MSUB -o logs/${MOAB_JOBID}.o
#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=0:12:00:00
#MSUB -l pmem=16000mb
#MSUB -t [1-4]

cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

AGENT=${1:-exponential_decay}
ID=${2:-0}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
NUM_RUNS=100
VERSION=extended_momentum
RESULTS_DIR="data_extended_momentum"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job ${MOAB_JOBID} of user ${MOAB_USER} using ${MOAB_NODECOUNT} nodes on partition ${MOAB_PARTITION}";


if [ ${MOAB_JOBARRAYINDEX} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    # python train_hpo.py --data_dir $RESULTS_DIR/heterogeneous/$FC1 --output_path smac_$AGENT\_$FC1
    # python train_hpo.py --data_dir $RESULTS_DIR/homogeneous/$AGENT/$FC1 --output_path smac_$AGENT\_$FC1
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC1 --output_path smac_$AGENT\_$FC1
elif [ ${MOAB_JOBARRAYINDEX} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    # python train_hpo.py --data_dir $RESULTS_DIR/heterogeneous/$FC2 --output_path smac_$AGENT\_$FC2
    # python train_hpo.py --data_dir $RESULTS_DIR/homogeneous/$AGENT/$FC2 --output_path smac_$AGENT\_$FC2
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC2 --output_path smac_$AGENT\_$FC2
elif [ ${MOAB_JOBARRAYINDEX} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    # python train_hpo.py --data_dir $RESULTS_DIR/heterogeneous/$FC3 --output_path smac_$AGENT\_$FC3
    # python train_hpo.py --data_dir $RESULTS_DIR/homogeneous/$AGENT/$FC3 --output_path smac_$AGENT\_$FC3
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC3 --output_path smac_$AGENT\_$FC3
elif [ ${MOAB_JOBARRAYINDEX} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID --results_dir $RESULTS_DIR
    # python train_hpo.py --data_dir $RESULTS_DIR/heterogeneous/$FC4 --output_path smac_$AGENT\_$FC4
    # python train_hpo.py --data_dir $RESULTS_DIR/homogeneous/$AGENT/$FC4 --output_path smac_$AGENT\_$FC4
    python train_hpo.py --data_dir $RESULTS_DIR/ToySGD/$AGENT/$ID/$FC4 --output_path smac_$AGENT\_$FC4
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
