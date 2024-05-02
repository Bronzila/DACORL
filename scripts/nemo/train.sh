#MSUB -N train_single_agent
#MSUB -e logs/${MOAB_JOBID}.e
#MSUB -o logs/${MOAB_JOBID}.o
#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=0:01:00:00
#MSUB -l pmem=16000mb
#MSUB -t [1-4]

cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

TEACHER=${1:-exponential_decay}
AGENT=${1:-td3_bc}
ID=combined
NUM_TRAIN_ITER=20000
VAL_FREQ=2000
NUM_RUNS=100
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
VERSION=default
DATA_DIR="data/test_agents"

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job ${MOAB_JOBID} of user ${MOAB_USER} using ${MOAB_NODECOUNT} nodes on partition ${MOAB_PARTITION}";


if [ ${MOAB_JOBARRAYINDEX} -eq 1 ]
then
    python train.py --data_dir $DATA_DIR/ToySGD/$TEACHER/$ID/$FC1 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --agent_type $AGENT
elif [ ${MOAB_JOBARRAYINDEX} -eq 2 ]
then
    python train.py --data_dir $DATA_DIR/ToySGD/$TEACHER/$ID/$FC2 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --agent_type $AGENT
elif [ ${MOAB_JOBARRAYINDEX} -eq 3 ]
then
    python train.py --data_dir $DATA_DIR/ToySGD/$TEACHER/$ID/$FC3 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --agent_type $AGENT
elif [ ${MOAB_JOBARRAYINDEX} -eq 4 ]
then
    python train.py --data_dir $DATA_DIR/ToySGD/$TEACHER/$ID/$FC4 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --agent_type $AGENT
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
