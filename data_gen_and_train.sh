#MSUB -N data_gen_and_train_single_agent
#MSUB -e logs/${MOAB_JOBID}.e
#MSUB -o logs/${MOAB_JOBID}.o
#MSUB -l nodes=1:ppn=4
#MSUB -l walltime=0:02:00:00
#MSUB -l pmem=16000mb
#MSUB -t [1-4]

cd /work/ws/nemo/fr_jf442-thesis-0/MTORL-DAC
source activate MTORL-DAC

AGENT=${1:-exponential_decay}
ID=${2:-0}
BS=20
NUM_TRAIN_ITER=10000
VAL_FREQ=2000
NUM_RUNS=100
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere
VERSION=default

# Print some information about the job to STDOUT
echo "Workingdir: $(pwd)";
echo "Started at $(date)";
echo "Running job ${MOAB_JOBID} of user ${MOAB_USER} using ${MOAB_NODECOUNT} nodes on partition ${MOAB_PARTITION}";


if [ ${MOAB_JOBARRAYINDEX} -eq 1 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC1\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID
    python train.py --data_dir data/ToySGD/$AGENT/$ID/$FC1 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${MOAB_JOBARRAYINDEX} -eq 2 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC2\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID
    python train.py --data_dir data/ToySGD/$AGENT/$ID/$FC2 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${MOAB_JOBARRAYINDEX} -eq 3 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC3\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID
    python train.py --data_dir data/ToySGD/$AGENT/$ID/$FC3 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
elif [ ${MOAB_JOBARRAYINDEX} -eq 4 ]
then
    python data_gen.py --save_run_data --save_rep_buffer --env $FC4\_$VERSION --agent $AGENT --num_runs $NUM_RUNS --id $ID
    python train.py --data_dir data/ToySGD/$AGENT/$ID/$FC4 --num_train_iter $NUM_TRAIN_ITER --num_eval_runs $NUM_RUNS --val_freq $VAL_FREQ --batch_size $BS
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";