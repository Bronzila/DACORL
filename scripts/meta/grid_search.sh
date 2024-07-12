#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080 # partition (queue)
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%A[%a].%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J HPO_grid # sets the job name. 
#SBATCH -a 1-8 # array size

LRs=(0.01 0.00871 0.00743 0.00614 0.00486 0.00357 0.00229 0.001)
start=`date +%s`

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
conda activate DACORL

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    python -W ignore grid_search.py ${LRs[0]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    python -W ignore grid_search.py ${LRs[1]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    python -W ignore grid_search.py ${LRs[2]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    python -W ignore grid_search.py ${LRs[3]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 5 ]
then
    python -W ignore grid_search.py ${LRs[4]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 6 ]
then
    python -W ignore grid_search.py ${LRs[5]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 7 ]
then
    python -W ignore grid_search.py ${LRs[6]}
elif [ ${SLURM_ARRAY_TASK_ID} -eq 8 ]
then
    python -W ignore grid_search.py ${LRs[7]}
fi

# Print some Information about the end-time to STDOUT
end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";