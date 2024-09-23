#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake #gki_cpu-caskadelake # relea_gpu-rtx2080 mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH -t 0-0:10:00
#SBATCH -o logs/%A.%N.o       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A.%N.e       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J data_merge # sets the job name. 
#SBATCH --mem 16GB
#SBATCH -a 1-4 

source ~/.bashrc
cd /work/dlclarge2/gieringl-DACORL/MTORL-DAC
source activate DACORL

SOURCE=${1:-data/data_homogeneous_10}
DESTINATION=${2:-data/data_homogeneous_10/combined}
FC1=Ackley
FC2=Rastrigin
FC3=Rosenbrock
FC4=Sphere


if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
    for TEACHER in exponential_decay step_decay sgdr constant
    do
        python -W ignore combine_buffers.py --root_dir $SOURCE --combined_dir $DESTINATION --teacher $TEACHER --function $FC1
    done
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
    for TEACHER in exponential_decay step_decay sgdr constant
    do
        python -W ignore combine_buffers.py --root_dir $SOURCE --combined_dir $DESTINATION --teacher $TEACHER --function $FC2
    done
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]
then
    for TEACHER in exponential_decay step_decay sgdr constant
    do
        python -W ignore combine_buffers.py --root_dir $SOURCE --combined_dir $DESTINATION --teacher $TEACHER --function $FC3
    done
elif [ ${SLURM_ARRAY_TASK_ID} -eq 4 ]
then
    for TEACHER in exponential_decay step_decay sgdr constant
    do
        python -W ignore combine_buffers.py --root_dir $SOURCE --combined_dir $DESTINATION --teacher $TEACHER --function $FC4
    done
fi
