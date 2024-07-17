#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # metaadm_cpu-test gki_cpu-caskadelake     # partition (queue)
#SBATCH -o logs/%A[%a].%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/%A[%a].%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J Percentages              # sets the job name. 
#SBATCH -a 1-10 # array size
#SBATCH -t 0-3:00:00
#SBATCH --mem 8GB

cd /work/dlclarge1/fixj-thesis/MTORL-DAC
source ~/.bashrc
conda activate MTORL-DAC

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Single Teacher percentages";
    python generate_perc_table.py --path data_single_64/ToySGD/ --mean --lowest --iqm --auc
elif [ 2 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Homo Concat percentages";
    python generate_perc_table.py --path data_homo_256_60k/ToySGD/ --id combined --mean --lowest --iqm --auc
elif [ 3 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Homo Perf-Based percentages";
    python generate_perc_table.py --path data_homo_256_perf_based_60k/ToySGD/ --id combined --mean --lowest --iqm --auc --baseline data_homo_256_60k/ToySGD/
elif [ 4 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Homo Expert percentages";
    python generate_perc_table.py --path data_homo_256_expert_60k/ToySGD/ --id combined --mean --lowest --iqm --auc --baseline data_homo_256_60k/ToySGD/
elif [ 5 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Hetero Concat percentages";
    python generate_perc_table.py --custom_path data_hetero_256_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc
elif [ 6 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Hetero Perf-Based percentages";
    python generate_perc_table.py --custom_path data_hetero_256_perf_based_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc --baseline data_hetero_256_60k/ToySGD/
elif [ 7 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Hetero Expert percentages";
    python generate_perc_table.py --custom_path data_hetero_256_expert_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc --baseline data_hetero_256_60k/ToySGD/
elif [ 8 -eq $SLURM_ARRAY_TASK_ID  ]
then
    echo "Mixed Concat percentages";
    python generate_perc_table.py --custom_path data_hetero_256_mixed_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc
elif [ 9 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Mixed Perf-Based percentages";
    python generate_perc_table.py --custom_path data_hetero_256_mixed_perf_based_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc --baseline data_hetero_256_mixed_60k/ToySGD/
elif [ 10 -eq $SLURM_ARRAY_TASK_ID ]
then
    echo "Mixed Expert percentages";
    python generate_perc_table.py --custom_path data_hetero_256_mixed_expert_60k/ToySGD/ --agents combined combined_e_c combined_e_sg combined_e_sg_c combined_e_st combined_e_st_c combined_e_st_sg combined_sg_c combined_st_c combined_st_sg combined_st_sg_c --mean --lowest --iqm --auc --baseline data_hetero_256_mixed_60k/ToySGD/
fi

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
