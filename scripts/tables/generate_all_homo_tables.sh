sbatch scripts/tables/generate_homo_tables.sh data_homo_256_60k/ --bosch
sbatch scripts/tables/generate_homo_tables.sh data_homo_256_perf_based_60k/ data_homo_256_60k/ --bosch
sbatch scripts/tables/generate_homo_tables.sh data_homo_256_expert_60k/ data_homo_256_60k/ --bosch