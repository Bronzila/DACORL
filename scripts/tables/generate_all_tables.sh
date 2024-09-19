sbatch scripts/tables/generate_single_tables.sh 
sbatch scripts/tables/generate_single_tables_interp.sh 

./scripts/tables/generate_all_homo_tables.sh
./scripts/tables/generate_all_homo_tables_interp.sh
./scripts/tables/generate_all_hetero_tables.sh
./scripts/tables/generate_all_hetero_tables_interp.sh