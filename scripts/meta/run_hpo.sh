RESULTS_DIR=${1:-data}

strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/hpo.sh exponential_decay $i $RESULTS_DIR
	sbatch --bosch scripts/meta/hpo.sh step_decay $i $RESULTS_DIR
	sbatch --bosch scripts/meta/hpo.sh sgdr $i $RESULTS_DIR
	sbatch --bosch scripts/meta/hpo.sh constant $i $RESULTS_DIR
done
