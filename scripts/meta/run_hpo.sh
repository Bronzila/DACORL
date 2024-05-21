RESULTS_DIR=${1:-data}
ID=${2:-0}

strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/hpo.sh exponential_decay $RESULTS_DIR $i $ID
	sbatch --bosch scripts/meta/hpo.sh step_decay $RESULTS_DIR $i $ID
	sbatch --bosch scripts/meta/hpo.sh sgdr $RESULTS_DIR $i $ID
	sbatch --bosch scripts/meta/hpo.sh constant $RESULTS_DIR $i $ID
done
