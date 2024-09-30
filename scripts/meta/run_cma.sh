RESULTS_DIR=${1:-data}
HIDDEN=${2:-64}

strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql" "td3")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/cma_datagen_train.sh $RESULTS_DIR $i $HIDDEN
done