RESULTS_DIR=${1:-data}
ID=${2:-0}

strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql" "td3")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/plotting_cma.sh $i $RESULTS_DIR $ID
done
