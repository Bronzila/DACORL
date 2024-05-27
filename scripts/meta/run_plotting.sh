RESULTS_DIR=${1:-data}
ID=${2:-0}

strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/plotting.sh $i $RESULTS_DIR $ID
done
