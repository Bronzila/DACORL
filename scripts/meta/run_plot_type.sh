strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql")
for i in "${strings[@]}"
do
	sbatch --bosch scripts/meta/plot_type.sh $i
done
