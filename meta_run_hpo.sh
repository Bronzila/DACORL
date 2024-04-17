strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac")
for i in "${strings[@]}"
do
	sbatch --bosch meta_hpo.sh exponential_decay $i
	sbatch --bosch meta_hpo.sh step_decay $i
	sbatch --bosch meta_hpo.sh sgdr $i
	sbatch --bosch meta_hpo.sh constant $i
done
