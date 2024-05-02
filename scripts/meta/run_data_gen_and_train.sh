strings=("bc" "td3_bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql")
for i in "${strings[@]}"
do
    sbatch --bosch scripts/meta/data_gen_and_train.sh exponential_decay $i
    sbatch --bosch scripts/meta/data_gen_and_train.sh step_decay $i
    sbatch --bosch scripts/meta/data_gen_and_train.sh sgdr $i
    sbatch --bosch scripts/meta/data_gen_and_train.sh constant $i
done