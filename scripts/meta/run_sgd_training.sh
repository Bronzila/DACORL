#!/bin/bash
RESULTS_DIR=${1:-data}


strings=("bc" "cql" "awac" "edac" "sac_n" "lb_sac" "iql" "td3")
for i in "${strings[@]}"
do
    sbatch --bosch --begin 2024-08-02T03:00 scripts/meta/sgd_train_single.sh "$RESULTS_DIR" "$i"
done
