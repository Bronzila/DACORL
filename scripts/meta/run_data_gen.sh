#!/bin/bash

RESULTS_DIR=${1:-data}
BENCH=${2:-SGD}
IDS=${3:-0}

for (( id=0; id<=IDS; id++ ))
do
    sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR exponential_decay $BENCH $id
    sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR step_decay $BENCH $id
    sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR sgdr $BENCH $id
    sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR constant $BENCH $id
done