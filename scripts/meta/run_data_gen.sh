#!/bin/bash

RESULTS_DIR=${1:-data}
BENCH=${2:-SGD}
ID=${3:-0}

sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR exponential_decay $BENCH $ID
sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR step_decay $BENCH $ID
sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR sgdr $BENCH $ID
sbatch --bosch scripts/meta/data_gen.sh $RESULTS_DIR constant $BENCH $ID