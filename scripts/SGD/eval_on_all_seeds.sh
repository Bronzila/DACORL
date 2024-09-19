SEEDS="209652396 398764591 924231285 1478610112 441365315"

# for SEED in $SEEDS
# do
#     sbatch scripts/SGD/eval_SGD.sh SGD_data/single_20_cpu 0 $SEED 30000 --bosch
# done

# for SEED in $SEEDS
# do
#     sbatch scripts/SGD/eval_SGD.sh SGD_data/homo_20_cpu combined $SEED 60000 --bosch
# done

for SEED in $SEEDS
do
    sbatch scripts/SGD/eval_SGD_hetero.sh SGD_data/hetero_20_cpu $SEED 60000 --bosch
done