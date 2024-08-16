SEEDS="209652396 398764591 924231285 1478610112 441365315"

for SEED in $SEEDS
do
    # sbatch scripts/SGD/train_single.sh $SEED --bosch
    # sbatch scripts/SGD/train_homo.sh $SEED --bosch
    sbatch scripts/SGD/train_hetero.sh $SEED --bosch
done