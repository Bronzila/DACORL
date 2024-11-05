SEEDS="0 1 2 3 4"
AGENT=td3_bc

for SEED in $SEEDS
do
    sbatch scripts/LayerwiseSGD/train_single.sh $SEED $AGENT --bosch
    sbatch scripts/LayerwiseSGD/train_homo.sh $SEED $AGENT --bosch
    sbatch scripts/LayerwiseSGD/train_hetero.sh $SEED $AGENT --bosch
done