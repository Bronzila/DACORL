SEEDS="0 1 2 3 4"
AGENT=td3_bc

for SEED in $SEEDS
do
    sbatch scripts/LayerwiseSGD/eval_single.sh $SEED $AGENT --bosch
done