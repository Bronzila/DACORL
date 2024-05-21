for id in {0..10}
do
    sbatch --bosch scripts/meta/data_gen.sh exponential_decay $id
    sbatch --bosch scripts/meta/data_gen.sh step_decay $id
    sbatch --bosch scripts/meta/data_gen.sh sgdr $id
    sbatch --bosch scripts/meta/data_gen.sh constant $id
done