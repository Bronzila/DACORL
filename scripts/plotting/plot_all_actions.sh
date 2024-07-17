# sbatch ./scripts/plotting/plotting_action_single.sh data_single_64/ --bosch

# sbatch ./scripts/plotting/plotting_action_homo.sh data_homo_256_60k/ --bosch
# sbatch ./scripts/plotting/plotting_action_homo.sh data_homo_256_perf_based_60k/ --bosch
# sbatch ./scripts/plotting/plotting_action_homo.sh data_homo_256_expert_60k/ --bosch

# sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_60k/ --bosch
# sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_perf_based_60k --bosch
# sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_expert_60k --bosch

sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_mixed_60k/ --bosch
sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_mixed_perf_based_60k --bosch
sbatch ./scripts/plotting/plotting_action_hetero.sh data_hetero_256_mixed_expert_60k --bosch