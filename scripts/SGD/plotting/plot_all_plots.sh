sbatch ./scripts/SGD/plotting/plotting_action_single.sh SGD_data/single_20_cpu --bosch
# sbatch ./scripts/SGD/plotting/plotting_action_homo.sh SGD_data/homo_20_cpu --bosch
# sbatch ./scripts/SGD/plotting/plotting_action_hetero.sh SGD_data/hetero_20_cpu --bosch

sbatch ./scripts/SGD/plotting/plotting_comparison_single.sh SGD_data/single_20_cpu --bosch
# sbatch ./scripts/SGD/plotting/plotting_comparison_homo.sh SGD_data/homo_20_cpu --bosch
# sbatch ./scripts/SGD/plotting/plotting_comparison_hetero.sh SGD_data/hetero_20_cpu --bosch