source activate MTORL-DAC

# Generate data

python data_gen.py --save_run_data --save_rep_buffer --env Ackley_extended --agent exponential_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rastrigin_extended --agent exponential_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rosenbrock_extended --agent exponential_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Sphere_extended --agent exponential_decay --num_runs 100

python data_gen.py --save_run_data --save_rep_buffer --env Ackley_extended --agent step_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rastrigin_extended --agent step_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rosenbrock_extended --agent step_decay --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Sphere_extended --agent step_decay --num_runs 100

python data_gen.py --save_run_data --save_rep_buffer --env Ackley_extended --agent sgdr --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rastrigin_extended --agent sgdr --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rosenbrock_extended --agent sgdr --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Sphere_extended --agent sgdr --num_runs 100

python data_gen.py --save_run_data --save_rep_buffer --env Ackley_extended --agent constant --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rastrigin_extended --agent constant --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Rosenbrock_extended --agent constant --num_runs 100
python data_gen.py --save_run_data --save_rep_buffer --env Sphere_extended --agent constant --num_runs 100

# Check generated data

python3.10 check_fbest.py --path data/ToySGD/exponential_decay/0/Ackley/aggregated_run_data.csv --lowest --mean
python3.10 check_fbest.py --path data/ToySGD/exponential_decay/0/Rastrigin/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/exponential_decay/0/Rosenbrock/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/exponential_decay/0/Sphere/aggregated_run_data.csv --lowest --mean

# python3.10 check_fbest.py --path data/ToySGD/step_decay/0/Ackley/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/step_decay/0/Rastrigin/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/step_decay/0/Rosenbrock/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/step_decay/0/Sphere/aggregated_run_data.csv --lowest --mean

# python3.10 check_fbest.py --path data/ToySGD/sgdr/0/Ackley/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/sgdr/0/Rastrigin/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/sgdr/0/Rosenbrock/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/sgdr/0/Sphere/aggregated_run_data.csv --lowest --mean

# python3.10 check_fbest.py --path data/ToySGD/constant/0/Ackley/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/constant/0/Rastrigin/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/constant/0/Rosenbrock/aggregated_run_data.csv --lowest --mean
# python3.10 check_fbest.py --path data/ToySGD/constant/0/Sphere/aggregated_run_data.csv --lowest --mean

# Train

python train.py --data_dir data/ToySGD/exponential_decay/0/Ackley --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/exponential_decay/0/Rastrigin --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/exponential_decay/0/Rosenbrock --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/exponential_decay/0/Sphere --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256

python train.py --data_dir data/ToySGD/step_decay/0/Ackley --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/step_decay/0/Rastrigin --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/step_decay/0/Rosenbrock --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/step_decay/0/Sphere --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256

python train.py --data_dir data/ToySGD/sgdr/0/Ackley --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/sgdr/0/Rastrigin --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/sgdr/0/Rosenbrock --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/sgdr/0/Sphere --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256

python train.py --data_dir data/ToySGD/constant/0/Ackley --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/constant/0/Rastrigin --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/constant/0/Rosenbrock --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256
python train.py --data_dir data/ToySGD/constant/0/Sphere --num_train_iter 10000 --num_eval_runs 100 --val_freq 2000 --batch_size 256