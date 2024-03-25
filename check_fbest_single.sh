source activate MTORL-DAC
id=combined
root_dir=data_multi

for agent in exponential_decay #step_decay sgdr constant
do
    for function in Ackley Rastrigin Rosenbrock Sphere
    do
        python3.10 check_fbest.py --path $root_dir/ToySGD/$agent/$id/$function/results --results --mean --lowest
    done
done