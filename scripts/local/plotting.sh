source activate DACORL
id=0
root_dir=data/check_lb-sac
fidelity=10000

for teacher in exponential_decay step_decay sgdr constant
do
    for function in Ackley Rastrigin Rosenbrock Sphere
    do
        for agent in lb_sac #bc td3_bc cql awac edac  sac_n
        do
            python -W ignore plotting.py --data_dir $root_dir/ToySGD/$teacher/$id/$function/ --agent_path results/$agent/$fidelity --action --num_runs 0 --teacher
        done
    done
done