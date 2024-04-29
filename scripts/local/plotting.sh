source activate DACORL
id=0
root_dir=data/hpo_extended_velocity
fidelity=10000

for teacher in step_decay sgdr constant exponential_decay 
do
    for function in Rastrigin Rosenbrock Sphere Ackley
    do
        for agent in bc td3_bc cql awac edac lb_sac sac_n iql dt
        do
            python -W ignore plotting.py --data_dir $root_dir/ToySGD/$teacher/$id/$function/ --agent_path results/$agent/$fidelity --action --num_runs 0 --teacher
        done
    done
done