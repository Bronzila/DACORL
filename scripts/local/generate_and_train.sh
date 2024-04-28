source activate DACORL
id=0
root_dir=data/new_debug_agents
fidelity=1000
num_runs=100
state_version=extended_velocity

for teacher in exponential_decay step_decay sgdr constant
do
    for function in Ackley Rastrigin Rosenbrock Sphere
    do
        for agent in bc td3_bc cql awac edac lb_sac sac_n
        do
            python -W ignore data_gen.py --results_dir $root_dir --env $function\_$state_version --agent $teacher --num_runs $num_runs  --save_run_data --save_rep_buffer
            python -W ignore train.py --data_dir $root_dir/ToySGD/$teacher/$id/$function/ --agent_type $agent --num_train_iter $fidelity --num_eval_runs $num_runs --val_freq $fidelity  --no-use_wandb
        done
    done
done