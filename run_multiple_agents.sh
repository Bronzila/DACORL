source activate MTORL-DAC

agent=exponential_decay
env=Ackley_default
num_runs=100

for id in {0..10}
do
    python3.10 data_gen.py --agent $agent --env $env --save_rep_buffer --save_run_data --num_runs $num_runs --id $id
done