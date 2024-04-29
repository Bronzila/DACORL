# source activate MTORL-DAC

agent=exponential_decay
# env=Ackley_default
# num_runs=100

for id in {0..10}
do
    msub data_gen.sh $id
done