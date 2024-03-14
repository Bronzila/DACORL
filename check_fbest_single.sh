AGENT=constant
FUNCTION=Ackley

source activate MTORL-DAC

python3.10 check_fbest.py --path data/ToySGD/$AGENT/0/$FUNCTION//results/td3_bc/2000/eval_data.csv --lowest --mean
python3.10 check_fbest.py --path data/ToySGD/$AGENT/0/$FUNCTION//results/td3_bc/4000/eval_data.csv --lowest --mean
python3.10 check_fbest.py --path data/ToySGD/$AGENT/0/$FUNCTION//results/td3_bc/6000/eval_data.csv --lowest --mean
python3.10 check_fbest.py --path data/ToySGD/$AGENT/0/$FUNCTION//results/td3_bc/8000/eval_data.csv --lowest --mean
python3.10 check_fbest.py --path data/ToySGD/$AGENT/0/$FUNCTION//results/td3_bc/10000/eval_data.csv --lowest --mean