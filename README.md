# DACORL

This repository contains work on applying multi-teacher offline reinforcement learning to dynamically adjust the learning rate during neural network optimization.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Development](#development)
3.1 [Contributing](#contributing)
3.2 [Pre-commit](#pre-commit)

## Installation

It is recommended to create a new Conda environment before installing the repository.

```bash
conda create -n DACORL python=3.10
conda activate DACORL

git clone --recurse-submodules https://github.com/Bronzila/DACORL.git && cd DACORL
pip install -r CORL/requirements/requirements_dev.txt
pip install -e DACBench[all,dev]
pip install -e .
```

## Usage

To generate a dataset, train agents, and evaluate them, use the `main.py` script. We utilize the [Hydra](https://hydra.cc/) framework for configuration management.

By default, all three tasks (data generation, training, and evaluation) run consecutively. You can separate these tasks by specifying the mode using `mode=data_gen|train|eval`. Note that training will automatically trigger evaluation upon completion.

It is also required to specify the `result_dir` when running any job. You can override default configuration values as needed. Refer to the [`hydra_conf/config.yaml`](hydra_conf/config.yaml) file and other configuration files in the `hydra_conf` directory for more details.

```bash
python main.py result_dir=data/test_experiment
```

## Development

### Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes with clear messages.
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

### Pre-commit

To ensure code quality and consistency, we use `pre-commit` for automatic code formatting and linting. Therefore, please install the development dependencies and the `pre-commit` hooks, which will be run automatically before each commit.

```bash
pip install .[dev]

# make sure that you are in DACORL/
pre-commit install
```
