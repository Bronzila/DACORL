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

### Multi-Teacher Experiments

> **_NOTE:_**  Multi-Teacher experiments only differ in the way of generating data by generating data for multiple teachers and then combining the generated datasets.

Conducting multi-teacher experiments is also easily possible by using the `main.py` script. Here we differntiate between two teacher combination strategies: `homogeneous` and `heterogeneous`, which can be parameterized using the `combination` configuration field. In the following we will quickly introduce the two different combination strategies and how to use them:

#### Homogeneous

In homogeneous combinations, teachers of the same type (e.g., step decay) but with different configurations (e.g., decay rate of 0.9 over 9 steps) are combined. To run homogeneous combination experiments, please set `combination=homogeneous`. The current implementation generates data using five teachers and then concatenates the datasets.

#### Heterogeneous

In heterogeneous combinations, teachers of varying type and configuration are combined. To run heterogeneous experiments, please set `combination=heterogeneous`. To define which (default) teachers you want to combine, please use the `teacher` field. Here we use the following notation:
Teacher types are abbreviated:

- E = Exponential decay
- ST = Step decay
- SG = SGDR
- C = Constant

Using these abbreviations you can combine teachers by separating them using a "-". To combine the exponential decay, step decay and constant teacher for example, use `teacher=E-ST-C`.

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
