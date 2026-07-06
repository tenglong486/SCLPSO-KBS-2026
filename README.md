# SCLPSO-KBS-2026
Code and data files for the paper "Stabilized Comprehensive Learning Particle Swarm Optimization for Benchmark Optimization and Urban Air-Quality Model Tuning".
# SCLPSO Code and Data

This repository provides the source code and data used in the paper:

**Stabilized Comprehensive Learning Particle Swarm Optimization for Benchmark Optimization and Urban Air-Quality Model Tuning**

## Repository structure

- `code/`: Python scripts for the SCLPSO algorithm, CEC2017 benchmark experiments, ablation experiments, comparative experiments, and city-level air-quality model tuning.
- `data/CEC2017/input_data/`: CEC2017 benchmark input files required by the benchmark implementation.
- `data/dataset/`: city-level air-quality datasets for Beijing, Shanghai, and Hefei.

## Code

The `code` folder contains the main experimental scripts used in this study, including:

- standalone SCLPSO implementation;
- CEC2017 benchmark experiments;
- progressive mechanism analysis;
- comparative experiments with other optimizers;
- SCLPSO-assisted hyperparameter tuning for SVR, RandomForest, ExtraTrees, and MLP models.

## Data

The `data` folder contains:

- CEC2017 benchmark input files;
- city-level air-quality datasets for Beijing, Shanghai, and Hefei.

## Requirements

The main Python packages used in the experiments include:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
netCDF4
tqdm
