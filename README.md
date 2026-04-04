This is the code for this paper "On the Generalization Properties of Selective State-Space Models for Filtering Tasks for Unknown Systems" by Alex Tang*, M. Emrullah Ildiz*, Batin Kurt, Samet Oymak, and Necmiye Ozay.

# SSMs for Filtering

This repository provides a full experimental pipeline for training and evaluating sequence models on filtering tasks under unknown system dynamics. The code supports both linear time-invariant (LTI) and drone systems, with experiments that cover:

- Standard in-distribution evaluation
- Length generalization
- Colored-noise robustness
- Dynamics-switching robustness

The pipeline compares learned models (Selective SSM, GPT-2, Mamba) against classical filters (KF or EKF depending on the system).

## Repository Structure

- `main.py`: top-level pipeline runner (data generation, training, testing, plotting)
- `configs/`: all experiment and system configs
- `src/data_collection/`: simulation-based dataset generation
- `src/training/`: model training
- `src/testing/`: Monte Carlo evaluation
- `src/plotting/`: plotting RMS error curves
- `src/sequence_models/`: backbone model implementations
- `src/estimators/`: KF and EKF baselines

## Setup

Use Python 3.10+ (recommended).

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Notes:

- `torch` and `mamba-ssm` may require environment-specific installation steps depending on your CUDA/PyTorch stack.
- If you need a CUDA-specific PyTorch build, install it using the official PyTorch instructions before or after the requirements file, depending on your platform.
- Run commands from the repository root (the folder containing `main.py`).

## Configuration Overview

Main execution flags are in `configs/main.yaml`:

- `generate_data_flag`
- `train_flag`
- `test_flag`
- `plot_flag`

Set these booleans to control which stages run when calling `python main.py`.

Key stage configs:

- Data generation: `configs/generate_data.yaml`
- Training: `configs/train.yaml`
- Testing: `configs/test.yaml`
- Plotting: `configs/plot.yaml`

System dynamics/noise parameters:

- LTI: `configs/lti_system.yaml`
- Drone: `configs/drone.yaml`

## Running the Pipeline

Run the full pipeline:

```bash
python main.py
```

Typical workflow:

1. Generate datasets (`generate_data_flag: true`)
2. Train selected models (`train_flag: true`)
3. Evaluate on selected test cases (`test_flag: true`)
4. Plot RMS curves (`plot_flag: true`)

To run only one stage, set the other flags to `false` in `configs/main.yaml`.

## Data Generation

`src/data_collection/generate_training_data.py` generates case-specific datasets for:

- `standard`
- `length_generalization`
- `colored_noise`
- `dynamics_switching`

Output naming follows this pattern:

```text
data/<system>/N<n_traj>-S<n_steps>-<input_tag>-<matrix_tag>[-colored_noise][-dynamics_switching]/<system>_dataset.pkl
```

## Training

`configs/train.yaml` controls:

- Which data variant(s) are used (`standard`, `colored_noise`, `length_generalization`)
- Forecasting setup (`H`, `L`)
- Model list (`models: [ssm, gpt2]` by default)
- Optimization hyperparameters

Trained artifacts are saved under `outputs_dir` (default `outputs`) with run directories encoding model type and settings.

## Testing

`src/testing/test.py` performs Monte Carlo evaluation and saves one pickle per case to `save_dir` (default `test`):

- `lti_test_results_standard.pkl`
- `lti_test_results_length_generalization.pkl`
- `lti_test_results_colored_noise.pkl`
- `lti_test_results_dynamics_switching.pkl`

Each test result includes:

- Baseline RMS over time/horizon (`baseline`)
- Per-model RMS over time/horizon (`model_*` entries)
- Per-trajectory RMS arrays (`rms_ah_by_traj`)
- Simulation settings metadata

## Plotting

`src/plotting/plot_results.py` reads per-case test pickles from `configs/plot.yaml` and generates:

- A combined 2x2 case figure
- Per-case standalone figures

Default output directory is `plots`.

## Supported Systems and Baselines

- `system: lti` uses Kalman Filter (KF) baseline
- `system: drone` uses Extended Kalman Filter (EKF) baseline

Choose the system in both `configs/generate_data.yaml` and `configs/test.yaml`.

## Quick Start

1. Set desired system and data paths in config files.
2. In `configs/main.yaml`, enable desired pipeline stages.
3. Run:

```bash
python main.py
```

4. Check outputs in:

- `data` for generated datasets
- `outputs` for trained models
- `test` for serialized test metrics
- `plots` for RMS figures

