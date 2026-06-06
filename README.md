# Tabular ML Workflow

This repository provides a workflow for applying machine learning to tabular data: benchmarking candidate model architectures, tuning and selecting a final model, calibrating probabilities for inference, and generating feature-importance explanations for trained models.

## Workflow

- `benchmark_models.py` screens candidate model families and compares baseline performance to identify strong model architectures.
- `select_features.py` selects features from labeled train and test tables, with optional one-run bootstrapping of training rows.
- `run_select_features_bootstrap_array.sh` runs repeated bootstrapped feature-selection replicates as a Slurm array.
- `summarise_bootstrap_features.py` summarises bootstrap-selected features, applies a selection-frequency threshold, and writes final reduced train and test tables.
- `tune_models.py` tunes shortlisted models with Optuna, selects the best-performing model, saves final uncalibrated and calibrated models, and saves best-parameter uncalibrated CV fold models for downstream explanation.
- `explain_model.py` generates SHAP-based explanations from either a single `--model-path` or a fold-model directory for averaged SHAP explanations.

## Data contract

- `train.csv` and `test.csv` are labeled tabular datasets.
- The first column is treated as the row or sample index (`index_col=0`).
- Both files must contain the label column, which is named `label` by default.
- All remaining columns are treated as input features, and `train.csv` and `test.csv` should share the same feature schema.
- `tune_models.py` supports both numeric and categorical feature columns through preprocessing; `benchmark_models.py` expects the same table format, but categorical handling is not explicitly implemented there.

## Components

- scikit-learn pipelines for preprocessing and model workflows
- Optuna for automated hyperparameter tuning
- MLflow for tracking tuned models, fold models, metrics, and artifacts
- SHAP for single-model and fold-averaged feature-level explanations

## Potential applications

- Classification from structured assay or measurement data
- Phenotype, condition, or outcome prediction
- Sample, cell-type, or subtype annotation
- Any tabular classification problem where interpretability matters
