# Tabular ML Workflow

This repository provides a workflow for applying machine learning to tabular data: benchmarking candidate model architectures, tuning and selecting a final model, calibrating probabilities for inference, and generating feature-importance explanations for trained models.

## Workflow

- `benchmark_models.py` screens candidate model families and compares baseline performance to identify strong model architectures.
- `tune_models.py` tunes shortlisted models with Optuna, selects the best-performing model, calibrates probabilities, and saves the final model and run outputs.
- `explain_model.py` generates SHAP-based explanations to identify important features and to assess whether learned feature-prediction relationships are consistent with prior knowledge.

## Potential applications

- Classification from structured assay or measurement data
- Phenotype, condition, or outcome prediction
- Sample, cell-type, or subtype annotation
- Any tabular classification problem where interpretability matters

## Components

- scikit-learn pipelines for preprocessing and model workflows
- Optuna for automated hyperparameter tuning
- MLflow for experiment tracking
- SHAP for feature-level explanations
