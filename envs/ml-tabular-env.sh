#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="ml-tabular-env"

mamba create -y -n "${ENV_NAME}" python=3.12
mamba run -n "${ENV_NAME}" pip install "flaml[automl]"
mamba run -n "${ENV_NAME}" pip install lazypredict[boost]
mamba run -n "${ENV_NAME}" pip install mlflow optuna
mamba run -n "${ENV_NAME}" pip install shap

printf "\nEnvironment created. Activate it with:\n"
printf "mamba activate %s\n" "${ENV_NAME}"
