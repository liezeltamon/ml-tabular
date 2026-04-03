# Environments

## Contents
- Environment-related files needed for project, for example conda environment YAML files and notes on how to recreate environments

## `ml-tabular-env`

Environment created with:

```bash
mamba create -n ml-tabular-env python=3.12
mamba activate ml-tabular-env
pip install "flaml[automl]"
pip install lazypredict[boost]
pip install mlflow optuna
```

Why not `lazypredict[all]`?

- `lazypredict[all]` pulls in a wider set of optional dependencies
- one of those dependencies tries to install `numba`
- the `numba` version selected during install does not support Python 3.12
- this causes the install to fail with an error saying only Python versions `>=3.8,<3.12` are supported

`lazypredict[boost]` worked on Python 3.12 and is a better fit here if you mainly want boosted tree models in the benchmark shortlist.
