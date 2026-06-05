# %% Benchmark model architectures
# env: ml-tabular-env
# sbatch -J benchmark_models_progb_vs_nonprogb_selected_models -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python benchmark_models.py --out-dir results/benchmark_models/progb_vs_nonprogb_selected_models"
# sbatch -J gpu_benchmark_models_progb_vs_nonprogb_include_tree_based -p gpu_interactive --gres gpu:1 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python benchmark_models.py --out-dir results/benchmark_models/progb_vs_nonprogb_selected_models_include_tree_based_gpu --use-gpu"
# sbatch -J benchmark_models_progb_vs_nonprogb_include_tree_based -p long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python benchmark_models.py --out-dir results/benchmark_models/progb_vs_nonprogb_selected_models_include_tree_based --use-gpu"
# sbatch -J benchmark_models_progb_vs_nonprogb_selectkbest_p005 -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python benchmark_models.py --out-dir results/benchmark_models/progb_vs_nonprogb_selectkbest_p005 --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/test.csv --target-column is_progb --out-dir results/benchmark_models/progb_vs_nonprogb_selectkbest_p005_selected_models_include_tree_based"
# sbatch -J benchmark_models_progb_vs_nonprogb_selectkbest_p005_all -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python benchmark_models.py --out-dir results/benchmark_models/progb_vs_nonprogb_selectkbest_p005 --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/test.csv --target-column is_progb --out-dir results/benchmark_models/progb_vs_nonprogb_selectkbest_p005"

import argparse
import os
import subprocess
import time

import pandas as pd

from lazypredict.Supervised import LazyClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier,  RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

import xgboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

os.chdir(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        universal_newlines=True,
    ).strip()
)

# %% Parameters

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out-dir",
    default="results/benchmark_models/",
    help="Output directory for benchmark result CSVs.",
)
parser.add_argument(
    "--train-path",
    default="../data/train.csv",
    help="Training CSV path. Read with index_col=0.",
)
parser.add_argument(
    "--test-path",
    default="../data/test.csv",
    help="Test CSV path. Read with index_col=0.",
)
parser.add_argument(
    "--target-column",
    default="is_progb",
    help="Target column name in train and test CSVs.",
)
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Whether to use GPU acceleration for supported models (only for lazypredict).",
)
args = parser.parse_args()

benchmark_method = "lazypredict"    # "lazypredict" or "flaml"
lazypredict_sorter_key = "ROC AUC"  # "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken"
categorical_encoder = "onehot" # "onehot", "ordinal", "target", "binary"
classifiers = "all" #[
#     LinearSVC, LogisticRegression, SGDClassifier, RidgeClassifier, RidgeClassifierCV, LinearDiscriminantAnalysis,
#     RandomForestClassifier, xgboost.XGBClassifier, lightgbm.LGBMClassifier,
#     #RandomForestClassifier, DecisionTreeClassifier GradientBoostingClassifier,
#     DummyClassifier,
# ] # "all" to use all available classifiers, or a list of specific classifiers to benchmark (e.g. [RandomForestClassifier, DecisionTreeClassifier])
use_gpu = args.use_gpu  # Only for lazypredict and supported models, not for flaml

target_column = args.target_column
train_path = args.train_path
test_path = args.test_path

out_dir = args.out_dir

# %% ----- MAIN -----

os.makedirs(out_dir, exist_ok=True)

# %% Load data

train_df = pd.read_csv(train_path, index_col=0)
test_df = pd.read_csv(test_path, index_col=0)

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# %% Benchmark models

if benchmark_method == "lazypredict":

    clf = LazyClassifier(
        verbose=1,                          # Show progress
        ignore_warnings=True,               # Suppress warnings
        custom_metric=None,                 # Use default metrics
        predictions=True,                   # Return predictions
        # See https://github.com/shankarpandala/lazypredict/issues/346 to pass custom classifiers
        # e.g. classifiers=[RandomForestClassifier, DecisionTreeClassifier]
        classifiers=classifiers,            # List of classifiers to benchmark (default: all available)
        categorical_encoder=categorical_encoder, # Encoding: "onehot" (default), "ordinal", "target", "binary"
        timeout=60,                         # Max time per model in seconds
        cv=5,                               # Cross-validation folds (optional)
        use_gpu=use_gpu                     # Enable GPU acceleration
    )
    lazypredict_start_time = time.perf_counter()
    print("Starting LazyPredict fit...", flush=True)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    lazypredict_elapsed_seconds = time.perf_counter() - lazypredict_start_time
    print(
        f"LazyPredict fit completed in {lazypredict_elapsed_seconds:.2f} seconds "
        f"({lazypredict_elapsed_seconds / 60:.2f} minutes).",
        flush=True,
    )

    # Save model x performance metrics
    models = models.sort_values(by=lazypredict_sorter_key, ascending=False)
    print(models)
    models.reset_index().to_csv(
        os.path.join(out_dir, "lazypredict_results.csv"), index=False
    )

    # Save predictions (sample x model)
    predictions.to_csv(os.path.join(out_dir, "lazypredict_predictions.csv"), index=False)

elif benchmark_method == "flaml":
    raise NotImplementedError(
        "FLAML workflow is still to be implemented and tested in benchmark_models.py."
    )

    # %% FLAML - Some code from https://medium.com/@mouse3mic3/a-practical-guide-to-automated-machine-learning-in-python-using-flaml-c73a714887a4

    # Check for using model outside of flaml - https://microsoft.github.io/FLAML/docs/FAQ/

    # import pickle
    # from flaml import AutoML
    # from sklearn.metrics import classification_report
    #
    # model_config = {
    #     "task": "classification",
    #     "time_budget": 300,     # in seconds
    #     "max_iter": 1000,       # num of model fitting allowed
    #     # "metric": sensitivity_metric,   # custom metric function
    #     # "estimator_list": ["lgbm", "histgb", "svm"],  # list of estimators used
    #     "eval_method": "cv",    # use RepeatedKFold
    #     # "split_type": RepeatedKFold(n_splits=3, n_repeats=10),
    #     # "custom_hp": custom_params,
    #     "ensemble": True,       # combine the two (or more) best performing estimators into one, effectively making an ensemble of classifiers
    #     "verbose": 3,
    #     "log_file_name": "training_log",
    #     "log_type": "all",
    # }
    #
    # automl = AutoML()
    # automl.add_learner("svm", SVM) # SVM is custom learner to be defined by user, check link how
    # automl.fit(X_train, y_train, **model_config)
    #
    # # Save Model
    # with open("flaml.pkl", "wb") as f:
    #     pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    #
    # # automl.add_learner("mylgbm", MyLGBMEstimator)
    # # automl.fit(
    # #     X_train=X_train,
    # #     y_train=y_train,
    # #     task="classification",
    # #     #metric=custom_metric,
    # #     estimator_list=["mylgbm"],
    # #     time_budget=60,
    # # )
    #
    # # Best model and best hyperparameters
    # automl.model.estimator
    #
    # # Best configuration per estimator
    # automl.best_config_per_estimator
    #
    # # Display which iteration resulted in the best model and how long it takes to find it.
    # print(
    #     "The ",
    #     automl.best_iteration,
    #     "-th iteration is the best, completed in ",
    #     round(automl.time_to_find_best_model, 1),
    #     " seconds.",
    #     sep="",
    # )
    #
    # # Predict test data
    # y_test_pred = automl.predict(X_test)
    #
    # # Display metrics
    # print(classification_report(y_test, y_test_pred))

else:
    raise ValueError("benchmark_method must be 'lazypredict' or 'flaml'")
