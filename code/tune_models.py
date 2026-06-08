# %% Tune shortlisted models and save the final pipeline
# sbatch -J parallel.tune_models -p long --mem=250G --cpus-per-task=11 --output=%x.log.out --error=%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name cytof_annotation_parallel"

# sbatch -J tune_models_progb_vs_nonprogb_selectkbest_p005_top5 -p long --mem=100G --cpus-per-task=11 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name progb_vs_nonprogb_selectkbest_p005_top5 --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/test.csv --target-column is_progb"

# sbatch -J tune_models_progb_vs_nonprogb_selectkbest_p005_nocorr_smartcorr_c09 -p long --mem=100G --cpus-per-task=11 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09 --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/test.csv --target-column is_progb"

# sbatch -J tune_models_progb_vs_nonprogb_selectkbest_p005_nocorr_summary -p long --mem=100G --cpus-per-task=11 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name progb_vs_nonprogb_selectkbest_p005_no_correlated_selection --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/test.csv --target-column is_progb"

# sbatch -J tune_models_progb_vs_nonprogb_sfp_auc_nocorr_summary -p long --mem=300G --cpus-per-task=11 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/test.csv --target-column is_progb"

# sbatch -J tune_models_progb_vs_nonprogb_sfp_auc_nocorr_smartcorr_c09 -p long --mem=300G --cpus-per-task=11 --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection_smartcorrelation_c09 --train-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection_smartcorrelation_c09/train.csv --test-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection_smartcorrelation_c09/test.csv --target-column is_progb"

import argparse
import joblib
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
import time

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

# %% Parameters

top_models_to_tune = [
    "XGBClassifier",
    "AdaBoostClassifier",
    "RandomForestClassifier",
    "LinearSVC",
    "LinearDiscriminantAnalysis",
    "LogisticRegression"
]

random_state = 123
test_size = 0.2
cv_folds = 5
optuna_n_trials = 30
scoring_metric = "roc_auc" #"roc_auc_ovr"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-jobs",
    dest="n_jobs",
    type=int,
    default=1,
    help="Number of Optuna trials to run in parallel within each model family.",
)
parser.add_argument(
    "--mlflow-experiment-name",
    dest="mlflow_experiment_name",
    type=str,
    required=True,
    help="MLflow experiment name. Output files are written to results/tune_models/<experiment_name>.",
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
    default="label",
    help="Target column name in train and test CSVs.",
)
parser.add_argument(
    "--ci-bootstrap-n",
    type=int,
    default=200,
    help="Number of held-out test bootstrap resamples used for score CIs.",
)
parser.add_argument(
    "--ci-alpha",
    type=float,
    default=0.05,
    help="Alpha for confidence intervals. Default 0.05 gives 95%% intervals.",
)
args = parser.parse_args()

n_jobs = args.n_jobs
if n_jobs < 1:
    raise ValueError("n_jobs must be at least 1")
ci_bootstrap_n = args.ci_bootstrap_n
if ci_bootstrap_n < 1:
    raise ValueError("ci_bootstrap_n must be at least 1")
ci_alpha = args.ci_alpha
if not 0 < ci_alpha < 1:
    raise ValueError("ci_alpha must be between 0 and 1")

mlflow_experiment_name = args.mlflow_experiment_name
out_dir = os.path.join("..", "results", "tune_models", mlflow_experiment_name)
os.makedirs(out_dir, exist_ok=True)
# MLflow tracking uses Path(...).as_uri(), which requires an absolute path.
# Keeping out_dir absolute also makes printed/logged artifact paths unambiguous.
out_dir = os.path.abspath(out_dir)
tracking_dir = os.path.join(out_dir, "mlruns")
os.makedirs(tracking_dir, exist_ok=True)
mlflow_tracking_uri = Path(tracking_dir).as_uri()
train_path = args.train_path
test_path = args.test_path
target_column = args.target_column

tables_dir = os.path.join(out_dir, "tables")
plots_dir = os.path.join(out_dir, "plots")
calibrated_models_dir = os.path.join(out_dir, "calibrated_models")
uncalibrated_models_dir = os.path.join(out_dir, "uncalibrated_models")
cv_fold_model_root_dir = os.path.join(out_dir, "cv_fold_models")

for output_subdir in [
    tables_dir,
    plots_dir,
    calibrated_models_dir,
    uncalibrated_models_dir,
    cv_fold_model_root_dir,
]:
    os.makedirs(output_subdir, exist_ok=True)

optuna_comparison_path = os.path.join(tables_dir, "optuna_model_comparison.csv")
model_score_confidence_intervals_path = os.path.join(
    tables_dir,
    "model_score_confidence_intervals.csv",
)
plot_model_comparison_path = os.path.join(plots_dir, "plot_model_comparison.png")
plot_model_comparison_confidence_path = os.path.join(
    plots_dir,
    "plot_model_comparison_confidence.png",
)
plot_cv_score_spread_path = os.path.join(plots_dir, "plot_cv_score_spread.png")

# Set to False if preprocessing can produce a sparse matrix, for example after one-hot encoding.
stdscaler_with_mean = True

MODEL_NAME_MAP = {
    "LogisticRegression": "logreg",
    "LinearSVC": "linearsvc",
    "LinearDiscriminantAnalysis": "lda",
    "SVC": "svc",
    "XGBClassifier": "xgb",
    "RandomForestClassifier": "rf",
    "AdaBoostClassifier": "adaboost",
    "ExtraTreesClassifier": "extratrees",
    "LGBMClassifier": "lgbm",
    "CatBoostClassifier": "catboost",
}

# %% Prepare data

if train_path is None and test_path is None:
    data_source = "breast_cancer_debug"
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
elif train_path is not None and test_path is not None:
    data_source = "external_csv"
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    if target_column not in train_df.columns:
        raise ValueError(f"target_column '{target_column}' not found in train data")
    if target_column not in test_df.columns:
        raise ValueError(f"target_column '{target_column}' not found in test data")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
else:
    raise ValueError("train_path and test_path must either both be set or both be None")

numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median"))]
)

categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

def build_pipeline(trial, model_name, fit_label=None):
    if model_name == "logreg":
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            penalty="l2",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler(with_mean=stdscaler_with_mean)),
                ("model", model),
            ]
        )

    if model_name == "linearsvc":
        model = LinearSVC(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            max_iter=5000,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler(with_mean=stdscaler_with_mean)),
                ("model", model),
            ]
        )

    if model_name == "lda":
        solver = trial.suggest_categorical("solver", ["svd", "lsqr"])
        shrinkage = None
        if solver == "lsqr":
            shrinkage = trial.suggest_categorical("shrinkage", [None, "auto"])
        model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler(with_mean=stdscaler_with_mean)),
                ("model", model),
            ]
        )

    if model_name == "svc":
        model = SVC(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            probability=True,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler(with_mean=stdscaler_with_mean)),
                ("model", model),
            ]
        )

    if model_name == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            random_state=random_state,
            n_jobs=1,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "adaboost":
        model = AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 2.0, log=True),
            random_state=random_state,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            bootstrap=trial.suggest_categorical("bootstrap", [False, True]),
            random_state=random_state,
            n_jobs=1,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "lgbm":
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            num_leaves=trial.suggest_int("num_leaves", 16, 256),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            n_jobs=1,
            random_state=random_state,
            verbose=-1,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "catboost":
        catboost_info_dir = os.path.join(out_dir, "catboost_info")
        os.makedirs(catboost_info_dir, exist_ok=True)
        if fit_label is not None:
            catboost_train_dir = os.path.join(catboost_info_dir, fit_label)
        elif hasattr(trial, "number"):
            catboost_train_dir = os.path.join(
                catboost_info_dir,
                f"trial_{trial.number}",
            )
        else:
            catboost_train_dir = os.path.join(catboost_info_dir, "run")
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 500),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            random_state=random_state,
            thread_count=1,
            train_dir=catboost_train_dir,
            verbose=0,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    raise ValueError(f"Unsupported model_name: {model_name}")


class FixedTrial:
    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, low, high, log=False):
        return self.params[name]

    def suggest_int(self, name, low, high, log=False):
        return self.params[name]

    def suggest_categorical(self, name, choices):
        return self.params[name]


def log_data_source_params():
    mlflow.log_param("data_source", data_source)
    if data_source == "external_csv":
        mlflow.log_param("train_path", train_path)
        mlflow.log_param("test_path", test_path)
        mlflow.log_param("target_column", target_column)


def log_metric_if_valid(name, value):
    if not np.isnan(value):
        mlflow.log_metric(name, float(value))


def get_model_scores_for_metric(model, X, scoring_metric, num_classes):
    pred = np.asarray(model.predict(X)).reshape(-1)
    proba = None
    proba_or_score = None

    if scoring_metric in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo"}:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if num_classes == 2:
                proba_or_score = proba[:, 1]
            else:
                proba_or_score = proba

        elif num_classes == 2 and hasattr(model, "decision_function"):
            proba_or_score = np.asarray(model.decision_function(X)).reshape(-1)

    return {
        "pred": pred,
        "proba_or_score": proba_or_score,
        "proba": proba,
    }


def compute_metric_from_predictions(
    y_true,
    pred,
    proba_or_score,
    scoring_metric,
    num_classes,
):
    y_true = np.asarray(y_true)
    pred = None if pred is None else np.asarray(pred).reshape(-1)

    try:
        if scoring_metric in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo"}:
            if proba_or_score is None:
                return np.nan
            if num_classes == 2:
                return roc_auc_score(y_true, proba_or_score)
            multi_class_mode = "ovo" if scoring_metric == "roc_auc_ovo" else "ovr"
            return roc_auc_score(
                y_true,
                proba_or_score,
                multi_class=multi_class_mode,
                average="macro",
            )

        if scoring_metric == "accuracy":
            if pred is None:
                return np.nan
            return accuracy_score(y_true, pred)
    except ValueError:
        return np.nan

    return np.nan


def compute_score(model, X, y, scoring_metric, num_classes):
    model_scores = get_model_scores_for_metric(
        model,
        X,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
    )
    return compute_metric_from_predictions(
        y,
        model_scores["pred"],
        model_scores["proba_or_score"],
        scoring_metric=scoring_metric,
        num_classes=num_classes,
    )


def bootstrap_metric_ci(
    y_true,
    pred,
    proba_or_score,
    scoring_metric,
    num_classes,
    n_bootstrap,
    alpha,
    random_state,
):
    y_true = np.asarray(y_true)
    pred = None if pred is None else np.asarray(pred).reshape(-1)
    proba_or_score = (
        None
        if proba_or_score is None
        else np.asarray(proba_or_score)
    )
    rng = np.random.default_rng(random_state)
    class_to_indices = {
        class_label: np.where(y_true == class_label)[0]
        for class_label in np.unique(y_true)
    }
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_to_indices.values()
            ]
        )
        rng.shuffle(sampled_indices)

        sampled_pred = None if pred is None else pred[sampled_indices]
        sampled_score = (
            None
            if proba_or_score is None
            else proba_or_score[sampled_indices]
        )
        score = compute_metric_from_predictions(
            y_true[sampled_indices],
            sampled_pred,
            sampled_score,
            scoring_metric=scoring_metric,
            num_classes=num_classes,
        )
        if not np.isnan(score):
            bootstrap_scores.append(float(score))

    if not bootstrap_scores:
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid": 0,
        }

    return {
        "ci_low": float(np.percentile(bootstrap_scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))),
        "n_valid": len(bootstrap_scores),
    }


def percentile_interval(values, alpha):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {
            "score": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_valid": 0,
        }
    return {
        "score": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 100 * alpha / 2)),
        "ci_high": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "n_valid": len(values),
    }


def save_best_param_cv_fold_models(
    best_model_name,
    best_params,
    X_train,
    y_train,
    cv,
    scoring_metric,
    num_classes,
    cv_fold_model_dir,
    cv_fold_model_summary_path,
    cv_fold_assignments_path,
):
    os.makedirs(cv_fold_model_dir, exist_ok=True)
    summary_rows = []
    assignment_rows = []

    for fold_idx, (train_indices, validation_indices) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        fold_pipeline = build_pipeline(
            FixedTrial(best_params),
            best_model_name,
            fit_label=f"cv_fold_{fold_idx}",
        )
        X_fold_train = X_train.iloc[train_indices]
        y_fold_train = y_train.iloc[train_indices]
        X_fold_validation = X_train.iloc[validation_indices]
        y_fold_validation = y_train.iloc[validation_indices]

        fold_pipeline.fit(X_fold_train, y_fold_train)
        validation_score = compute_score(
            fold_pipeline,
            X_fold_validation,
            y_fold_validation,
            scoring_metric=scoring_metric,
            num_classes=num_classes,
        )

        model_path = os.path.join(cv_fold_model_dir, f"fold_{fold_idx}.pkl")
        joblib.dump(fold_pipeline, model_path)

        summary_rows.append(
            {
                "fold": fold_idx,
                "model_path": model_path,
                "model_family": best_model_name,
                "scoring_metric": scoring_metric,
                "validation_score": validation_score,
                "n_train_rows": len(train_indices),
                "n_validation_rows": len(validation_indices),
            }
        )

        assignment_rows.extend(
            {
                "row_index": str(X_train.index[row_index]),
                "fold": fold_idx,
                "role": "train",
            }
            for row_index in train_indices
        )
        assignment_rows.extend(
            {
                "row_index": str(X_train.index[row_index]),
                "fold": fold_idx,
                "role": "validation",
            }
            for row_index in validation_indices
        )

    cv_fold_model_summary_df = pd.DataFrame(summary_rows)
    cv_fold_assignments_df = pd.DataFrame(assignment_rows)
    cv_fold_model_summary_df.to_csv(cv_fold_model_summary_path, index=False)
    cv_fold_assignments_df.to_csv(cv_fold_assignments_path, index=False)

    return cv_fold_model_summary_df, cv_fold_assignments_df


def plot_model_comparison(results_df, scoring_metric, out_path):
    plot_df = results_df.copy()
    score_columns = [
        ("best_score", "selection_cv", "#4C78A8"),
        ("uncalibrated_test_score", "uncalibrated_test", "#F58518"),
        ("calibrated_test_score", "calibrated_test", "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(plot_df))
    bar_width = 0.25

    for score_idx, (score_column, label, color) in enumerate(score_columns):
        offsets = x_positions + (score_idx - 1) * bar_width
        values = plot_df[score_column].to_numpy(dtype=float)
        ax.bar(offsets, values, width=bar_width, label=label, color=color)

        for x_pos, value in zip(offsets, values):
            if not np.isnan(value):
                ax.text(
                    x_pos,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["model_name"], rotation=45, ha="right")
    ax.set_xlabel("Model family")
    ax.set_ylabel(f"Score: {scoring_metric}")
    ax.set_title("Model comparison: CV selection vs held-out test")
    ax.legend()

    best_idx = plot_df["best_score"].idxmax()
    best_name = plot_df.loc[best_idx, "model_name"]
    best_score_value = plot_df.loc[best_idx, "best_score"]
    ax.axhline(best_score_value, color="darkorange", linestyle="--", linewidth=1)
    ax.text(
        0.02,
        0.98,
        f"Selected model: {best_name}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_model_comparison_confidence(confidence_df, scoring_metric, out_path):
    plot_df = confidence_df[
        confidence_df["score_source"].isin(
            ["selection_cv", "uncalibrated_test", "calibrated_test"]
        )
    ].copy()
    score_sources = [
        ("selection_cv", "selection_cv", "#4C78A8"),
        ("uncalibrated_test", "uncalibrated_test", "#F58518"),
        ("calibrated_test", "calibrated_test", "#54A24B"),
    ]
    model_names = list(plot_df["model_name"].drop_duplicates())

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(model_names))
    bar_width = 0.25

    for score_idx, (score_source, label, color) in enumerate(score_sources):
        source_df = (
            plot_df[plot_df["score_source"] == score_source]
            .set_index("model_name")
            .reindex(model_names)
        )
        offsets = x_positions + (score_idx - 1) * bar_width
        values = source_df["score"].to_numpy(dtype=float)
        ci_low = source_df["ci_low"].to_numpy(dtype=float)
        ci_high = source_df["ci_high"].to_numpy(dtype=float)
        lower_errors = np.maximum(values - ci_low, 0.0)
        upper_errors = np.maximum(ci_high - values, 0.0)
        yerr = np.vstack(
            [
                np.where(np.isnan(lower_errors), 0.0, lower_errors),
                np.where(np.isnan(upper_errors), 0.0, upper_errors),
            ]
        )

        ax.bar(
            offsets,
            values,
            width=bar_width,
            yerr=yerr,
            capsize=3,
            label=label,
            color=color,
        )

        for x_pos, value in zip(offsets, values):
            if not np.isnan(value):
                ax.text(
                    x_pos,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_xlabel("Model family")
    ax.set_ylabel(f"Score: {scoring_metric}")
    ax.set_title("Model comparison with confidence intervals")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cv_score_spread(results_df, scoring_metric, out_path):
    cv_plot_rows = []
    for _, row in results_df.iterrows():
        for fold_idx, fold_score in enumerate(row["cv_scores"], start=1):
            cv_plot_rows.append(
                {
                    "model_name": row["model_name"],
                    "fold": fold_idx,
                    "score": fold_score,
                }
            )

    cv_plot_df = pd.DataFrame(cv_plot_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = list(cv_plot_df["model_name"].unique())

    for x_pos, model_name in enumerate(model_names):
        group = cv_plot_df[cv_plot_df["model_name"] == model_name]
        ax.scatter(
            [x_pos] * len(group),
            group["score"],
            alpha=0.8,
            color="steelblue",
        )
        ax.hlines(
            group["score"].mean(),
            x_pos - 0.25,
            x_pos + 0.25,
            colors="black",
            linewidth=2,
        )

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_xlabel("Model family")
    ax.set_ylabel(f"Fold CV score: {scoring_metric}")
    ax.set_title("Cross-validation score spread by model")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_model_scores(
    best_model_name,
    selection_cv_score,
    uncalibrated_test_score,
    calibrated_test_score,
    scoring_metric,
    out_path,
):
    score_summary_df = pd.DataFrame(
        {
            "dataset": [
                "selection_cv",
                "uncalibrated_test",
                "calibrated_test",
            ],
            "score": [
                selection_cv_score,
                uncalibrated_test_score,
                calibrated_test_score,
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        score_summary_df["dataset"],
        score_summary_df["score"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax.set_ylabel(f"Score: {scoring_metric}")
    ax.set_title(f"Selection vs held-out performance for {best_model_name}")
    ax.tick_params(axis="x", rotation=20)

    for i, value in enumerate(score_summary_df["score"]):
        if not np.isnan(value):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarise_calibration(
    model,
    X,
    y,
    scoring_metric,
    num_classes,
    bin_edges,
    thresholds,
    ci_bootstrap_n,
    ci_alpha,
    random_state,
):
    y_series = pd.Series(y).reset_index(drop=True)
    model_scores = get_model_scores_for_metric(
        model,
        X,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
    )
    pred = pd.Series(model_scores["pred"]).reset_index(drop=True)
    score_value = compute_metric_from_predictions(
        y,
        model_scores["pred"],
        model_scores["proba_or_score"],
        scoring_metric=scoring_metric,
        num_classes=num_classes,
    )
    score_ci = bootstrap_metric_ci(
        y,
        model_scores["pred"],
        model_scores["proba_or_score"],
        scoring_metric=scoring_metric,
        num_classes=num_classes,
        n_bootstrap=ci_bootstrap_n,
        alpha=ci_alpha,
        random_state=random_state,
    )

    if model_scores["proba"] is None:
        bin_rows = [
            {
                "bin_lower": float(bin_edges[bin_idx]),
                "bin_upper": float(bin_edges[bin_idx + 1]),
                "count": 0,
                "mean_confidence": np.nan,
                "empirical_accuracy": np.nan,
            }
            for bin_idx in range(len(bin_edges) - 1)
        ]
        threshold_rows = [
            {
                "threshold": float(threshold),
                "retained_count": np.nan,
                "retained_fraction": np.nan,
                "retained_accuracy": np.nan,
            }
            for threshold in thresholds
        ]
        summary = {
            "score": float(score_value) if not np.isnan(score_value) else np.nan,
            "score_ci_low": score_ci["ci_low"],
            "score_ci_high": score_ci["ci_high"],
            "score_ci_method": "stratified_test_bootstrap",
            "score_ci_n_valid": score_ci["n_valid"],
            "log_loss": np.nan,
            "ece": np.nan,
        }
        return (
            summary,
            pd.DataFrame(bin_rows),
            pd.DataFrame(threshold_rows),
        )

    proba = model_scores["proba"]
    confidence = proba.max(axis=1)
    correct = (pred.values == y_series.values).astype(float)
    log_loss_value = log_loss(y_series, proba, labels=list(model.classes_))

    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=False)
    bin_rows = []
    ece = 0.0

    for bin_idx in range(len(bin_edges) - 1):
        mask = bin_ids == bin_idx
        count = int(mask.sum())

        if count == 0:
            mean_confidence = np.nan
            empirical_accuracy = np.nan
        else:
            mean_confidence = float(confidence[mask].mean())
            empirical_accuracy = float(correct[mask].mean())
            ece += (count / len(y_series)) * abs(empirical_accuracy - mean_confidence)

        bin_rows.append(
            {
                "bin_lower": float(bin_edges[bin_idx]),
                "bin_upper": float(bin_edges[bin_idx + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
            }
        )

    threshold_rows = []
    for threshold in thresholds:
        mask = confidence >= threshold
        retained_count = int(mask.sum())
        retained_fraction = retained_count / len(y_series)
        retained_accuracy = float(correct[mask].mean()) if retained_count else np.nan
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "retained_count": retained_count,
                "retained_fraction": float(retained_fraction),
                "retained_accuracy": retained_accuracy,
            }
        )

    summary = {
        "score": float(score_value) if not np.isnan(score_value) else np.nan,
        "score_ci_low": score_ci["ci_low"],
        "score_ci_high": score_ci["ci_high"],
        "score_ci_method": "stratified_test_bootstrap",
        "score_ci_n_valid": score_ci["n_valid"],
        "log_loss": float(log_loss_value),
        "ece": float(ece),
    }

    return (
        summary,
        pd.DataFrame(bin_rows),
        pd.DataFrame(threshold_rows),
    )


def plot_reliability_comparison(calibration_bins_df, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"uncalibrated": "#F58518", "calibrated": "#54A24B"}

    for model_version, group in calibration_bins_df.groupby("model_version"):
        plot_df = group.dropna(subset=["mean_confidence", "empirical_accuracy"])
        if plot_df.empty:
            continue

        ax.plot(
            plot_df["mean_confidence"],
            plot_df["empirical_accuracy"],
            marker="o",
            linewidth=2,
            label=model_version,
            color=colors.get(model_version, "steelblue"),
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability comparison on held-out test data")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_objective(model_name):
    def objective(trial):
        pipeline = build_pipeline(trial, model_name)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring_metric,
            n_jobs=1,
            error_score="raise",
        )
        if np.all(np.isnan(scores)):
            raise ValueError(
                f"All CV scores are NaN for model_name={model_name!r} "
                f"with scoring_metric={scoring_metric!r}."
            )
        mean_score = float(np.nanmean(scores))
        trial.set_user_attr("cv_scores", scores.tolist())
        return mean_score

    return objective

# %% ----- MAIN -----

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

overall_start = time.perf_counter()
results = []

for lazy_name in top_models_to_tune:
    if lazy_name not in MODEL_NAME_MAP:
        print(f"Skipping unsupported model from LazyPredict: {lazy_name}")
        continue

    model_name = MODEL_NAME_MAP[lazy_name]

    with mlflow.start_run(run_name=f"optuna_{model_name}"):
        log_data_source_params()
        mlflow.log_param("model_family", model_name)
        mlflow.log_param("lazy_name", lazy_name)
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_param("scoring", scoring_metric)
        mlflow.log_param("optuna_n_trials", optuna_n_trials)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("test_size", test_size)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(model_name),
            n_trials=optuna_n_trials,
            n_jobs=n_jobs,
        )

        mlflow.log_metric("best_score", study.best_value)

        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)

        results.append(
            {
                "lazy_name": lazy_name,
                "model_name": model_name,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "cv_scores": study.best_trial.user_attrs["cv_scores"],
            }
        )

results_df = pd.DataFrame(results).sort_values("best_score", ascending=False)
best_model_name = results_df.iloc[0]["model_name"]
num_classes = len(np.unique(y_train))

bin_edges = np.linspace(0.0, 1.0, 11)
confidence_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
model_metric_rows = []
model_score_ci_rows = []

for _, model_row in results_df.iterrows():
    lazy_name = model_row["lazy_name"]
    model_name = model_row["model_name"]
    best_params = model_row["best_params"]
    selection_cv_score = float(model_row["best_score"])

    uncalibrated_model_path = os.path.join(
        uncalibrated_models_dir,
        f"{model_name}__model_uncalibrated.pkl",
    )
    calibrated_model_path = os.path.join(
        calibrated_models_dir,
        f"{model_name}__model_calibrated.pkl",
    )
    calibration_comparison_path = os.path.join(
        tables_dir,
        f"{model_name}__calibration_comparison.csv",
    )
    calibration_bins_path = os.path.join(
        tables_dir,
        f"{model_name}__calibration_bins.csv",
    )
    confidence_threshold_summary_path = os.path.join(
        tables_dir,
        f"{model_name}__confidence_threshold_summary.csv",
    )
    model_scores_plot_path = os.path.join(
        plots_dir,
        f"{model_name}_model_scores.png",
    )
    reliability_plot_path = os.path.join(
        plots_dir,
        f"{model_name}__reliability_comparison.png",
    )
    cv_fold_model_dir = os.path.join(
        cv_fold_model_root_dir,
        model_name,
        "uncalibrated",
    )
    cv_fold_model_summary_path = os.path.join(
        tables_dir,
        f"{model_name}__cv_fold_model_summary.csv",
    )
    cv_fold_assignments_path = os.path.join(
        tables_dir,
        f"{model_name}__cv_fold_assignments.csv",
    )

    cv_fold_model_summary_df, cv_fold_assignments_df = save_best_param_cv_fold_models(
        best_model_name=model_name,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
        cv_fold_model_dir=cv_fold_model_dir,
        cv_fold_model_summary_path=cv_fold_model_summary_path,
        cv_fold_assignments_path=cv_fold_assignments_path,
    )
    selection_cv_interval = percentile_interval(
        cv_fold_model_summary_df["validation_score"],
        alpha=ci_alpha,
    )
    model_score_ci_rows.append(
        {
            "lazy_name": lazy_name,
            "model_name": model_name,
            "score_source": "selection_cv",
            "score": selection_cv_interval["score"],
            "ci_low": selection_cv_interval["ci_low"],
            "ci_high": selection_cv_interval["ci_high"],
            "ci_method": "cv_fold_percentile",
            "n_valid": selection_cv_interval["n_valid"],
        }
    )

    uncalibrated_pipeline = build_pipeline(
        FixedTrial(best_params),
        model_name,
        fit_label=f"{model_name}_uncalibrated",
    )
    uncalibrated_pipeline.fit(X_train, y_train)

    calibration_pipeline = build_pipeline(
        FixedTrial(best_params),
        model_name,
        fit_label=f"{model_name}_calibrated",
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=calibration_pipeline,
        method="sigmoid",
        cv=cv,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)

    uncalibrated_summary, uncalibrated_bins_df, uncalibrated_thresholds_df = (
        summarise_calibration(
            uncalibrated_pipeline,
            X_test,
            y_test,
            scoring_metric=scoring_metric,
            num_classes=num_classes,
            bin_edges=bin_edges,
            thresholds=confidence_thresholds,
            ci_bootstrap_n=ci_bootstrap_n,
            ci_alpha=ci_alpha,
            random_state=random_state,
        )
    )
    (
        calibrated_summary,
        calibrated_bins_df,
        calibrated_thresholds_df,
    ) = summarise_calibration(
        calibrated_model,
        X_test,
        y_test,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
        bin_edges=bin_edges,
        thresholds=confidence_thresholds,
        ci_bootstrap_n=ci_bootstrap_n,
        ci_alpha=ci_alpha,
        random_state=random_state,
    )
    model_score_ci_rows.extend(
        [
            {
                "lazy_name": lazy_name,
                "model_name": model_name,
                "score_source": "uncalibrated_test",
                "score": uncalibrated_summary["score"],
                "ci_low": uncalibrated_summary["score_ci_low"],
                "ci_high": uncalibrated_summary["score_ci_high"],
                "ci_method": uncalibrated_summary["score_ci_method"],
                "n_valid": uncalibrated_summary["score_ci_n_valid"],
            },
            {
                "lazy_name": lazy_name,
                "model_name": model_name,
                "score_source": "calibrated_test",
                "score": calibrated_summary["score"],
                "ci_low": calibrated_summary["score_ci_low"],
                "ci_high": calibrated_summary["score_ci_high"],
                "ci_method": calibrated_summary["score_ci_method"],
                "n_valid": calibrated_summary["score_ci_n_valid"],
            },
        ]
    )

    calibration_comparison_df = pd.DataFrame(
        [
            {
                "model_family": model_name,
                "model_version": "uncalibrated",
                "selection_cv_score": selection_cv_score,
                "test_score": uncalibrated_summary["score"],
                "test_score_ci_low": uncalibrated_summary["score_ci_low"],
                "test_score_ci_high": uncalibrated_summary["score_ci_high"],
                "test_score_ci_method": uncalibrated_summary["score_ci_method"],
                "test_score_ci_n_valid": uncalibrated_summary["score_ci_n_valid"],
                "log_loss": uncalibrated_summary["log_loss"],
                "ece": uncalibrated_summary["ece"],
            },
            {
                "model_family": model_name,
                "model_version": "calibrated",
                "selection_cv_score": selection_cv_score,
                "test_score": calibrated_summary["score"],
                "test_score_ci_low": calibrated_summary["score_ci_low"],
                "test_score_ci_high": calibrated_summary["score_ci_high"],
                "test_score_ci_method": calibrated_summary["score_ci_method"],
                "test_score_ci_n_valid": calibrated_summary["score_ci_n_valid"],
                "log_loss": calibrated_summary["log_loss"],
                "ece": calibrated_summary["ece"],
            },
        ]
    )

    calibration_bins_df = pd.concat(
        [
            uncalibrated_bins_df.assign(
                model_family=model_name,
                model_version="uncalibrated",
            ),
            calibrated_bins_df.assign(
                model_family=model_name,
                model_version="calibrated",
            ),
        ],
        ignore_index=True,
    )

    confidence_threshold_summary_df = pd.concat(
        [
            uncalibrated_thresholds_df.assign(
                model_family=model_name,
                model_version="uncalibrated",
            ),
            calibrated_thresholds_df.assign(
                model_family=model_name,
                model_version="calibrated",
            ),
        ],
        ignore_index=True,
    )

    calibration_comparison_df.to_csv(calibration_comparison_path, index=False)
    calibration_bins_df.to_csv(calibration_bins_path, index=False)
    confidence_threshold_summary_df.to_csv(
        confidence_threshold_summary_path,
        index=False,
    )

    joblib.dump(uncalibrated_pipeline, uncalibrated_model_path)
    joblib.dump(calibrated_model, calibrated_model_path)

    plot_model_scores(
        best_model_name=model_name,
        selection_cv_score=selection_cv_score,
        uncalibrated_test_score=uncalibrated_summary["score"],
        calibrated_test_score=calibrated_summary["score"],
        scoring_metric=scoring_metric,
        out_path=model_scores_plot_path,
    )

    plot_reliability_comparison(
        calibration_bins_df=calibration_bins_df,
        out_path=reliability_plot_path,
    )

    model_metric_rows.append(
        {
            "lazy_name": lazy_name,
            "model_name": model_name,
            "uncalibrated_test_score": uncalibrated_summary["score"],
            "uncalibrated_test_score_ci_low": uncalibrated_summary["score_ci_low"],
            "uncalibrated_test_score_ci_high": uncalibrated_summary["score_ci_high"],
            "uncalibrated_test_score_ci_n_valid": uncalibrated_summary["score_ci_n_valid"],
            "uncalibrated_test_score_ci_method": uncalibrated_summary["score_ci_method"],
            "calibrated_test_score": calibrated_summary["score"],
            "calibrated_test_score_ci_low": calibrated_summary["score_ci_low"],
            "calibrated_test_score_ci_high": calibrated_summary["score_ci_high"],
            "calibrated_test_score_ci_n_valid": calibrated_summary["score_ci_n_valid"],
            "calibrated_test_score_ci_method": calibrated_summary["score_ci_method"],
            "uncalibrated_test_log_loss": uncalibrated_summary["log_loss"],
            "calibrated_test_log_loss": calibrated_summary["log_loss"],
            "uncalibrated_test_ece": uncalibrated_summary["ece"],
            "calibrated_test_ece": calibrated_summary["ece"],
        }
    )

    with mlflow.start_run(run_name=f"model_{model_name}"):
        log_data_source_params()
        mlflow.log_param("model_family", model_name)
        mlflow.log_param("lazy_name", lazy_name)
        mlflow.log_param("selected_by_cv", model_name == best_model_name)
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_param("scoring_metric", scoring_metric)
        mlflow.log_param("calibration_method", "sigmoid")
        mlflow.log_param("calibration_cv_folds", cv.get_n_splits())
        mlflow.log_param("calibration_ensemble", False)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("ci_bootstrap_n", ci_bootstrap_n)
        mlflow.log_param("ci_alpha", ci_alpha)
        for key, value in best_params.items():
            mlflow.log_param(f"best_{key}", value)
        log_metric_if_valid("selection_cv_score", selection_cv_score)
        log_metric_if_valid(
            "selection_cv_refit_score",
            selection_cv_interval["score"],
        )
        log_metric_if_valid(
            "selection_cv_refit_ci_low",
            selection_cv_interval["ci_low"],
        )
        log_metric_if_valid(
            "selection_cv_refit_ci_high",
            selection_cv_interval["ci_high"],
        )
        log_metric_if_valid("uncalibrated_test_score", uncalibrated_summary["score"])
        log_metric_if_valid(
            "uncalibrated_test_score_ci_low",
            uncalibrated_summary["score_ci_low"],
        )
        log_metric_if_valid(
            "uncalibrated_test_score_ci_high",
            uncalibrated_summary["score_ci_high"],
        )
        log_metric_if_valid(
            "uncalibrated_test_score_ci_n_valid",
            uncalibrated_summary["score_ci_n_valid"],
        )
        log_metric_if_valid("calibrated_test_score", calibrated_summary["score"])
        log_metric_if_valid(
            "calibrated_test_score_ci_low",
            calibrated_summary["score_ci_low"],
        )
        log_metric_if_valid(
            "calibrated_test_score_ci_high",
            calibrated_summary["score_ci_high"],
        )
        log_metric_if_valid(
            "calibrated_test_score_ci_n_valid",
            calibrated_summary["score_ci_n_valid"],
        )
        log_metric_if_valid(
            "uncalibrated_test_log_loss",
            uncalibrated_summary["log_loss"],
        )
        log_metric_if_valid("calibrated_test_log_loss", calibrated_summary["log_loss"])
        log_metric_if_valid("uncalibrated_test_ece", uncalibrated_summary["ece"])
        log_metric_if_valid("calibrated_test_ece", calibrated_summary["ece"])

        mlflow.log_artifact(uncalibrated_model_path)
        mlflow.log_artifact(calibrated_model_path)
        mlflow.log_artifacts(
            cv_fold_model_dir,
            artifact_path=f"cv_fold_models/{model_name}/uncalibrated",
        )
        mlflow.log_artifact(cv_fold_model_summary_path)
        mlflow.log_artifact(cv_fold_assignments_path)
        mlflow.log_artifact(calibration_comparison_path)
        mlflow.log_artifact(calibration_bins_path)
        mlflow.log_artifact(confidence_threshold_summary_path)
        mlflow.log_artifact(model_scores_plot_path)
        mlflow.log_artifact(reliability_plot_path)
        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path=f"{model_name}_sklearn_model",
        )

model_metrics_df = pd.DataFrame(model_metric_rows)
results_df = results_df.merge(
    model_metrics_df,
    on=["lazy_name", "model_name"],
    how="left",
).sort_values("best_score", ascending=False)
results_df.to_csv(optuna_comparison_path, index=False)

model_score_ci_df = pd.DataFrame(model_score_ci_rows)
model_order = {
    model_name: model_idx
    for model_idx, model_name in enumerate(results_df["model_name"].tolist())
}
score_source_order = {
    "selection_cv": 0,
    "uncalibrated_test": 1,
    "calibrated_test": 2,
}
model_score_ci_df["_model_order"] = model_score_ci_df["model_name"].map(model_order)
model_score_ci_df["_score_source_order"] = model_score_ci_df["score_source"].map(
    score_source_order
)
model_score_ci_df = model_score_ci_df.sort_values(
    ["_model_order", "_score_source_order"]
).drop(columns=["_model_order", "_score_source_order"])
model_score_ci_df.to_csv(model_score_confidence_intervals_path, index=False)

plot_model_comparison(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=plot_model_comparison_path,
)

plot_model_comparison_confidence(
    confidence_df=model_score_ci_df,
    scoring_metric=scoring_metric,
    out_path=plot_model_comparison_confidence_path,
)

plot_cv_score_spread(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=plot_cv_score_spread_path,
)

with mlflow.start_run(run_name="model_comparison_summary"):
    log_data_source_params()
    mlflow.log_artifact(optuna_comparison_path)
    mlflow.log_artifact(model_score_confidence_intervals_path)
    mlflow.log_artifact(plot_model_comparison_path)
    mlflow.log_artifact(plot_model_comparison_confidence_path)
    mlflow.log_artifact(plot_cv_score_spread_path)

print(results_df)

best_model_row = results_df.iloc[0]

overall_duration = time.perf_counter() - overall_start

print("Best model:", best_model_row["model_name"])
print("n_jobs:", n_jobs)
print("Selection CV score:", best_model_row["best_score"])
print("Uncalibrated test score:", best_model_row["uncalibrated_test_score"])
print("Calibrated test score:", best_model_row["calibrated_test_score"])
print("Uncalibrated test log loss:", best_model_row["uncalibrated_test_log_loss"])
print("Calibrated test log loss:", best_model_row["calibrated_test_log_loss"])
print("Uncalibrated test ECE:", best_model_row["uncalibrated_test_ece"])
print("Calibrated test ECE:", best_model_row["calibrated_test_ece"])
print("Saved model comparison to", optuna_comparison_path)
print("Saved CV fold models to", cv_fold_model_root_dir)
print(
    f"Total runtime (seconds): {overall_duration:.2f} "
    f"({overall_duration / 60:.2f} minutes)"
)
