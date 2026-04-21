# %% Tune shortlisted models and save the final pipeline
# sbatch -J parallel.tune_models -p long --mem=250G --cpus-per-task=11 --output=%x.log.out --error=%x.log.err --wrap="python tune_models.py --n-jobs 10 --mlflow-experiment-name cytof_annotation_parallel"

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier

# %% Parameters

random_state = 123
test_size = 0.2
cv_folds = 5
optuna_n_trials = 30
scoring_metric = "roc_auc_ovr"

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
args = parser.parse_args()

n_jobs = args.n_jobs
if n_jobs < 1:
    raise ValueError("n_jobs must be at least 1")

mlflow_experiment_name = args.mlflow_experiment_name
out_dir = os.path.join("..", "results", "tune_models", mlflow_experiment_name)
os.makedirs(out_dir, exist_ok=True)
# MLflow tracking uses Path(...).as_uri(), which requires an absolute path.
# Keeping out_dir absolute also makes printed/logged artifact paths unambiguous.
out_dir = os.path.abspath(out_dir)
tracking_dir = os.path.join(out_dir, "mlruns")
os.makedirs(tracking_dir, exist_ok=True)
mlflow_tracking_uri = Path(tracking_dir).as_uri()
train_path = "../data/train.csv"
test_path = "../data/test.csv"
target_column = "label"

optuna_comparison_path = os.path.join(out_dir, "optuna_model_comparison.csv")
calibration_comparison_path = os.path.join(out_dir, "calibration_comparison.csv")
calibration_bins_path = os.path.join(out_dir, "calibration_bins.csv")
confidence_threshold_summary_path = os.path.join(
    out_dir,
    "confidence_threshold_summary.csv",
)
final_model_calibrated_path = os.path.join(out_dir, "final_model_calibrated.pkl")
final_model_uncalibrated_path = os.path.join(out_dir, "final_model_uncalibrated.pkl")
plot_model_comparison_path = os.path.join(out_dir, "plot_model_comparison.png")
plot_cv_score_spread_path = os.path.join(out_dir, "plot_cv_score_spread.png")
plot_final_model_scores_path = os.path.join(out_dir, "plot_final_model_scores.png")
plot_reliability_comparison_path = os.path.join(
    out_dir,
    "plot_reliability_comparison.png",
)

# Set to False if preprocessing can produce a sparse matrix, for example after one-hot encoding.
stdscaler_with_mean = True

top_models_to_tune = [
    #"LinearSVC",
    #"SVC",
    #"XGBClassifier",
    "CatBoostClassifier",
    #"LogisticRegression",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "LGBMClassifier",
]

MODEL_NAME_MAP = {
    "LogisticRegression": "logreg",
    "LinearSVC": "linearsvc",
    "SVC": "svc",
    "XGBClassifier": "xgb",
    "RandomForestClassifier": "rf",
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


def compute_score(model, X, y, scoring_metric, num_classes):
    if scoring_metric in {"roc_auc", "roc_auc_ovr", "roc_auc_ovo"} and hasattr(
        model, "predict_proba"
    ):
        proba = model.predict_proba(X)
        if num_classes == 2:
            return roc_auc_score(y, proba[:, 1])
        multi_class_mode = "ovo" if scoring_metric == "roc_auc_ovo" else "ovr"
        return roc_auc_score(y, proba, multi_class=multi_class_mode, average="macro")

    if scoring_metric == "accuracy":
        pred = np.asarray(model.predict(X)).reshape(-1)
        return accuracy_score(y, pred)

    return np.nan


def plot_model_comparison(results_df, scoring_metric, out_path):
    plot_df = results_df.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["model_name"], plot_df["best_score"], color="steelblue")
    ax.set_xlabel("Model family")
    ax.set_ylabel(f"Best CV score: {scoring_metric}")
    ax.set_title("Model comparison")
    ax.tick_params(axis="x", rotation=45)

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


def plot_final_model_scores(
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


def summarise_calibration(model, X, y, scoring_metric, num_classes, bin_edges, thresholds):
    y_series = pd.Series(y).reset_index(drop=True)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    pred = np.asarray(pred).reshape(-1)
    pred = pd.Series(pred).reset_index(drop=True)
    confidence = proba.max(axis=1)
    correct = (pred.values == y_series.values).astype(float)

    score_value = compute_score(
        model,
        X,
        y,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
    )
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
        )
        mean_score = float(np.mean(scores))
        trial.set_user_attr("cv_scores", scores.tolist())
        return mean_score

    return objective

# %% ----- MAIN -----

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(mlflow_experiment_name)

overall_start = time.perf_counter()
results = []
studies = {}

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

        studies[model_name] = study
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
results_df.to_csv(optuna_comparison_path, index=False)

plot_model_comparison(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=plot_model_comparison_path,
)

plot_cv_score_spread(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=plot_cv_score_spread_path,
)

with mlflow.start_run(run_name="model_comparison_summary"):
    log_data_source_params()
    mlflow.log_artifact(optuna_comparison_path)
    mlflow.log_artifact(plot_model_comparison_path)
    mlflow.log_artifact(plot_cv_score_spread_path)

print(results_df)

best_model_name = results_df.iloc[0]["model_name"]
best_lazy_name = results_df.iloc[0]["lazy_name"]
best_study = studies[best_model_name]
num_classes = len(np.unique(y_train))

uncalibrated_final_pipeline = build_pipeline(
    FixedTrial(best_study.best_params),
    best_model_name,
    fit_label="final_uncalibrated",
)
uncalibrated_final_pipeline.fit(X_train, y_train)

best_pipeline = build_pipeline(
    FixedTrial(best_study.best_params),
    best_model_name,
    fit_label="final_calibrated",
)
calibrated_model = CalibratedClassifierCV(
    estimator=best_pipeline,
    method="sigmoid",
    cv=cv,
    ensemble=False,
)
calibrated_model.fit(X_train, y_train)

selection_cv_score = float(best_study.best_value)
uncalibrated_test_score = compute_score(
    uncalibrated_final_pipeline,
    X_test,
    y_test,
    scoring_metric=scoring_metric,
    num_classes=num_classes,
)
calibrated_test_score = compute_score(
    calibrated_model,
    X_test,
    y_test,
    scoring_metric=scoring_metric,
    num_classes=num_classes,
)

bin_edges = np.linspace(0.0, 1.0, 11)
confidence_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

uncalibrated_summary, uncalibrated_bins_df, uncalibrated_thresholds_df = (
    summarise_calibration(
        uncalibrated_final_pipeline,
        X_test,
        y_test,
        scoring_metric=scoring_metric,
        num_classes=num_classes,
        bin_edges=bin_edges,
        thresholds=confidence_thresholds,
    )
)
calibrated_summary, calibrated_bins_df, calibrated_thresholds_df = summarise_calibration(
    calibrated_model,
    X_test,
    y_test,
    scoring_metric=scoring_metric,
    num_classes=num_classes,
    bin_edges=bin_edges,
    thresholds=confidence_thresholds,
)

calibration_comparison_df = pd.DataFrame(
    [
        {
            "model_version": "uncalibrated",
            "selection_cv_score": selection_cv_score,
            "test_score": uncalibrated_summary["score"],
            "log_loss": uncalibrated_summary["log_loss"],
            "ece": uncalibrated_summary["ece"],
        },
        {
            "model_version": "calibrated",
            "selection_cv_score": selection_cv_score,
            "test_score": calibrated_summary["score"],
            "log_loss": calibrated_summary["log_loss"],
            "ece": calibrated_summary["ece"],
        },
    ]
)

calibration_bins_df = pd.concat(
    [
        uncalibrated_bins_df.assign(model_version="uncalibrated"),
        calibrated_bins_df.assign(model_version="calibrated"),
    ],
    ignore_index=True,
)

confidence_threshold_summary_df = pd.concat(
    [
        uncalibrated_thresholds_df.assign(model_version="uncalibrated"),
        calibrated_thresholds_df.assign(model_version="calibrated"),
    ],
    ignore_index=True,
)

calibration_comparison_df.to_csv(calibration_comparison_path, index=False)
calibration_bins_df.to_csv(calibration_bins_path, index=False)
confidence_threshold_summary_df.to_csv(
    confidence_threshold_summary_path,
    index=False,
)

joblib.dump(uncalibrated_final_pipeline, final_model_uncalibrated_path)
joblib.dump(calibrated_model, final_model_calibrated_path)

plot_final_model_scores(
    best_model_name=best_model_name,
    selection_cv_score=selection_cv_score,
    uncalibrated_test_score=uncalibrated_test_score,
    calibrated_test_score=calibrated_test_score,
    scoring_metric=scoring_metric,
    out_path=plot_final_model_scores_path,
)

plot_reliability_comparison(
    calibration_bins_df=calibration_bins_df,
    out_path=plot_reliability_comparison_path,
)

with mlflow.start_run(run_name="final_model"):
    log_data_source_params()
    mlflow.log_param("winning_model_family", best_model_name)
    mlflow.log_param("winning_lazy_name", best_lazy_name)
    mlflow.log_param("calibration_method", "sigmoid")
    mlflow.log_param("calibration_cv_folds", cv.get_n_splits())
    mlflow.log_param("calibration_ensemble", False)
    mlflow.log_param("n_jobs", n_jobs)
    log_metric_if_valid("selection_cv_score", selection_cv_score)
    log_metric_if_valid("uncalibrated_test_score", uncalibrated_summary["score"])
    log_metric_if_valid("calibrated_test_score", calibrated_summary["score"])
    log_metric_if_valid("uncalibrated_test_log_loss", uncalibrated_summary["log_loss"])
    log_metric_if_valid("calibrated_test_log_loss", calibrated_summary["log_loss"])
    log_metric_if_valid("uncalibrated_test_ece", uncalibrated_summary["ece"])
    log_metric_if_valid("calibrated_test_ece", calibrated_summary["ece"])

    mlflow.log_artifact(final_model_calibrated_path)
    mlflow.log_artifact(final_model_uncalibrated_path)
    mlflow.log_artifact(optuna_comparison_path)
    mlflow.log_artifact(calibration_comparison_path)
    mlflow.log_artifact(calibration_bins_path)
    mlflow.log_artifact(confidence_threshold_summary_path)
    mlflow.log_artifact(plot_final_model_scores_path)
    mlflow.log_artifact(plot_reliability_comparison_path)
    mlflow.sklearn.log_model(calibrated_model, artifact_path="final_sklearn_model")

overall_duration = time.perf_counter() - overall_start

print("Best model:", best_model_name)
print("n_jobs:", n_jobs)
print("Selection CV score:", selection_cv_score)
print("Uncalibrated test score:", uncalibrated_test_score)
print("Calibrated test score:", calibrated_test_score)
print("Uncalibrated test log loss:", uncalibrated_summary["log_loss"])
print("Calibrated test log loss:", calibrated_summary["log_loss"])
print("Uncalibrated test ECE:", uncalibrated_summary["ece"])
print("Calibrated test ECE:", calibrated_summary["ece"])
print("Saved calibrated final model to", final_model_calibrated_path)
print("Saved uncalibrated final model to", final_model_uncalibrated_path)
print(
    f"Total runtime (seconds): {overall_duration:.2f} "
    f"({overall_duration / 60:.2f} minutes)"
)
