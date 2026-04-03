# %% Tune shortlisted models and save the final pipeline

import joblib
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
scoring_metric = "roc_auc"
mlflow_experiment_name = "tabular_model_selection"
out_dir = "../results/tune_models"
os.makedirs(out_dir, exist_ok=True)

# Set to False if preprocessing can produce a sparse matrix, for example after one-hot encoding.
stdscaler_with_mean = True

# %% Prepare data

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

top_models_to_tune = [
    "LogisticRegression",
    "LinearSVC",
    "SVC",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
]

MODEL_NAME_MAP = {
    "LogisticRegression": "logreg",
    "LinearSVC": "linearsvc",
    "SVC": "svc",
    "XGBClassifier": "xgb",
    "LGBMClassifier": "lgbm",
    "CatBoostClassifier": "catboost",
}


def build_pipeline(trial, model_name):
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
            random_state=random_state,
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
            random_state=random_state,
            verbose=-1,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_name == "catboost":
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 500),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            random_state=random_state,
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


def compute_score(model, X, y, scoring_metric, num_classes):
    if scoring_metric == "roc_auc" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if num_classes == 2:
            return roc_auc_score(y, proba[:, 1])
        return roc_auc_score(y, proba, multi_class="ovr", average="macro")

    if scoring_metric == "accuracy":
        pred = model.predict(X)
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
    train_score,
    cv_score,
    test_score,
    scoring_metric,
    out_path,
):
    score_summary_df = pd.DataFrame(
        {
            "dataset": ["train", "cv", "test"],
            "score": [train_score, cv_score, test_score],
        }
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        score_summary_df["dataset"],
        score_summary_df["score"],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    ax.set_ylabel(f"Score: {scoring_metric}")
    ax.set_title(f"Generalization check for {best_model_name}")

    for i, value in enumerate(score_summary_df["score"]):
        if not np.isnan(value):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")

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

mlflow.set_experiment(mlflow_experiment_name)

results = []
studies = {}

for lazy_name in top_models_to_tune:
    if lazy_name not in MODEL_NAME_MAP:
        print(f"Skipping unsupported model from LazyPredict: {lazy_name}")
        continue

    model_name = MODEL_NAME_MAP[lazy_name]

    with mlflow.start_run(run_name=f"optuna_{model_name}"):
        mlflow.log_param("model_family", model_name)
        mlflow.log_param("lazy_name", lazy_name)
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_param("scoring", scoring_metric)
        mlflow.log_param("optuna_n_trials", optuna_n_trials)
        mlflow.log_param("test_size", test_size)

        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(model_name), n_trials=optuna_n_trials)

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
results_df.to_csv("optuna_model_comparison.csv", index=False)

plot_model_comparison(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=os.path.join(out_dir, "plot_model_comparison.png"),
)

plot_cv_score_spread(
    results_df=results_df,
    scoring_metric=scoring_metric,
    out_path=os.path.join(out_dir, "plot_cv_score_spread.png"),
)

with mlflow.start_run(run_name="model_comparison_summary"):
    mlflow.log_artifact("optuna_model_comparison.csv")
    mlflow.log_artifact(os.path.join(out_dir, "plot_model_comparison.png"))
    mlflow.log_artifact(os.path.join(out_dir, "plot_cv_score_spread.png"))

print(results_df)

best_model_name = results_df.iloc[0]["model_name"]
best_lazy_name = results_df.iloc[0]["lazy_name"]
best_study = studies[best_model_name]
num_classes = len(np.unique(y_train))

final_pipeline = build_pipeline(FixedTrial(best_study.best_params), best_model_name)
final_pipeline.fit(X_train, y_train)

train_score = compute_score(
    final_pipeline,
    X_train,
    y_train,
    scoring_metric=scoring_metric,
    num_classes=num_classes,
)

test_score = compute_score(
    final_pipeline,
    X_test,
    y_test,
    scoring_metric=scoring_metric,
    num_classes=num_classes,
)

joblib.dump(final_pipeline, "final_model.pkl")

plot_final_model_scores(
    best_model_name=best_model_name,
    train_score=train_score,
    cv_score=best_study.best_value,
    test_score=test_score,
    scoring_metric=scoring_metric,
    out_path=os.path.join(out_dir, "plot_final_model_scores.png"),
)

with mlflow.start_run(run_name="final_model"):
    mlflow.log_param("winning_model_family", best_model_name)
    mlflow.log_param("winning_lazy_name", best_lazy_name)
    mlflow.log_metric("best_score", best_study.best_value)
    if not np.isnan(train_score):
        mlflow.log_metric("train_score", float(train_score))

    if not np.isnan(test_score):
        mlflow.log_metric("test_score", float(test_score))

    mlflow.log_artifact("final_model.pkl")
    mlflow.log_artifact("optuna_model_comparison.csv")
    mlflow.log_artifact(os.path.join(out_dir, "plot_final_model_scores.png"))
    mlflow.sklearn.log_model(final_pipeline, artifact_path="final_sklearn_model")

print("Best model:", best_model_name)
print("Best score:", best_study.best_value)
print("Train score:", train_score)
print("Test score:", test_score)
print("Saved final model to final_model.pkl")
