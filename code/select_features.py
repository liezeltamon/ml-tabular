# %% Select features particularly when dealing with high-dimensional data with high degree of correlation
# env: ml-tabular-env

# sbatch -J select_features_progb_vs_nonprogb -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python select_features.py --out-dir results/select_features/progb_vs_nonprogb --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv"

# sbatch -J select_features_progb_vs_nonprogb_selectkbest_p005 -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python select_features.py --out-dir results/select_features/progb_vs_nonprogb_selectkbest_p005 --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv --univariate-method select_k_best --univariate-pvalue-threshold 0.05"

import argparse
import os
import subprocess

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SelectBySingleFeaturePerformance,
    SmartCorrelatedSelection,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        universal_newlines=True,
    ).strip()
)

# %% Parameters

parser = argparse.ArgumentParser()
parser.add_argument("--train-path", default="data/train.csv")
parser.add_argument("--test-path", default="data/test.csv")
parser.add_argument("--target-column", default="is_progb")
parser.add_argument("--out-dir", default="results/select_features")
parser.add_argument("--correlation-threshold", type=float, default=0.8)
parser.add_argument("--correlation-method", default="pearson")
parser.add_argument("--smart-selection-method", default="variance")
parser.add_argument("--skip-univariate-selection", action="store_true")
parser.add_argument(
    "--univariate-method",
    choices=["select_k_best", "single_feature_performance"],
    default="select_k_best",
)
parser.add_argument("--univariate-pvalue-threshold", type=float, default=0.05)
parser.add_argument("--univariate-scoring", default="roc_auc")
parser.add_argument("--univariate-cv", type=int, default=5)
parser.add_argument("--univariate-threshold", type=float, default=None)
parser.add_argument("--no-scale", dest="scale", action="store_false")
parser.add_argument("--skip-correlated-selection", action="store_true")
parser.set_defaults(scale=True)
args = parser.parse_args()


# %% Helpers

def assert_valid_input(train_df, test_df, target_column):
    assert target_column in train_df.columns, (
        f"{target_column} not found in train columns"
    )
    assert target_column in test_df.columns, (
        f"{target_column} not found in test columns"
    )

    train_features = train_df.drop(columns=[target_column]).columns
    test_features = test_df.drop(columns=[target_column]).columns
    assert train_features.equals(test_features), (
        "Train and test feature columns differ before feature selection"
    )

    non_numeric = train_df[train_features].select_dtypes(exclude="number").columns
    assert len(non_numeric) == 0, (
        "Feature selection expects numeric features. "
        f"Non-numeric columns found: {non_numeric.tolist()}"
    )


def scale_for_selection(X_train):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    return X_train_scaled


def fit_transform_train_selector(
    selector,
    X_train_selection,
    step_name,
    summary_rows,
    dropped_rows,
    y_train=None,
):
    n_features_before = X_train_selection.shape[1]
    if y_train is None:
        X_train_selection = selector.fit_transform(X_train_selection)
    else:
        X_train_selection = selector.fit_transform(X_train_selection, y_train)
    n_features_after = X_train_selection.shape[1]

    features_to_drop = list(getattr(selector, "features_to_drop_", []))
    dropped_rows.extend(
        {
            "feature": feature,
            "step": step_name,
        }
        for feature in features_to_drop
    )
    summary_rows.append(
        {
            "step": step_name,
            "n_features_before": n_features_before,
            "n_features_after": n_features_after,
            "n_dropped": n_features_before - n_features_after,
        }
    )
    return X_train_selection, selector


def make_correlated_feature_sets_df(selector, selected_features):
    selected_features = set(selected_features)
    dropped_features = set(getattr(selector, "features_to_drop_", []))
    correlated_sets = getattr(selector, "correlated_feature_sets_", [])

    rows = []
    for group_id, feature_set in enumerate(correlated_sets):
        features = list(feature_set)
        selected_in_group = sorted(set(features).intersection(selected_features))
        dropped_in_group = sorted(set(features).intersection(dropped_features))
        rows.append(
            {
                "group_id": group_id,
                "selected_feature": ";".join(selected_in_group),
                "correlated_features": ";".join(features),
                "dropped_features": ";".join(dropped_in_group),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "group_id",
            "selected_feature",
            "correlated_features",
            "dropped_features",
        ],
    )


# %% ----- MAIN -----

os.makedirs(args.out_dir, exist_ok=True)


# %% Load data

train_df = pd.read_csv(args.train_path, index_col=0)
test_df = pd.read_csv(args.test_path, index_col=0)

assert_valid_input(train_df, test_df, args.target_column)

X_train = train_df.drop(columns=[args.target_column])
y_train = train_df[args.target_column]

X_test = test_df.drop(columns=[args.target_column])
y_test = test_df[args.target_column]


# %% Select features on train only

summary_rows = [
    {
        "step": "input",
        "n_features_before": X_train.shape[1],
        "n_features_after": X_train.shape[1],
        "n_dropped": 0,
    }
]
dropped_rows = []

if args.scale:
    X_train_selection = scale_for_selection(X_train)
else:
    X_train_selection = X_train.copy()

X_train_selection, constant_selector = fit_transform_train_selector(
    DropConstantFeatures(missing_values="raise"),
    X_train_selection,
    "drop_constant",
    summary_rows,
    dropped_rows,
)

X_train_selection, duplicate_selector = fit_transform_train_selector(
    DropDuplicateFeatures(missing_values="raise"),
    X_train_selection,
    "drop_duplicate",
    summary_rows,
    dropped_rows,
)

univariate_selector = None
univariate_feature_performance = pd.DataFrame(
    columns=["feature", "score", "pvalue", "score_std", "selected", "method"]
)
if not args.skip_univariate_selection:
    if args.univariate_method == "select_k_best":
        n_features_before = X_train_selection.shape[1]
        univariate_selector = SelectKBest(score_func=f_classif, k="all")
        univariate_selector.fit(X_train_selection, y_train)

        scores = pd.Series(
            univariate_selector.scores_,
            index=X_train_selection.columns,
            name="score",
        )
        pvalues = pd.Series(
            univariate_selector.pvalues_,
            index=X_train_selection.columns,
            name="pvalue",
        )
        selected_features_univariate = pvalues[
            pvalues < args.univariate_pvalue_threshold
        ].index.tolist()
        if len(selected_features_univariate) == 0:
            raise ValueError(
                "No features passed SelectKBest p-value threshold "
                f"{args.univariate_pvalue_threshold}"
            )

        features_to_drop = [
            feature
            for feature in X_train_selection.columns
            if feature not in selected_features_univariate
        ]
        dropped_rows.extend(
            {"feature": feature, "step": "select_k_best_pvalue"}
            for feature in features_to_drop
        )
        X_train_selection = X_train_selection[selected_features_univariate]
        n_features_after = X_train_selection.shape[1]
        summary_rows.append(
            {
                "step": "select_k_best_pvalue",
                "n_features_before": n_features_before,
                "n_features_after": n_features_after,
                "n_dropped": n_features_before - n_features_after,
            }
        )
        univariate_feature_performance = pd.DataFrame(
            {
                "feature": scores.index,
                "score": scores.values,
                "pvalue": pvalues.values,
                "score_std": pd.NA,
                "selected": scores.index.isin(selected_features_univariate),
                "method": "select_k_best",
            }
        )
    else:
        X_train_selection, univariate_selector = fit_transform_train_selector(
            SelectBySingleFeaturePerformance(
                estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
                scoring=args.univariate_scoring,
                cv=args.univariate_cv,
                threshold=args.univariate_threshold,
            ),
            X_train_selection,
            "select_by_single_feature_performance",
            summary_rows,
            dropped_rows,
            y_train=y_train,
        )
        univariate_selected_features = set(univariate_selector.variables_) - set(
            univariate_selector.features_to_drop_
        )
        univariate_feature_performance = pd.DataFrame(
            {
                "feature": list(univariate_selector.feature_performance_.keys()),
                "score": list(univariate_selector.feature_performance_.values()),
                "pvalue": pd.NA,
                "score_std": [
                    univariate_selector.feature_performance_std_.get(feature)
                    for feature in univariate_selector.feature_performance_.keys()
                ],
                "selected": [
                    feature in univariate_selected_features
                    for feature in univariate_selector.feature_performance_.keys()
                ],
                "method": "single_feature_performance",
            }
        )

correlated_selector = None
if not args.skip_correlated_selection:
    X_train_selection, correlated_selector = fit_transform_train_selector(
        SmartCorrelatedSelection(
            variables=None,
            method=args.correlation_method,
            threshold=args.correlation_threshold,
            missing_values="raise",
            selection_method=args.smart_selection_method,
            estimator=None,
        ),
        X_train_selection,
        "smart_correlated_selection",
        summary_rows,
        dropped_rows,
    )

selected_features = list(X_train_selection.columns)


# %% Save benchmark-compatible reduced train and test CSVs

train_reduced = pd.concat([X_train[selected_features], y_train], axis=1)
test_reduced = pd.concat([X_test[selected_features], y_test], axis=1)

assert train_reduced.drop(columns=[args.target_column]).columns.equals(
    test_reduced.drop(columns=[args.target_column]).columns
), "Reduced train and test feature columns differ"
assert train_reduced[args.target_column].equals(y_train), (
    "Train target changed during feature selection"
)
assert test_reduced[args.target_column].equals(y_test), (
    "Test target changed during feature selection"
)

train_reduced.to_csv(os.path.join(args.out_dir, "train.csv"))
test_reduced.to_csv(os.path.join(args.out_dir, "test.csv"))


# %% Save feature selection metadata

pd.DataFrame({"feature": selected_features}).to_csv(
    os.path.join(args.out_dir, "selected_features.csv"),
    index=False,
)

pd.DataFrame(dropped_rows, columns=["feature", "step"]).to_csv(
    os.path.join(args.out_dir, "dropped_features.csv"),
    index=False,
)

univariate_feature_performance = univariate_feature_performance[
    ["feature", "score", "pvalue", "score_std", "selected", "method"]
]
univariate_feature_performance.to_csv(
    os.path.join(args.out_dir, "univariate_feature_performance.csv"),
    index=False,
)

if correlated_selector is None:
    correlated_feature_sets = pd.DataFrame(
        columns=[
            "group_id",
            "selected_feature",
            "correlated_features",
            "dropped_features",
        ]
    )
else:
    correlated_feature_sets = make_correlated_feature_sets_df(
        correlated_selector,
        selected_features,
    )
correlated_feature_sets.to_csv(
    os.path.join(args.out_dir, "correlated_feature_sets.csv"),
    index=False,
)

pd.DataFrame(summary_rows).to_csv(
    os.path.join(args.out_dir, "feature_selection_summary.csv"),
    index=False,
)

print(pd.DataFrame(summary_rows))
print(f"Saved reduced train/test and metadata to {args.out_dir}")
