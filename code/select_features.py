# %% Select features particularly when dealing with high-dimensional data with high degree of correlation
# env: ml-tabular-env

# sbatch -J select_features_progb_vs_nonprogb -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python select_features.py --out-dir results/select_features/progb_vs_nonprogb --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv"

# sbatch -J select_features_progb_vs_nonprogb_selectkbest_p005 -p short,long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python select_features.py --out-dir results/select_features/progb_vs_nonprogb_selectkbest_p005 --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv --univariate-method select_k_best --univariate-pvalue-threshold 0.05"

import argparse
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy import stats

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
parser.add_argument("--test-path", default=None)
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
parser.add_argument(
    "--min-non-missing-per-class",
    type=int,
    default=None,
    help="Drop features with fewer than this many non-missing train samples in any target class before feature selection.",
)
parser.add_argument("--univariate-scoring", default="roc_auc")
parser.add_argument("--univariate-cv", type=int, default=5)
parser.add_argument("--univariate-threshold", type=float, default=None)
parser.add_argument("--no-scale", dest="scale", action="store_false")
parser.add_argument("--skip-correlated-selection", action="store_true")
parser.add_argument(
    "--write-reduced-data",
    action="store_true",
    help="Write train.csv, and test.csv when --test-path is provided, reduced to selected features.",
)
parser.add_argument(
    "--bootstrap-train",
    action="store_true",
    help="Sample train rows with replacement before feature selection.",
)
parser.add_argument(
    "--bootstrap-seed",
    type=int,
    default=123,
    help="Random seed used when --bootstrap-train is set.",
)
parser.add_argument(
    "--no-stratified-bootstrap",
    dest="stratified_bootstrap",
    action="store_false",
    help="Disable class-stratified bootstrap resampling.",
)
parser.set_defaults(scale=True, stratified_bootstrap=True)
args = parser.parse_args()


# %% Helpers

def assert_valid_input(train_df, test_df, target_column):
    assert target_column in train_df.columns, (
        f"{target_column} not found in train columns"
    )

    train_features = train_df.drop(columns=[target_column]).columns
    non_numeric = train_df[train_features].select_dtypes(exclude="number").columns
    assert len(non_numeric) == 0, (
        "Feature selection expects numeric features. "
        f"Non-numeric columns found: {non_numeric.tolist()}"
    )

    if test_df is not None:
        assert target_column in test_df.columns, (
            f"{target_column} not found in test columns"
        )
        test_features = test_df.drop(columns=[target_column]).columns
        assert train_features.equals(test_features), (
            "Train and test feature columns differ before feature selection"
        )


def bootstrap_train_df(train_df, target_column, seed, stratified):
    rng = np.random.default_rng(seed)

    if not stratified:
        sampled_positions = rng.choice(len(train_df), size=len(train_df), replace=True)
        return train_df.iloc[sampled_positions].copy()

    sampled_parts = []
    for _, group in train_df.groupby(target_column, sort=False):
        sampled_positions = rng.choice(len(group), size=len(group), replace=True)
        sampled_parts.append(group.iloc[sampled_positions])

    return pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=seed).copy()


def filter_features_by_min_non_missing_per_class(
    X_train,
    y_train,
    min_non_missing_per_class,
):
    non_missing_counts = (
        X_train.notna()
        .reset_index(drop=True)
        .groupby(pd.Series(y_train).reset_index(drop=True), sort=False)
        .sum()
    )
    min_counts = non_missing_counts.min(axis=0)
    dropped = min_counts[min_counts < min_non_missing_per_class]
    dropped_features_df = pd.DataFrame(
        {
            "feature": dropped.index,
            "min_non_missing_per_class": dropped.values,
            "threshold": min_non_missing_per_class,
        }
    )

    X_train_filtered = X_train.loc[:, min_counts >= min_non_missing_per_class]
    if X_train_filtered.shape[1] == 0:
        raise ValueError(
            "No features remain after --min-non-missing-per-class="
            f"{min_non_missing_per_class}"
        )

    return X_train_filtered, dropped_features_df


def missingness_aware_f_classif(X, y):
    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    classes = pd.unique(y_series)
    scores = np.full(X_df.shape[1], np.nan, dtype=float)
    pvalues = np.full(X_df.shape[1], np.nan, dtype=float)

    for col_idx, column in enumerate(X_df.columns):
        values = X_df[column]
        groups = []
        valid_feature = True
        for class_value in classes:
            group_values = values[(y_series == class_value) & values.notna()].to_numpy(
                dtype=float
            )
            if len(group_values) < 2:
                valid_feature = False
                break
            groups.append(group_values)

        if not valid_feature:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score, pvalue = stats.f_oneway(*groups)
        if np.isfinite(score) and np.isfinite(pvalue):
            scores[col_idx] = score
            pvalues[col_idx] = pvalue

    return scores, pvalues


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


def run_feature_selection(X_train, y_train, args):
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
        DropConstantFeatures(missing_values="ignore"),
        X_train_selection,
        "drop_constant",
        summary_rows,
        dropped_rows,
    )

    X_train_selection, duplicate_selector = fit_transform_train_selector(
        DropDuplicateFeatures(missing_values="ignore"),
        X_train_selection,
        "drop_duplicate",
        summary_rows,
        dropped_rows,
    )

    univariate_feature_performance = pd.DataFrame(
        columns=["feature", "score", "pvalue", "score_std", "selected", "method"]
    )
    if not args.skip_univariate_selection:
        if args.univariate_method == "select_k_best":
            n_features_before = X_train_selection.shape[1]
            has_missing_for_select_k_best = X_train_selection.isna().any().any()
            if has_missing_for_select_k_best:
                print(
                    "Training features contain missing values; using "
                    "missingness-aware ANOVA scorer for SelectKBest."
                )
                X_train_selection_for_score = X_train_selection.copy()

                def score_func(X, y):
                    return missingness_aware_f_classif(X_train_selection_for_score, y)

                univariate_method = "select_k_best_missingness_aware"
                X_train_fit = X_train_selection.fillna(0)
            else:
                score_func = f_classif
                univariate_method = "select_k_best"
                X_train_fit = X_train_selection

            univariate_selector = SelectKBest(score_func=score_func, k="all")
            univariate_selector.fit(X_train_fit, y_train)

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
                pvalues.notna() & (pvalues < args.univariate_pvalue_threshold)
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
            if has_missing_for_select_k_best and X_train_selection.isna().any().any():
                print(
                    "Warning: selected features still contain missing values after "
                    "SelectKBest; smart correlation may fail unless it is skipped "
                    "or missingness is handled first."
                )
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
                    "method": univariate_method,
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

    return {
        "selected_features": selected_features,
        "dropped_features": pd.DataFrame(dropped_rows, columns=["feature", "step"]),
        "univariate_feature_performance": univariate_feature_performance[
            ["feature", "score", "pvalue", "score_std", "selected", "method"]
        ],
        "correlated_feature_sets": correlated_feature_sets,
        "feature_selection_summary": pd.DataFrame(summary_rows),
    }


# %% ----- MAIN -----

if args.test_path == "":
    args.test_path = None
if (
    args.min_non_missing_per_class is not None
    and args.min_non_missing_per_class < 1
):
    raise ValueError("--min-non-missing-per-class must be >= 1")
os.makedirs(args.out_dir, exist_ok=True)


# %% Load data

train_df = pd.read_csv(args.train_path, index_col=0)
test_df = (
    pd.read_csv(args.test_path, index_col=0)
    if args.test_path is not None
    else None
)

assert_valid_input(train_df, test_df, args.target_column)

if args.bootstrap_train:
    train_df_selection = bootstrap_train_df(
        train_df,
        target_column=args.target_column,
        seed=args.bootstrap_seed,
        stratified=args.stratified_bootstrap,
    )
else:
    train_df_selection = train_df.copy()

X_train = train_df_selection.drop(columns=[args.target_column])
y_train = train_df_selection[args.target_column]

if args.min_non_missing_per_class is not None:
    X_train, dropped_min_non_missing_df = (
        filter_features_by_min_non_missing_per_class(
            X_train,
            y_train,
            args.min_non_missing_per_class,
        )
    )
    dropped_min_non_missing_df.to_csv(
        os.path.join(
            args.out_dir,
            "features_dropped_min_non_missing_per_class.csv",
        ),
        index=False,
    )
    print(
        "Dropped "
        f"{len(dropped_min_non_missing_df)} features with fewer than "
        f"{args.min_non_missing_per_class} non-missing samples in any class."
    )


# %% Select features on train only

selection_outputs = run_feature_selection(X_train, y_train, args)


# %% Save feature selection metadata

pd.DataFrame({"feature": selection_outputs["selected_features"]}).to_csv(
    os.path.join(args.out_dir, "selected_features.csv"),
    index=False,
)

selection_outputs["dropped_features"].to_csv(
    os.path.join(args.out_dir, "dropped_features.csv"),
    index=False,
)

selection_outputs["univariate_feature_performance"].to_csv(
    os.path.join(args.out_dir, "univariate_feature_performance.csv"),
    index=False,
)

selection_outputs["correlated_feature_sets"].to_csv(
    os.path.join(args.out_dir, "correlated_feature_sets.csv"),
    index=False,
)

summary_df = selection_outputs["feature_selection_summary"].copy()
summary_df.insert(0, "bootstrap_train", bool(args.bootstrap_train))
summary_df.insert(
    1,
    "bootstrap_seed",
    args.bootstrap_seed if args.bootstrap_train else pd.NA,
)
summary_df.insert(2, "stratified_bootstrap", bool(args.stratified_bootstrap))
summary_df.to_csv(
    os.path.join(args.out_dir, "feature_selection_summary.csv"),
    index=False,
)

if args.write_reduced_data:
    selected_features = selection_outputs["selected_features"]
    output_columns = selected_features + [args.target_column]
    original_train_features = train_df.drop(columns=[args.target_column]).columns
    missing_train_features = sorted(
        set(selected_features).difference(original_train_features)
    )

    if test_df is not None:
        original_test_features = test_df.drop(columns=[args.target_column]).columns
        missing_test_features = sorted(
            set(selected_features).difference(original_test_features)
        )
    else:
        missing_test_features = []

    if missing_train_features or missing_test_features:
        missing_train_preview = ", ".join(missing_train_features[:10])
        missing_test_preview = ", ".join(missing_test_features[:10])
        raise ValueError(
            "Selected features are missing from original input data. "
            f"Train missing: {missing_train_preview or 'none'}. "
            f"Test missing: {missing_test_preview or 'none'}."
        )

    train_df.loc[:, output_columns].to_csv(
        os.path.join(args.out_dir, "train.csv"),
    )
    if test_df is not None:
        test_df.loc[:, output_columns].to_csv(
            os.path.join(args.out_dir, "test.csv"),
        )
        print(f"Saved reduced train/test data to {args.out_dir}")
    else:
        print(f"Saved reduced train data to {args.out_dir}")

print(summary_df)
print(f"Selected features: {len(selection_outputs['selected_features'])}")
print(f"Saved feature-selection metadata to {args.out_dir}")
