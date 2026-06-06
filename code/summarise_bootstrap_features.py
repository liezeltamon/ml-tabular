# %% Summarise bootstrap feature-selection runs into a robust selected feature set
# env: ml-tabular-env

# mkdir -p logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005
# sbatch -J summarise_bootstrap_p005 -p short,long --mem=15G --output=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005/%x.log.out --error=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005/%x.log.err --wrap="python summarise_bootstrap_features.py --bootstrap-root results/select_features/progb_vs_nonprogb_selectkbest_p005/bootstraps --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv --target-column is_progb --out-dir results/select_features/progb_vs_nonprogb_selectkbest_p005_bootstrap90 --selection-threshold 0.9"

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

os.chdir(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        universal_newlines=True,
    ).strip()
)


def bootstrap_sort_key(path):
    match = re.search(r"bootstrap_(\d+)$", path.name)
    if match is None:
        return (1, path.name)
    return (0, int(match.group(1)))


def split_feature_list(value):
    if pd.isna(value):
        return []
    return [
        feature
        for feature in str(value).split(";")
        if feature and feature.lower() != "nan"
    ]


def read_selected_features(bootstrap_dir):
    selected_path = bootstrap_dir / "selected_features.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}")

    selected_df = pd.read_csv(selected_path)
    if "feature" not in selected_df.columns:
        raise ValueError(f"{selected_path} must contain a 'feature' column")

    return selected_df["feature"].dropna().astype(str).tolist()


def read_correlated_pairs(bootstrap_dir):
    correlated_path = bootstrap_dir / "correlated_feature_sets.csv"
    if not correlated_path.exists():
        return set()

    correlated_df = pd.read_csv(correlated_path)
    required_columns = {"selected_feature", "correlated_features"}
    missing_columns = required_columns.difference(correlated_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{correlated_path} missing columns: {missing}")

    pairs = set()
    for _, row in correlated_df.iterrows():
        selected_features = split_feature_list(row["selected_feature"])
        correlated_features = split_feature_list(row["correlated_features"])
        for selected_feature in selected_features:
            for correlated_feature in correlated_features:
                if selected_feature == correlated_feature:
                    continue
                pairs.add((selected_feature, correlated_feature))

    return pairs


def assert_valid_original_data(train_df, test_df, target_column):
    if target_column not in train_df.columns:
        raise ValueError(f"{target_column} not found in train columns")
    if target_column not in test_df.columns:
        raise ValueError(f"{target_column} not found in test columns")

    train_features = train_df.drop(columns=[target_column]).columns
    test_features = test_df.drop(columns=[target_column]).columns
    if not train_features.equals(test_features):
        raise ValueError("Train and test feature columns differ")


def reduce_and_save_data(train_df, test_df, selected_features, target_column, out_dir):
    output_columns = selected_features + [target_column]
    input_features = train_df.drop(columns=[target_column]).columns
    missing_features = sorted(
        set(selected_features).difference(input_features)
    )
    if missing_features:
        missing_preview = ", ".join(missing_features[:10])
        raise ValueError(
            "Selected bootstrap features are missing from input data: "
            f"{missing_preview}"
        )

    train_df.loc[:, output_columns].to_csv(out_dir / "train.csv")
    test_df.loc[:, output_columns].to_csv(out_dir / "test.csv")


parser = argparse.ArgumentParser(
    description="Summarise bootstrap feature-selection runs."
)
parser.add_argument("--bootstrap-root", required=True)
parser.add_argument("--train-path", required=True)
parser.add_argument("--test-path", required=True)
parser.add_argument("--target-column", default="is_progb")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--selection-threshold", type=float, default=0.9)
args = parser.parse_args()

if not 0 < args.selection_threshold <= 1:
    raise ValueError("--selection-threshold must be in the interval (0, 1].")

bootstrap_root = Path(args.bootstrap_root)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

if not bootstrap_root.exists():
    raise FileNotFoundError(f"Bootstrap root not found: {bootstrap_root}")

bootstrap_dirs = sorted(
    [path for path in bootstrap_root.iterdir() if path.is_dir()],
    key=bootstrap_sort_key,
)
if not bootstrap_dirs:
    raise ValueError(f"No bootstrap directories found under {bootstrap_root}")

feature_counter = Counter()
pair_counter = Counter()
success_rows = []

for bootstrap_dir in bootstrap_dirs:
    bootstrap_id_match = re.search(r"bootstrap_(\d+)$", bootstrap_dir.name)
    bootstrap_id = (
        int(bootstrap_id_match.group(1))
        if bootstrap_id_match is not None
        else pd.NA
    )

    try:
        selected_features = read_selected_features(bootstrap_dir)
        correlated_pairs = read_correlated_pairs(bootstrap_dir)
        feature_counter.update(set(selected_features))
        pair_counter.update(correlated_pairs)
        success_rows.append(
            {
                "bootstrap_dir": str(bootstrap_dir),
                "bootstrap_id": bootstrap_id,
                "status": "success",
                "n_selected_features": len(set(selected_features)),
                "n_correlated_pairs": len(correlated_pairs),
                "error": pd.NA,
            }
        )
    except Exception as error:
        success_rows.append(
            {
                "bootstrap_dir": str(bootstrap_dir),
                "bootstrap_id": bootstrap_id,
                "status": "failed",
                "n_selected_features": 0,
                "n_correlated_pairs": 0,
                "error": str(error),
            }
        )

success_summary_df = pd.DataFrame(success_rows)
success_summary_df.to_csv(out_dir / "bootstrap_success_summary.csv", index=False)

n_bootstraps = int((success_summary_df["status"] == "success").sum())
if n_bootstraps == 0:
    raise ValueError("No successful bootstrap runs were found.")

feature_frequency_df = pd.DataFrame(
    [
        {
            "feature": feature,
            "n_selected": n_selected,
            "n_bootstraps": n_bootstraps,
            "selection_frequency": n_selected / n_bootstraps,
            "selected_final": n_selected / n_bootstraps >= args.selection_threshold,
        }
        for feature, n_selected in feature_counter.items()
    ]
)
feature_frequency_df = feature_frequency_df.sort_values(
    ["selected_final", "selection_frequency", "n_selected", "feature"],
    ascending=[False, False, False, True],
).reset_index(drop=True)
feature_frequency_df.to_csv(
    out_dir / "bootstrap_feature_frequency.csv",
    index=False,
)

pair_frequency_df = pd.DataFrame(
    [
        {
            "selected_feature": selected_feature,
            "correlated_feature": correlated_feature,
            "n_pair_observed": n_pair_observed,
            "n_bootstraps": n_bootstraps,
            "pair_frequency": n_pair_observed / n_bootstraps,
        }
        for (
            selected_feature,
            correlated_feature,
        ), n_pair_observed in pair_counter.items()
    ]
)
if pair_frequency_df.empty:
    pair_frequency_df = pd.DataFrame(
        columns=[
            "selected_feature",
            "correlated_feature",
            "n_pair_observed",
            "n_bootstraps",
            "pair_frequency",
        ]
    )
else:
    pair_frequency_df = pair_frequency_df.sort_values(
        [
            "pair_frequency",
            "n_pair_observed",
            "selected_feature",
            "correlated_feature",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
pair_frequency_df.to_csv(
    out_dir / "bootstrap_correlated_pair_frequency.csv",
    index=False,
)

selected_final_features = feature_frequency_df.loc[
    feature_frequency_df["selected_final"],
    "feature",
].tolist()
if not selected_final_features:
    raise ValueError(
        "No features met the final bootstrap selection threshold "
        f"{args.selection_threshold}."
    )

pd.DataFrame({"feature": selected_final_features}).to_csv(
    out_dir / "selected_features.csv",
    index=False,
)

train_df = pd.read_csv(args.train_path, index_col=0)
test_df = pd.read_csv(args.test_path, index_col=0)
assert_valid_original_data(train_df, test_df, args.target_column)
reduce_and_save_data(
    train_df,
    test_df,
    selected_final_features,
    args.target_column,
    out_dir,
)

print(f"Successful bootstrap runs: {n_bootstraps}")
print(f"Final selected features: {len(selected_final_features)}")
print(f"Saved bootstrap summary outputs to {out_dir}")
