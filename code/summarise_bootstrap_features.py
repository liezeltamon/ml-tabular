# %% Summarise bootstrap feature-selection runs into a robust selected feature set
# env: ml-tabular-env

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005
# sbatch -J summarise_bootstrap_p005 -p short,long --mem=15G \
#   --output=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005/%x.log.out \
#   --error=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005/%x.log.err \
#   --wrap="python summarise_bootstrap_features.py \
#     --bootstrap-root /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005/bootstraps \
#     --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv \
#     --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv \
#     --target-column is_progb \
#     --out-dir /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005 \
#     --selection-threshold 0.9"

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection
# sbatch -J summarise_bootstrap_p005_nocorr -p short,long --mem=15G \
#   --output=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/%x.log.out \
#   --error=logs/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/%x.log.err \
#   --wrap="python summarise_bootstrap_features.py \
#     --bootstrap-root /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/bootstraps \
#     --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv \
#     --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv \
#     --target-column is_progb \
#     --out-dir /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection \
#     --selection-threshold 0.9 \
#     --no-correlation-outputs"

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection
# sbatch -J summarise_bootstrap_sfp_auc_nocorr -p short,long --mem=15G \
#   --output=logs/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/%x.log.out \
#   --error=logs/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/%x.log.err \
#   --wrap="python summarise_bootstrap_features.py \
#     --bootstrap-root /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/bootstraps \
#     --train-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv \
#     --test-path /well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv \
#     --target-column is_progb \
#     --out-dir /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection \
#     --selection-threshold 0.9 \
#     --no-correlation-outputs"

import argparse
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

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
        raise FileNotFoundError(f"Missing {correlated_path}")

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


def missing_required_outputs(bootstrap_dir, no_correlation_outputs):
    required_output_names = ["selected_features.csv"]
    if not no_correlation_outputs:
        required_output_names.append("correlated_feature_sets.csv")
    return [
        output_name
        for output_name in required_output_names
        if not (bootstrap_dir / output_name).exists()
    ]


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


def save_feature_frequency_histogram(feature_frequency_df, selection_threshold, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_bootstraps = (
        int(feature_frequency_df["n_bootstraps"].iloc[0])
        if not feature_frequency_df.empty and "n_bootstraps" in feature_frequency_df
        else 0
    )

    n_features_total = len(feature_frequency_df)
    n_features_passing = (
        int(feature_frequency_df["selected_final"].sum())
        if not feature_frequency_df.empty and "selected_final" in feature_frequency_df
        else 0
    )

    if feature_frequency_df.empty:
        ax.text(
            0.5,
            0.5,
            "No bootstrap-selected features found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        frequencies = feature_frequency_df["selection_frequency"].to_numpy()
        ax.hist(
            frequencies,
            bins=np.linspace(0, 1, 51),
            density=True,
            alpha=0.65,
            color="steelblue",
            edgecolor="white",
            label="Feature frequency",
        )

        if gaussian_kde is not None and len(np.unique(frequencies)) > 1:
            try:
                density = gaussian_kde(frequencies)
                x_values = np.linspace(0, 1, 300)
                ax.plot(
                    x_values,
                    density(x_values),
                    color="black",
                    linewidth=1.8,
                    label="KDE density",
                )
            except Exception:
                pass

        max_frequency = float(np.max(frequencies))
        ax.axvline(
            max_frequency,
            color="darkorange",
            linestyle=":",
            linewidth=1.6,
            label=f"Max observed = {max_frequency:.3f}",
        )

    ax.axvline(
        selection_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.6,
        label=f"Selection threshold = {selection_threshold:.3f}",
    )
    ax.set_xlabel("Selection frequency across complete bootstraps")
    ax.set_ylabel("Density")
    ax.set_title(
        "Bootstrap feature selection frequency distribution\n"
        f"Complete bootstraps considered: {n_bootstraps}\n"
        f"Features passing threshold: {n_features_passing}/{n_features_total}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


parser = argparse.ArgumentParser(
    description="Summarise bootstrap feature-selection runs."
)
parser.add_argument("--bootstrap-root", required=True)
parser.add_argument("--train-path", required=True)
parser.add_argument("--test-path", required=True)
parser.add_argument("--target-column", default="is_progb")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--selection-threshold", type=float, default=0.9)
parser.add_argument(
    "--no-correlation-outputs",
    action="store_true",
    help=(
        "Summarise runs where no correlation-related outputs are expected. "
        "Only selected_features.csv is required, and no correlated-pair "
        "frequency table is written."
    ),
)
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

    missing_outputs = missing_required_outputs(
        bootstrap_dir,
        args.no_correlation_outputs,
    )
    if missing_outputs:
        success_rows.append(
            {
                "bootstrap_dir": str(bootstrap_dir),
                "bootstrap_id": bootstrap_id,
                "status": "ignored_missing_required_outputs",
                "n_selected_features": 0,
                "n_correlated_pairs": 0,
                "error": "Missing required output(s): "
                + ", ".join(missing_outputs),
            }
        )
        continue

    try:
        selected_features = read_selected_features(bootstrap_dir)
        if args.no_correlation_outputs:
            correlated_pairs = set()
        else:
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
save_feature_frequency_histogram(
    feature_frequency_df,
    args.selection_threshold,
    out_dir / "bootstrap_feature_frequency_histogram.png",
)

pair_frequency_path = out_dir / "bootstrap_correlated_pair_frequency.csv"
if args.no_correlation_outputs:
    if pair_frequency_path.exists():
        pair_frequency_path.unlink()
else:
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
        pair_frequency_path,
        index=False,
    )

selected_final_features = feature_frequency_df.loc[
    feature_frequency_df["selected_final"],
    "feature",
].tolist()
if not selected_final_features:
    if feature_frequency_df.empty:
        max_frequency_message = "No feature frequencies were observed."
    else:
        max_frequency = feature_frequency_df["selection_frequency"].max()
        max_frequency_message = (
            "Maximum observed selection_frequency was "
            f"{max_frequency:.3f}."
        )
    raise ValueError(
        "No features met the final bootstrap selection threshold "
        f"{args.selection_threshold}. {max_frequency_message}"
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
