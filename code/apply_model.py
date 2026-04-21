# %% Apply a saved model to unseen data and summarise prediction confidence
# sbatch -J apply_model -p long --mem=40G --output=%x.log.out --error=%x.log.err --wrap="python apply_model.py --model-path ../results/tune_models/cytof_annotation_parallel_calibrated_uncalibrated/final_model_calibrated.pkl --data-path ../data/test.csv --mlflow-experiment-name cytof_annotation_parallel_calibrated_uncalibrated --label-key label"

import argparse
from collections import deque
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# %% Functions

def recover_feature_names_in(loaded_model):
    candidate_queue = deque([("loaded_model", loaded_model)])
    visited_ids = set()
    candidate_sources = []

    while candidate_queue:
        source_name, current = candidate_queue.popleft()

        if current is None:
            continue

        current_id = id(current)
        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)

        feature_names = getattr(current, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = np.asarray(feature_names, dtype=str)
            if feature_names.ndim == 1 and len(feature_names) > 0:
                return feature_names.tolist(), source_name
            candidate_sources.append(source_name)

        for attr_name in [
            "estimator",
            "base_estimator",
            "preprocessor",
            "transformer",
            "feature_transformer",
        ]:
            if hasattr(current, attr_name):
                candidate_queue.append(
                    (f"{source_name}.{attr_name}", getattr(current, attr_name))
                )

        named_steps = getattr(current, "named_steps", None)
        if named_steps is not None:
            for step_name, step_obj in named_steps.items():
                candidate_queue.append((f"{source_name}.named_steps['{step_name}']", step_obj))

        steps = getattr(current, "steps", None)
        if steps is not None:
            for step_name, step_obj in steps:
                candidate_queue.append((f"{source_name}.steps['{step_name}']", step_obj))

    checked_sources = ", ".join(candidate_sources) if candidate_sources else "none"
    raise ValueError(
        "Could not recover fitted feature names from the saved model. "
        "Checked feature_names_in_ on nested model objects and found no usable values. "
        f"Visited sources: {checked_sources}."
    )

# %% Argument parsing

script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Apply a saved classifier to unseen data and export confidence diagnostics."
)
parser.add_argument(
    "--model-path",
    required=True,
    help="Path to a saved fitted joblib model.",
)
parser.add_argument(
    "--data-path",
    required=True,
    help="Path to the unseen CSV to score.",
)
parser.add_argument(
    "--mlflow-experiment-name",
    required=True,
    help="Experiment name used to derive the default results directory.",
)
parser.add_argument(
    "--label-key",
    default="pred_label",
    help="Column name to use for predicted labels in the output CSV.",
)
parser.add_argument(
    "--out-dir",
    default=None,
    help=(
        "Optional output directory. Defaults to "
        "../results/apply_model/<experiment_name>/<model_stem>__<data_stem>."
    ),
)
args = parser.parse_args()

# %% Path resolution

model_path = Path(args.model_path).resolve()
data_path = Path(args.data_path).resolve()

if args.out_dir is not None:
    out_dir = Path(args.out_dir).resolve()
else:
    out_dir = (
        script_dir.parent
        / "results"
        / "apply_model"
        / args.mlflow_experiment_name
        / f"{model_path.stem}__{data_path.stem}"
    )

predictions_path = out_dir / "predictions.csv"
plot_confidence_distribution_path = out_dir / "plot_confidence_distribution.png"
plot_predicted_label_fraction_path = out_dir / "plot_predicted_label_fraction.png"

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

out_dir.mkdir(parents=True, exist_ok=True)

# %% ----- MAIN -----

# %% Load model and validate prediction interface

loaded_model = joblib.load(model_path)

if not hasattr(loaded_model, "predict"):
    raise TypeError("Loaded model does not expose predict().")
if not hasattr(loaded_model, "predict_proba"):
    raise TypeError("Loaded model does not expose predict_proba().")
if not hasattr(loaded_model, "classes_"):
    raise TypeError("Loaded model appears unfitted because classes_ is missing.")

model_feature_names, feature_source = recover_feature_names_in(loaded_model)

if len(model_feature_names) != len(set(model_feature_names)):
    duplicate_model_features = (
        pd.Series(model_feature_names)[
            pd.Series(model_feature_names).duplicated(keep=False)
        ]
        .tolist()
    )
    raise ValueError(
        "Recovered duplicate feature names from the saved model metadata: "
        f"{sorted(set(duplicate_model_features))}"
    )

# %% Load and align unseen data

data_df = pd.read_csv(data_path, index_col=0)

if len(data_df.columns) != len(set(data_df.columns)):
    duplicate_input_columns = (
        pd.Series(data_df.columns)[pd.Series(data_df.columns).duplicated(keep=False)]
        .tolist()
    )
    raise ValueError(
        "Input data contains duplicate column names: "
        f"{sorted(set(duplicate_input_columns))}"
    )

input_columns = data_df.columns.tolist()
input_column_set = set(input_columns)
model_feature_set = set(model_feature_names)
input_feature_order = [column for column in input_columns if column in model_feature_set]

missing_features = [column for column in model_feature_names if column not in input_column_set]
extra_columns = [column for column in input_columns if column not in model_feature_set]

if missing_features:
    raise ValueError(
        "Input data is missing feature columns required by the saved model: "
        f"{missing_features}"
    )

X = data_df.loc[:, model_feature_names].copy()
input_feature_order_matches_model = input_feature_order == model_feature_names

if len(X) == 0:
    raise ValueError("Input data contains zero rows after feature alignment.")

# %% Predict and compute confidence metrics

y_pred = loaded_model.predict(X)
print("Raw y_pred shape:", np.asarray(y_pred).shape)
y_pred = np.asarray(y_pred).reshape(-1)
print("Flattened y_pred shape:", y_pred.shape)
y_proba = np.asarray(loaded_model.predict_proba(X), dtype=float)

if y_proba.ndim != 2:
    raise ValueError(f"Expected predict_proba() to return a 2D array, got {y_proba.ndim}D.")
if y_proba.shape[0] != len(X):
    raise ValueError(
        "The number of probability rows does not match the number of input rows."
    )

class_labels = np.asarray(loaded_model.classes_)
n_classes = len(class_labels)

if y_proba.shape[1] != n_classes:
    raise ValueError(
        "The probability matrix width does not match the number of fitted classes. "
        f"Received {y_proba.shape[1]} probability columns for {n_classes} classes."
    )

sorted_proba = np.sort(y_proba, axis=1)[:, ::-1]
confidence = sorted_proba[:, 0]

if n_classes >= 2:
    confidence_margin_top1_top2 = sorted_proba[:, 0] - sorted_proba[:, 1]
else:
    confidence_margin_top1_top2 = np.full(len(X), np.nan)

if n_classes >= 3:
    confidence_margin_top2_top3 = sorted_proba[:, 1] - sorted_proba[:, 2]
else:
    confidence_margin_top2_top3 = np.full(len(X), np.nan)

safe_proba = np.clip(y_proba, 1e-15, 1.0)
confidence_entropy = -(safe_proba * np.log(safe_proba)).sum(axis=1)

if n_classes > 1:
    confidence_entropy_normalized = confidence_entropy / np.log(n_classes)
else:
    confidence_entropy_normalized = np.zeros(len(X), dtype=float)

prediction_df = pd.DataFrame(
    {
        args.label_key: y_pred,
        "confidence": confidence,
        "confidence_margin_top1_top2": confidence_margin_top1_top2,
        "confidence_margin_top2_top3": confidence_margin_top2_top3,
        "confidence_entropy": confidence_entropy,
        "confidence_entropy_normalized": confidence_entropy_normalized,
    },
    index=data_df.index,
)

index_label = data_df.index.name if data_df.index.name else "index"
prediction_df.to_csv(predictions_path, index=True, index_label=index_label)

# %% Summaries used by plots and console output

label_summary_df = (
    prediction_df.groupby(args.label_key, dropna=False)
    .agg(
        predicted_count=(args.label_key, "size"),
        median_confidence=("confidence", "median"),
    )
    .sort_values(
        ["median_confidence", "predicted_count"],
        ascending=[False, False],
    )
)
ordered_labels = label_summary_df.index.tolist()
predicted_label_counts = label_summary_df["predicted_count"]
predicted_label_fractions = predicted_label_counts / predicted_label_counts.sum()

# %% Plot predicted label fractions

label_tick_fontsize = 8

fig, ax = plt.subplots(
    figsize=(10, max(4.5, 0.35 * len(predicted_label_counts) + 2.5))
)
bars = ax.barh(
    [str(label) for label in predicted_label_fractions.index],
    predicted_label_fractions.values,
    color="steelblue",
)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel("Fraction of predictions")
ax.set_ylabel("Predicted label")
ax.set_title("Predicted label fraction")
ax.tick_params(axis="y", labelsize=label_tick_fontsize)
ax.invert_yaxis()

max_fraction = (
    float(predicted_label_fractions.max()) if len(predicted_label_fractions) > 0 else 0.0
)
text_offset = max(max_fraction * 0.015, 0.005)
ax.set_xlim(0, max_fraction + text_offset * 8)

for bar, count in zip(bars, predicted_label_counts.values):
    ax.text(
        bar.get_width() + text_offset,
        bar.get_y() + bar.get_height() / 2,
        str(int(count)),
        ha="left",
        va="center",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(plot_predicted_label_fraction_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# %% Plot confidence diagnostics

plot_metrics = [
    ("confidence", "Confidence (max probability)"),
    ("confidence_margin_top1_top2", "Margin top-1 minus top-2"),
    ("confidence_entropy_normalized", "Normalized entropy"),
]

if n_classes >= 3:
    plot_metrics.append(
        ("confidence_margin_top2_top3", "Margin top-2 minus top-3")
    )

n_rows = len(plot_metrics)
fig, axes = plt.subplots(
    n_rows,
    2,
    figsize=(16, max(4.5 * n_rows, 0.32 * len(ordered_labels) * n_rows)),
    squeeze=False,
    gridspec_kw={"width_ratios": [1.0, 1.35]},
)

for row_idx, (metric_name, metric_title) in enumerate(plot_metrics):
    overall_ax = axes[row_idx, 0]
    by_class_ax = axes[row_idx, 1]

    metric_values = prediction_df[metric_name].dropna().to_numpy()
    if metric_values.size == 0:
        overall_ax.text(
            0.5,
            0.5,
            "No values available",
            ha="center",
            va="center",
            transform=overall_ax.transAxes,
        )
        overall_ax.set_axis_off()
    else:
        overall_ax.hist(metric_values, bins=30, color="steelblue", edgecolor="black")
        overall_ax.set_xlabel(metric_title)
        overall_ax.set_ylabel("Count")
        overall_ax.set_title(f"{metric_title}: overall")

    grouped_metric_values = [
        prediction_df.loc[prediction_df[args.label_key] == label, metric_name]
        .dropna()
        .to_numpy()
        for label in ordered_labels
    ]
    grouped_metric_values = [
        values for values in grouped_metric_values if values.size > 0
    ]
    grouped_labels = [
        str(label)
        for label in ordered_labels
        if prediction_df.loc[prediction_df[args.label_key] == label, metric_name]
        .dropna()
        .size
        > 0
    ]

    if not grouped_metric_values:
        by_class_ax.text(
            0.5,
            0.5,
            "No class-wise values available",
            ha="center",
            va="center",
            transform=by_class_ax.transAxes,
        )
        by_class_ax.set_axis_off()
    else:
        by_class_ax.boxplot(
            grouped_metric_values,
            labels=grouped_labels,
            showfliers=False,
            vert=False,
        )
        by_class_ax.set_xlabel(metric_title)
        by_class_ax.set_ylabel("Predicted label")
        by_class_ax.set_title(f"{metric_title}: by predicted class")
        by_class_ax.tick_params(axis="y", labelsize=label_tick_fontsize)
        by_class_ax.invert_yaxis()

fig.subplots_adjust(left=0.08, right=0.98, wspace=0.55, hspace=0.35)
fig.tight_layout()
fig.savefig(plot_confidence_distribution_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# %% Console summary

print(f"Loaded model from: {model_path}")
print(f"Recovered feature names from: {feature_source}")
print(f"Loaded unseen data from: {data_path}")
print(f"Scored rows: {len(X)}")
print(f"Classes: {n_classes}")

if extra_columns:
    print(f"Dropped extra input columns not used by the model: {extra_columns}")
else:
    print("Dropped extra input columns not used by the model: none")

if input_feature_order_matches_model:
    print("Input feature order already matched the model feature order.")
else:
    print("Reordered input features to match the model feature order.")

print("Predicted label counts:")
for label, count in predicted_label_counts.items():
    print(f"  {label}: {count}")

print(f"Saved predictions to: {predictions_path}")
print(f"Saved confidence plot to: {plot_confidence_distribution_path}")
print(f"Saved predicted label fraction plot to: {plot_predicted_label_fraction_path}")
