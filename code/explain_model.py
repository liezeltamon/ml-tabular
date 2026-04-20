# sbatch -J explain_model -p long --mem=100G --output=%x.log.out --error=%x.log.err --wrap="python explain_model.py --model-path final_model.pkl --data-path ../data/test.csv --label-column label --out-dir ../results/explain_model/test_data --max-samples 1000000"

import argparse
from pathlib import Path

import joblib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# %% Parameters and defaults

SUPPORTED_ESTIMATOR_NAMES = {
    "LGBMClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "CatBoostClassifier",
    "XGBClassifier",
}


def normalize_shap_values(shap_values, n_classes):
    if isinstance(shap_values, list):
        normalized = [np.asarray(values) for values in shap_values]
    else:
        values = np.asarray(getattr(shap_values, "values", shap_values))

        if values.ndim == 2:
            if n_classes != 2:
                raise ValueError(
                    "Received 2D SHAP values for a model with more than two classes."
                )
            normalized = [-values, values]
        elif values.ndim == 3:
            if values.shape[1] == n_classes:
                normalized = [values[:, class_idx, :] for class_idx in range(n_classes)]
            elif values.shape[2] == n_classes:
                normalized = [values[:, :, class_idx] for class_idx in range(n_classes)]
            else:
                raise ValueError(
                    f"Could not align SHAP values with {n_classes} classes. Received shape {values.shape}."
                )
        else:
            raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    if len(normalized) != n_classes:
        raise ValueError(
            f"Expected SHAP outputs for {n_classes} classes, received {len(normalized)}."
        )

    return [np.asarray(class_values) for class_values in normalized]


def sanitize_label(value):
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    )
    return sanitized.strip("_") or "class"


def save_heatmap(heatmap_df, output_path, title):
    n_rows, n_cols = heatmap_df.shape
    fig_width = max(8, 0.7 * n_cols + 3)
    fig_height = max(4, 0.45 * n_rows + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(heatmap_df.columns, rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Class")
    ax.set_title(title)

    if n_rows <= 15 and n_cols <= 20:
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                value = heatmap_df.iat[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )

    fig.colorbar(image, ax=ax, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bar_grid(plot_items, output_path):
    if not plot_items:
        return

    n_panels = len(plot_items)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.6 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, plot_item in zip(flat_axes, plot_items):
        class_label = plot_item["class_label"]
        plot_df = plot_item["plot_df"]

        ax.barh(
            plot_df["display_feature"],
            plot_df["mean_abs_shap"],
            color="steelblue",
        )
        ax.set_title(str(class_label), fontsize=10)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.set_ylabel("Feature", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_beeswarm_grid(plot_items, output_path, top_features):
    if not plot_items:
        return

    n_panels = len(plot_items)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.8 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, plot_item in zip(flat_axes, plot_items):
        class_label = plot_item["class_label"]
        class_shap_values = plot_item["class_shap_values"]
        class_feature_df = plot_item["class_feature_df"]

        plt.sca(ax)
        shap.summary_plot(
            class_shap_values,
            class_feature_df,
            max_display=top_features,
            show=False,
            color_bar=False,
            plot_size=None,
        )
        ax.set_title(str(class_label), fontsize=10)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dependence_grid(plot_items, output_path):
    if not plot_items:
        return

    n_panels = len(plot_items)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.8 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, plot_item in zip(flat_axes, plot_items):
        shap.dependence_plot(
            plot_item["feature_name"],
            plot_item["class_shap_values"],
            plot_item["class_feature_df"],
            interaction_index="auto",
            ax=ax,
            show=False,
        )
        ax.set_title(str(plot_item["feature_name"]), fontsize=10)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %% Argument parsing

script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Explain class predictions from a saved sklearn pipeline with SHAP."
)
parser.add_argument(
    "--model-path",
    required=True,
    help="Path to a saved joblib pipeline such as final_model.pkl.",
)
parser.add_argument(
    "--data-path",
    default=str(script_dir.parent / "data" / "test.csv"),
    help="Path to the labeled CSV used for explanation.",
)
parser.add_argument(
    "--label-column",
    default="label",
    help="Name of the target column in the explanation dataset.",
)
parser.add_argument(
    "--out-dir",
    default=None,
    help="Directory for outputs. Defaults to ../results/explain_model/<model_stem>.",
)
parser.add_argument(
    "--top-features",
    type=int,
    default=5,
    help="Number of top features to show in class-level plots.",
)
parser.add_argument(
    "--heatmap-top-features-per-class",
    type=int,
    default=10,
    help="Number of top features per class used to build the heatmap union.",
)
parser.add_argument(
    "--max-samples",
    type=int,
    default=2000,
    help="Maximum number of rows used for SHAP computation.",
)
parser.add_argument(
    "--random-state",
    type=int,
    default=123,
    help="Random seed for reproducible sampling.",
)
args = parser.parse_args()

# %% Path resolution and validation

model_path = Path(args.model_path).resolve()
data_path = Path(args.data_path).resolve()

if args.out_dir is not None:
    out_dir = Path(args.out_dir).resolve()
else:
    out_dir = script_dir.parent / "results" / "explain_model" / model_path.stem

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

out_dir.mkdir(parents=True, exist_ok=True)

# %% Model loading and estimator checks

pipeline = joblib.load(model_path)

if not isinstance(pipeline, Pipeline):
    raise TypeError("Expected a fitted sklearn Pipeline saved with joblib.")
if len(pipeline.steps) < 2:
    raise ValueError(
        "Expected a pipeline with preprocessing steps and a final estimator."
    )

estimator = pipeline.steps[-1][1]
estimator_name = estimator.__class__.__name__
feature_transformer = pipeline[:-1]

if estimator_name not in SUPPORTED_ESTIMATOR_NAMES:
    supported = ", ".join(sorted(SUPPORTED_ESTIMATOR_NAMES))
    raise ValueError(
        f"Unsupported estimator '{estimator_name}'. Supported estimators: {supported}."
    )
if not hasattr(estimator, "predict_proba"):
    raise ValueError(
        "The final estimator must support predict_proba for class-level explanation."
    )
if not hasattr(estimator, "classes_"):
    raise ValueError("The final estimator appears unfitted because classes_ is missing.")
if not hasattr(feature_transformer, "transform"):
    raise ValueError("The pipeline preprocessing block must support transform().")

# %% Dataset loading

data_df = pd.read_csv(data_path, index_col=0)

if args.label_column not in data_df.columns:
    raise ValueError(
        f"Label column '{args.label_column}' not found in {data_path}."
    )

X = data_df.drop(columns=[args.label_column])
y_true = data_df[args.label_column]

# %% Predictions and sampling

y_pred = pipeline.predict(X)
y_proba = pipeline.predict_proba(X)
predicted_probabilities = y_proba.max(axis=1)

all_indices = np.arange(len(y_true))
if args.max_samples is None or args.max_samples <= 0 or len(y_true) <= args.max_samples:
    sample_indices = all_indices
else:
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    can_stratify = len(unique_classes) > 1 and np.all(class_counts >= 2)

    if can_stratify:
        try:
            sample_indices, _ = train_test_split(
                all_indices,
                train_size=args.max_samples,
                random_state=args.random_state,
                stratify=y_true,
            )
            sample_indices = np.sort(sample_indices)
        except ValueError:
            rng = np.random.default_rng(args.random_state)
            sample_indices = np.sort(
                rng.choice(all_indices, size=args.max_samples, replace=False)
            )
    else:
        rng = np.random.default_rng(args.random_state)
        sample_indices = np.sort(
            rng.choice(all_indices, size=args.max_samples, replace=False)
        )

X_sample = X.iloc[sample_indices].copy()
y_true_sample = y_true.iloc[sample_indices].reset_index(drop=True)
y_pred_sample = (
    pd.Series(y_pred, index=X.index).iloc[sample_indices].reset_index(drop=True)
)

# %% Preprocessing and SHAP computation

X_transformed_sample = feature_transformer.transform(X_sample)
if hasattr(X_transformed_sample, "toarray"):
    X_transformed_sample = X_transformed_sample.toarray()
else:
    X_transformed_sample = np.asarray(X_transformed_sample)

try:
    raw_feature_names = feature_transformer.get_feature_names_out()
except TypeError:
    raw_feature_names = feature_transformer.get_feature_names_out(X_sample.columns)
except AttributeError:
    raw_feature_names = np.array(
        [f"feature_{i}" for i in range(X_transformed_sample.shape[1])]
    )

raw_feature_names = np.asarray(raw_feature_names, dtype=str)
display_feature_names = np.array(
    [
        name.split("__", 1)[1] if "__" in name else name
        for name in raw_feature_names
    ],
    dtype=str,
)

duplicate_mask = (
    pd.Series(display_feature_names).duplicated(keep=False).to_numpy()
)
display_feature_names[duplicate_mask] = raw_feature_names[duplicate_mask]

if X_transformed_sample.shape[1] != len(raw_feature_names):
    raise ValueError(
        "The transformed feature matrix width does not match the recovered feature names."
    )

class_labels = np.asarray(estimator.classes_)
explainer = shap.TreeExplainer(estimator)
shap_by_class = normalize_shap_values(
    explainer.shap_values(X_transformed_sample),
    n_classes=len(class_labels),
)

# Keep the sampled rows aligned across SHAP, predictions, and plots.
sample_feature_df = pd.DataFrame(
    X_transformed_sample,
    columns=display_feature_names,
)

# %% Prediction summary output

prediction_summary_df = pd.DataFrame(
    {
        "row_index": X.index.astype(str),
        "true_label": y_true.to_numpy(),
        "pred_label": y_pred,
        "pred_probability": predicted_probabilities,
        "is_correct": y_true.to_numpy() == y_pred,
        "used_for_shap": False,
    }
)
prediction_summary_df.loc[sample_indices, "used_for_shap"] = True
prediction_summary_df.to_csv(out_dir / "prediction_summary.csv", index=False)

# %% Grouped outputs

group_configs = [
    ("by_true_label", y_true_sample.to_numpy()),
    ("by_pred_label", y_pred_sample.to_numpy()),
]

display_name_map = pd.Series(
    display_feature_names,
    index=raw_feature_names,
).to_dict()

for grouping_name, group_labels in group_configs:
    group_dir = out_dir / grouping_name
    group_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = group_dir / "tables"
    bars_dir = group_dir / "bars"
    beeswarms_dir = group_dir / "beeswarms"
    dependence_dir = group_dir / "dependence"

    tables_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    beeswarms_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir.mkdir(parents=True, exist_ok=True)

    class_counts = pd.Series(group_labels).value_counts().reindex(
        class_labels,
        fill_value=0,
    )
    class_counts.rename_axis("class_name").reset_index(name="n_cells").to_csv(
        group_dir / "class_counts.csv",
        index=False,
    )

    heatmap_feature_union = []
    heatmap_rows = []
    bar_grid_items = []
    beeswarm_grid_items = []

    for class_idx, class_label in enumerate(class_labels):
        class_mask = np.asarray(group_labels == class_label)
        safe_label = sanitize_label(class_label)

        if not np.any(class_mask):
            pd.DataFrame(
                columns=[
                    "raw_feature",
                    "display_feature",
                    "mean_abs_shap",
                    "mean_shap",
                ]
            ).to_csv(tables_dir / f"{safe_label}__top_features.csv", index=False)
            heatmap_rows.append(pd.Series(dtype=float, name=str(class_label)))
            continue

        class_shap_values = shap_by_class[class_idx][class_mask]
        class_feature_df = sample_feature_df.iloc[class_mask]

        summary_df = pd.DataFrame(
            {
                "raw_feature": raw_feature_names,
                "display_feature": display_feature_names,
                "mean_abs_shap": np.mean(np.abs(class_shap_values), axis=0),
                "mean_shap": np.mean(class_shap_values, axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        summary_df.head(args.top_features).to_csv(
            tables_dir / f"{safe_label}__top_features.csv",
            index=False,
        )

        plot_df = summary_df.head(args.top_features).iloc[::-1]
        bar_grid_items.append(
            {
                "class_label": class_label,
                "plot_df": plot_df,
            }
        )
        fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(plot_df) + 1)))
        ax.barh(
            plot_df["display_feature"],
            plot_df["mean_abs_shap"],
            color="steelblue",
        )
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("Feature")
        ax.set_title(f"{class_label}: top features")
        fig.tight_layout()
        fig.savefig(bars_dir / f"{safe_label}__bar.png", dpi=300)
        plt.close(fig)

        plt.figure(figsize=(8, max(4, 0.55 * args.top_features + 1)))
        shap.summary_plot(
            class_shap_values,
            class_feature_df,
            max_display=args.top_features,
            show=False,
        )
        plt.title(f"{class_label}: SHAP distribution")
        plt.tight_layout()
        plt.savefig(
            beeswarms_dir / f"{safe_label}__beeswarm.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        beeswarm_grid_items.append(
            {
                "class_label": class_label,
                "class_shap_values": class_shap_values,
                "class_feature_df": class_feature_df,
            }
        )

        dependence_feature_names = summary_df.head(args.top_features)[
            "display_feature"
        ].tolist()
        dependence_grid_items = []

        for feature_name in dependence_feature_names:
            safe_feature = sanitize_label(feature_name)

            dependence_grid_items.append(
                {
                    "feature_name": feature_name,
                    "class_shap_values": class_shap_values,
                    "class_feature_df": class_feature_df,
                }
            )

            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            shap.dependence_plot(
                feature_name,
                class_shap_values,
                class_feature_df,
                interaction_index="auto",
                ax=ax,
                show=False,
            )
            ax.set_title(f"{class_label}: {feature_name}", fontsize=10)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xlabel(ax.get_xlabel(), fontsize=9)
            ax.set_ylabel(ax.get_ylabel(), fontsize=9)
            fig.tight_layout()
            fig.savefig(
                dependence_dir / f"{safe_label}__dependence__{safe_feature}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        save_dependence_grid(
            dependence_grid_items,
            dependence_dir / f"{safe_label}__dependence_grid.png",
        )

        # Use the union of each class's strongest features to build the heatmap.
        heatmap_top_df = summary_df.head(args.heatmap_top_features_per_class)
        heatmap_feature_union.extend(heatmap_top_df["raw_feature"].tolist())
        heatmap_rows.append(
            heatmap_top_df.set_index("raw_feature")["mean_abs_shap"].rename(
                str(class_label)
            )
        )

    heatmap_features = pd.Index(heatmap_feature_union).unique()
    if len(heatmap_features) == 0:
        heatmap_df = pd.DataFrame(index=[str(label) for label in class_labels])
    else:
        heatmap_df = pd.DataFrame(
            heatmap_rows,
            index=[str(label) for label in class_labels],
        ).reindex(columns=heatmap_features, fill_value=0.0)
        column_order = heatmap_df.max(axis=0).sort_values(ascending=False).index
        heatmap_df = heatmap_df.loc[:, column_order]

    heatmap_df.to_csv(group_dir / "class_feature_heatmap_values.csv")
    heatmap_display_df = heatmap_df.rename(columns=display_name_map)

    save_bar_grid(
        bar_grid_items,
        group_dir / "all_classes__bar_grid.png",
    )
    save_beeswarm_grid(
        beeswarm_grid_items,
        group_dir / "all_classes__beeswarm_grid.png",
        args.top_features,
    )

    if heatmap_display_df.shape[1] > 0:
        save_heatmap(
            heatmap_display_df,
            group_dir / "class_feature_heatmap.png",
            f"{grouping_name}: class-feature SHAP summary",
        )

# %% Final summary

print(f"Saved explanation outputs to: {out_dir}")
