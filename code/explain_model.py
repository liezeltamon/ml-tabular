# sbatch -J explain_model -p long --mem=100G --output=%x.log.out --error=%x.log.err --wrap="python explain_model.py --model-path final_model_calibrated.pkl --data-path ../data/test.csv --label-column label --out-dir ../results/explain_model/test_data --max-samples 1000000 --mean-abs-shap-threshold 0"

# sbatch -J explain_model_lda_progb_vs_nonprogb_selectkbest_p005_top5 -p long --mem=100G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python explain_model.py --model-path results/tune_models/progb_vs_nonprogb_selectkbest_p005_top5/final_model_uncalibrated.pkl --data-path results/select_features/progb_vs_nonprogb_selectkbest_p005/test.csv --background-data-path results/select_features/progb_vs_nonprogb_selectkbest_p005/train.csv --label-column is_progb --out-dir results/explain_model/progb_vs_nonprogb_selectkbest_p005_top5/uncalibrated --max-samples 0 --max-background-samples 0 --mean-abs-shap-threshold 0"

# sbatch -J explain_model_xgb_progb_vs_nonprogb_selectkbest_p005_nocorr_smartcorr_c09_cvfolds -p long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python explain_model.py --model-dir /well/immune-rep/users/yfg436/git/ml-tabular/results/tune_models/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/cv_fold_models/xgb/uncalibrated --data-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/test.csv /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/train.csv --background-data-path /well/immune-rep/users/yfg436/git/ml-tabular/results/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/train.csv --label-column is_progb --out-dir results/explain_model/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection_smartcorrelation_c09/xgb_cvfolds_uncalibrated --max-samples 0 --max-background-samples 0.9 --mean-abs-shap-threshold 0"

# sbatch -J explain_model_lda_progb_vs_nonprogb_selectkbest_p005_nocorr_summary_cvfolds -p long --mem=50G --output=logs/%x.log.out --error=logs/%x.log.err --wrap="python explain_model.py --model-dir /well/immune-rep/users/yfg436/git/ml-tabular/results/tune_models/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/cv_fold_models/lda/uncalibrated --data-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/test.csv /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/train.csv --background-data-path /well/immune-rep/users/yfg436/git/ml-tabular/results/summarise_bootstrap_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/train.csv --label-column is_progb --out-dir results/explain_model/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/lda_cvfolds_uncalibrated --max-samples 0 --max-background-samples 0.9 --mean-abs-shap-threshold 0"

import argparse
import joblib
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

os.chdir(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        universal_newlines=True,
    ).strip()
)

# %% Parameters and defaults

SUPPORTED_ESTIMATOR_NAMES = {
    "LGBMClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "CatBoostClassifier",
    "XGBClassifier",
    "LinearDiscriminantAnalysis",
}

def reshape_shap_values_container(shap_values, class_labels):
    class_labels = np.asarray(class_labels)
    n_classes = len(class_labels)

    if isinstance(shap_values, list):
        shap_by_output = [np.asarray(values) for values in shap_values]
        shap_output_labels = class_labels
    else:
        values = np.asarray(getattr(shap_values, "values", shap_values))

        if values.ndim == 2:
            if n_classes != 2:
                raise ValueError(
                    "Received 2D SHAP values for a model with more than two classes; "
                    "there is no class/output axis to reshape safely."
                )
            shap_by_output = [values]
            shap_output_labels = np.asarray([class_labels[-1]])
        elif values.ndim == 3:
            if values.shape[1] == n_classes:
                shap_by_output = [
                    values[:, class_idx, :] for class_idx in range(n_classes)
                ]
                shap_output_labels = class_labels
            elif values.shape[2] == n_classes:
                shap_by_output = [
                    values[:, :, class_idx] for class_idx in range(n_classes)
                ]
                shap_output_labels = class_labels
            else:
                raise ValueError(
                    f"Could not align SHAP values with {n_classes} classes. Received shape {values.shape}."
                )
        else:
            raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

    if len(shap_by_output) != len(shap_output_labels):
        raise ValueError(
            "Number of SHAP output matrices does not match number of output labels."
        )

    if len(shap_by_output) not in {1, n_classes}:
        raise ValueError(
            f"Expected 1 or {n_classes} SHAP outputs, received {len(shap_by_output)}."
        )

    return [np.asarray(values) for values in shap_by_output], shap_output_labels


def get_output_shap_values(shap_by_output, shap_output_labels, output_label):
    output_matches = np.where(np.asarray(shap_output_labels) == output_label)[0]
    if len(output_matches) == 0:
        available = ", ".join(map(str, shap_output_labels))
        raise ValueError(
            f"Could not find SHAP output for label {output_label!r}. "
            f"Available SHAP outputs: {available}"
        )
    return shap_by_output[output_matches[0]]


def build_row_matched_shap_values(shap_by_output, shap_output_labels, group_labels):
    shap_output_labels = np.asarray(shap_output_labels)
    group_labels = np.asarray(group_labels)
    shap_by_output = [np.asarray(values) for values in shap_by_output]

    if not shap_by_output:
        raise ValueError("Cannot build row-matched SHAP values without SHAP outputs.")

    n_features = shap_by_output[0].shape[1]
    matched_values = []
    matched_row_indices = []

    for row_idx, group_label in enumerate(group_labels):
        output_matches = np.where(shap_output_labels == group_label)[0]
        if len(output_matches) == 0:
            continue

        output_values = shap_by_output[output_matches[0]]
        matched_values.append(output_values[row_idx])
        matched_row_indices.append(row_idx)

    if not matched_values:
        return (
            np.empty((0, n_features), dtype=shap_by_output[0].dtype),
            np.asarray([], dtype=int),
        )

    return np.vstack(matched_values), np.asarray(matched_row_indices, dtype=int)


def sanitize_label(value):
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    )
    return sanitized.strip("_") or "class"


def to_dense_array(values):
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def resolve_sample_size(n_rows, max_samples):
    if max_samples is None or max_samples <= 0:
        return n_rows
    if 0 < max_samples < 1:
        return max(1, int(np.ceil(n_rows * max_samples)))
    return int(np.ceil(max_samples))


def select_sample_indices(row_labels, max_samples, random_state):
    all_indices = np.arange(len(row_labels))
    sample_size = resolve_sample_size(len(row_labels), max_samples)

    if len(row_labels) <= sample_size:
        return all_indices

    unique_classes, class_counts = np.unique(row_labels, return_counts=True)
    can_stratify = len(unique_classes) > 1 and np.all(class_counts >= 2)

    if can_stratify:
        try:
            sample_indices, _ = train_test_split(
                all_indices,
                train_size=sample_size,
                random_state=random_state,
                stratify=row_labels,
            )
            return np.sort(sample_indices)
        except ValueError:
            pass

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(all_indices, size=sample_size, replace=False))


def load_background_data(background_data_path, label_column, reference_columns):
    background_df = pd.read_csv(background_data_path, index_col=0)
    background_labels = None

    if label_column in background_df.columns:
        background_labels = background_df[label_column].copy()
        background_df = background_df.drop(columns=[label_column])

    if list(background_df.columns) != list(reference_columns):
        raise ValueError(
            "Background feature columns must exactly match explanation data columns."
        )

    return background_df, background_labels


def load_explanation_data(data_paths, label_column):
    loaded_dfs = []
    data_sources = []
    reference_feature_columns = None

    for data_path in data_paths:
        data_df = pd.read_csv(data_path, index_col=0)

        if label_column not in data_df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in {data_path}."
            )

        feature_columns = data_df.drop(columns=[label_column]).columns.tolist()
        if reference_feature_columns is None:
            reference_feature_columns = feature_columns
        elif feature_columns != reference_feature_columns:
            raise ValueError(
                "All explanation data files must have identical ordered feature "
                f"columns. Mismatch found in {data_path}."
            )

        loaded_dfs.append(data_df)
        data_sources.extend([sanitize_label(data_path.stem)] * data_df.shape[0])

    combined_df = pd.concat(loaded_dfs, axis=0)
    return combined_df, pd.Series(data_sources, index=combined_df.index)


def build_explainer(
    estimator,
    X_transformed_background,
    class_labels,
    display_feature_names,
):
    explainer = shap.Explainer(
        estimator,
        X_transformed_background,
        algorithm="auto",
        output_names=class_labels,
        feature_names=display_feature_names,
    )
    return explainer, explainer.__class__.__name__


def load_model_paths(model_path, model_dir, model_glob):
    if model_path is not None:
        return [Path(model_path).resolve()]

    resolved_model_dir = Path(model_dir).resolve()
    model_paths = sorted(resolved_model_dir.glob(model_glob))
    if not model_paths:
        raise FileNotFoundError(
            f"No model files matched {model_glob!r} under {resolved_model_dir}."
        )
    return model_paths


def get_pipeline_components(pipeline, model_path):
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Expected a fitted sklearn Pipeline saved with joblib: {model_path}"
        )
    if len(pipeline.steps) < 2:
        raise ValueError(
            f"Expected a pipeline with preprocessing steps and a final estimator: {model_path}"
        )

    estimator = pipeline.steps[-1][1]
    estimator_name = estimator.__class__.__name__
    feature_transformer = pipeline[:-1]

    if estimator_name not in SUPPORTED_ESTIMATOR_NAMES:
        supported = ", ".join(sorted(SUPPORTED_ESTIMATOR_NAMES))
        raise ValueError(
            f"Unsupported estimator '{estimator_name}' in {model_path}. "
            f"Supported estimators: {supported}."
        )
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"The final estimator in {model_path} must support predict_proba."
        )
    if not hasattr(estimator, "classes_"):
        raise ValueError(
            f"The final estimator in {model_path} appears unfitted because classes_ is missing."
        )
    if not hasattr(feature_transformer, "transform"):
        raise ValueError(
            f"The pipeline preprocessing block in {model_path} must support transform()."
        )

    return estimator, estimator_name, feature_transformer


def get_transformed_feature_names(feature_transformer, X_sample, n_features):
    try:
        raw_feature_names = feature_transformer.get_feature_names_out()
    except TypeError:
        raw_feature_names = feature_transformer.get_feature_names_out(X_sample.columns)
    except AttributeError:
        raw_feature_names = np.array(
            [f"feature_{i}" for i in range(n_features)]
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

    return raw_feature_names, display_feature_names


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
        figsize=(4.8 * n_cols, 5.8 * n_rows),
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


def save_beeswarm_grid(plot_items, output_path):
    if not plot_items:
        return

    n_panels = len(plot_items)
    n_cols = int(np.ceil(np.sqrt(n_panels)))
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.0 * n_cols, 7.2 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, plot_item in zip(flat_axes, plot_items):
        class_label = plot_item["class_label"]
        class_shap_values = plot_item["class_shap_values"]
        class_feature_df = plot_item["class_feature_df"]
        max_display = plot_item["max_display"]

        plt.sca(ax)
        shap.summary_plot(
            class_shap_values,
            class_feature_df,
            max_display=max_display,
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


def style_dependence_plot(fig, ax, title, fontsize=5):
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)

    for plot_ax in fig.axes:
        legend = plot_ax.get_legend()
        if legend is not None:
            legend.get_title().set_fontsize(fontsize)
            for text in legend.get_texts():
                text.set_fontsize(fontsize)

        if plot_ax is not ax:
            plot_ax.set_title(plot_ax.get_title(), fontsize=fontsize)
            plot_ax.set_xlabel(plot_ax.get_xlabel(), fontsize=fontsize)
            plot_ax.set_ylabel(plot_ax.get_ylabel(), fontsize=fontsize)
            plot_ax.tick_params(axis="both", labelsize=fontsize)


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
        style_dependence_plot(fig, ax, str(plot_item["feature_name"]))

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_display_summary(summary_df, top_features, mean_abs_shap_threshold):
    display_df = summary_df[
        summary_df["mean_abs_shap"] > mean_abs_shap_threshold
    ]
    if top_features is not None:
        display_df = display_df.head(top_features)
    return display_df.copy()


def save_mean_abs_shap_distribution(summary_df, output_path, mean_abs_shap_threshold):
    values = summary_df["mean_abs_shap"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if len(values) > 0:
        bins = min(50, max(10, int(np.sqrt(len(values)))))
        ax.hist(values, bins=bins, color="steelblue", alpha=0.85)
    else:
        ax.text(
            0.5,
            0.5,
            "No features",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    if mean_abs_shap_threshold is not None:
        ax.axvline(
            mean_abs_shap_threshold,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {mean_abs_shap_threshold:g}",
        )
        ax.legend()

    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature count")
    ax.set_title("Mean |SHAP| distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_category_outputs(
    category_label,
    safe_label,
    category_shap_values,
    category_model_shap_values,
    model_labels,
    category_feature_df,
    raw_feature_names,
    display_feature_names,
    top_features,
    heatmap_top_features,
    mean_abs_shap_threshold,
    tables_dir,
    bars_dir,
    beeswarms_dir,
    dependence_dir,
    mean_abs_shap_distributions_dir,
):
    fold_mode = category_model_shap_values is not None and len(model_labels) > 1
    base_columns = [
        "raw_feature",
        "display_feature",
        "mean_abs_shap",
        "mean_shap",
    ]
    fold_columns = []
    if fold_mode:
        fold_columns = [
            f"{model_label}_mean_abs_shap"
            for model_label in model_labels
        ] + [
            "mean_abs_shap_sd",
            "mean_abs_shap_min",
            "mean_abs_shap_max",
        ]

    if category_shap_values.shape[0] == 0:
        pd.DataFrame(
            columns=base_columns + fold_columns
        ).to_csv(tables_dir / f"{safe_label}__top_features.csv", index=False)
        return {
            "heatmap_features": [],
            "heatmap_row": pd.Series(dtype=float, name=str(category_label)),
            "bar_grid_item": None,
            "beeswarm_grid_item": None,
        }

    if fold_mode:
        fold_mean_abs_shap = np.vstack(
            [
                np.mean(np.abs(model_shap_values), axis=0)
                for model_shap_values in category_model_shap_values
            ]
        )
        mean_abs_shap = fold_mean_abs_shap.mean(axis=0)
    else:
        fold_mean_abs_shap = None
        mean_abs_shap = np.mean(np.abs(category_shap_values), axis=0)

    summary_df = pd.DataFrame(
        {
            "feature_index": np.arange(len(raw_feature_names)),
            "raw_feature": raw_feature_names,
            "display_feature": display_feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": np.mean(category_shap_values, axis=0),
        }
    )

    if fold_mode:
        for model_idx, model_label in enumerate(model_labels):
            summary_df[f"{model_label}_mean_abs_shap"] = fold_mean_abs_shap[
                model_idx,
            ]
        summary_df["mean_abs_shap_sd"] = fold_mean_abs_shap.std(axis=0)
        summary_df["mean_abs_shap_min"] = fold_mean_abs_shap.min(axis=0)
        summary_df["mean_abs_shap_max"] = fold_mean_abs_shap.max(axis=0)

    summary_df = summary_df.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    summary_df.drop(columns=["feature_index"]).to_csv(
        tables_dir / f"{safe_label}__top_features.csv",
        index=False,
    )

    save_mean_abs_shap_distribution(
        summary_df,
        mean_abs_shap_distributions_dir
        / f"{safe_label}__mean_abs_shap_distribution.png",
        mean_abs_shap_threshold,
    )
    save_mean_abs_shap_distribution(
        summary_df[summary_df["mean_abs_shap"] > 0],
        mean_abs_shap_distributions_dir
        / f"{safe_label}__mean_abs_shap_distribution_nonzero.png",
        mean_abs_shap_threshold,
    )

    display_summary_df = select_display_summary(
        summary_df,
        top_features,
        mean_abs_shap_threshold,
    )
    if display_summary_df.empty:
        return {
            "heatmap_features": [],
            "heatmap_row": pd.Series(dtype=float, name=str(category_label)),
            "bar_grid_item": None,
            "beeswarm_grid_item": None,
        }

    selected_feature_indices = display_summary_df["feature_index"].to_numpy()
    selected_feature_names = display_summary_df["display_feature"].to_numpy()
    selected_shap_values = category_shap_values[:, selected_feature_indices]
    selected_feature_df = category_feature_df.loc[:, selected_feature_names]

    plot_df = display_summary_df.iloc[::-1]
    bar_grid_item = {
        "class_label": category_label,
        "plot_df": plot_df,
    }
    fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(plot_df) + 1)))
    ax.barh(
        plot_df["display_feature"],
        plot_df["mean_abs_shap"],
        color="steelblue",
    )
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    ax.set_title(f"{category_label}: top features")
    fig.tight_layout()
    fig.savefig(bars_dir / f"{safe_label}__bar.png", dpi=300)
    plt.close(fig)

    plt.figure(figsize=(20, max(4, 0.55 * len(display_summary_df) + 1)))
    shap.summary_plot(
        selected_shap_values,
        selected_feature_df,
        max_display=len(display_summary_df),
        show=False,
        plot_size=None,
    )
    plt.title(f"{category_label}: SHAP distribution")
    plt.tight_layout()
    plt.savefig(
        beeswarms_dir / f"{safe_label}__beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    beeswarm_grid_item = {
        "class_label": category_label,
        "class_shap_values": selected_shap_values,
        "class_feature_df": selected_feature_df,
        "max_display": len(display_summary_df),
    }

    dependence_feature_names = display_summary_df["display_feature"].tolist()
    dependence_grid_items = []

    for feature_name in dependence_feature_names:
        safe_feature = sanitize_label(feature_name)

        dependence_grid_items.append(
            {
                "feature_name": feature_name,
                "class_shap_values": selected_shap_values,
                "class_feature_df": selected_feature_df,
            }
        )

        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        shap.dependence_plot(
            feature_name,
            selected_shap_values,
            selected_feature_df,
            interaction_index="auto",
            ax=ax,
            show=False,
        )
        style_dependence_plot(fig, ax, f"{category_label}: {feature_name}")
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

    heatmap_top_df = display_summary_df
    if heatmap_top_features is not None:
        heatmap_top_df = heatmap_top_df.head(heatmap_top_features)
    return {
        "heatmap_features": heatmap_top_df["raw_feature"].tolist(),
        "heatmap_row": heatmap_top_df.set_index("raw_feature")[
            "mean_abs_shap"
        ].rename(str(category_label)),
        "bar_grid_item": bar_grid_item,
        "beeswarm_grid_item": beeswarm_grid_item,
    }

# %% Argument parsing

script_dir = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Explain class predictions from a saved sklearn pipeline with SHAP."
)
model_input_group = parser.add_mutually_exclusive_group(required=True)
model_input_group.add_argument(
    "--model-path",
    help="Path to a saved joblib pipeline such as final_model_calibrated.pkl.",
)
model_input_group.add_argument(
    "--model-dir",
    help="Directory containing saved fold model pipelines.",
)
parser.add_argument(
    "--model-glob",
    default="fold_*.pkl",
    help="Glob used with --model-dir to select saved fold model pipelines.",
)
parser.add_argument(
    "--data-path",
    nargs="+",
    default=[str(script_dir.parent / "data" / "test.csv")],
    help=(
        "One or more labeled CSVs used for explanation. Multiple files are "
        "row-bound."
    ),
)
parser.add_argument(
    "--background-data-path",
    default=None,
    help=(
        "Optional CSV used as SHAP background/reference data. If it contains "
        "the label column, that column is dropped after optional stratified "
        "background sampling."
    ),
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
    default=None,
    help=(
        "Optional maximum number of features to show in class-level plots. "
        "By default, all features above --mean-abs-shap-threshold are shown."
    ),
)
parser.add_argument(
    "--heatmap-top-features-per-class",
    type=int,
    default=None,
    help=(
        "Optional maximum number of features per class used to build the heatmap union. "
        "By default, all displayed features contribute."
    ),
)
parser.add_argument(
    "--mean-abs-shap-threshold",
    type=float,
    default=0.0,
    help=(
        "Minimum mean absolute SHAP value for displayed features. "
        "Features must be strictly greater than this threshold."
    ),
)
parser.add_argument(
    "--max-samples",
    type=float,
    default=2000,
    help=(
        "Rows used for SHAP computation. Values <= 0 use all rows; values "
        "between 0 and 1 sample that fraction; values >= 1 sample up to that "
        "many rows."
    ),
)
parser.add_argument(
    "--max-background-samples",
    type=float,
    default=0,
    help=(
        "Rows used for SHAP background. Values <= 0 use all background rows; "
        "values between 0 and 1 sample that fraction; values >= 1 sample up "
        "to that many rows."
    ),
)
parser.add_argument(
    "--random-state",
    type=int,
    default=123,
    help="Random seed for reproducible sampling.",
)
args = parser.parse_args()

if args.mean_abs_shap_threshold is not None and args.mean_abs_shap_threshold < 0:
    raise ValueError("--mean-abs-shap-threshold must be >= 0.")
if args.top_features is not None and args.top_features <= 0:
    raise ValueError("--top-features must be > 0 when provided.")
if (
    args.heatmap_top_features_per_class is not None
    and args.heatmap_top_features_per_class <= 0
):
    raise ValueError("--heatmap-top-features-per-class must be > 0 when provided.")

# %% Path resolution and validation

model_paths = load_model_paths(args.model_path, args.model_dir, args.model_glob)
data_paths = [Path(data_path).resolve() for data_path in args.data_path]
background_data_path = (
    Path(args.background_data_path).resolve()
    if args.background_data_path is not None
    else None
)

if args.out_dir is not None:
    out_dir = Path(args.out_dir).resolve()
elif args.model_path is not None:
    out_dir = script_dir.parent / "results" / "explain_model" / model_paths[0].stem
else:
    out_dir = script_dir.parent / "results" / "explain_model" / model_paths[0].parent.name

for model_path in model_paths:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
for data_path in data_paths:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
if background_data_path is not None and not background_data_path.exists():
    raise FileNotFoundError(f"Background data file not found: {background_data_path}")

out_dir.mkdir(parents=True, exist_ok=True)

# %% Model loading and estimator checks

model_records = []
for model_path in model_paths:
    pipeline = joblib.load(model_path)
    estimator, estimator_name, feature_transformer = get_pipeline_components(
        pipeline,
        model_path,
    )
    model_records.append(
        {
            "path": model_path,
            "label": sanitize_label(model_path.stem),
            "pipeline": pipeline,
            "estimator": estimator,
            "estimator_name": estimator_name,
            "feature_transformer": feature_transformer,
        }
    )

reference_record = model_records[0]
estimator_name = reference_record["estimator_name"]
class_labels = np.asarray(reference_record["estimator"].classes_)
model_labels = [record["label"] for record in model_records]
is_fold_model_mode = len(model_records) > 1

for model_record in model_records[1:]:
    if model_record["estimator_name"] != estimator_name:
        raise ValueError(
            "All fold models must use the same estimator type. "
            f"Expected {estimator_name}, found {model_record['estimator_name']} "
            f"in {model_record['path']}."
        )
    if not np.array_equal(np.asarray(model_record["estimator"].classes_), class_labels):
        raise ValueError(
            f"Class labels in {model_record['path']} do not match the first model."
        )

# %% Dataset loading

data_df, data_source = load_explanation_data(data_paths, args.label_column)

X = data_df.drop(columns=[args.label_column])
y_true = data_df[args.label_column]

# %% Predictions and sampling

if is_fold_model_mode:
    y_proba_by_model = [
        model_record["pipeline"].predict_proba(X)
        for model_record in model_records
    ]
    y_proba = np.mean(np.stack(y_proba_by_model, axis=0), axis=0)
    y_pred = class_labels[np.argmax(y_proba, axis=1)]
else:
    y_pred = reference_record["pipeline"].predict(X)
    y_proba = reference_record["pipeline"].predict_proba(X)
predicted_probabilities = y_proba.max(axis=1)

sample_indices = select_sample_indices(
    y_true,
    max_samples=args.max_samples,
    random_state=args.random_state,
)

X_sample = X.iloc[sample_indices].copy()
y_true_sample = y_true.iloc[sample_indices].reset_index(drop=True)
y_pred_sample = (
    pd.Series(y_pred, index=X.index).iloc[sample_indices].reset_index(drop=True)
)

# %% Preprocessing and SHAP computation

if background_data_path is not None:
    X_background, background_labels = load_background_data(
        background_data_path,
        label_column=args.label_column,
        reference_columns=X.columns,
    )
    background_indices = select_sample_indices(
        (
            background_labels.to_numpy()
            if background_labels is not None
            else np.zeros(len(X_background))
        ),
        max_samples=args.max_background_samples,
        random_state=args.random_state,
    )
    X_background = X_background.iloc[background_indices].copy()
    background_source = str(background_data_path)
else:
    X_background = None
    background_labels = None

X_transformed_sample_by_model = []
shap_by_output_by_model = []
explainer_names = []
raw_feature_names = None
display_feature_names = None
background_rows = None
shap_output_labels = None

for model_record in model_records:
    feature_transformer = model_record["feature_transformer"]
    estimator = model_record["estimator"]
    X_transformed_sample_model = to_dense_array(
        feature_transformer.transform(X_sample)
    )
    model_raw_feature_names, model_display_feature_names = (
        get_transformed_feature_names(
            feature_transformer,
            X_sample,
            X_transformed_sample_model.shape[1],
        )
    )

    if X_transformed_sample_model.shape[1] != len(model_raw_feature_names):
        raise ValueError(
            "The transformed feature matrix width does not match the recovered "
            f"feature names for {model_record['path']}."
        )

    if raw_feature_names is None:
        raw_feature_names = model_raw_feature_names
        display_feature_names = model_display_feature_names
    else:
        if not np.array_equal(model_raw_feature_names, raw_feature_names):
            raise ValueError(
                f"Transformed feature names in {model_record['path']} do not match "
                "the first model."
            )

    if X_background is None:
        X_transformed_background = X_transformed_sample_model
        background_source = "sampled explanation data"
    else:
        X_transformed_background = to_dense_array(
            feature_transformer.transform(X_background)
        )
    background_rows = X_transformed_background.shape[0]

    if (
        X_transformed_background is not None
        and X_transformed_background.shape[1] != X_transformed_sample_model.shape[1]
    ):
        raise ValueError(
            "The transformed background matrix width does not match the explanation "
            f"matrix for {model_record['path']}."
        )

    explainer, model_explainer_name = build_explainer(
        estimator,
        X_transformed_background,
        class_labels,
        model_display_feature_names,
    )
    shap_by_output_model, model_shap_output_labels = reshape_shap_values_container(
        explainer(X_transformed_sample_model),
        class_labels=class_labels,
    )
    if shap_output_labels is None:
        shap_output_labels = model_shap_output_labels
    elif not np.array_equal(model_shap_output_labels, shap_output_labels):
        raise ValueError(
            f"SHAP output labels in {model_record['path']} do not match the first model."
        )

    X_transformed_sample_by_model.append(X_transformed_sample_model)
    shap_by_output_by_model.append(shap_by_output_model)
    explainer_names.append(model_explainer_name)

X_transformed_sample = np.mean(
    np.stack(X_transformed_sample_by_model, axis=0),
    axis=0,
)
shap_by_output = [
    np.mean(
        np.stack(
            [
                model_shap_by_output[output_idx]
                for model_shap_by_output in shap_by_output_by_model
            ],
            axis=0,
        ),
        axis=0,
    )
    for output_idx in range(len(shap_output_labels))
]
explainer_name = explainer_names[0]

if any(name != explainer_name for name in explainer_names):
    raise ValueError(
        "All fold models must use the same SHAP explainer type."
    )

pd.DataFrame(
    {
        "model_label": model_labels,
        "model_path": [str(model_record["path"]) for model_record in model_records],
        "estimator_name": [
            model_record["estimator_name"] for model_record in model_records
        ],
        "explainer_name": explainer_names,
        "shap_output_labels": [
            ",".join(map(str, shap_output_labels))
            for _ in model_records
        ],
    }
).to_csv(
    out_dir / "model_explainers.csv",
    index=False,
)

sample_feature_df = pd.DataFrame(
    X_transformed_sample,
    columns=display_feature_names,
)

# %% Prediction summary output

prediction_summary_df = pd.DataFrame(
    {
        "row_index": X.index.astype(str),
        "data_source": data_source.to_numpy(),
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
    mean_abs_shap_distributions_dir = group_dir / "mean_abs_shap_distributions"

    tables_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)
    beeswarms_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir.mkdir(parents=True, exist_ok=True)
    mean_abs_shap_distributions_dir.mkdir(parents=True, exist_ok=True)

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
    heatmap_row_labels = []
    bar_grid_items = []
    beeswarm_grid_items = []

    all_classes_shap_values, all_classes_row_indices = build_row_matched_shap_values(
        shap_by_output,
        shap_output_labels,
        group_labels,
    )
    if is_fold_model_mode:
        all_classes_model_shap_values = []
        for model_shap_by_output in shap_by_output_by_model:
            model_all_classes_shap_values, model_all_classes_row_indices = (
                build_row_matched_shap_values(
                    model_shap_by_output,
                    shap_output_labels,
                    group_labels,
                )
            )
            if not np.array_equal(
                model_all_classes_row_indices,
                all_classes_row_indices,
            ):
                raise ValueError(
                    "Fold-level row-matched SHAP rows do not match averaged SHAP rows."
                )
            all_classes_model_shap_values.append(model_all_classes_shap_values)
    else:
        all_classes_model_shap_values = None

    all_classes_outputs = save_category_outputs(
        category_label="all_classes",
        safe_label="all_classes",
        category_shap_values=all_classes_shap_values,
        category_model_shap_values=all_classes_model_shap_values,
        model_labels=model_labels,
        category_feature_df=sample_feature_df.iloc[all_classes_row_indices],
        raw_feature_names=raw_feature_names,
        display_feature_names=display_feature_names,
        top_features=args.top_features,
        heatmap_top_features=args.heatmap_top_features_per_class,
        mean_abs_shap_threshold=args.mean_abs_shap_threshold,
        tables_dir=tables_dir,
        bars_dir=bars_dir,
        beeswarms_dir=beeswarms_dir,
        dependence_dir=dependence_dir,
        mean_abs_shap_distributions_dir=mean_abs_shap_distributions_dir,
    )
    heatmap_feature_union.extend(all_classes_outputs["heatmap_features"])
    heatmap_rows.append(all_classes_outputs["heatmap_row"])
    heatmap_row_labels.append("all_classes")
    if all_classes_outputs["bar_grid_item"] is not None:
        bar_grid_items.append(all_classes_outputs["bar_grid_item"])
    if all_classes_outputs["beeswarm_grid_item"] is not None:
        beeswarm_grid_items.append(all_classes_outputs["beeswarm_grid_item"])

    for output_idx, class_label in enumerate(shap_output_labels):
        class_mask = np.asarray(group_labels == class_label)
        safe_label = sanitize_label(class_label)
        class_shap_values_all = shap_by_output[output_idx]

        if not np.any(class_mask):
            empty_outputs = save_category_outputs(
                category_label=class_label,
                safe_label=safe_label,
                category_shap_values=class_shap_values_all[class_mask],
                category_model_shap_values=(
                    [
                        model_shap_by_output[output_idx][class_mask]
                        for model_shap_by_output in shap_by_output_by_model
                    ]
                    if is_fold_model_mode
                    else None
                ),
                model_labels=model_labels,
                category_feature_df=sample_feature_df.iloc[class_mask],
                raw_feature_names=raw_feature_names,
                display_feature_names=display_feature_names,
                top_features=args.top_features,
                heatmap_top_features=args.heatmap_top_features_per_class,
                mean_abs_shap_threshold=args.mean_abs_shap_threshold,
                tables_dir=tables_dir,
                bars_dir=bars_dir,
                beeswarms_dir=beeswarms_dir,
                dependence_dir=dependence_dir,
                mean_abs_shap_distributions_dir=mean_abs_shap_distributions_dir,
            )
            heatmap_rows.append(empty_outputs["heatmap_row"])
            heatmap_row_labels.append(str(class_label))
            continue

        class_shap_values = class_shap_values_all[class_mask]
        if is_fold_model_mode:
            class_model_shap_values = [
                model_shap_by_output[output_idx][class_mask]
                for model_shap_by_output in shap_by_output_by_model
            ]
        else:
            class_model_shap_values = None
        class_feature_df = sample_feature_df.iloc[class_mask]

        class_outputs = save_category_outputs(
            category_label=class_label,
            safe_label=safe_label,
            category_shap_values=class_shap_values,
            category_model_shap_values=class_model_shap_values,
            model_labels=model_labels,
            category_feature_df=class_feature_df,
            raw_feature_names=raw_feature_names,
            display_feature_names=display_feature_names,
            top_features=args.top_features,
            heatmap_top_features=args.heatmap_top_features_per_class,
            mean_abs_shap_threshold=args.mean_abs_shap_threshold,
            tables_dir=tables_dir,
            bars_dir=bars_dir,
            beeswarms_dir=beeswarms_dir,
            dependence_dir=dependence_dir,
            mean_abs_shap_distributions_dir=mean_abs_shap_distributions_dir,
        )
        heatmap_feature_union.extend(class_outputs["heatmap_features"])
        heatmap_rows.append(class_outputs["heatmap_row"])
        heatmap_row_labels.append(str(class_label))
        if class_outputs["bar_grid_item"] is not None:
            bar_grid_items.append(class_outputs["bar_grid_item"])
        if class_outputs["beeswarm_grid_item"] is not None:
            beeswarm_grid_items.append(class_outputs["beeswarm_grid_item"])

    heatmap_features = pd.Index(heatmap_feature_union).unique()
    if len(heatmap_features) == 0:
        heatmap_df = pd.DataFrame(
            index=heatmap_row_labels
        )
    else:
        heatmap_df = pd.DataFrame(
            heatmap_rows,
            index=heatmap_row_labels,
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
    )

    if heatmap_display_df.shape[1] > 0:
        save_heatmap(
            heatmap_display_df,
            group_dir / "class_feature_heatmap.png",
            f"{grouping_name}: class-feature SHAP summary",
        )

# %% Final summary

print(f"Estimator: {estimator_name}")
print(f"SHAP explainer: {explainer_name}")
print(f"Explained models: {len(model_records)}")
print(f"Explained rows: {len(X_sample)}")
if background_rows is not None:
    print(f"Background rows: {background_rows}")
    print(f"Background source: {background_source}")
print(f"Saved explanation outputs to: {out_dir}")
