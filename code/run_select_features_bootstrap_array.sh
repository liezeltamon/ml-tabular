#!/usr/bin/env bash
set -euo pipefail

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/select_features/progb_vs_nonprogb_selectkbest_p005
# sbatch --array=0-99 -J select_features_bootstrap_p005 -p short,long --mem=200G --output=logs/select_features/progb_vs_nonprogb_selectkbest_p005/%x_%A_%a.log.out --error=logs/select_features/progb_vs_nonprogb_selectkbest_p005/%x_%A_%a.log.err run_select_features_bootstrap_array.sh

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection
# SKIP_CORRELATED_SELECTION=true sbatch --array=0-99 -J select_features_bootstrap_p005_nocorr -p short,long --mem=200G --output=logs/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/%x_%A_%a.log.out --error=logs/select_features/progb_vs_nonprogb_selectkbest_p005_no_correlated_selection/%x_%A_%a.log.err run_select_features_bootstrap_array.sh

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc
# UNIVARIATE_METHOD=single_feature_performance sbatch --array=0-99 -J select_features_bootstrap_sfp_auc -p short,long --mem=200G --output=logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc/%x_%A_%a.log.out --error=logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc/%x_%A_%a.log.err run_select_features_bootstrap_array.sh

# cd /well/immune-rep/users/yfg436/git/ml-tabular/code
# mkdir -p logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection
# UNIVARIATE_METHOD=single_feature_performance SKIP_CORRELATED_SELECTION=true sbatch --array=0-99 -J select_features_bootstrap_sfp_auc_nocorr -p short,long --mem=200G --output=logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/%x_%A_%a.log.out --error=logs/select_features/progb_vs_nonprogb_singlefeatureperformance_auc_no_correlated_selection/%x_%A_%a.log.err run_select_features_bootstrap_array.sh

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-99." >&2
    exit 1
fi

SCRIPT_DIR="/well/immune-rep/users/yfg436/git/ml-tabular/code"
REPO_ROOT="/well/immune-rep/users/yfg436/git/ml-tabular"
cd "${SCRIPT_DIR}"

SKIP_CORRELATED_SELECTION="${SKIP_CORRELATED_SELECTION:-false}"
if [[ "${SKIP_CORRELATED_SELECTION}" != "true" && "${SKIP_CORRELATED_SELECTION}" != "false" ]]; then
    echo "SKIP_CORRELATED_SELECTION must be true or false." >&2
    exit 1
fi

UNIVARIATE_METHOD="${UNIVARIATE_METHOD:-select_k_best}"
if [[ "${UNIVARIATE_METHOD}" != "select_k_best" && "${UNIVARIATE_METHOD}" != "single_feature_performance" ]]; then
    echo "UNIVARIATE_METHOD must be select_k_best or single_feature_performance." >&2
    exit 1
fi

UNIVARIATE_SELECTION_ARGS=()
if [[ "${UNIVARIATE_METHOD}" == "single_feature_performance" ]]; then
    BASE_RUN_ID="progb_vs_nonprogb_singlefeatureperformance_auc"
    UNIVARIATE_SELECTION_ARGS+=(
        --univariate-method single_feature_performance
        --univariate-scoring roc_auc
        --univariate-cv 5
        --univariate-threshold 0.5
    )
else
    BASE_RUN_ID="progb_vs_nonprogb_selectkbest_p005"
    UNIVARIATE_SELECTION_ARGS+=(
        --univariate-method select_k_best
        --univariate-pvalue-threshold 0.05
    )
fi

CORRELATED_SELECTION_ARGS=()
if [[ "${SKIP_CORRELATED_SELECTION}" == "true" ]]; then
    RUN_ID="${BASE_RUN_ID}_no_correlated_selection"
    CORRELATED_SELECTION_ARGS+=(--skip-correlated-selection)
else
    RUN_ID="${BASE_RUN_ID}"
fi
LOG_DIR="${SCRIPT_DIR}/logs/select_features/${RUN_ID}"

BOOTSTRAP_ID="${SLURM_ARRAY_TASK_ID}"
BOOTSTRAP_SEED="${SLURM_ARRAY_TASK_ID}"

TRAIN_PATH="/well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/train.csv"
TEST_PATH="/well/immune-rep/users/yfg436/git/sle/results/prediction/create_input_table/group_id_nonprogb_progb_missingness0_minuniqueNone/test.csv"
OUT_DIR="${REPO_ROOT}/results/select_features/${RUN_ID}/bootstraps/bootstrap_${BOOTSTRAP_ID}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_DIR}"

echo "Run ID: ${RUN_ID}"
echo "Bootstrap ID: ${BOOTSTRAP_ID}"
echo "Bootstrap seed: ${BOOTSTRAP_SEED}"
echo "Univariate method: ${UNIVARIATE_METHOD}"
echo "Skip correlated selection: ${SKIP_CORRELATED_SELECTION}"
echo "Log directory: ${LOG_DIR}"
echo "Output directory: ${OUT_DIR}"

python select_features.py \
    --out-dir "${OUT_DIR}" \
    --train-path "${TRAIN_PATH}" \
    --test-path "${TEST_PATH}" \
    --target-column is_progb \
    "${UNIVARIATE_SELECTION_ARGS[@]}" \
    --bootstrap-train \
    --bootstrap-seed "${BOOTSTRAP_SEED}" \
    "${CORRELATED_SELECTION_ARGS[@]}"
