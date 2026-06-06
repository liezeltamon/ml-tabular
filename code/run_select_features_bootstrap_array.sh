#!/usr/bin/env bash
set -euo pipefail

# mkdir -p logs/select_features/progb_vs_nonprogb_selectkbest_p005
# sbatch --array=0-99 -J select_features_bootstrap_p005 -p short,long --mem=100G --output=logs/select_features/progb_vs_nonprogb_selectkbest_p005/%x_%A_%a.log.out --error=logs/select_features/progb_vs_nonprogb_selectkbest_p005/%x_%A_%a.log.err run_select_features_bootstrap_array.sh

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-99." >&2
    exit 1
fi

SCRIPT_DIR="/well/immune-rep/users/yfg436/git/ml-tabular/code"
REPO_ROOT="/well/immune-rep/users/yfg436/git/ml-tabular"
cd "${SCRIPT_DIR}"

RUN_ID="progb_vs_nonprogb_selectkbest_p005"
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
echo "Log directory: ${LOG_DIR}"
echo "Output directory: ${OUT_DIR}"

python select_features.py \
    --out-dir "${OUT_DIR}" \
    --train-path "${TRAIN_PATH}" \
    --test-path "${TEST_PATH}" \
    --target-column is_progb \
    --univariate-method select_k_best \
    --univariate-pvalue-threshold 0.05 \
    --bootstrap-train \
    --bootstrap-seed "${BOOTSTRAP_SEED}"
