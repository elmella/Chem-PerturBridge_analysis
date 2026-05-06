#!/bin/bash
set -euo pipefail

if [ "${1:-}" = "subsampling" ]; then
    echo "> Work with a subsample"
    SUFFIX="_subsample"
    SUBDIR="subsample"
else
    echo "> Work with a full version"
    SUFFIX=""
    SUBDIR="full"
fi

UV_BIN="${UV_BIN:-uv}"
QOS="${QOS:-}"
PARTITION="${PARTITION:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
NUMPY_THREADS="${NUMPY_THREADS:-${CPUS_PER_TASK}}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/${SUBDIR}}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-${OUTPUT_DIR}/tasks}"
LOGS_DIR="${LOGS_DIR:-./logs/retrieval/${SUBDIR}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cross_dataset_retrieval_parallel${SUFFIX}}"

QUERY_DATASETS="${QUERY_DATASETS:-all}"
DB_DATASETS="${DB_DATASETS:-all}"
REPRESENTATIONS="${REPRESENTATIONS:-all}"
SKIP_REPRESENTATIONS="${SKIP_REPRESENTATIONS:-}"
METRICS="${METRICS:-cosine,pearson,mrrmse}"
CELL_TYPES="${CELL_TYPES:-}"
INCLUDE_SELF_DATASET="${INCLUDE_SELF_DATASET:-0}"
STRICT_MISSING="${STRICT_MISSING:-0}"
SPLIT_BY_CELL_TYPE="${SPLIT_BY_CELL_TYPE:-1}"
DATASET_PATH_ARGS="${DATASET_PATH_ARGS:-}"

PREP_TIME="${PREP_TIME:-24:00:00}"
TASK_TIME="${TASK_TIME:-24:00:00}"
MERGE_TIME="${MERGE_TIME:-24:00:00}"
PREP_MEM="${PREP_MEM:-200G}"
TASK_MEM="${TASK_MEM:-100G}"
MERGE_MEM="${MERGE_MEM:-200G}"

mkdir -p "${OUTPUT_DIR}" "${TASK_OUTPUT_DIR}" "${LOGS_DIR}"

SBATCH_CLUSTER_ARGS=()
if [ -n "${QOS}" ]; then
    SBATCH_CLUSTER_ARGS+=(--qos="${QOS}")
fi
if [ -n "${PARTITION}" ]; then
    SBATCH_CLUSTER_ARGS+=(--partition="${PARTITION}")
fi

TASK_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_tasks.csv"

PRECOMPUTE_CMD="${UV_BIN} run python -m chem_perturbridge_analysis.retrieval.parallel_cli precompute \
    --output-dir ${OUTPUT_DIR} \
    --output-prefix ${OUTPUT_PREFIX} \
    --query-datasets ${QUERY_DATASETS} \
    --db-datasets ${DB_DATASETS} \
    --representations ${REPRESENTATIONS} \
    --verbose"
if [ -n "${SKIP_REPRESENTATIONS}" ]; then
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD} --skip-representations ${SKIP_REPRESENTATIONS}"
fi
if [ -n "${CELL_TYPES}" ]; then
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD} --cell-types ${CELL_TYPES}"
fi
if [ "${INCLUDE_SELF_DATASET}" = "1" ]; then
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD} --include-self-dataset"
fi
if [ "${SPLIT_BY_CELL_TYPE}" = "1" ]; then
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD} --split-by-cell-type"
fi
if [ -n "${DATASET_PATH_ARGS}" ]; then
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD} ${DATASET_PATH_ARGS}"
fi

RUN_TASK_CMD="OMP_NUM_THREADS=${NUMPY_THREADS} OPENBLAS_NUM_THREADS=${NUMPY_THREADS} MKL_NUM_THREADS=${NUMPY_THREADS} NUMEXPR_NUM_THREADS=${NUMPY_THREADS} ${UV_BIN} run python -m chem_perturbridge_analysis.retrieval.parallel_cli run-task \
    --task-file ${TASK_FILE} \
    --task-id \${SLURM_ARRAY_TASK_ID} \
    --output-dir ${TASK_OUTPUT_DIR} \
    --output-prefix ${OUTPUT_PREFIX} \
    --metrics ${METRICS} \
    --verbose"
if [ -n "${CELL_TYPES}" ]; then
    RUN_TASK_CMD="${RUN_TASK_CMD} --cell-types ${CELL_TYPES}"
fi
if [ -n "${DATASET_PATH_ARGS}" ]; then
    RUN_TASK_CMD="${RUN_TASK_CMD} ${DATASET_PATH_ARGS}"
fi

MERGE_CMD="${UV_BIN} run python -m chem_perturbridge_analysis.retrieval.parallel_cli merge \
    --task-file ${TASK_FILE} \
    --task-output-dir ${TASK_OUTPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --output-prefix ${OUTPUT_PREFIX} \
    --verbose"
if [ "${STRICT_MISSING}" = "1" ]; then
    MERGE_CMD="${MERGE_CMD} --strict-missing"
fi

echo "> Stage 1/3: Precompute true matches and task matrix"
sbatch -W -J retrieval_precompute \
    -t "${PREP_TIME}" \
    -n 1 \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${PREP_MEM}" \
    -e "${LOGS_DIR}/retrieval_precompute.%j.err" \
    -o "${LOGS_DIR}/retrieval_precompute.%j.out" \
    --wrap="${PRECOMPUTE_CMD}"

if [ ! -f "${TASK_FILE}" ]; then
    echo "> ERROR: task file not found: ${TASK_FILE}"
    exit 1
fi
N_TASKS=$(($(wc -l < "${TASK_FILE}") - 1))
if [ "${N_TASKS}" -le 0 ]; then
    echo "> No retrieval tasks to run (task matrix is empty)."
    exit 0
fi

echo "> Stage 2/3: Run ${N_TASKS} retrieval tasks in a Slurm array"
sbatch -W -J retrieval_task \
    -t "${TASK_TIME}" \
    -n 1 \
    --array=1-"${N_TASKS}" \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${TASK_MEM}" \
    -e "${LOGS_DIR}/retrieval_task.%A_%a.err" \
    -o "${LOGS_DIR}/retrieval_task.%A_%a.out" \
    --wrap="${RUN_TASK_CMD}"

echo "> Stage 3/3: Merge task outputs"
sbatch -W -J retrieval_merge \
    -t "${MERGE_TIME}" \
    -n 1 \
    "${SBATCH_CLUSTER_ARGS[@]}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MERGE_MEM}" \
    -e "${LOGS_DIR}/retrieval_merge.%j.err" \
    -o "${LOGS_DIR}/retrieval_merge.%j.out" \
    --wrap="${MERGE_CMD}"

echo "> Done. Final outputs:"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_detail.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_summary_by_cell_type.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_summary_overall.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_pair_matches/*.h5ad"
