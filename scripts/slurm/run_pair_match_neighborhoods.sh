#!/bin/bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
QOS="${QOS:-cpu_normal}"
PARTITION="${PARTITION:-cpu_p}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
NUMPY_THREADS="${NUMPY_THREADS:-${CPUS_PER_TASK}}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/pair_match_neighborhoods}"
LOGS_DIR="${LOGS_DIR:-./logs/pair_match_neighborhoods}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-pair_match_neighborhoods}"

QUERY_DATASETS="${QUERY_DATASETS:-all}"
DB_DATASETS="${DB_DATASETS:-all}"
CELL_TYPES="${CELL_TYPES:-}"
INCLUDE_SELF_DATASET="${INCLUDE_SELF_DATASET:-0}"
REPRESENTATION="${REPRESENTATION:-logFC}"
LOGFC_LAYER="${LOGFC_LAYER:-logFC}"
PVALUE_LAYER="${PVALUE_LAYER:-P.Value}"
DATASET_PATH_ARGS="${DATASET_PATH_ARGS:-}"

TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MEM="${MEM:-200G}"

mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}"

CMD="OMP_NUM_THREADS=${NUMPY_THREADS} OPENBLAS_NUM_THREADS=${NUMPY_THREADS} MKL_NUM_THREADS=${NUMPY_THREADS} NUMEXPR_NUM_THREADS=${NUMPY_THREADS} ${UV_BIN} run python -m op3_analysis.pair_match_neighborhoods \
    --output-dir ${OUTPUT_DIR} \
    --output-prefix ${OUTPUT_PREFIX} \
    --query-datasets ${QUERY_DATASETS} \
    --db-datasets ${DB_DATASETS} \
    --representation ${REPRESENTATION} \
    --logfc-layer ${LOGFC_LAYER} \
    --pvalue-layer ${PVALUE_LAYER} \
    --verbose"

if [ -n "${CELL_TYPES}" ]; then
    CMD="${CMD} --cell-types ${CELL_TYPES}"
fi
if [ "${INCLUDE_SELF_DATASET}" = "1" ]; then
    CMD="${CMD} --include-self-dataset"
fi
if [ -n "${DATASET_PATH_ARGS}" ]; then
    CMD="${CMD} ${DATASET_PATH_ARGS}"
fi

echo "> Running matched-pair agreement + neighborhood evaluation"
sbatch -W -J pair_match_nbhd \
    -t "${TIME_LIMIT}" \
    -n 1 \
    --qos="${QOS}" \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    -e "${LOGS_DIR}/pair_match_nbhd.%j.err" \
    -o "${LOGS_DIR}/pair_match_nbhd.%j.out" \
    --wrap="${CMD}"

echo "> Done. Final outputs:"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_truth_matches.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_truth_summary.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_detail.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_summary_by_cell_type.csv"
echo "  - ${OUTPUT_DIR}/${OUTPUT_PREFIX}_summary_overall.csv"
