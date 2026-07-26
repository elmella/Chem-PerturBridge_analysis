#!/bin/bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
PRECOMPUTE_SCRIPT="${PRECOMPUTE_SCRIPT:-scripts/precompute_replicate_signature_similarity.py}"
QOS="${QOS:-cpu_normal}"
PARTITION="${PARTITION:-cpu_p}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
NUMPY_THREADS="${NUMPY_THREADS:-${CPUS_PER_TASK}}"

PREP_MEM="${PREP_MEM:-120G}"
TASK_MEM="${TASK_MEM:-200G}"
MERGE_MEM="${MERGE_MEM:-120G}"
PREP_TIME="${PREP_TIME:-24:00:00}"
TASK_TIME="${TASK_TIME:-24:00:00}"
MERGE_TIME="${MERGE_TIME:-12:00:00}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/replicate_signature_similarity_sep_rep}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-${OUTPUT_DIR}/task_outputs}"
LOGS_DIR="${LOGS_DIR:-./logs/replicate_signature_similarity}"
DATASETS="${DATASETS:-all}"
TOP_K="${TOP_K:-50}"
MIN_REPLICATES_PER_CONDITION="${MIN_REPLICATES_PER_CONDITION:-2}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
CONDITIONS_PER_TASK="${CONDITIONS_PER_TASK:-250}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-32}"
STRICT_MISSING="${STRICT_MISSING:-1}"
TEST_ONE_LINE_PER_DATASET="${TEST_ONE_LINE_PER_DATASET:-0}"
TEST_MAX_CONDITIONS_PER_DATASET="${TEST_MAX_CONDITIONS_PER_DATASET:-0}"
COMPUTE_BASELINE_METRICS="${COMPUTE_BASELINE_METRICS:-0}"
COMPUTE_DEG_METRICS="${COMPUTE_DEG_METRICS:-0}"
COMPUTE_RETRIEVAL_METRICS="${COMPUTE_RETRIEVAL_METRICS:-0}"
MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME="${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME:-10}"
MAX_BASELINE_PEERS="${MAX_BASELINE_PEERS:-512}"
PEER_SAMPLING_SEED="${PEER_SAMPLING_SEED:-20260505}"
QUICK_TEST_RUN="${QUICK_TEST_RUN:-0}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-3}"
TASK_ARRAY_SPEC="${TASK_ARRAY_SPEC:-}"
RESHARD_TASKS="${RESHARD_TASKS:-0}"
SKIP_EXISTING_DATASETS="${SKIP_EXISTING_DATASETS:-0}"
EXISTING_RESULTS_DIR="${EXISTING_RESULTS_DIR:-${OUTPUT_DIR}}"
ALLOW_SKIP_EXISTING_SAME_OUTPUT_DIR="${ALLOW_SKIP_EXISTING_SAME_OUTPUT_DIR:-0}"
COMBINE_WITH_EXISTING_RESULTS="${COMBINE_WITH_EXISTING_RESULTS:-${SKIP_EXISTING_DATASETS}}"

if [ "${QUICK_TEST_RUN}" = "1" ]; then
    TEST_ONE_LINE_PER_DATASET=1
    if [ "${TEST_MAX_CONDITIONS_PER_DATASET}" = "0" ]; then
        TEST_MAX_CONDITIONS_PER_DATASET=16
    fi
    COMPUTE_BASELINE_METRICS=1
    COMPUTE_DEG_METRICS=1
    COMPUTE_RETRIEVAL_METRICS=1
    if [ "${CONDITIONS_PER_TASK}" = "250" ]; then
        CONDITIONS_PER_TASK=32
    fi
    if [ "${MAX_CONCURRENT_TASKS}" = "32" ]; then
        MAX_CONCURRENT_TASKS=8
    fi
fi

mkdir -p "${OUTPUT_DIR}" "${TASK_OUTPUT_DIR}" "${LOGS_DIR}" "${UV_CACHE_DIR}"

TASK_FILE="${OUTPUT_DIR}/task_manifest.tsv"
TASK_CONFIG_FILE="${OUTPUT_DIR}/task_inputs/task_config.json"

validate_reused_prepare_outputs() {
    env \
        TASK_CONFIG_FILE="${TASK_CONFIG_FILE}" \
        PRECOMPUTE_SCRIPT="${PRECOMPUTE_SCRIPT}" \
        DATASETS="${DATASETS}" \
        MIN_REPLICATES_PER_CONDITION="${MIN_REPLICATES_PER_CONDITION}" \
        TEST_ONE_LINE_PER_DATASET="${TEST_ONE_LINE_PER_DATASET}" \
        TEST_MAX_CONDITIONS_PER_DATASET="${TEST_MAX_CONDITIONS_PER_DATASET}" \
        MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME="${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME}" \
        MAX_BASELINE_PEERS="${MAX_BASELINE_PEERS}" \
        PEER_SAMPLING_SEED="${PEER_SAMPLING_SEED}" \
        python - <<'PY'
import json
import os
import pathlib
import sys
import ast

config_path = pathlib.Path(os.environ["TASK_CONFIG_FILE"])
if not config_path.exists():
    sys.exit(0)

def current_default_datasets():
    script_path = pathlib.Path(os.environ["PRECOMPUTE_SCRIPT"])
    tree = ast.parse(script_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "DEFAULT_SOURCE_DATASET_DIRS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        return [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    raise RuntimeError(f"Could not read DEFAULT_SOURCE_DATASET_DIRS from {script_path}")

config = json.loads(config_path.read_text())
requested = {
    "min_replicates_per_condition": int(os.environ["MIN_REPLICATES_PER_CONDITION"]),
    "test_one_line_per_dataset": os.environ["TEST_ONE_LINE_PER_DATASET"] == "1",
    "test_max_conditions_per_dataset": int(os.environ["TEST_MAX_CONDITIONS_PER_DATASET"]),
    "min_retrieval_compounds_per_line_time": int(os.environ["MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME"]),
}
datasets_arg = os.environ["DATASETS"].strip()
if not datasets_arg or datasets_arg.lower() == "all":
    requested["datasets"] = current_default_datasets()
else:
    requested["datasets"] = [item.strip() for item in datasets_arg.split(",") if item.strip()]

mismatches = []
for key, expected in requested.items():
    actual = config.get(key)
    if actual != expected:
        mismatches.append(f"{key}: existing={actual!r}, requested={expected!r}")

if mismatches:
    print("\\n".join(mismatches))
    sys.exit(2)
PY
}

refresh_saved_metric_flags() {
    env \
        TASK_CONFIG_FILE="${TASK_CONFIG_FILE}" \
        CONDITIONS_PER_TASK="${CONDITIONS_PER_TASK}" \
        COMPUTE_BASELINE_METRICS="${COMPUTE_BASELINE_METRICS}" \
        COMPUTE_DEG_METRICS="${COMPUTE_DEG_METRICS}" \
        COMPUTE_RETRIEVAL_METRICS="${COMPUTE_RETRIEVAL_METRICS}" \
        MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME="${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME}" \
        MAX_BASELINE_PEERS="${MAX_BASELINE_PEERS}" \
        PEER_SAMPLING_SEED="${PEER_SAMPLING_SEED}" \
        python - <<'PY'
import json
import os
import pathlib

config_path = pathlib.Path(os.environ["TASK_CONFIG_FILE"])
if not config_path.exists():
    raise FileNotFoundError(f"Missing task config: {config_path}")

config = json.loads(config_path.read_text())
config["conditions_per_task"] = int(os.environ["CONDITIONS_PER_TASK"])
config["compute_baseline_metrics"] = os.environ["COMPUTE_BASELINE_METRICS"] == "1"
config["compute_deg_metrics"] = os.environ["COMPUTE_DEG_METRICS"] == "1"
config["compute_retrieval_metrics"] = os.environ["COMPUTE_RETRIEVAL_METRICS"] == "1"
config["min_retrieval_compounds_per_line_time"] = int(os.environ["MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME"])
config["max_baseline_peers"] = int(os.environ["MAX_BASELINE_PEERS"])
config["peer_sampling_seed"] = int(os.environ["PEER_SAMPLING_SEED"])
config["peer_baseline_engine_version"] = 2
config_path.write_text(json.dumps(config, indent=2))
PY
}

resolve_missing_datasets() {
    env \
        PRECOMPUTE_SCRIPT="${PRECOMPUTE_SCRIPT}" \
        DATASETS="${DATASETS}" \
        EXISTING_RESULTS_DIR="${EXISTING_RESULTS_DIR}" \
        python - <<'PY'
import ast
import csv
import os
import pathlib
import sys

def current_default_datasets(script_path: pathlib.Path):
    tree = ast.parse(script_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "DEFAULT_SOURCE_DATASET_DIRS" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        return [
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
    raise RuntimeError(f"Could not read DEFAULT_SOURCE_DATASET_DIRS from {script_path}")

def requested_datasets(script_path: pathlib.Path):
    defaults = current_default_datasets(script_path)
    datasets_arg = os.environ["DATASETS"].strip()
    if not datasets_arg or datasets_arg.lower() == "all":
        return defaults

    requested = [item.strip() for item in datasets_arg.split(",") if item.strip()]
    unknown = [dataset_name for dataset_name in requested if dataset_name not in defaults]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return requested

def completed_datasets(results_dir: pathlib.Path):
    condition_summary_path = results_dir / "condition_metric_summary.tsv"
    if condition_summary_path.exists():
        with condition_summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return {
                str(row.get("dataset_name", "")).strip()
                for row in reader
                if str(row.get("dataset_name", "")).strip()
            }

    dataset_summary_path = results_dir / "dataset_metric_summary.tsv"
    if dataset_summary_path.exists():
        completed = set()
        with dataset_summary_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                dataset_name = str(row.get("dataset_name", "")).strip()
                if not dataset_name:
                    continue
                n_conditions_raw = str(row.get("n_conditions", "1")).strip()
                try:
                    n_conditions = float(n_conditions_raw)
                except ValueError:
                    n_conditions = 0.0
                if n_conditions > 0:
                    completed.add(dataset_name)
        return completed

    return set()

script_path = pathlib.Path(os.environ["PRECOMPUTE_SCRIPT"])
results_dir = pathlib.Path(os.environ["EXISTING_RESULTS_DIR"])
requested = requested_datasets(script_path)
completed = completed_datasets(results_dir)
missing = [dataset_name for dataset_name in requested if dataset_name not in completed]

print(",".join(missing))
print(
    f"> skip-existing reference: {results_dir} "
    f"({len(completed)} completed, {len(missing)} missing from {len(requested)} requested)",
    file=sys.stderr,
)
if completed:
    print("> completed datasets: " + ", ".join(sorted(completed)), file=sys.stderr)
if missing:
    print("> datasets selected for this run: " + ", ".join(missing), file=sys.stderr)
PY
}

same_output_dir_as_existing_results() {
    env \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        EXISTING_RESULTS_DIR="${EXISTING_RESULTS_DIR}" \
        python - <<'PY'
import os
import pathlib

output_dir = pathlib.Path(os.environ["OUTPUT_DIR"]).expanduser().resolve()
existing_dir = pathlib.Path(os.environ["EXISTING_RESULTS_DIR"]).expanduser().resolve()
print("1" if output_dir == existing_dir else "0")
PY
}

if ! [[ "${START_STAGE}" =~ ^[123]$ ]]; then
    echo "> ERROR: START_STAGE must be 1, 2, or 3; got ${START_STAGE}"
    exit 1
fi
if ! [[ "${END_STAGE}" =~ ^[123]$ ]]; then
    echo "> ERROR: END_STAGE must be 1, 2, or 3; got ${END_STAGE}"
    exit 1
fi
if [ "${START_STAGE}" -gt "${END_STAGE}" ]; then
    echo "> ERROR: START_STAGE (${START_STAGE}) must be <= END_STAGE (${END_STAGE})"
    exit 1
fi

if [ "${SKIP_EXISTING_DATASETS}" = "1" ]; then
    if [ "$(same_output_dir_as_existing_results)" = "1" ] && [ "${ALLOW_SKIP_EXISTING_SAME_OUTPUT_DIR}" != "1" ]; then
        echo "> ERROR: SKIP_EXISTING_DATASETS=1 needs a separate OUTPUT_DIR from EXISTING_RESULTS_DIR."
        echo "> Missing-only runs rewrite task manifests and merged summaries, so using the same directory would replace combined outputs."
        echo "> Example:"
        echo "  SKIP_EXISTING_DATASETS=1 EXISTING_RESULTS_DIR=${EXISTING_RESULTS_DIR} OUTPUT_DIR=${OUTPUT_DIR}_missing ${0}"
        echo "> To intentionally write missing-only outputs into the same directory, set ALLOW_SKIP_EXISTING_SAME_OUTPUT_DIR=1."
        exit 1
    fi

    if ! RESOLVED_DATASETS=$(resolve_missing_datasets); then
        echo "> ERROR: failed to resolve missing datasets from ${EXISTING_RESULTS_DIR}"
        exit 1
    fi
    if [ -z "${RESOLVED_DATASETS}" ]; then
        echo "> No missing datasets to run."
        exit 0
    fi
    DATASETS="${RESOLVED_DATASETS}"
fi

PREP_CMD="UV_CACHE_DIR=${UV_CACHE_DIR} ${UV_BIN} run python ${PRECOMPUTE_SCRIPT} prepare \
    --output-dir ${OUTPUT_DIR} \
    --datasets ${DATASETS} \
    --top-k ${TOP_K} \
    --min-replicates-per-condition ${MIN_REPLICATES_PER_CONDITION} \
    --progress-every ${PROGRESS_EVERY} \
    --min-retrieval-compounds-per-line-time ${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME} \
    --max-baseline-peers ${MAX_BASELINE_PEERS} \
    --peer-sampling-seed ${PEER_SAMPLING_SEED} \
    --conditions-per-task ${CONDITIONS_PER_TASK}"
if [ "${TEST_ONE_LINE_PER_DATASET}" = "1" ]; then
    PREP_CMD="${PREP_CMD} --test-one-line-per-dataset"
fi
if [ "${TEST_MAX_CONDITIONS_PER_DATASET}" -gt 0 ]; then
    PREP_CMD="${PREP_CMD} --test-max-conditions-per-dataset ${TEST_MAX_CONDITIONS_PER_DATASET}"
fi
if [ "${COMPUTE_BASELINE_METRICS}" = "1" ]; then
    PREP_CMD="${PREP_CMD} --compute-baseline-metrics"
fi
if [ "${COMPUTE_DEG_METRICS}" = "1" ]; then
    PREP_CMD="${PREP_CMD} --compute-deg-metrics"
fi
if [ "${COMPUTE_RETRIEVAL_METRICS}" = "1" ]; then
    PREP_CMD="${PREP_CMD} --compute-retrieval-metrics"
fi

RUN_TASK_CMD="UV_CACHE_DIR=${UV_CACHE_DIR} OMP_NUM_THREADS=${NUMPY_THREADS} OPENBLAS_NUM_THREADS=${NUMPY_THREADS} MKL_NUM_THREADS=${NUMPY_THREADS} NUMEXPR_NUM_THREADS=${NUMPY_THREADS} ${UV_BIN} run python ${PRECOMPUTE_SCRIPT} run-task \
    --output-dir ${OUTPUT_DIR} \
    --task-file ${TASK_FILE} \
    --task-id \${SLURM_ARRAY_TASK_ID} \
    --task-output-dir ${TASK_OUTPUT_DIR} \
    --top-k ${TOP_K} \
    --peer-sampling-seed ${PEER_SAMPLING_SEED} \
    --min-retrieval-compounds-per-line-time ${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME}"
if [ "${COMPUTE_BASELINE_METRICS}" = "1" ]; then
    RUN_TASK_CMD="${RUN_TASK_CMD} --compute-baseline-metrics"
fi
if [ "${COMPUTE_DEG_METRICS}" = "1" ]; then
    RUN_TASK_CMD="${RUN_TASK_CMD} --compute-deg-metrics"
fi
if [ "${COMPUTE_RETRIEVAL_METRICS}" = "1" ]; then
    RUN_TASK_CMD="${RUN_TASK_CMD} --compute-retrieval-metrics"
fi
# The cap affects only individual-peer scoring; the legacy centroid always uses all peers.
RUN_TASK_CMD="${RUN_TASK_CMD} --max-baseline-peers ${MAX_BASELINE_PEERS}"

RESHARD_CMD="UV_CACHE_DIR=${UV_CACHE_DIR} ${UV_BIN} run python ${PRECOMPUTE_SCRIPT} reshard \
    --output-dir ${OUTPUT_DIR} \
    --conditions-per-task ${CONDITIONS_PER_TASK}"

MERGE_CMD="UV_CACHE_DIR=${UV_CACHE_DIR} ${UV_BIN} run python ${PRECOMPUTE_SCRIPT} merge \
    --output-dir ${OUTPUT_DIR} \
    --task-file ${TASK_FILE} \
    --task-output-dir ${TASK_OUTPUT_DIR} \
    --top-k ${TOP_K}"
if [ "${STRICT_MISSING}" = "1" ]; then
    MERGE_CMD="${MERGE_CMD} --strict-missing"
fi
if [ "${COMBINE_WITH_EXISTING_RESULTS}" = "1" ]; then
    MERGE_CMD="${MERGE_CMD} --existing-results-dir ${EXISTING_RESULTS_DIR}"
fi

if [ "${START_STAGE}" -le 1 ] && [ "${END_STAGE}" -ge 1 ]; then
    echo "> Stage 1/3: Prepare metadata, retained conditions, and task shards"
    if [ "${QUICK_TEST_RUN}" = "1" ]; then
        echo "  quick test mode: enabled"
    fi
    if [ "${TEST_ONE_LINE_PER_DATASET}" = "1" ]; then
        echo "  test mode: one line per dataset"
    fi
    if [ "${TEST_MAX_CONDITIONS_PER_DATASET}" -gt 0 ]; then
        echo "  test condition cap per dataset: ${TEST_MAX_CONDITIONS_PER_DATASET}"
    fi
    if [ "${COMPUTE_BASELINE_METRICS}" = "1" ]; then
        echo "  baseline metrics: enabled"
    fi

    echo "  per-peer baseline cap: ${MAX_BASELINE_PEERS} peers per condition"
    echo "  per-peer sampling seed: ${PEER_SAMPLING_SEED}"
    if [ "${COMPUTE_DEG_METRICS}" = "1" ]; then
        echo "  DEG metrics: enabled"
    fi
    if [ "${COMPUTE_RETRIEVAL_METRICS}" = "1" ]; then
        echo "  retrieval metrics: enabled (min compounds per line-time = ${MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME})"
    fi
    sbatch -W -J replicate_similarity_prepare \
        -t "${PREP_TIME}" \
        -n 1 \
        --qos="${QOS}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${PREP_MEM}" \
        -e "${LOGS_DIR}/replicate_similarity_prepare.%j.err" \
        -o "${LOGS_DIR}/replicate_similarity_prepare.%j.out" \
        --wrap="${PREP_CMD}"
elif [ "${START_STAGE}" -gt 1 ]; then
    echo "> Skipping stage 1 and reusing existing prepare outputs from ${OUTPUT_DIR}"
    if [ -f "${TASK_CONFIG_FILE}" ]; then
        if ! CONFIG_MISMATCHES=$(validate_reused_prepare_outputs 2>&1); then
            echo "> ERROR: existing prepare outputs are incompatible with the requested run settings."
            if [ -n "${CONFIG_MISMATCHES}" ]; then
                echo "${CONFIG_MISMATCHES}" | sed 's/^/  - /'
            fi
            echo "> Rerun from stage 1, for example:"
            echo "  START_STAGE=1 ${0}"
            exit 1
        fi
        refresh_saved_metric_flags
    fi
fi

if [ ! -f "${TASK_FILE}" ]; then
    echo "> ERROR: task file not found: ${TASK_FILE}"
    exit 1
fi

if [ "${START_STAGE}" -le 2 ] && [ "${END_STAGE}" -ge 2 ] && [ "${START_STAGE}" -gt 1 ]; then
    RESHARD_REASON=""
    if [ "${RESHARD_TASKS}" = "1" ]; then
        RESHARD_REASON="RESHARD_TASKS=1"
    elif [ -f "${TASK_CONFIG_FILE}" ]; then
        EXISTING_CONDITIONS_PER_TASK=$(python -c "import json, pathlib; print(json.loads(pathlib.Path('${TASK_CONFIG_FILE}').read_text()).get('conditions_per_task', ''))")
        if [ -n "${EXISTING_CONDITIONS_PER_TASK}" ] && [ "${EXISTING_CONDITIONS_PER_TASK}" != "${CONDITIONS_PER_TASK}" ]; then
            RESHARD_REASON="conditions_per_task changed (${EXISTING_CONDITIONS_PER_TASK} -> ${CONDITIONS_PER_TASK})"
        fi
    fi

    if [ -n "${RESHARD_REASON}" ]; then
        echo "> Rebuilding task shards before stage 2 because ${RESHARD_REASON}"
        eval "${RESHARD_CMD}"
    fi
fi

N_TASKS=$(($(wc -l < "${TASK_FILE}") - 1))
if [ "${N_TASKS}" -le 0 ]; then
    echo "> No replicate-scoring tasks to run."
    exit 0
fi

ARRAY_SPEC="1-${N_TASKS}"
if [ -n "${TASK_ARRAY_SPEC}" ]; then
    ARRAY_SPEC="${TASK_ARRAY_SPEC}"
elif [ "${MAX_CONCURRENT_TASKS}" -gt 0 ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS}"
fi

if [ "${START_STAGE}" -le 2 ] && [ "${END_STAGE}" -ge 2 ]; then
    echo "> Stage 2/3: Run ${N_TASKS} replicate-scoring tasks as a Slurm array (${ARRAY_SPEC})"
    sbatch -W -J replicate_similarity_task \
        -t "${TASK_TIME}" \
        -n 1 \
        --array="${ARRAY_SPEC}" \
        --qos="${QOS}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${TASK_MEM}" \
        -e "${LOGS_DIR}/replicate_similarity_task.%A_%a.err" \
        -o "${LOGS_DIR}/replicate_similarity_task.%A_%a.out" \
        --wrap="${RUN_TASK_CMD}"
elif [ "${START_STAGE}" -gt 2 ] || [ "${END_STAGE}" -lt 2 ]; then
    echo "> Skipping stage 2"
fi

if [ "${START_STAGE}" -le 3 ] && [ "${END_STAGE}" -ge 3 ]; then
    echo "> Stage 3/3: Merge task outputs"
    sbatch -W -J replicate_similarity_merge \
        -t "${MERGE_TIME}" \
        -n 1 \
        --qos="${QOS}" \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem="${MERGE_MEM}" \
        -e "${LOGS_DIR}/replicate_similarity_merge.%j.err" \
        -o "${LOGS_DIR}/replicate_similarity_merge.%j.out" \
        --wrap="${MERGE_CMD}"
else
    echo "> Skipping stage 3"
fi

echo "> Done. Outputs:"
echo "  - ${OUTPUT_DIR}/dataset_selection_summary.tsv"
echo "  - ${OUTPUT_DIR}/retained_replicate_conditions.tsv"
echo "  - ${OUTPUT_DIR}/line_global_gene_counts.tsv"
echo "  - ${OUTPUT_DIR}/condition_metric_summary.tsv"
echo "  - ${OUTPUT_DIR}/dataset_metric_summary.tsv"
echo "  - ${OUTPUT_DIR}/dataset_line_metric_summary.tsv"
echo "  - ${OUTPUT_DIR}/dataset_t_strength_relationship_summary.tsv"
echo "  - ${OUTPUT_DIR}/dataset_line_t_strength_relationship_summary.tsv"
if [ "${COMPUTE_DEG_METRICS}" = "1" ]; then
    echo "  - ${OUTPUT_DIR}/condition_deg_metric_summary.tsv"
    echo "  - ${OUTPUT_DIR}/dataset_deg_metric_summary.tsv"
    echo "  - ${OUTPUT_DIR}/dataset_line_deg_metric_summary.tsv"
fi
if [ "${COMPUTE_RETRIEVAL_METRICS}" = "1" ]; then
    echo "  - ${OUTPUT_DIR}/condition_retrieval_summary.tsv"
    echo "  - ${OUTPUT_DIR}/line_time_retrieval_summary.tsv"
    echo "  - ${OUTPUT_DIR}/dataset_retrieval_summary.tsv"
    echo "  - ${OUTPUT_DIR}/dataset_line_retrieval_summary.tsv"
fi
