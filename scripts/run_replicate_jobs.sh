#!/usr/bin/env bash
# Finish every pending replicate-scoring job, in order. Safe to rerun at any
# time -- after a reboot, rerun it and it picks up where each job stopped:
#
#   nohup scripts/run_replicate_jobs.sh > logs/replicate_jobs.log 2>&1 &
#
# Scoring is CPU-bound and every condition is independent, so on a larger
# instance raise the worker count; the scorer's own default is capped at 32:
#
#   SCORING_WORKERS=126 RETRIEVAL_THREADS=64 nohup scripts/run_replicate_jobs.sh ...
#
# A job counts as finished once its shards have been merged into
# condition_metric_summary.tsv. Unfinished jobs resume from their own saved
# config (scripts/resume_replicate_run.py), so completed shards are reused and
# flags cannot drift between attempts.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
SCORER=scripts/precompute_replicate_signature_similarity.py
WORKER_FLAGS=()
if [[ -n "${SCORING_WORKERS:-}" ]]; then
  WORKER_FLAGS=(--workers "$SCORING_WORKERS")
fi

# Shared settings, identical across jobs so the tables are comparable.
COMMON_FLAGS=(
  --compute-baseline-metrics --compute-deg-metrics --deg-definitions p05
  --compute-normalized-deg --compute-normalized-cosine --compute-normalized-spearman
  --normalization-scales all --max-baseline-peers 256 --peer-sampling-seed 20260505
  --conditions-per-task 25 --progress off
)

# Each job: output directory, then its dataset list.
#  - replicate_full_v1: every non-L1000 dataset not in the published run.
#  - replicate_l1000_cigs_v1: the published run's six datasets. L1000's
#    retained conditions depend on which non-L1000 datasets share its run, so
#    this set is kept equal to the published one to reproduce its L1000 rows.
JOBS=(
  "results/replicate_full_v1|op3,dilimap_train_val,gdpx2,sciplex,tahoe,vcpi_0002,vcpi_0001,novartis_batch_2500"
  "results/replicate_l1000_cigs_v1|cigs_mce,cigs_tcm,sciplex,tahoe,l1000_phase1,l1000_phase2"
)

for job in "${JOBS[@]}"; do
  out="${job%%|*}"
  datasets="${job##*|}"
  if [[ -f "$out/condition_metric_summary.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out already merged; skipping"
    continue
  fi
  if [[ -f "$out/task_inputs/task_config.json" && -f "$out/task_manifest.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) resuming $out"
    "$PY" scripts/resume_replicate_run.py --output-dir "$out" "${WORKER_FLAGS[@]}"
  else
    echo "[jobs] $(date -u +%H:%M:%S) starting $out ($datasets)"
    "$PY" "$SCORER" --datasets "$datasets" --output-dir "$out" "${COMMON_FLAGS[@]}" "${WORKER_FLAGS[@]}"
  fi
  status=$?
  if [[ $status -ne 0 || ! -f "$out/condition_metric_summary.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out stopped (exit $status); rerun this script to continue"
    exit "$status"
  fi
  echo "[jobs] $(date -u +%H:%M:%S) $out merged"
done
echo "[jobs] $(date -u +%H:%M:%S) all replicate scoring jobs complete"

# Replicate retrieval (scripts/replicate_retrieval.py), from each scoring
# run's prepared inputs so the conditions match Tables 7/8/10. sci-Plex and
# Tahoe appear in both scoring runs with identical conditions, so they are
# retrieved once, from the first. Each run checkpoints per stratum and
# refuses to resume under different settings.
RETRIEVAL_THREADS="${RETRIEVAL_THREADS:-16}"
RETRIEVAL_JOBS=(
  "results/replicate_full_v1|results/replicate_retrieval_v1|op3,dilimap_train_val,gdpx2,sciplex,tahoe,vcpi_0002,vcpi_0001,novartis_batch_2500"
  "results/replicate_l1000_cigs_v1|results/replicate_retrieval_l1000_cigs_v1|cigs_mce,cigs_tcm,l1000_phase1,l1000_phase2"
)
for job in "${RETRIEVAL_JOBS[@]}"; do
  IFS="|" read -r prepared out datasets <<< "$job"
  if [[ -f "$out/tables/replicate_retrieval_table.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out already complete; skipping"
    continue
  fi
  echo "[jobs] $(date -u +%H:%M:%S) retrieval into $out ($datasets)"
  if ! "$PY" scripts/replicate_retrieval.py --prepared-dir "$prepared" --output-dir "$out" \
      --datasets "$datasets" --threads "$RETRIEVAL_THREADS"; then
    echo "[jobs] $(date -u +%H:%M:%S) $out stopped; rerun this script to continue"
    exit 1
  fi
done

# One table across all twelve datasets.
COMBINED=results/replicate_retrieval_all12
if [[ ! -f "$COMBINED/tables/replicate_retrieval_table.tsv" ]]; then
  mkdir -p "$COMBINED"
  "$PY" - <<'PY'
import pandas as pd
parts = [pd.read_csv(f"{d}/condition_retrieval_summary.tsv", sep="\t", dtype={"pubchem_cid": str})
         for d in ("results/replicate_retrieval_v1", "results/replicate_retrieval_l1000_cigs_v1")]
merged = pd.concat(parts, ignore_index=True)
assert not merged.duplicated(["dataset_name", "condition_key", "similarity_metric", "scale_variant"]).any()
merged.to_csv("results/replicate_retrieval_all12/condition_retrieval_summary.tsv", sep="\t", index=False)
print(f"[jobs] combined {merged.dataset_name.nunique()} datasets, {len(merged):,} rows")
PY
  "$PY" scripts/replicate_retrieval.py --output-dir "$COMBINED" --tables-only --threads 8
fi

# Individual-peer baseline for replicate Spearman on the moderated t-statistic
# (Table 10's t rows). Same dataset groupings, peer cap and seed as the main
# runs, so conditions and sampled peers are identical and only the t-peer
# columns are new. DEG metrics stay on so genes are masked exactly as in the
# main runs; the normalized families, which dominate cost, are left off.
TPEER_FLAGS=(
  --compute-baseline-metrics --compute-deg-metrics --deg-definitions p05
  --compute-t-peers --max-baseline-peers 256 --peer-sampling-seed 20260505
  --conditions-per-task 25 --progress off
)
TPEER_JOBS=(
  "results/replicate_tpeers_full_v1|op3,dilimap_train_val,gdpx2,sciplex,tahoe,vcpi_0002,vcpi_0001,novartis_batch_2500"
  "results/replicate_tpeers_l1000_cigs_v1|cigs_mce,cigs_tcm,sciplex,tahoe,l1000_phase1,l1000_phase2"
)
for job in "${TPEER_JOBS[@]}"; do
  out="${job%%|*}"
  datasets="${job##*|}"
  if [[ -f "$out/condition_metric_summary.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out already merged; skipping"
    continue
  fi
  if [[ -f "$out/task_inputs/task_config.json" && -f "$out/task_manifest.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) resuming $out"
    "$PY" scripts/resume_replicate_run.py --output-dir "$out" "${WORKER_FLAGS[@]}"
  else
    echo "[jobs] $(date -u +%H:%M:%S) starting $out ($datasets)"
    "$PY" "$SCORER" --datasets "$datasets" --output-dir "$out" "${TPEER_FLAGS[@]}" "${WORKER_FLAGS[@]}"
  fi
  if [[ ! -f "$out/condition_metric_summary.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out stopped; rerun this script to continue"
    exit 1
  fi
  echo "[jobs] $(date -u +%H:%M:%S) $out merged"
done
echo "[jobs] $(date -u +%H:%M:%S) all jobs complete"
