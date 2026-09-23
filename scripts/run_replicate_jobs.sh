#!/usr/bin/env bash
# Finish every pending replicate-scoring job, in order. Safe to rerun at any
# time -- after a reboot, rerun it and it picks up where each job stopped:
#
#   nohup scripts/run_replicate_jobs.sh > logs/replicate_jobs.log 2>&1 &
#
# A job counts as finished once its shards have been merged into
# condition_metric_summary.tsv. Unfinished jobs resume from their own saved
# config (scripts/resume_replicate_run.py), so completed shards are reused and
# flags cannot drift between attempts.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
SCORER=scripts/precompute_replicate_signature_similarity.py

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
    "$PY" scripts/resume_replicate_run.py --output-dir "$out"
  else
    echo "[jobs] $(date -u +%H:%M:%S) starting $out ($datasets)"
    "$PY" "$SCORER" --datasets "$datasets" --output-dir "$out" "${COMMON_FLAGS[@]}"
  fi
  status=$?
  if [[ $status -ne 0 || ! -f "$out/condition_metric_summary.tsv" ]]; then
    echo "[jobs] $(date -u +%H:%M:%S) $out stopped (exit $status); rerun this script to continue"
    exit "$status"
  fi
  echo "[jobs] $(date -u +%H:%M:%S) $out merged"
done
echo "[jobs] $(date -u +%H:%M:%S) all replicate jobs complete"
