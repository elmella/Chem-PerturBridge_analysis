#!/usr/bin/env python3
"""Resume an interrupted replicate-scoring run from its own saved config.

``precompute_replicate_signature_similarity.py`` already checkpoints: each task
shard writes ``task_outputs/task_NNNNNN_condition_metric_summary.tsv``, and
``--prepared-only`` reuses every non-empty shard and re-runs only the rest. It
also refuses to resume when the requested flags differ from the ones the run
was prepared with, which is the right call -- mixing a 512-peer shard with a
256-peer shard would produce a table whose rows were not comparable.

The catch is that resuming correctly means retyping a dozen flags exactly. This
command reads them back out of ``task_inputs/task_config.json`` instead, so the
resume cannot drift from the original run.

Usage::

    uv run python scripts/resume_replicate_run.py --output-dir results/replicate_new5_full
    uv run python scripts/resume_replicate_run.py --output-dir results/... --dry-run

Whatever is left to do is reported first, so an interrupted run's remaining
cost is visible before it restarts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SCORER = SCRIPT_DIR / "precompute_replicate_signature_similarity.py"

# Config key -> command-line flag, for the settings the scorer fingerprints.
STORE_TRUE_FLAGS = {
    "compute_baseline_metrics": "--compute-baseline-metrics",
    "compute_deg_metrics": "--compute-deg-metrics",
    "compute_retrieval_metrics": "--compute-retrieval-metrics",
    "compute_normalized_cosine": "--compute-normalized-cosine",
    "compute_normalized_spearman": "--compute-normalized-spearman",
    "compute_normalized_deg": "--compute-normalized-deg",
    "test_one_line_per_dataset": "--test-one-line-per-dataset",
}
VALUE_FLAGS = {
    "deg_definitions": "--deg-definitions",
    "normalization_scales": "--normalization-scales",
    "population_stats_root": "--population-stats-root",
    "min_retrieval_compounds_per_line_time": "--min-retrieval-compounds-per-line-time",
    "max_baseline_peers": "--max-baseline-peers",
    "peer_sampling_seed": "--peer-sampling-seed",
    "min_replicates_per_condition": "--min-replicates-per-condition",
    "conditions_per_task": "--conditions-per-task",
    "test_max_conditions_per_dataset": "--test-max-conditions-per-dataset",
}


def load_config(output_dir: Path) -> dict:
    config_path = output_dir / "task_inputs" / "task_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No saved task config at {config_path}. This directory was never "
            "prepared, so there is nothing to resume -- start a fresh run."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def shard_progress(output_dir: Path) -> tuple[int, int, list[int]]:
    """Completed and total shard counts, plus the pending task ids."""
    manifest = output_dir / "task_manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"No task manifest at {manifest}; cannot tell what is pending."
        )
    with manifest.open(encoding="utf-8") as handle:
        total = max(sum(1 for _ in handle) - 1, 0)
    task_outputs = output_dir / "task_outputs"
    pending: list[int] = []
    for task_id in range(1, total + 1):
        shard = task_outputs / f"task_{task_id:06d}_condition_metric_summary.tsv"
        if not (shard.is_file() and shard.stat().st_size > 0):
            pending.append(task_id)
    return total - len(pending), total, pending


def build_command(output_dir: Path, config: dict, workers: int | None) -> list[str]:
    command = [
        sys.executable,
        str(SCORER),
        "--output-dir",
        str(output_dir),
        "--prepared-only",
    ]
    datasets = config.get("datasets")
    if datasets:
        command += ["--datasets", ",".join(str(name) for name in datasets)]
    for key, flag in STORE_TRUE_FLAGS.items():
        if config.get(key):
            command.append(flag)
    for key, flag in VALUE_FLAGS.items():
        value = config.get(key)
        if value is None:
            continue
        command += [flag, str(value)]
    if workers is not None:
        command += ["--workers", str(int(workers))]
    command += ["--progress", "off"]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="The interrupted run's output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override worker count; omit to use the scorer's default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resume command and what is pending, then stop.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    config = load_config(output_dir)
    completed, total, pending = shard_progress(output_dir)

    print(f"[resume] {output_dir}")
    print(f"[resume] datasets: {', '.join(config.get('datasets', []) ) or 'unknown'}")
    print(
        f"[resume] peers={config.get('max_baseline_peers')} "
        f"shard={config.get('conditions_per_task')} conditions "
        f"seed={config.get('peer_sampling_seed')}"
    )
    print(f"[resume] shards complete: {completed:,}/{total:,}")
    if not pending:
        print("[resume] nothing pending; rerun to merge the finished shards.")
    else:
        preview = ", ".join(str(task_id) for task_id in pending[:8])
        if len(pending) > 8:
            preview += ", ..."
        print(f"[resume] pending {len(pending):,} shards: {preview}")

    command = build_command(output_dir, config, args.workers)
    print("[resume] command:")
    print("  " + " ".join(command))
    if args.dry_run:
        return 0
    # Inherit stdout/stderr so the scorer's own progress lines stream through.
    return subprocess.call(command, cwd=str(REPO_ROOT), env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
