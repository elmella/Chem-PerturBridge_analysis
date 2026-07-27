#!/usr/bin/env python3
"""Summarize nested retrieval peer-cap sensitivity with compound BCa CIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table
from scripts.cross_source_parallel import (
    atomic_write_frame,
    output_directory_lock,
)


METRICS = (
    "source_individual_n_peers",
    "source_individual_mean_similarity",
    "source_individual_sd_similarity",
    "source_individual_fraction_below_observed",
    "source_individual_corrected_percentile",
    "target_individual_n_peers",
    "target_individual_mean_similarity",
    "target_individual_sd_similarity",
    "target_individual_fraction_below_observed",
    "target_individual_corrected_percentile",
    "source_peer_normalized_rank",
    "target_peer_normalized_rank",
    "delta_vs_source_peer_normalized_rank",
    "delta_vs_target_peer_normalized_rank",
)
IDENTITY_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "direction",
    "cell_type",
    "time_key",
    "query_obs_id",
    "query_pubchem_cid",
    "similarity_metric",
    "scale_variant",
    "peer_cap_order",
    "peer_cap_label",
)
OUTPUT_NAME = "retrieval_peer_sensitivity_cluster_bca_ci.tsv"


def build_summary(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    workers: int,
    bootstrap_batch_size: int,
    progress: bool,
) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Peer sensitivity metrics are missing columns: {missing}")
    if frame.duplicated(
        [
            "dataset_a",
            "dataset_b",
            "direction",
            "cell_type",
            "time_key",
            "query_obs_id",
            "query_pubchem_cid",
            "similarity_metric",
            "scale_variant",
            "peer_cap_label",
        ]
    ).any():
        raise ValueError("Peer sensitivity metrics contain duplicate identities")
    return cluster_bca_nested_mean_ci_table(
        frame,
        group_cols=[
            "dataset_a",
            "dataset_b",
            "similarity_metric",
            "scale_variant",
            "peer_cap_order",
            "peer_cap_label",
        ],
        metric_cols=list(METRICS),
        cluster_col="query_pubchem_cid",
        inner_cols=["direction", "cell_type", "time_key"],
        outer_cols=["direction"],
        n_boot=n_boot,
        seed=seed,
        summary_level="retrieval_peer_sensitivity_dataset_pair",
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
        progress_desc="peer sensitivity BCa",
    ).sort_values(
        [
            "dataset_a",
            "dataset_b",
            "similarity_metric",
            "scale_variant",
            "peer_cap_order",
            "metric",
        ]
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize retrieval peer-cap sensitivity sidecars."
    )
    parser.add_argument("--peer-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "off"),
        default="auto",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    path = Path(args.peer_metrics)
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    usecols = [
        column
        for column in header
        if column in {*IDENTITY_COLUMNS, *METRICS}
    ]
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype={
            column: "string"
            for column in IDENTITY_COLUMNS
            if column not in {"peer_cap_order"}
        },
        low_memory=False,
    )
    progress = args.progress == "always" or (
        args.progress == "auto" and sys.stderr.isatty()
    )
    summary = build_summary(
        frame,
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
        workers=int(args.workers),
        bootstrap_batch_size=int(args.bootstrap_batch_size),
        progress=progress,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_directory_lock(output_dir):
        output_path = output_dir / OUTPUT_NAME
        atomic_write_frame(output_path, summary)
    print(
        f"[peer-sensitivity-summary] wrote {len(summary):,} rows -> "
        f"{output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
