#!/usr/bin/env python3
"""Raw-scale reviewer Tables 4, 5 and 6 for all nine cross-source pairs.

The published raw Tables 4 and 6 cover only the six original manuscript pairs:
raw CIGS rows were missing from the local cache when they were built. The
``raw_cigs_v1`` DEG and signature runs rescored all nine pairs on raw logFC,
so this builds the raw tables from them with the same summary as the
normalized tables (``summarize_reviewer_final_tables.py``): per-compound
means within each dataset pair, then a compound-clustered BCa bootstrap.

The scorers write raw individual-peer metrics under a ``pb_`` prefix; the
normalized runs write the same metrics as ``<scale>__w4_*``. Renaming
``pb_*`` to ``w4_*`` lets both go through the one metric list, so raw and
normalized rows are directly comparable.

Table 5 (raw direction agreement) is rebuilt too, as a check: it was already
published for all nine pairs from the production run, so matching it
validates the CIGS rows that are new in Tables 4 and 6.

Usage::

    uv run python scripts/summarize_raw_cross_source_tables.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_source_parallel import atomic_write_frame
from scripts.summarize_reviewer_final_tables import (
    TABLE5_METRICS,
    TABLE5_SCORER_ALIASES,
    _ci_presentation,
    _dataset_pair_labels,
    _require_columns,
    build_table5_ci,
)
from scripts.summarize_reviewer_minimal_metrics import (
    W4_DEG_METRICS,
    W4_IDENTITY_COLUMNS,
    W4_SIGNATURE_METRICS,
    _read_metrics,
    _w4_ci,
)

DEFAULT_DEG_METRICS = Path(
    "results/parallel_cross_source/deg/raw_cigs_v1/deg_scored_metrics.tsv"
)
DEFAULT_SIGNATURE_METRICS = Path(
    "results/parallel_cross_source/signature/raw_cigs_v1/"
    "signature_scored_metrics.tsv"
)
DEFAULT_SUMMARY_DIR = Path(
    "results/parallel_cross_source/reviewer_minimal_summary"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/parallel_cross_source/reviewer_raw_tables_all9"
)


# The one raw column whose name is not the table metric's with pb_ for w4_:
# the raw scorer drops "_sym" (its values equal observed_deg_lfc_spearman_sym_p05).
SCORER_COLUMN_OVERRIDES = {
    "w4_observed_deg_lfc_spearman_sym_p05": "pb_observed_deg_lfc_spearman_p05",
}


def scorer_column(metric: str) -> str:
    """Raw scorer column for a ``w4_*`` table metric."""
    if metric in SCORER_COLUMN_OVERRIDES:
        return SCORER_COLUMN_OVERRIDES[metric]
    if not metric.startswith("w4_"):
        raise ValueError(f"expected a w4_ metric, got {metric!r}")
    return "pb_" + metric[len("w4_"):]


def raw_long_frame(frame: pd.DataFrame, metrics: tuple[str, ...], label: str) -> pd.DataFrame:
    columns = {scorer_column(metric): metric for metric in metrics}
    identity = [column for column in W4_IDENTITY_COLUMNS if column in frame.columns]
    _require_columns(frame, [*identity, *columns], label=label)
    result = frame[[*identity, *columns]].rename(columns=columns)
    result.insert(2, "scale_variant", "raw")
    return result


def build_raw_ci(frame, metrics, *, summary_level, args) -> pd.DataFrame:
    return _w4_ci(
        raw_long_frame(frame, metrics, f"{summary_level} metrics"),
        metrics=metrics,
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
        summary_level=summary_level,
        workers=int(args.workers),
        bootstrap_batch_size=int(args.bootstrap_batch_size),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deg-metrics", type=Path, default=DEFAULT_DEG_METRICS)
    parser.add_argument("--signature-metrics", type=Path, default=DEFAULT_SIGNATURE_METRICS)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    pair_labels = _dataset_pair_labels(args.summary_dir)
    identity = set(W4_IDENTITY_COLUMNS)

    deg = _read_metrics(
        args.deg_metrics,
        label="raw DEG metrics",
        columns={
            *identity,
            *(scorer_column(metric) for metric in W4_DEG_METRICS),
            *TABLE5_METRICS,
            *TABLE5_SCORER_ALIASES.values(),
        },
    )
    signature = _read_metrics(
        args.signature_metrics,
        label="raw signature metrics",
        columns={*identity, *(scorer_column(metric) for metric in W4_SIGNATURE_METRICS)},
    )

    tables = {
        "table4_raw": build_raw_ci(deg, W4_DEG_METRICS, summary_level="table_4_raw_dataset_pair", args=args),
        "table5_direction_agreement": build_table5_ci(
            deg,
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            workers=int(args.workers),
            bootstrap_batch_size=int(args.bootstrap_batch_size),
        ),
        "table6_raw": build_raw_ci(signature, W4_SIGNATURE_METRICS, summary_level="table_6_raw_dataset_pair", args=args),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ci in tables.items():
        atomic_write_frame(args.output_dir / f"{name}_cluster_bca_ci.tsv", ci)
        atomic_write_frame(
            args.output_dir / f"{name}.tsv",
            _ci_presentation(ci, pair_labels=pair_labels),
        )
        print(f"[raw-tables] {name}: {len(ci):,} rows, {ci[['dataset_a', 'dataset_b']].drop_duplicates().shape[0]} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
