#!/usr/bin/env python3
"""Add the replicate t-peer columns to the 12-dataset replicate input.

The t-peer runs (``replicate_tpeers_*``) rescored the same conditions with the
same peer cap and seed as the main replicate runs, but with only the raw
families switched on, so the only new columns are the individual-peer baseline
for replicate Spearman on t. This joins those columns onto the combined
12-dataset condition summary that Tables 7/8/10 are built from.

Before joining, it checks that the t-peer runs are the same conditions: the
condition keys must match one to one, and every column the two share must be
identical (NaN-aware, to within 1e-12). sci-Plex and Tahoe are in both t-peer runs, like the
main runs; their two copies must agree too. Any disagreement stops the merge,
so a stale or differently-configured run cannot slip into the tables.

Usage::

    uv run python scripts/merge_replicate_tpeers.py
    uv run python scripts/replicate_reviewer_tables.py \
        --input-dir results/replicate_all12_tpeers_input \
        --output-dir results/parallel_cross_source/replicate_reviewer_tables_all12_tpeers
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["dataset_name", "condition_key"]
T_PEER_COLUMNS = [
    "mean_peer_baseline_spearman_t",
    "mean_peer_baseline_sd_spearman_t",
    "mean_peer_baseline_fraction_below_observed_spearman_t",
    "mean_peer_baseline_corrected_percentile_spearman_t",
    "mean_replicate_minus_peer_baseline_spearman_t",
    "n_valid_peer_baseline_t_pairs",
]
# Shared numeric columns must agree to within this; summation order leaves
# differences of ~1e-33 on DEG metrics whose values are essentially zero.
ABS_TOLERANCE = 1e-12


def tie_sensitive(column: str) -> bool:
    """signed_overlap_t_top50 takes the top 50 genes by |t| with an unstable
    sort, so exact |t| ties at the cutoff resolve differently between runs.
    No table reads it; differences are reported, not treated as mismatches."""
    return "signed_overlap_t_top50" in column


def run_dependent(column: str) -> bool:
    """Columns that legitimately differ between runs of the same conditions.

    The *_global metrics use the gene set shared by every dataset in the run,
    so they change with which other datasets a run includes; the peer config
    fingerprint records the run's dataset list. No table reads either.
    """
    return column.endswith("_global") or column in {"n_global_shared_genes", "peer_config_fingerprint"}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"pubchem_cid": str}, low_memory=False)


def column_mismatches(left: pd.DataFrame, right: pd.DataFrame, columns) -> dict[str, int]:
    """Rows that differ per shared column; both frames aligned on KEY."""
    out = {}
    for column in columns:
        a, b = left[column], right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            same = (np.abs(a - b) <= ABS_TOLERANCE) | (np.isnan(a) & np.isnan(b))
        else:
            same = (a.astype(str) == b.astype(str)).to_numpy()
        if (~same).any():
            out[column] = int((~same).sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, default=Path("results/replicate_all12_input"))
    parser.add_argument(
        "--tpeer-dirs",
        nargs="+",
        type=Path,
        default=[Path("results/replicate_tpeers_full_v1"), Path("results/replicate_tpeers_l1000_cigs_v1")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/replicate_all12_tpeers_input"))
    args = parser.parse_args()

    base = read(args.base_dir / "condition_metric_summary.tsv")
    if base.duplicated(KEY).any():
        raise SystemExit("base summary has duplicate condition keys")

    parts = []
    for order, directory in enumerate(args.tpeer_dirs):
        part = read(directory / "condition_metric_summary.tsv")
        part["_run"] = order
        parts.append(part)
    tpeer = pd.concat(parts, ignore_index=True)
    shared = [column for column in tpeer.columns if column in base.columns and column not in KEY]
    invariant = [column for column in shared if not run_dependent(column)]

    # Datasets present in more than one t-peer run must agree with each other.
    duplicated = tpeer[tpeer.duplicated(KEY, keep=False)]
    if not duplicated.empty:
        first = duplicated.drop_duplicates(KEY, keep="first").set_index(KEY).sort_index()
        last = duplicated.drop_duplicates(KEY, keep="last").set_index(KEY).sort_index()
        diffs = column_mismatches(first, last, [*invariant, *T_PEER_COLUMNS])
        hard = {k: v for k, v in diffs.items() if not tie_sensitive(k)}
        if hard:
            raise SystemExit(f"t-peer runs disagree on shared datasets: {hard}")
        print(f"[merge] {len(first):,} conditions scored in both t-peer runs agree "
              f"({sorted(duplicated.dataset_name.unique())})")
        # Keep, per dataset, the copy from the run the base summary came from:
        # the one whose run-dependent columns match it too.
        base_global = base.set_index(KEY)["n_global_shared_genes"]
        keep = []
        for dataset, rows in tpeer.groupby("dataset_name", sort=False):
            runs = rows["_run"].unique()
            if len(runs) == 1:
                keep.append(rows)
                continue
            matching = [
                run for run in runs
                if rows.loc[rows._run == run].set_index(KEY)["n_global_shared_genes"]
                .reindex(base_global.loc[[dataset]].index).equals(base_global.loc[[dataset]])
            ]
            if len(matching) != 1:
                raise SystemExit(f"cannot tell which t-peer run matches the base for {dataset}")
            keep.append(rows.loc[rows._run == matching[0]])
            print(f"[merge] {dataset}: using {args.tpeer_dirs[matching[0]]}")
        tpeer = pd.concat(keep, ignore_index=True)

    base_keys = set(map(tuple, base[KEY].to_numpy()))
    tpeer = tpeer.drop(columns="_run")
    tpeer_keys = set(map(tuple, tpeer[KEY].to_numpy()))
    if base_keys != tpeer_keys:
        raise SystemExit(
            f"condition sets differ: {len(base_keys - tpeer_keys)} only in base, "
            f"{len(tpeer_keys - base_keys)} only in t-peer runs"
        )

    left = base.set_index(KEY).sort_index()
    right = tpeer.set_index(KEY).sort_index().loc[left.index]
    diffs = column_mismatches(left, right, shared)
    # The base came from runs with the same dataset groupings, so its
    # run-dependent columns must match too -- except the fingerprint, which
    # also records the flags each run was made with.
    diffs.pop("peer_config_fingerprint", None)
    hard = {k: v for k, v in diffs.items() if not tie_sensitive(k)}
    if hard:
        raise SystemExit(f"t-peer runs disagree with the base summary: {hard}")
    for column, count in diffs.items():
        print(f"[merge] {column}: {count} of {len(left):,} conditions differ (unstable-sort ties; not a table metric)")
    print(f"[merge] {len(shared) - len(diffs)} other shared columns identical across {len(left):,} conditions (tolerance {ABS_TOLERANCE:g})")

    merged = base.merge(tpeer[[*KEY, *T_PEER_COLUMNS]], on=KEY, how="left", validate="one_to_one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "condition_metric_summary.tsv", sep="\t", index=False)
    # DEG summaries are untouched by the t-peer columns; carry them over as is.
    shutil.copy2(args.base_dir / "condition_deg_metric_summary.tsv", args.output_dir / "condition_deg_metric_summary.tsv")
    n_valid = merged["n_valid_peer_baseline_t_pairs"].fillna(0).gt(0)
    print(f"[merge] wrote {len(merged):,} conditions; t-peer baseline present for {int(n_valid.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
