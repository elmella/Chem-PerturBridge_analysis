#!/usr/bin/env python3
"""Add the moderated-t replicate columns to the 12-dataset replicate input.

The t runs (``replicate_tdegcos_*``, and before them ``replicate_tpeers_*``)
rescored the same conditions with the same peer cap and seed as the main
replicate runs, with only the raw families plus the t families switched on.
This joins every column they have and the combined 12-dataset input lacks --
the t-peer baseline for Table 10 Spearman, and with ``--compute-t-deg-cosine``
Table 7's DEG-restricted Spearman and Table 10's cosine on t -- onto both
condition summaries (``condition_metric_summary.tsv`` feeds Table 10,
``condition_deg_metric_summary.tsv`` Tables 7 and 8).

Nothing joins unless the t runs are the same conditions: condition keys must
match one to one, and every column shared with the base must agree to within
1e-12. sci-Plex and Tahoe are in both groupings; their two copies must agree
too, and the copy from the run the base used is kept. ``--check-dirs`` names
earlier merged inputs (e.g. the t-peer merge) whose shared columns the result
must also reproduce, so a rerun cannot silently change a published number.

Two kinds of column are exempt, because they legitimately differ between runs
and no table reads them: the ``*_global`` metrics, which use the gene set
shared by every dataset in a run, and ``signed_overlap_t_top50``, whose
unstable top-50 sort resolves exact |t| ties differently on a few conditions.

Usage::

    uv run python scripts/merge_replicate_t_columns.py
    uv run python scripts/replicate_reviewer_tables.py \
        --input-dir results/replicate_all12_t_input \
        --output-dir results/parallel_cross_source/replicate_reviewer_tables_all12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["dataset_name", "condition_key"]
SUMMARY_FILES = ("condition_metric_summary.tsv", "condition_deg_metric_summary.tsv")
# Shared numeric columns must agree to within this; summation order leaves
# differences of ~1e-33 on DEG metrics whose values are essentially zero.
ABS_TOLERANCE = 1e-12


def run_dependent(column: str) -> bool:
    """Columns that differ between runs of the same conditions by design:
    the *_global metrics depend on which other datasets share a run, and the
    peer config fingerprint records the run's own flags and dataset list."""
    return column.endswith("_global") or column in {"n_global_shared_genes", "peer_config_fingerprint"}


def tie_sensitive(column: str) -> bool:
    """signed_overlap_t_top50 takes the top 50 genes by |t| with an unstable
    sort, so exact |t| ties at the cutoff resolve differently between runs."""
    return "signed_overlap_t_top50" in column


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"pubchem_cid": str}, low_memory=False)


def column_mismatches(left: pd.DataFrame, right: pd.DataFrame, columns) -> dict[str, int]:
    """Rows that differ per column; both frames indexed identically on KEY."""
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


def require_agreement(diffs: dict[str, int], what: str, n: int) -> None:
    hard = {k: v for k, v in diffs.items() if not tie_sensitive(k) and not run_dependent(k)}
    if hard:
        raise SystemExit(f"{what}: {hard}")
    for column, count in diffs.items():
        if tie_sensitive(column) and not run_dependent(column):
            print(f"[merge] {what}: {column} differs on {count} of {n:,} conditions (unstable-sort ties; no table reads it)")


def choose_run_per_dataset(base: pd.DataFrame, runs: list[pd.DataFrame], names: list[str]) -> dict[str, int]:
    """For a dataset in several runs, the run whose run-dependent gene count matches the base."""
    base_global = base.set_index(KEY)["n_global_shared_genes"]
    chosen: dict[str, int] = {}
    for dataset in sorted(base["dataset_name"].unique()):
        candidates = [i for i, run in enumerate(runs) if (run["dataset_name"] == dataset).any()]
        if not candidates:
            raise SystemExit(f"no t run covers {dataset}")
        if len(candidates) == 1:
            chosen[dataset] = candidates[0]
            continue
        expected = base_global.loc[[dataset]]
        matching = [
            i for i in candidates
            if runs[i].set_index(KEY)["n_global_shared_genes"].reindex(expected.index).equals(expected)
        ]
        if len(matching) != 1:
            raise SystemExit(f"cannot tell which t run matches the base for {dataset}")
        chosen[dataset] = matching[0]
        print(f"[merge] {dataset}: in {len(candidates)} runs; using {names[matching[0]]}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, default=Path("results/replicate_all12_input"))
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        default=[Path("results/replicate_tdegcos_full_v1"), Path("results/replicate_tdegcos_l1000_cigs_v1")],
    )
    parser.add_argument(
        "--check-dirs",
        nargs="*",
        type=Path,
        default=[Path("results/replicate_all12_tpeers_input")],
        help="Earlier merged inputs whose shared columns the result must reproduce.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/replicate_all12_t_input"))
    args = parser.parse_args()
    names = [str(d) for d in args.run_dirs]

    chosen: dict[str, int] | None = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in SUMMARY_FILES:
        base = read(args.base_dir / filename)
        if base.duplicated(KEY).any():
            raise SystemExit(f"{filename}: base has duplicate condition keys")
        runs = [read(d / filename) for d in args.run_dirs]
        if chosen is None:
            chosen = choose_run_per_dataset(base, runs, names)

        # Datasets scored in several runs must agree across them.
        for dataset in sorted(base["dataset_name"].unique()):
            copies = [run[run.dataset_name == dataset].set_index(KEY).sort_index() for run in runs
                      if (run["dataset_name"] == dataset).any()]
            for other in copies[1:]:
                if not other.index.equals(copies[0].index):
                    raise SystemExit(f"{filename}: runs disagree on {dataset}'s conditions")
                diffs = column_mismatches(copies[0], other, [c for c in copies[0].columns if c in other.columns])
                require_agreement(diffs, f"{filename}: runs disagree on {dataset}", len(other))

        joined = pd.concat(
            [run[run.dataset_name.isin([d for d, i in chosen.items() if i == k])] for k, run in enumerate(runs)],
            ignore_index=True,
        )
        if joined.duplicated(KEY).any() or set(map(tuple, joined[KEY].to_numpy())) != set(map(tuple, base[KEY].to_numpy())):
            raise SystemExit(f"{filename}: t runs do not cover the base conditions one to one")

        left = base.set_index(KEY).sort_index()
        right = joined.set_index(KEY).sort_index().loc[left.index]
        shared = [c for c in right.columns if c in left.columns]
        require_agreement(column_mismatches(left, right, shared), f"{filename}: t runs disagree with the base", len(left))
        new_columns = [c for c in right.columns if c not in left.columns and not run_dependent(c)]
        merged = base.merge(joined[[*KEY, *new_columns]], on=KEY, how="left", validate="one_to_one")

        for check_dir in args.check_dirs:
            earlier = read(check_dir / filename).set_index(KEY).sort_index()
            if not earlier.index.equals(left.index):
                raise SystemExit(f"{filename}: {check_dir} covers different conditions")
            current = merged.set_index(KEY).sort_index()
            checked = [c for c in earlier.columns if c in current.columns]
            require_agreement(column_mismatches(earlier, current, checked), f"{filename}: result differs from {check_dir}", len(current))
            print(f"[merge] {filename}: reproduces all {len(checked)} columns of {check_dir}")

        merged.to_csv(args.output_dir / filename, sep="\t", index=False)
        print(f"[merge] {filename}: {len(merged):,} conditions, {len(shared)} shared columns agree, "
              f"{len(new_columns)} new: {new_columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
