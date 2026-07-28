#!/usr/bin/env python3
"""Notebook-free reviewer tables for within-dataset replicate agreement.

The expensive H5AD scoring remains in
``precompute_replicate_signature_similarity.py``.  That scorer writes one
condition-level row per replicate-supported condition, including the original
centroid baseline and the reviewer-requested individual-signature peer
baseline.  This module turns those rows into Tables 7, 8, and 10 with
PubChem-clustered BCa confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_bootstrap_ci import (
    cluster_bca_nested_mean_ci_table,
    summarize_ci_half_width_ranges,
)


DEFAULT_INPUT_DIR = (
    REPO_ROOT / "results" / "replicate_signature_similarity_sep_rep_combined"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "parallel_cross_source"
    / "replicate_reviewer_tables"
)
DEFAULT_SEED = 20260505

TABLE_INPUT_FILES = {
    7: "condition_deg_metric_summary.tsv",
    8: "condition_deg_metric_summary.tsv",
    10: "condition_metric_summary.tsv",
}

TABLE_METRICS: dict[int, dict[str, str]] = {
    7: {
        "observed_deg_lfc_spearman_sym_p05": (
            "mean_replicate_deg_lfc_spearman_sym_p05"
        ),
        "centroid_baseline_deg_lfc_spearman_sym_p05": (
            "mean_baseline_pair_deg_lfc_spearman_sym_p05"
        ),
        "delta_vs_centroid_deg_lfc_spearman_sym_p05": (
            "mean_delta_vs_baseline_pair_deg_lfc_spearman_sym_p05"
        ),
        "individual_peer_mean_deg_lfc_spearman_sym_p05": (
            "mean_peer_baseline_deg_lfc_spearman_sym_p05"
        ),
        "individual_peer_sd_deg_lfc_spearman_sym_p05": (
            "mean_peer_baseline_deg_lfc_spearman_sym_sd_p05"
        ),
        "individual_peer_fraction_below_observed_deg_lfc_spearman_sym_p05": (
            "mean_peer_baseline_deg_lfc_spearman_sym_fraction_below_observed_p05"
        ),
        "individual_peer_corrected_percentile_deg_lfc_spearman_sym_p05": (
            "mean_peer_baseline_deg_lfc_spearman_sym_corrected_percentile_p05"
        ),
        "delta_vs_individual_peer_deg_lfc_spearman_sym_p05": (
            "mean_delta_vs_peer_baseline_deg_lfc_spearman_sym_p05"
        ),
    },
    8: {
        "observed_direction_agreement_p05": (
            "mean_replicate_direction_agreement_p05"
        ),
        "centroid_baseline_direction_agreement_p05": (
            "mean_baseline_pair_direction_agreement_p05"
        ),
        "delta_vs_centroid_direction_agreement_p05": (
            "mean_delta_vs_baseline_pair_direction_agreement_p05"
        ),
        "individual_peer_mean_direction_agreement_p05": (
            "mean_peer_baseline_direction_agreement_p05"
        ),
        "individual_peer_sd_direction_agreement_p05": (
            "mean_peer_baseline_direction_agreement_sd_p05"
        ),
        "individual_peer_fraction_below_observed_direction_agreement_p05": (
            "mean_peer_baseline_direction_agreement_fraction_below_observed_p05"
        ),
        "individual_peer_corrected_percentile_direction_agreement_p05": (
            "mean_peer_baseline_direction_agreement_corrected_percentile_p05"
        ),
        "delta_vs_individual_peer_direction_agreement_p05": (
            "mean_delta_vs_peer_baseline_direction_agreement_p05"
        ),
    },
    10: {
        "observed_replicate_spearman_logfc": (
            "mean_replicate_spearman_logfc"
        ),
        "centroid_baseline_spearman_logfc": (
            "mean_replicate_baseline_spearman_logfc"
        ),
        "delta_vs_centroid_spearman_logfc": (
            "mean_replicate_minus_baseline_spearman_logfc"
        ),
        "individual_peer_mean_spearman_logfc": (
            "mean_peer_baseline_spearman_logfc"
        ),
        "individual_peer_sd_spearman_logfc": (
            "mean_peer_baseline_sd_spearman_logfc"
        ),
        "individual_peer_fraction_below_observed_spearman_logfc": (
            "mean_peer_baseline_fraction_below_observed_spearman_logfc"
        ),
        "individual_peer_corrected_percentile_spearman_logfc": (
            "mean_peer_baseline_corrected_percentile_spearman_logfc"
        ),
        "delta_vs_individual_peer_spearman_logfc": (
            "mean_replicate_minus_peer_baseline_spearman_logfc"
        ),
        "raw_observed_replicate_cosine": (
            "mean_replicate_cosine_logfc_raw"
        ),
        "raw_centroid_baseline_cosine": (
            "mean_replicate_baseline_cosine_logfc_raw"
        ),
        "raw_delta_vs_centroid_cosine": (
            "mean_replicate_minus_baseline_cosine_logfc_raw"
        ),
        "raw_individual_peer_mean_cosine": (
            "mean_peer_baseline_cosine_logfc_raw"
        ),
        "raw_individual_peer_sd_cosine": (
            "mean_peer_baseline_sd_cosine_logfc_raw"
        ),
        "raw_individual_peer_fraction_below_observed_cosine": (
            "mean_peer_baseline_fraction_below_observed_cosine_logfc_raw"
        ),
        "raw_individual_peer_corrected_percentile_cosine": (
            "mean_peer_baseline_corrected_percentile_cosine_logfc_raw"
        ),
        "raw_delta_vs_individual_peer_cosine": (
            "mean_replicate_minus_peer_baseline_cosine_logfc_raw"
        ),
        "dataset_normalized_observed_replicate_cosine": (
            "mean_replicate_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_centroid_baseline_cosine": (
            "mean_replicate_baseline_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_delta_vs_centroid_cosine": (
            "mean_replicate_minus_baseline_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_individual_peer_mean_cosine": (
            "mean_peer_baseline_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_individual_peer_sd_cosine": (
            "mean_peer_baseline_sd_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_individual_peer_fraction_below_observed_cosine": (
            "mean_peer_baseline_fraction_below_observed_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_individual_peer_corrected_percentile_cosine": (
            "mean_peer_baseline_corrected_percentile_cosine_logfc_normalized_dataset"
        ),
        "dataset_normalized_delta_vs_individual_peer_cosine": (
            "mean_replicate_minus_peer_baseline_cosine_logfc_normalized_dataset"
        ),
        "dataset_cell_type_normalized_observed_replicate_cosine": (
            "mean_replicate_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_centroid_baseline_cosine": (
            "mean_replicate_baseline_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_delta_vs_centroid_cosine": (
            "mean_replicate_minus_baseline_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_individual_peer_mean_cosine": (
            "mean_peer_baseline_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_individual_peer_sd_cosine": (
            "mean_peer_baseline_sd_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_individual_peer_fraction_below_observed_cosine": (
            "mean_peer_baseline_fraction_below_observed_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_individual_peer_corrected_percentile_cosine": (
            "mean_peer_baseline_corrected_percentile_cosine_logfc_normalized_dataset_cell_type"
        ),
        "dataset_cell_type_normalized_delta_vs_individual_peer_cosine": (
            "mean_replicate_minus_peer_baseline_cosine_logfc_normalized_dataset_cell_type"
        ),
    },
}

TABLE_PRIMARY_METRICS = {
    7: (
        "observed_deg_lfc_spearman_sym_p05",
        "centroid_baseline_deg_lfc_spearman_sym_p05",
        "delta_vs_centroid_deg_lfc_spearman_sym_p05",
        "individual_peer_mean_deg_lfc_spearman_sym_p05",
        "delta_vs_individual_peer_deg_lfc_spearman_sym_p05",
        "individual_peer_corrected_percentile_deg_lfc_spearman_sym_p05",
    ),
    8: (
        "observed_direction_agreement_p05",
        "centroid_baseline_direction_agreement_p05",
        "delta_vs_centroid_direction_agreement_p05",
        "individual_peer_mean_direction_agreement_p05",
        "delta_vs_individual_peer_direction_agreement_p05",
        "individual_peer_corrected_percentile_direction_agreement_p05",
    ),
    10: (
        "observed_replicate_spearman_logfc",
        "centroid_baseline_spearman_logfc",
        "delta_vs_centroid_spearman_logfc",
        "individual_peer_mean_spearman_logfc",
        "delta_vs_individual_peer_spearman_logfc",
        "individual_peer_corrected_percentile_spearman_logfc",
        "raw_observed_replicate_cosine",
        "raw_centroid_baseline_cosine",
        "raw_delta_vs_centroid_cosine",
        "raw_individual_peer_mean_cosine",
        "raw_delta_vs_individual_peer_cosine",
        "raw_individual_peer_corrected_percentile_cosine",
        "dataset_normalized_observed_replicate_cosine",
        "dataset_normalized_centroid_baseline_cosine",
        "dataset_normalized_delta_vs_centroid_cosine",
        "dataset_normalized_individual_peer_mean_cosine",
        "dataset_normalized_delta_vs_individual_peer_cosine",
        "dataset_normalized_individual_peer_corrected_percentile_cosine",
        "dataset_cell_type_normalized_observed_replicate_cosine",
        "dataset_cell_type_normalized_centroid_baseline_cosine",
        "dataset_cell_type_normalized_delta_vs_centroid_cosine",
        "dataset_cell_type_normalized_individual_peer_mean_cosine",
        "dataset_cell_type_normalized_delta_vs_individual_peer_cosine",
        "dataset_cell_type_normalized_individual_peer_corrected_percentile_cosine",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fingerprint(
    *,
    table: int,
    input_path: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> str:
    stat = input_path.stat()
    source_path = Path(__file__).resolve()
    payload = {
        "table": int(table),
        "input_path": str(input_path.resolve()),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "metrics": TABLE_METRICS[int(table)],
        "source_sha256": _sha256(source_path),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _read_table_input(path: Path, table: int) -> pd.DataFrame:
    required = {
        "dataset_name",
        "pubchem_cid",
        *TABLE_METRICS[int(table)].values(),
    }
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    missing = sorted(required - set(header))
    if missing:
        raise KeyError(
            f"Table {table} input {path} is missing columns: {missing}. "
            "Rerun precompute_replicate_signature_similarity.py with "
            "--compute-baseline-metrics and --compute-deg-metrics."
        )
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=[column for column in header if column in required],
        dtype={"dataset_name": "string", "pubchem_cid": "string"},
        low_memory=False,
    )
    if frame.empty:
        raise ValueError(f"Table {table} input is empty: {path}")
    return frame


def build_table_ci(
    frame: pd.DataFrame,
    *,
    table: int,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = DEFAULT_SEED,
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> pd.DataFrame:
    """Build the long-form Table 7, 8, or 10 clustered-CI result."""
    table = int(table)
    if table not in TABLE_METRICS:
        raise ValueError(f"Unsupported table {table}; choose 7, 8, or 10")
    result = cluster_bca_nested_mean_ci_table(
        frame,
        group_cols=["dataset_name"],
        metric_cols=TABLE_METRICS[table],
        cluster_col="pubchem_cid",
        n_boot=int(bootstrap_iterations),
        seed=int(bootstrap_seed),
        summary_level=f"table_{table}_dataset",
        workers=int(workers),
        bootstrap_batch_size=int(bootstrap_batch_size),
        progress=bool(progress),
        progress_desc=f"Table {table} BCa",
    )
    metric_order = {
        metric: position
        for position, metric in enumerate(TABLE_METRICS[table])
    }
    result["_metric_order"] = result["metric"].map(metric_order)
    result = result.sort_values(
        ["dataset_name", "_metric_order"],
        kind="stable",
    ).drop(columns="_metric_order")
    return result.reset_index(drop=True)


def format_estimate(mean: object, low: object, high: object) -> str:
    values = pd.to_numeric(pd.Series([mean, low, high]), errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        return ""
    return f"{values.iloc[0]:.3f} [{values.iloc[1]:.3f}, {values.iloc[2]:.3f}]"


def build_presentation_table(ci: pd.DataFrame, *, table: int) -> pd.DataFrame:
    """Pivot primary reviewer metrics to one human-readable row per dataset."""
    focused = ci.loc[
        ci["metric"].isin(TABLE_PRIMARY_METRICS[int(table)])
    ].copy()
    focused["estimate_ci"] = [
        format_estimate(mean, low, high)
        for mean, low, high in zip(
            focused["mean"],
            focused["ci_low"],
            focused["ci_high"],
        )
    ]
    wide = focused.pivot(
        index="dataset_name",
        columns="metric",
        values="estimate_ci",
    )
    return (
        wide.reindex(columns=TABLE_PRIMARY_METRICS[int(table)])
        .reset_index()
        .rename_axis(columns=None)
    )


def output_paths(output_dir: Path, table: int) -> dict[str, Path]:
    return {
        "ci": output_dir / f"table_{table}_peer_baseline_cluster_bca_ci.tsv",
        "presentation": output_dir / f"table_{table}_peer_baseline.tsv",
        "ranges": output_dir / f"table_{table}_ci_half_width_ranges.tsv",
        "marker": output_dir / f"table_{table}_completion.json",
    }


def completion_is_valid(paths: Mapping[str, Path], fingerprint: str) -> bool:
    marker = paths["marker"]
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("fingerprint") != fingerprint:
        return False
    for name in ("ci", "presentation", "ranges"):
        path = paths[name]
        expected = payload.get("files", {}).get(name, {})
        if (
            not path.is_file()
            or int(expected.get("size", -1)) != path.stat().st_size
            or expected.get("sha256") != _sha256(path)
        ):
            return False
    return True


def run_table(
    *,
    table: int,
    input_dir: Path,
    output_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    workers: int,
    bootstrap_batch_size: int,
    progress: bool,
    force: bool,
) -> dict[str, Path]:
    table = int(table)
    input_path = Path(input_dir) / TABLE_INPUT_FILES[table]
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Table {table} condition metrics do not exist: {input_path}"
        )
    paths = output_paths(Path(output_dir), table)
    fingerprint = _fingerprint(
        table=table,
        input_path=input_path,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    if not force and completion_is_valid(paths, fingerprint):
        print(f"[Table {table}] compatible summary already complete; reusing it")
        return paths

    started = time.monotonic()
    frame = _read_table_input(input_path, table)
    print(
        f"[Table {table}] loaded {len(frame):,} condition rows from {input_path}"
    )
    ci = build_table_ci(
        frame,
        table=table,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
    )
    presentation = build_presentation_table(ci, table=table)
    ranges = summarize_ci_half_width_ranges(ci)
    _atomic_tsv(ci, paths["ci"])
    _atomic_tsv(presentation, paths["presentation"])
    _atomic_tsv(ranges, paths["ranges"])
    files = {
        name: {"size": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in paths.items()
        if name != "marker"
    }
    _atomic_json(
        {
            "table": table,
            "fingerprint": fingerprint,
            "input_path": str(input_path.resolve()),
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_seed": int(bootstrap_seed),
            "n_condition_rows": int(len(frame)),
            "n_ci_rows": int(len(ci)),
            "files": files,
        },
        paths["marker"],
    )
    print(
        f"[Table {table}] completed {len(ci):,} intervals in "
        f"{time.monotonic() - started:.1f}s -> {paths['presentation']}"
    )
    return paths


def _parse_tables(values: Sequence[str]) -> list[int]:
    requested = [
        int(item.strip())
        for value in values
        for item in str(value).split(",")
        if item.strip()
    ]
    if not requested:
        return [7, 8, 10]
    unknown = sorted(set(requested) - {7, 8, 10})
    if unknown:
        raise ValueError(f"Unsupported table(s): {unknown}; choose 7, 8, 10")
    return [table for table in (7, 8, 10) if table in set(requested)]


def build_parser(*, fixed_table: int | None = None) -> argparse.ArgumentParser:
    description = (
        f"Build reviewer Table {fixed_table} from replicate condition metrics."
        if fixed_table is not None
        else "Build reviewer Tables 7, 8, and 10 from replicate condition metrics."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing condition_metric_summary.tsv and "
            "condition_deg_metric_summary.tsv from the replicate precompute."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    if fixed_table is None:
        parser.add_argument(
            "--tables",
            action="append",
            default=[],
            metavar="7,8,10",
            help="Tables to build; repeat or use a comma-separated list.",
        )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "off"),
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fixed_table: int | None = None,
) -> int:
    args = build_parser(fixed_table=fixed_table).parse_args(argv)
    if args.bootstrap_iterations < 1:
        raise SystemExit("--bootstrap-iterations must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.bootstrap_batch_size < 1:
        raise SystemExit("--bootstrap-batch-size must be positive")
    tables = (
        [int(fixed_table)]
        if fixed_table is not None
        else _parse_tables(args.tables)
    )
    progress = args.progress == "always" or (
        args.progress == "auto" and os.isatty(2)
    )
    for table in tables:
        run_table(
            table=table,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
            workers=args.workers,
            bootstrap_batch_size=args.bootstrap_batch_size,
            progress=progress,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
