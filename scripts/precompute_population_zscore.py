#!/usr/bin/env python3
"""Precompute reusable per-gene population-standardization statistics.

This command writes compact ``.npz`` sufficient-statistic caches only.  It never
materializes standardized ``.h5ad`` copies.

Examples
--------
Explicit line files::

    uv run python scripts/precompute_population_zscore.py \
      --source cigs_mce=CVCL_0023=/data/cigs_mce/CVCL_0023_de.h5ad \
      --source cigs_mce=CVCL_1055=/data/cigs_mce/CVCL_1055_de.h5ad

Discover all direct ``.h5ad`` children of dataset directories::

    uv run python scripts/precompute_population_zscore.py \
      --dataset-dir cigs_mce=/data/cigs_mce/group_rep/results \
      --dataset-dir cigs_tcm=/data/cigs_tcm/group_rep/results \
      --row-chunk-size 1024 \
      --workers 2

``--workers`` parallelizes independent line files only. Dataset-wide statistics are
still pooled in canonical cell-type order, so serial and parallel runs are numerically
identical.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

# Pin BLAS before pandas/numpy load: each spawned worker would otherwise try
# to claim every core, and the resulting oversubscription is the usual reason
# a run gets slower as --workers rises.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dataset_layout import dge_dataset_dir  # noqa: E402
from scripts.lazy_h5ad import open_lazy_h5ad  # noqa: E402
from scripts.population_zscore import (  # noqa: E402
    DEFAULT_ROW_CHUNK_SIZE,
    discover_dataset_population_sources,
    infer_line_cell_type,
    load_or_fit_cell_type_population_stats_from_source_stats,
    load_or_fit_dataset_population_stats_from_source_stats,
    load_or_fit_population_stats,
    resolve_source_cell_type,
    stats_qc_record,
)


SCOPE_BOTH = "both"
SCOPE_DATASET_CELL_TYPE = "dataset-cell-type"
SCOPE_DATASET = "dataset"


def default_worker_count(*, cap: int) -> int:
    """Spawned workers to use when nothing is requested.

    Parallelism here is per line file, and most datasets have only a handful --
    Novartis, VCPI-0001, VCPI-0002 and DILImap have exactly one each -- so the
    file count, not the core count, usually sets the limit. The cap stays small
    for that reason; a dataset with more files than workers still streams
    through them in canonical order.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 2
    return int(min(cap, max(2, cores - 2)))


def _fit_line_task(
    task: tuple[str, str, Path, Path, str, int, bool],
):
    (
        dataset_name,
        cell_type,
        source_path,
        cache_root,
        layer_name,
        row_chunk_size,
        force,
    ) = task
    return load_or_fit_population_stats(
        source_path=source_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        cache_root=cache_root,
        layer_name=layer_name,
        row_chunk_size=row_chunk_size,
        force=force,
        verbose=True,
    )


def _pool_split_cell_types(
    *,
    dataset_name: str,
    line_sources: dict,
    line_stats_by_cell: dict,
    cache_root: Path,
    force: bool,
) -> list:
    """Write a per-cell-type cache wherever several files share one cell type.

    Line caches are keyed by filename, which is normally the cell type. GDPx2
    splits each line across seeding densities, so its files key as
    ``CL_0000515_0.0625`` while every downstream scorer looks the population up
    as ``CL_0000515``. Those groups are pooled into the name the scorers use;
    sources that already agree are left completely alone.
    """
    groups: dict[str, list[str]] = {}
    for file_key, source_path in line_sources.items():
        if file_key not in line_stats_by_cell:
            continue
        adata = open_lazy_h5ad(source_path)
        try:
            resolved = resolve_source_cell_type(adata.obs, file_key)
        finally:
            adata.close()
        groups.setdefault(resolved, []).append(file_key)

    pooled = []
    for cell_type, file_keys in sorted(groups.items()):
        if len(file_keys) == 1 and file_keys[0] == cell_type:
            continue  # the file's own cache is already at the right path
        pooled.append(
            load_or_fit_cell_type_population_stats_from_source_stats(
                source_stats=[line_stats_by_cell[key] for key in sorted(file_keys)],
                dataset_name=dataset_name,
                cell_type=cell_type,
                cache_root=cache_root,
                force=force,
                verbose=True,
            )
        )
    return pooled


def configured_dataset_dirs(repo_root: Path) -> list[tuple[str, Path]]:
    """Return the line-level source directories used by the three notebooks.

    Locations are resolved rather than hardcoded: the tarball-only datasets
    unpack under ``<dataset>/group_rep_extracted/`` with inconsistent internal
    prefixes, and a hardcoded shape drops a source without saying so.
    """
    data_root = Path(repo_root) / "data" / "theislab_temp"
    dataset_names = (
        "l1000_phase1",
        "l1000_phase2",
        "tahoe",
        "cigs_mce",
        "novartis_batch_2500",
        "vcpi_0001",
        "cigs_tcm",
        "vcpi_0002",
        "gdpx2",
        "sciplex",
        "dilimap_train_val",
        "op3",
    )
    resolved: list[tuple[str, Path]] = []
    for dataset_name in dataset_names:
        directory = dge_dataset_dir(data_root, dataset_name, "group_rep")
        if directory.is_dir() and any(directory.glob("*.h5ad")):
            resolved.append((dataset_name, directory))
    return resolved


def _split_spec(value: str, *, expected_parts: int, label: str) -> list[str]:
    parts = value.split("=", maxsplit=expected_parts - 1)
    if len(parts) != expected_parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            f"{label} must contain {expected_parts} non-empty '='-separated fields"
        )
    return [part.strip() for part in parts]


def parse_source_spec(value: str) -> tuple[str, str, Path]:
    dataset_name, cell_type, path = _split_spec(
        value,
        expected_parts=3,
        label="--source DATASET=CELL_TYPE=PATH",
    )
    return dataset_name, cell_type, Path(path)


def parse_dataset_dir_spec(value: str) -> tuple[str, Path]:
    dataset_name, path = _split_spec(
        value,
        expected_parts=2,
        label="--dataset-dir DATASET=PATH",
    )
    return dataset_name, Path(path)


def infer_cell_type(path: Path) -> str:
    """Backward-compatible wrapper for existing callers and tests."""
    return infer_line_cell_type(path)


def collect_sources(
    *,
    explicit_sources: Sequence[tuple[str, str, Path]],
    dataset_dirs: Sequence[tuple[str, Path]],
    recursive: bool,
) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}

    def add(dataset_name: str, cell_type: str, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Population source does not exist: {path}")
        existing = sources.setdefault(dataset_name, {}).get(cell_type)
        if existing is not None and existing.resolve() != path.resolve():
            raise ValueError(
                f"Conflicting sources for {dataset_name}/{cell_type}: "
                f"{existing} and {path}"
            )
        sources[dataset_name][cell_type] = path

    for dataset_name, cell_type, path in explicit_sources:
        add(dataset_name, cell_type, path)

    for dataset_name, directory in dataset_dirs:
        for cell_type, path in discover_dataset_population_sources(
            directory,
            recursive=recursive,
        ).items():
            add(dataset_name, cell_type, path)
    if not sources:
        raise ValueError("Provide at least one --source or --dataset-dir")
    return {
        dataset_name: dict(sorted(line_sources.items()))
        for dataset_name, line_sources in sorted(sources.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute dataset×cell-type and dataset-wide per-gene population "
            "z-score statistics without writing standardized H5AD files."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=parse_source_spec,
        metavar="DATASET=CELL_TYPE=PATH",
        help="Add one explicit line-level H5AD source (repeatable).",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        type=parse_dataset_dir_spec,
        metavar="DATASET=PATH",
        help="Discover line-level H5AD files in a dataset directory (repeatable).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover H5AD files below each --dataset-dir.",
    )
    parser.add_argument(
        "--all-configured",
        action="store_true",
        help=(
            "Discover every currently available dataset source directory used by "
            "the three cross-source notebooks."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "results" / "w4_population_zscore_stats",
    )
    parser.add_argument(
        "--scope",
        choices=(SCOPE_BOTH, SCOPE_DATASET_CELL_TYPE, SCOPE_DATASET),
        default=SCOPE_BOTH,
    )
    parser.add_argument("--layer-name", default="logFC")
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_ROW_CHUNK_SIZE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(cap=8),
        help=(
            "Independent line files to process concurrently. Defaults to a "
            "quarter of the core count, capped at 8; a dataset with a single "
            "line file gains nothing from more. Use 1 for serial behavior."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the requested caches even when fingerprints match.",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        help="Optionally save one TSV row per generated or reloaded cache.",
    )
    parser.add_argument(
        "--page-cache-prewarm",
        choices=("blocking", "background", "off"),
        help=(
            "Read each source sequentially before scanning it, so the small "
            "gzip-chunk reads are served from RAM. 'blocking' (default) warms "
            "then scans; 'background' lets the scan race the warmer. Sets "
            "CPB_PAGE_CACHE_PREWARM for this run and its workers."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    if getattr(args, "page_cache_prewarm", None):
        # Set before workers are spawned so they inherit the same policy.
        os.environ["CPB_PAGE_CACHE_PREWARM"] = args.page_cache_prewarm
    if args.row_chunk_size < 1:
        raise ValueError("--row-chunk-size must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    dataset_dirs = list(args.dataset_dir)
    if args.all_configured:
        dataset_dirs.extend(configured_dataset_dirs(REPO_ROOT))
    sources = collect_sources(
        explicit_sources=args.source,
        dataset_dirs=dataset_dirs,
        recursive=args.recursive,
    )
    n_lines = sum(len(line_sources) for line_sources in sources.values())
    print(
        f"[w4_precompute] {len(sources):,} datasets, {n_lines:,} line files; "
        f"scope={args.scope}; workers={args.workers}; "
        f"cache_root={args.cache_root}",
        flush=True,
    )

    qc_records: list[dict[str, object]] = []
    for dataset_index, (dataset_name, line_sources) in enumerate(
        sources.items(),
        start=1,
    ):
        print(
            f"[w4_precompute] dataset {dataset_index}/{len(sources)}: "
            f"{dataset_name} ({len(line_sources)} lines)",
            flush=True,
        )
        line_tasks = [
            (
                dataset_name,
                cell_type,
                source_path,
                args.cache_root,
                args.layer_name,
                args.row_chunk_size,
                args.force,
            )
            for cell_type, source_path in line_sources.items()
        ]
        line_stats_by_cell = {}
        line_started_at = time.monotonic()
        if args.workers == 1 or len(line_tasks) == 1:
            for line_index, task in enumerate(line_tasks, start=1):
                cell_type = task[1]
                print(
                    f"[w4_precompute]   line {line_index}/{len(line_sources)}: "
                    f"{cell_type}",
                    flush=True,
                )
                stats = _fit_line_task(task)
                line_stats_by_cell[cell_type] = stats
        else:
            worker_count = min(args.workers, len(line_tasks))
            print(
                f"[w4_precompute]   processing {len(line_tasks)} lines with "
                f"{worker_count} workers",
                flush=True,
            )
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                future_to_cell = {
                    executor.submit(_fit_line_task, task): task[1]
                    for task in line_tasks
                }
                for completed_count, future in enumerate(
                    as_completed(future_to_cell),
                    start=1,
                ):
                    cell_type = future_to_cell[future]
                    stats = future.result()
                    line_stats_by_cell[cell_type] = stats
                    elapsed = time.monotonic() - line_started_at
                    print(
                        f"[w4_precompute]   completed "
                        f"{completed_count}/{len(line_tasks)}: {cell_type} "
                        f"(elapsed {elapsed / 60.0:.1f}m)",
                        flush=True,
                    )

        ordered_line_stats = [
            line_stats_by_cell[cell_type]
            for cell_type in sorted(line_stats_by_cell)
        ]
        if args.scope in {SCOPE_BOTH, SCOPE_DATASET_CELL_TYPE}:
            qc_records.extend(
                stats_qc_record(stats)
                for stats in ordered_line_stats
            )
            qc_records.extend(
                stats_qc_record(stats)
                for stats in _pool_split_cell_types(
                    dataset_name=dataset_name,
                    line_sources=line_sources,
                    line_stats_by_cell=line_stats_by_cell,
                    cache_root=args.cache_root,
                    force=args.force,
                )
            )

        if args.scope in {SCOPE_BOTH, SCOPE_DATASET}:
            dataset_stats = load_or_fit_dataset_population_stats_from_source_stats(
                source_stats=ordered_line_stats,
                dataset_name=dataset_name,
                cache_root=args.cache_root,
                force=args.force,
                verbose=True,
            )
            qc_records.append(stats_qc_record(dataset_stats))

        print(
            f"[w4_precompute] dataset {dataset_name} complete in "
            f"{(time.monotonic() - line_started_at) / 60.0:.1f}m",
            flush=True,
        )

    qc = pd.DataFrame(qc_records)
    if args.qc_output is not None:
        args.qc_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.qc_output.with_name(
            f".{args.qc_output.name}.tmp-{os.getpid()}"
        )
        try:
            qc.to_csv(temporary, sep="\t", index=False)
            os.replace(temporary, args.qc_output)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"[w4_precompute] saved QC to {args.qc_output}", flush=True)
    print("[w4_precompute] complete", flush=True)
    return qc


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
