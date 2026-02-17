from __future__ import annotations

import argparse
from pathlib import Path

from .cli import (
    _parse_dataset_overrides,
    _parse_metrics,
    _parse_representations,
    _parse_skip_representations,
    _split_csv,
)
from .config import RetrievalSettings, resolve_dataset_paths
from .parallel import (
    build_truth_matches_and_tasks,
    merge_task_outputs,
    run_single_task,
    write_precompute_outputs,
)


def _resolve_dataset_inputs(args: argparse.Namespace) -> tuple[dict[str, Path], list[str], list[str]]:
    dataset_overrides = _parse_dataset_overrides(args.dataset_path)
    dataset_paths = resolve_dataset_paths(dataset_overrides)
    dataset_names = list(dataset_paths.keys())

    query_datasets = (
        dataset_names if args.query_datasets.strip().lower() == "all" else _split_csv(args.query_datasets)
    )
    db_datasets = dataset_names if args.db_datasets.strip().lower() == "all" else _split_csv(args.db_datasets)
    missing_query = sorted(set(query_datasets) - set(dataset_names))
    missing_db = sorted(set(db_datasets) - set(dataset_names))
    if missing_query or missing_db:
        raise KeyError(
            f"Unknown dataset names. query missing={missing_query}, db missing={missing_db}, "
            f"available={dataset_names}"
        )
    return dataset_paths, query_datasets, db_datasets


def _add_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a dataset path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-datasets",
        default="all",
        help="Comma-separated query dataset names, or 'all'.",
    )
    parser.add_argument(
        "--db-datasets",
        default="all",
        help="Comma-separated db dataset names, or 'all'.",
    )
    parser.add_argument(
        "--cell-types",
        default=None,
        help="Optional comma-separated list of cell types to evaluate.",
    )
    parser.add_argument(
        "--include-self-dataset",
        action="store_true",
        help="Also include query_dataset == db_dataset pairs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable AnnData caching for cell-type slices.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel retrieval pipeline CLI with three stages: "
            "precompute truth matches + tasks, run single task, merge task outputs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    precompute_parser = subparsers.add_parser(
        "precompute",
        help="Precompute true matches and generate pair x representation task matrix.",
    )
    _add_dataset_selection_args(precompute_parser)
    precompute_parser.add_argument(
        "--representations",
        default="all",
        help="Comma-separated representations to schedule, or 'all'.",
    )
    precompute_parser.add_argument(
        "--skip-representations",
        default="",
        help="Comma-separated representations to skip (or 'none').",
    )
    precompute_parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for precompute CSV outputs.",
    )
    precompute_parser.add_argument(
        "--output-prefix",
        default="cross_dataset_retrieval_parallel",
        help="Prefix for output CSV files.",
    )
    precompute_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )
    precompute_parser.add_argument(
        "--split-by-cell-type",
        action="store_true",
        help="Schedule one task per (dataset pair, query cell type, representation).",
    )

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Run one retrieval task from a precomputed task CSV.",
    )
    run_task_parser.add_argument(
        "--task-file",
        required=True,
        help="Path to the CSV produced by 'precompute' (<prefix>_tasks.csv).",
    )
    run_task_parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task id to execute (matches task_id column; Slurm array friendly).",
    )
    run_task_parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a dataset path. Can be passed multiple times.",
    )
    run_task_parser.add_argument(
        "--cell-types",
        default=None,
        help="Optional comma-separated list of cell types to evaluate.",
    )
    run_task_parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated metrics to evaluate, or 'all'.",
    )
    run_task_parser.add_argument(
        "--output-dir",
        default="results/tasks",
        help="Directory for per-task detail CSV outputs.",
    )
    run_task_parser.add_argument(
        "--output-prefix",
        default="cross_dataset_retrieval_parallel",
        help="Prefix for task output filenames.",
    )
    run_task_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable AnnData caching for cell-type slices.",
    )
    run_task_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge per-task outputs into final detail + summaries.",
    )
    merge_parser.add_argument(
        "--task-file",
        required=True,
        help="Path to the CSV produced by 'precompute' (<prefix>_tasks.csv).",
    )
    merge_parser.add_argument(
        "--task-output-dir",
        required=True,
        help="Directory containing per-task detail CSV outputs.",
    )
    merge_parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for merged final outputs.",
    )
    merge_parser.add_argument(
        "--output-prefix",
        default="cross_dataset_retrieval_parallel",
        help="Prefix for merged output CSV files.",
    )
    merge_parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any expected task output file is missing.",
    )
    merge_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )

    return parser


def _run_precompute(args: argparse.Namespace) -> int:
    dataset_paths, query_datasets, db_datasets = _resolve_dataset_inputs(args)
    cell_type_filter = set(_split_csv(args.cell_types)) if args.cell_types else None
    settings = RetrievalSettings(
        include_representations=_parse_representations(args.representations),
        skip_representations=_parse_skip_representations(args.skip_representations),
    )

    truth_df, truth_summary_df, tasks_df = build_truth_matches_and_tasks(
        dataset_paths=dataset_paths,
        query_datasets=query_datasets,
        db_datasets=db_datasets,
        settings=settings,
        include_self_dataset=args.include_self_dataset,
        cell_type_filter=cell_type_filter,
        cache_cell_types=not args.no_cache,
        split_by_cell_type=args.split_by_cell_type,
        output_prefix=args.output_prefix,
        verbose=args.verbose,
    )
    output_dir = Path(args.output_dir)
    truth_path, truth_summary_path, tasks_path = write_precompute_outputs(
        truth_df=truth_df,
        truth_summary_df=truth_summary_df,
        tasks_df=tasks_df,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
    )
    print(f"Saved truth matches rows={len(truth_df)} -> {truth_path}")
    print(f"Saved truth summary rows={len(truth_summary_df)} -> {truth_summary_path}")
    print(f"Saved tasks rows={len(tasks_df)} -> {tasks_path}")
    return 0


def _run_task(args: argparse.Namespace) -> int:
    dataset_overrides = _parse_dataset_overrides(args.dataset_path)
    dataset_paths = resolve_dataset_paths(dataset_overrides)
    include_metrics = _parse_metrics(args.metrics)
    cell_type_filter = set(_split_csv(args.cell_types)) if args.cell_types else None

    detail_path = run_single_task(
        dataset_paths=dataset_paths,
        task_file=Path(args.task_file),
        task_id=args.task_id,
        include_metrics=include_metrics,
        task_output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        cell_type_filter=cell_type_filter,
        cache_cell_types=not args.no_cache,
        verbose=args.verbose,
    )
    print(f"Saved task detail -> {detail_path}")
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    detail_path, summary_cell_path, summary_overall_path, missing_files = merge_task_outputs(
        task_file=Path(args.task_file),
        task_output_dir=Path(args.task_output_dir),
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        strict_missing=args.strict_missing,
        verbose=args.verbose,
    )
    print(f"Saved merged detail -> {detail_path}")
    print(f"Saved merged summary by cell type -> {summary_cell_path}")
    print(f"Saved merged summary overall -> {summary_overall_path}")
    if missing_files:
        print(f"Warning: missing task outputs={len(missing_files)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "precompute":
        return _run_precompute(args)
    if args.command == "run-task":
        return _run_task(args)
    if args.command == "merge":
        return _run_merge(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
