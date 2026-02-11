from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_DATASET_PATHS, resolve_dataset_paths
from .engine import run_cross_dataset_retrieval


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_dataset_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --dataset-path value '{item}'. Expected format: dataset_name=/path/to/data"
            )
        name, raw_path = item.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError(
                f"Invalid --dataset-path value '{item}'. Expected format: dataset_name=/path/to/data"
            )
        overrides[name] = Path(raw_path)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-dataset retrieval benchmark for differential expression signatures."
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a dataset path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--query-datasets",
        default=",".join(DEFAULT_DATASET_PATHS.keys()),
        help="Comma-separated query dataset names, or 'all'.",
    )
    parser.add_argument(
        "--db-datasets",
        default=",".join(DEFAULT_DATASET_PATHS.keys()),
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
        help="Also run retrieval within the same dataset (default is cross-dataset only).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable AnnData caching for cell-type slices.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="cross_dataset_retrieval",
        help="Prefix for output CSV files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_overrides = _parse_dataset_overrides(args.dataset_path)
    dataset_paths = resolve_dataset_paths(dataset_overrides)
    dataset_names = list(dataset_paths.keys())

    query_datasets = dataset_names if args.query_datasets.strip().lower() == "all" else _split_csv(args.query_datasets)
    db_datasets = dataset_names if args.db_datasets.strip().lower() == "all" else _split_csv(args.db_datasets)

    missing_query = sorted(set(query_datasets) - set(dataset_names))
    missing_db = sorted(set(db_datasets) - set(dataset_names))
    if missing_query or missing_db:
        raise KeyError(
            f"Unknown dataset names. query missing={missing_query}, db missing={missing_db}, "
            f"available={dataset_names}"
        )

    cell_type_filter = set(_split_csv(args.cell_types)) if args.cell_types else None

    detail_df, summary_by_cell_df, summary_overall_df = run_cross_dataset_retrieval(
        dataset_paths=dataset_paths,
        query_datasets=query_datasets,
        db_datasets=db_datasets,
        include_self_dataset=args.include_self_dataset,
        cell_type_filter=cell_type_filter,
        cache_cell_types=not args.no_cache,
        verbose=args.verbose,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"{args.output_prefix}_detail.csv"
    summary_cell_path = output_dir / f"{args.output_prefix}_summary_by_cell_type.csv"
    summary_overall_path = output_dir / f"{args.output_prefix}_summary_overall.csv"

    detail_df.to_csv(detail_path, index=False)
    summary_by_cell_df.to_csv(summary_cell_path, index=False)
    summary_overall_df.to_csv(summary_overall_path, index=False)

    print(f"Saved detail rows={len(detail_df)} -> {detail_path}")
    print(f"Saved summary by cell type rows={len(summary_by_cell_df)} -> {summary_cell_path}")
    print(f"Saved summary overall rows={len(summary_overall_df)} -> {summary_overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

