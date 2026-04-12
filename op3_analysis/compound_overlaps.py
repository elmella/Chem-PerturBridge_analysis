from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable

import anndata as ad
import pandas as pd

from .retrieval.config import resolve_dataset_paths
from .retrieval.data import DatasetStore

NAME_COLUMNS = ("perturbagen_name", "perturbagen", "perturbation_label")
INVALID_CID_VALUES = frozenset({"", "nan", "none", "<na>"})
PRECOMPUTE_LOG_PATTERNS = ("*precompute*.out",)
TRUTH_SUMMARY_PATTERNS = ("*_truth_summary.csv",)
PRECOMPUTE_ELIGIBLE_RE = re.compile(
    r"^\[precompute\] pair (?P<query>.+?)->(?P<db>.+?): eligible_query_cell_types=(?P<count>\d+)\s*$"
)
PRECOMPUTE_NO_OVERLAP_RE = re.compile(
    r"^\[precompute\] pair (?P<query>.+?)->(?P<db>.+?): no overlapping cell types; skipping\s*$"
)


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


def _cid_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _pipe_join(values: Iterable[str]) -> str:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values
            if not pd.isna(value) and str(value).strip()
        }
    )
    return "|".join(cleaned)


def _read_obs_table(h5ad_path: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        return adata.obs.copy()
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def _iter_obs_tables(dataset_path: Path) -> Iterable[pd.DataFrame]:
    if dataset_path.is_dir():
        files = sorted(dataset_path.glob("*_de.h5ad"))
        if not files:
            files = sorted(dataset_path.glob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found under dataset directory: {dataset_path}")
        for path in files:
            yield _read_obs_table(path)
        return

    if dataset_path.is_file() and dataset_path.suffix == ".h5ad":
        yield _read_obs_table(dataset_path)
        return

    raise FileNotFoundError(f"Unsupported dataset path: {dataset_path}")


def _normalize_is_control(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"})


def summarize_dataset_cell_types(dataset_name: str, dataset_path: Path) -> list[str]:
    store = DatasetStore(dataset_name=dataset_name, dataset_path=dataset_path, cache_enabled=False)
    return sorted(store.list_cell_types())


def summarize_dataset_compounds(dataset_name: str, dataset_path: Path, verbose: bool = False) -> pd.DataFrame:
    counts: dict[str, int] = defaultdict(int)
    names: dict[str, set[str]] = defaultdict(set)

    for obs in _iter_obs_tables(dataset_path):
        if "pubchem_cid" not in obs.columns:
            raise KeyError(
                f"{dataset_name} is missing obs['pubchem_cid']; available columns: {list(obs.columns)}"
            )

        frame = obs.copy()
        if "pert_type" in frame.columns:
            pert_type = frame["pert_type"].astype("string").str.strip().str.lower()
            compound_mask = pert_type.eq("compound")
            if bool(compound_mask.any()):
                frame = frame.loc[compound_mask]

        if "is_control" in frame.columns:
            frame = frame.loc[~_normalize_is_control(frame["is_control"])]

        cids = frame["pubchem_cid"].astype("string").str.strip()
        valid_cids = cids.notna() & ~cids.str.lower().isin(INVALID_CID_VALUES)
        if not bool(valid_cids.any()):
            continue

        frame = frame.loc[valid_cids].copy()
        frame["pubchem_cid"] = cids.loc[valid_cids]
        name_column = next((column for column in NAME_COLUMNS if column in frame.columns), None)
        if name_column is not None:
            compound_names = frame[name_column].astype("string").str.strip()
            valid_names = compound_names.notna() & compound_names.ne("")
            named_rows = frame.loc[valid_names, ["pubchem_cid"]].copy()
            named_rows["compound_name"] = compound_names.loc[valid_names]
            named_rows = named_rows.drop_duplicates()
            for row in named_rows.itertuples(index=False):
                names[str(row.pubchem_cid)].add(str(row.compound_name))

        grouped_counts = frame.groupby("pubchem_cid", sort=False).size()
        for pubchem_cid, count in grouped_counts.items():
            counts[str(pubchem_cid)] += int(count)

    summary = pd.DataFrame({"pubchem_cid": sorted(counts, key=_cid_sort_key)})
    summary[f"{dataset_name}_n_rows"] = summary["pubchem_cid"].map(counts).astype(int)
    summary[f"{dataset_name}_perturbagen_names"] = summary["pubchem_cid"].map(
        lambda cid: "|".join(sorted(names.get(str(cid), set())))
    )

    if verbose:
        print(
            f"[compound-overlaps] dataset={dataset_name} unique_compounds={len(summary)} path={dataset_path}",
            flush=True,
        )
    return summary


def _discover_files(
    values: list[str],
    default_roots: tuple[Path, ...],
    patterns: tuple[str, ...],
) -> list[Path]:
    if values:
        roots = [Path(value).expanduser() for value in values]
        strict_missing = True
    else:
        roots = [root for root in default_roots if root.exists()]
        strict_missing = False

    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            if strict_missing:
                raise FileNotFoundError(f"Search path does not exist: {root}")
            continue

        candidates: list[Path]
        if root.is_dir():
            candidates = []
            for pattern in patterns:
                candidates.extend(sorted(path for path in root.rglob(pattern) if path.is_file()))
        elif root.is_file():
            candidates = [root]
        else:
            continue

        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(path)

    return discovered


def _parse_precompute_logs(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(paths, key=lambda item: (item.stat().st_mtime, str(item))):
        pair_entries: dict[tuple[str, str], dict[str, object]] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            match = PRECOMPUTE_ELIGIBLE_RE.match(line)
            if match is None:
                match = PRECOMPUTE_NO_OVERLAP_RE.match(line)
                if match is None:
                    continue
                count = 0
            else:
                count = int(match.group("count"))

            key = (match.group("query").strip(), match.group("db").strip())
            pair_entries[key] = {
                "precompute_eligible_query_cell_types": int(count),
                "precompute_log_path": str(path),
            }

        mtime = float(path.stat().st_mtime)
        for key, entry in pair_entries.items():
            previous = by_pair.get(key)
            if previous is None or mtime >= float(previous["_mtime"]):
                by_pair[key] = {
                    **entry,
                    "_mtime": mtime,
                }

    return by_pair


def _parse_truth_summaries(paths: list[Path]) -> dict[tuple[str, str], dict[str, object]]:
    required_columns = {
        "query_dataset",
        "db_dataset",
        "query_cell_type",
        "n_shared_pubchem_cids",
        "n_truth_queries",
    }
    by_pair: dict[tuple[str, str], dict[str, object]] = {}

    for path in sorted(paths, key=lambda item: (item.stat().st_mtime, str(item))):
        frame = pd.read_csv(path)
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise KeyError(
                f"Truth summary file {path} is missing required columns: {missing_columns}"
            )
        if frame.empty:
            continue

        frame = frame.copy()
        frame["query_dataset"] = frame["query_dataset"].astype("string").str.strip()
        frame["db_dataset"] = frame["db_dataset"].astype("string").str.strip()
        frame["query_cell_type"] = frame["query_cell_type"].astype("string").str.strip()
        frame["n_shared_pubchem_cids"] = pd.to_numeric(
            frame["n_shared_pubchem_cids"], errors="coerce"
        )
        frame["n_truth_queries"] = pd.to_numeric(frame["n_truth_queries"], errors="coerce")

        grouped = (
            frame.groupby(["query_dataset", "db_dataset"], dropna=False)
            .agg(
                truth_summary_cell_lines_with_shared_compounds=("query_cell_type", "nunique"),
                truth_summary_shared_line_compound_pairs=("n_shared_pubchem_cids", "sum"),
                truth_summary_mean_shared_compounds_per_line=("n_shared_pubchem_cids", "mean"),
                truth_summary_max_shared_compounds_per_line=("n_shared_pubchem_cids", "max"),
                truth_summary_total_truth_queries=("n_truth_queries", "sum"),
                truth_summary_shared_cell_type_list=("query_cell_type", _pipe_join),
            )
            .reset_index()
        )

        mtime = float(path.stat().st_mtime)
        for row in grouped.itertuples(index=False):
            key = (str(row.query_dataset), str(row.db_dataset))
            previous = by_pair.get(key)
            if previous is not None and mtime < float(previous["_mtime"]):
                continue

            by_pair[key] = {
                "truth_summary_cell_lines_with_shared_compounds": int(
                    row.truth_summary_cell_lines_with_shared_compounds
                ),
                "truth_summary_shared_line_compound_pairs": int(
                    row.truth_summary_shared_line_compound_pairs
                ),
                "truth_summary_mean_shared_compounds_per_line": float(
                    row.truth_summary_mean_shared_compounds_per_line
                ),
                "truth_summary_max_shared_compounds_per_line": int(
                    row.truth_summary_max_shared_compounds_per_line
                ),
                "truth_summary_total_truth_queries": int(row.truth_summary_total_truth_queries),
                "truth_summary_shared_cell_type_list": str(
                    row.truth_summary_shared_cell_type_list
                ),
                "truth_summary_path": str(path),
                "_mtime": mtime,
            }

    return by_pair


def _prefer_forward_value(
    forward: dict[str, object],
    reverse: dict[str, object],
    field: str,
) -> object | None:
    if field in forward:
        return forward[field]
    if field in reverse:
        return reverse[field]
    return None


def _mismatch_notes(
    forward: dict[str, object],
    reverse: dict[str, object],
    fields: Iterable[str],
) -> str:
    notes: list[str] = []
    for field in fields:
        if field not in forward or field not in reverse:
            continue
        if forward[field] != reverse[field]:
            notes.append(f"{field}:{forward[field]}!={reverse[field]}")
    return ";".join(notes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report Tahoe-vs-other overlaps from datasets and retrieval logs, and "
            "save the shared compound sets to separate CSV files."
        )
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override a dataset path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tahoe-dataset",
        default="tahoe",
        help="Dataset name to use as the Tahoe reference.",
    )
    parser.add_argument(
        "--compare-datasets",
        default="all",
        help="Comma-separated dataset names to compare against Tahoe, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/compound_overlaps",
        help="Directory for overlap CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for output filenames.",
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help=(
            "Optional explicit path for the pair summary CSV. Defaults to "
            "<output-dir>/<prefix><tahoe-dataset>_overlap_summary.csv."
        ),
    )
    parser.add_argument(
        "--log-path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Precompute log file or directory to search. Can be passed multiple times. "
            "Defaults to ./logs if present."
        ),
    )
    parser.add_argument(
        "--truth-summary-path",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Truth summary CSV file or directory to search. Can be passed multiple times. "
            "Defaults to ./results if present."
        ),
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

    if args.tahoe_dataset not in dataset_paths:
        raise KeyError(
            f"Unknown Tahoe dataset '{args.tahoe_dataset}'. Available datasets: {dataset_names}"
        )

    if args.compare_datasets.strip().lower() == "all":
        compare_datasets = [name for name in dataset_names if name != args.tahoe_dataset]
    else:
        compare_datasets = [name for name in _split_csv(args.compare_datasets) if name != args.tahoe_dataset]

    missing_compare = sorted(set(compare_datasets) - set(dataset_names))
    if missing_compare:
        raise KeyError(
            f"Unknown compare datasets: {missing_compare}. Available datasets: {dataset_names}"
        )
    if not compare_datasets:
        raise ValueError("No comparison datasets selected after excluding the Tahoe dataset itself.")

    tahoe_summary = summarize_dataset_compounds(
        dataset_name=args.tahoe_dataset,
        dataset_path=dataset_paths[args.tahoe_dataset],
        verbose=args.verbose,
    )
    tahoe_cell_types = summarize_dataset_cell_types(
        dataset_name=args.tahoe_dataset,
        dataset_path=dataset_paths[args.tahoe_dataset],
    )
    tahoe_compounds = set(tahoe_summary["pubchem_cid"].astype(str))

    log_paths = _discover_files(
        values=args.log_path,
        default_roots=(Path("logs"),),
        patterns=PRECOMPUTE_LOG_PATTERNS,
    )
    truth_summary_paths = _discover_files(
        values=args.truth_summary_path,
        default_roots=(Path("results"),),
        patterns=TRUTH_SUMMARY_PATTERNS,
    )
    precompute_by_pair = _parse_precompute_logs(log_paths)
    truth_summary_by_pair = _parse_truth_summaries(truth_summary_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    summary_rows: list[dict[str, object]] = []

    for dataset_name in compare_datasets:
        other_summary = summarize_dataset_compounds(
            dataset_name=dataset_name,
            dataset_path=dataset_paths[dataset_name],
            verbose=args.verbose,
        )
        other_cell_types = summarize_dataset_cell_types(
            dataset_name=dataset_name,
            dataset_path=dataset_paths[dataset_name],
        )
        shared_compounds = sorted(
            tahoe_compounds & set(other_summary["pubchem_cid"].astype(str)),
            key=_cid_sort_key,
        )
        shared_cell_types = sorted(set(tahoe_cell_types) & set(other_cell_types))
        overlap_df = pd.DataFrame({"pubchem_cid": shared_compounds})
        overlap_df = overlap_df.merge(tahoe_summary, on="pubchem_cid", how="left")
        overlap_df = overlap_df.merge(other_summary, on="pubchem_cid", how="left")

        output_path = output_dir / f"{file_prefix}{args.tahoe_dataset}__{dataset_name}_compound_overlap.csv"
        overlap_df.to_csv(output_path, index=False)

        forward_key = (args.tahoe_dataset, dataset_name)
        reverse_key = (dataset_name, args.tahoe_dataset)
        forward_log = precompute_by_pair.get(forward_key, {})
        reverse_log = precompute_by_pair.get(reverse_key, {})
        forward_truth = truth_summary_by_pair.get(forward_key, {})
        reverse_truth = truth_summary_by_pair.get(reverse_key, {})

        consistency_notes = _mismatch_notes(
            forward=forward_truth,
            reverse=reverse_truth,
            fields=(
                "truth_summary_cell_lines_with_shared_compounds",
                "truth_summary_shared_line_compound_pairs",
                "truth_summary_shared_cell_type_list",
            ),
        )

        summary_row = {
            "tahoe_dataset": args.tahoe_dataset,
            "other_dataset": dataset_name,
            "shared_cell_lines_dataset": int(len(shared_cell_types)),
            "shared_cell_lines_dataset_list": _pipe_join(shared_cell_types),
            "shared_compounds_global": int(len(shared_compounds)),
            "compound_overlap_csv": str(output_path),
            "shared_cell_lines_with_shared_compounds": _prefer_forward_value(
                forward_truth,
                reverse_truth,
                "truth_summary_cell_lines_with_shared_compounds",
            ),
            "shared_line_compound_pairs": _prefer_forward_value(
                forward_truth,
                reverse_truth,
                "truth_summary_shared_line_compound_pairs",
            ),
            "mean_shared_compounds_per_shared_line": _prefer_forward_value(
                forward_truth,
                reverse_truth,
                "truth_summary_mean_shared_compounds_per_line",
            ),
            "max_shared_compounds_per_shared_line": _prefer_forward_value(
                forward_truth,
                reverse_truth,
                "truth_summary_max_shared_compounds_per_line",
            ),
            "shared_cell_lines_truth_summary_list": _prefer_forward_value(
                forward_truth,
                reverse_truth,
                "truth_summary_shared_cell_type_list",
            ),
            "tahoe_to_other_log_eligible_cell_lines": forward_log.get(
                "precompute_eligible_query_cell_types"
            ),
            "other_to_tahoe_log_eligible_cell_lines": reverse_log.get(
                "precompute_eligible_query_cell_types"
            ),
            "tahoe_to_other_precompute_log": forward_log.get("precompute_log_path", ""),
            "other_to_tahoe_precompute_log": reverse_log.get("precompute_log_path", ""),
            "tahoe_to_other_truth_cell_lines_with_shared_compounds": forward_truth.get(
                "truth_summary_cell_lines_with_shared_compounds"
            ),
            "other_to_tahoe_truth_cell_lines_with_shared_compounds": reverse_truth.get(
                "truth_summary_cell_lines_with_shared_compounds"
            ),
            "tahoe_to_other_truth_shared_line_compound_pairs": forward_truth.get(
                "truth_summary_shared_line_compound_pairs"
            ),
            "other_to_tahoe_truth_shared_line_compound_pairs": reverse_truth.get(
                "truth_summary_shared_line_compound_pairs"
            ),
            "tahoe_to_other_truth_summary": forward_truth.get("truth_summary_path", ""),
            "other_to_tahoe_truth_summary": reverse_truth.get("truth_summary_path", ""),
            "consistency_notes": consistency_notes,
        }
        summary_rows.append(summary_row)

        print(
            f"Saved shared compounds rows={len(overlap_df)} "
            f"{args.tahoe_dataset}<->{dataset_name} -> {output_path} "
            f"(cell_lines={len(shared_cell_types)})",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("other_dataset").reset_index(drop=True)
    summary_output = (
        Path(args.summary_output)
        if args.summary_output
        else output_dir / f"{file_prefix}{args.tahoe_dataset}_overlap_summary.csv"
    )
    summary_df.to_csv(summary_output, index=False)
    print(f"Saved Tahoe overlap summary rows={len(summary_df)} -> {summary_output}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
