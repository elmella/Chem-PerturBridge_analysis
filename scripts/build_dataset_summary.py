#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import anndata as ad
import pandas as pd

DATA_ROOT = Path(os.environ.get("PERTURB_DATA_ROOT", "data/processed"))
DATASET_ORDER = [
    "cigs_mce",
    "cigs_tcm",
    "dilimap_train_val",
    "gdpx2",
    "l1000_phase1",
    "l1000_phase2",
    "novartis_batch_2500",
    "op3",
    "sciplex",
    "tahoe",
    "vcpi_0001",
    "vcpi_0002",
]
REPO_SEARCH_EXTENSIONS = {
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ipynb",
    ".csv",
    ".tsv",
}
SKIP_REPO_PATH_PARTS = {".git", ".venv", "__pycache__", "results"}
SOURCE_ALIASES = {
    "cigs_mce": {"cigs_mce", "cigs mce", "compounds_mce", "hts2"},
    "cigs_tcm": {"cigs_tcm", "cigs tcm", "compounds_tcm", "himap-seq"},
    "dilimap_train": {"dilimap_train", "dilimap", "dilimap_pubchem_cache"},
    "dilimap_train_val": {"dilimap_train_val", "dilimap", "dilimap_pubchem_cache"},
    "gdpx2": {"gdpx2", "gdpx2_compounds", "gdpx2_pubchem_cache"},
    "l1000_phase1": {"l1000_phase1", "lincs_phase1_level3_epsilon", "l1000"},
    "l1000_phase2": {"l1000_phase2", "lincs_phase2_level3", "l1000"},
    "novartis_batch_1000": {"novartis_batch_1000", "novartis moabox drug-seq", "drug-seq"},
    "novartis_batch_2500": {"novartis_batch_2500", "novartis moabox drug-seq", "drug-seq"},
    "op3": {"op3", "neurips2023 scperturb dge"},
    "sciplex": {"sciplex", "srivatsan20_sciplex3", "sci-plex"},
    "tahoe": {"tahoe", "tahoe100", "parse evercode"},
    "vcpi_0001": {"vcpi_0001", "vcpi-0001", "vcpi ginkgo"},
    "vcpi_0002": {"vcpi_0002", "vcpi-0002", "vcpi ginkgo"},
}
TECHNOLOGY_BY_ASSAY = {
    "L1000 mRNA profiling assay": "LINCS L1000 mRNA profiling assay",
    "DRUG-seq": "DRUG-seq",
    "HTS2": "HTS2",
    "HiMAP-seq": "HiMAP-seq",
    "SMARTSeq bulk RNA-seq": "SMART-seq bulk RNA-seq",
    "10x 3' v3.1": "10x Genomics Chromium Single Cell 3' v3.1 RNA-seq",
    "sci-Plex": "sci-Plex single-cell RNA-seq",
    "Parse Evercode Whole Transcriptome v3": "Parse Biosciences Evercode Whole Transcriptome v3",
}
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")
NUMERIC_SIG_FIGS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize selected perturbation datasets from processed group-level h5ad files."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root directory containing dataset folders (default: {DATA_ROOT}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used when searching for source URLs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help=(
            "Datasets to summarize. Use this to shard work across nodes by passing "
            "different subsets on different runs."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of dataset workers to run in parallel on the current node.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_summary.tsv"),
        help="Output table path. Extension .tsv writes tab-separated output; everything else writes CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log progress to stderr.",
    )
    parser.add_argument(
        "--no-chem-perturbridge-row",
        action="store_true",
        help="Do not append the aggregate overlap row. Useful when sharding by dataset across nodes.",
    )
    return parser.parse_args()


def log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def resolve_processed_h5ad(data_root: Path, dataset_name: str) -> Path:
    group_rep_dir = data_root / dataset_name / "pseudobulk_processed" / "group_rep"
    matches = sorted(group_rep_dir.glob("*.h5ad"))
    if not matches:
        raise FileNotFoundError(f"No .h5ad file found in {group_rep_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one .h5ad file in {group_rep_dir}, found {len(matches)}")
    return matches[0]


def read_obs_column(adata: ad.AnnData, column: str) -> pd.Series:
    return pd.Series(adata.obs[column], copy=False)


def normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.dropna().astype(str).str.strip()
    return normalized[normalized != ""]


def unique_sorted_strings(series: pd.Series) -> list[str]:
    values = normalize_string_series(series).unique().tolist()
    return sorted(values)


def unique_sorted_numeric_strings(series: pd.Series) -> list[str]:
    numeric = pd.to_numeric(series, errors="coerce")
    formatted = {format_numeric(float(value)) for value in numeric.dropna().tolist()}
    return sorted(formatted, key=float)


def format_numeric(value: float) -> str:
    formatted = f"{value:.{NUMERIC_SIG_FIGS}g}"
    return "0" if formatted == "-0" else formatted


def to_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    lowered = series.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def detect_system(obs: pd.DataFrame) -> str:
    context_values = unique_sorted_strings(obs["cell_type"])
    tissues = set(unique_sorted_strings(obs["tissue"])) if "tissue" in obs.columns else set()

    if any(value.startswith("CVCL_") for value in context_values):
        return "cell lines"
    if tissues == {"blood"}:
        return "primary PBMC"
    if context_values and all(value.startswith("CL_") for value in context_values):
        return "primary cells"
    return "cell lines"


def summarize_dataset(
    dataset_name: str,
    data_root: Path,
    repo_root: Path,
    verbose: bool = False,
) -> dict[str, object]:
    h5ad_path = resolve_processed_h5ad(data_root, dataset_name)
    log(f"[{dataset_name}] reading {h5ad_path}", verbose=verbose)
    adata = ad.read_h5ad(h5ad_path, backed="r")

    try:
        obs = adata.obs
        non_control_mask = ~to_boolean_series(read_obs_column(adata, "is_control"))
        perturbagen_series = read_obs_column(adata, "perturbagen")

        assay_values = unique_sorted_strings(read_obs_column(adata, "assay"))
        if len(assay_values) != 1:
            raise RuntimeError(f"[{dataset_name}] expected one assay value, found {assay_values}")
        assay = assay_values[0]

        record = {
            "dataset_name": dataset_name,
            "system": detect_system(obs),
            "technology": TECHNOLOGY_BY_ASSAY.get(assay, assay),
            "n_compounds": int(normalize_string_series(perturbagen_series[non_control_mask]).nunique()),
            "n_contexts": int(normalize_string_series(read_obs_column(adata, "cell_type")).nunique()),
            "n_samples": int(adata.n_obs),
            "timepoints_h": "; ".join(unique_sorted_numeric_strings(read_obs_column(adata, "pert_time_h"))),
            "doses_uM": "; ".join(unique_sorted_numeric_strings(read_obs_column(adata, "pert_dose_uM"))),
            "source_url": find_source_url_in_repo(dataset_name, repo_root),
        }
    finally:
        adata.file.close()

    log(f"[{dataset_name}] done", verbose=verbose)
    return record


def load_bridge_obs(dataset_name: str, data_root: Path, verbose: bool = False) -> pd.DataFrame:
    h5ad_path = resolve_processed_h5ad(data_root, dataset_name)
    log(f"[{dataset_name}] loading bridge metadata from {h5ad_path}", verbose=verbose)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs[
            ["cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM", "is_control"]
        ].copy()
    finally:
        adata.file.close()

    obs = obs.loc[~to_boolean_series(obs["is_control"])].copy()
    obs["dataset_name"] = dataset_name
    obs["cell_type"] = obs["cell_type"].astype(str).str.strip()
    obs["pubchem_cid"] = obs["pubchem_cid"].astype(str).str.strip()
    return obs[["dataset_name", "cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM"]]


def summarize_chem_perturbridge(
    dataset_names: list[str],
    data_root: Path,
    dataset_rows: list[dict[str, object]],
    verbose: bool = False,
) -> dict[str, object]:
    bridge_frames = [load_bridge_obs(name, data_root=data_root, verbose=verbose) for name in dataset_names]
    combined_bridge = pd.concat(bridge_frames, ignore_index=True)
    systems = sorted({str(row["system"]) for row in dataset_rows if str(row["system"]).strip()})

    context_values = normalize_string_series(combined_bridge["cell_type"])
    compound_values = normalize_string_series(combined_bridge["pubchem_cid"])
    valid_compounds = compound_values[~compound_values.str.lower().isin({"nan", "none", "<na>"})]

    return {
        "dataset_name": "Chem-PerturBridge",
        "system": "; ".join(systems),
        "technology": "union across selected datasets",
        "n_compounds": int(valid_compounds.nunique()),
        "n_contexts": int(context_values.nunique()),
        "n_samples": int(len(combined_bridge)),
        "timepoints_h": "; ".join(unique_sorted_numeric_strings(combined_bridge["pert_time_h"])),
        "doses_uM": "; ".join(unique_sorted_numeric_strings(combined_bridge["pert_dose_uM"])),
        "source_url": "NA",
    }


def find_source_url_in_repo(dataset_name: str, repo_root: Path) -> str:
    aliases = {alias.lower() for alias in SOURCE_ALIASES.get(dataset_name, {dataset_name})}
    for path in iter_repo_text_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        if not any(alias in lowered for alias in aliases):
            continue
        urls = URL_PATTERN.findall(text)
        if urls:
            return urls[0]
    return "NA"


def iter_repo_text_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_REPO_PATH_PARTS for part in path.parts):
            continue
        if path.name == "uv.lock":
            continue
        if path.suffix.lower() not in REPO_SEARCH_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > 5_000_000:
                continue
        except OSError:
            continue
        yield path


def build_summary(
    dataset_names: list[str],
    data_root: Path,
    repo_root: Path,
    jobs: int,
    verbose: bool,
    include_chem_perturbridge_row: bool,
) -> pd.DataFrame:
    if jobs <= 1 or len(dataset_names) == 1:
        rows = [
            summarize_dataset(name, data_root=data_root, repo_root=repo_root, verbose=verbose)
            for name in dataset_names
        ]
    else:
        max_workers = min(jobs, len(dataset_names), os.cpu_count() or jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    summarize_dataset,
                    name,
                    data_root,
                    repo_root,
                    False,
                )
                for name in dataset_names
            ]
            rows = [future.result() for future in futures]

    if include_chem_perturbridge_row:
        rows.append(
            summarize_chem_perturbridge(
                dataset_names=dataset_names,
                data_root=data_root,
                dataset_rows=rows,
                verbose=verbose,
            )
        )

    frame = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    return frame.sort_values(
        by="dataset_name",
        key=lambda series: series.map(lambda value: order.get(value, len(order))),
    ).reset_index(drop=True)


def write_summary(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if output_path.suffix.lower() == ".tsv" else ","
    frame.to_csv(output_path, sep=sep, index=False)


def main() -> None:
    args = parse_args()
    summary = build_summary(
        dataset_names=args.datasets,
        data_root=args.data_root,
        repo_root=args.repo_root,
        jobs=args.jobs,
        verbose=args.verbose,
        include_chem_perturbridge_row=not args.no_chem_perturbridge_row,
    )
    write_summary(summary, args.output)


if __name__ == "__main__":
    main()
