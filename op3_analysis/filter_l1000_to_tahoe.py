from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import anndata as ad
import pandas as pd

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from op3_analysis.retrieval.config import DEFAULT_DATASET_PATHS, FALLBACK_DATASET_PATHS
else:
    from .retrieval.config import DEFAULT_DATASET_PATHS, FALLBACK_DATASET_PATHS

INVALID_CID_VALUES = frozenset({"", "nan", "none", "<na>"})

DEFAULT_LEVEL5_DATASET_PATHS: dict[str, Path] = {
    "l1000_phase1_level5": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/l1000_phase1/level5/level5_phase1_not_filtered.h5ad"
    ),
    "l1000_phase2_level5": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/l1000_phase2/level5/level5_phase2_not_filtered.h5ad"
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


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


def _normalize_is_control(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"})


def _dataset_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_dir():
        files = sorted(dataset_path.glob("*_de.h5ad"))
        if not files:
            files = sorted(dataset_path.glob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found under dataset directory: {dataset_path}")
        return files

    if dataset_path.is_file() and dataset_path.suffix == ".h5ad":
        return [dataset_path]

    raise FileNotFoundError(f"Unsupported dataset path: {dataset_path}")


def _resolve_path(name: str, overrides: dict[str, Path]) -> Path:
    if name in overrides:
        path = overrides[name]
        if not path.exists():
            raise FileNotFoundError(f"Dataset path for '{name}' does not exist: {path}")
        return path

    if name in DEFAULT_LEVEL5_DATASET_PATHS:
        path = DEFAULT_LEVEL5_DATASET_PATHS[name]
        if not path.exists():
            raise FileNotFoundError(f"Dataset path for '{name}' does not exist: {path}")
        return path

    path = DEFAULT_DATASET_PATHS[name]
    if path.exists():
        return path

    fallback = FALLBACK_DATASET_PATHS.get(name)
    if fallback is not None and fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Dataset path for '{name}' does not exist: {path}"
        + (f" (fallback checked: {fallback})" if fallback else "")
    )


def resolve_input_specs(overrides: dict[str, Path]) -> list[DatasetSpec]:
    dataset_names = [
        "l1000_phase1",
        "l1000_phase2",
        "l1000_phase1_level5",
        "l1000_phase2_level5",
    ]
    return [DatasetSpec(name=name, path=_resolve_path(name, overrides)) for name in dataset_names]


def resolve_tahoe_path(overrides: dict[str, Path]) -> Path:
    return _resolve_path("tahoe", overrides)


def _valid_pubchem_cids(obs: pd.DataFrame) -> pd.Series:
    if "pubchem_cid" not in obs.columns:
        raise KeyError(f"Expected obs['pubchem_cid']; available columns: {list(obs.columns)}")

    cids = obs["pubchem_cid"].astype("string").str.strip()
    return cids.notna() & ~cids.str.lower().isin(INVALID_CID_VALUES)


def _base_compound_mask(obs: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=obs.index, dtype=bool)

    if "pert_type" in obs.columns:
        pert_type = obs["pert_type"].astype("string").str.strip().str.lower()
        compound_mask = pert_type.eq("compound")
        if bool(compound_mask.any()):
            mask &= compound_mask

    if "is_control" in obs.columns:
        mask &= ~_normalize_is_control(obs["is_control"])

    mask &= _valid_pubchem_cids(obs)
    return mask


def collect_tahoe_compounds(tahoe_path: Path, verbose: bool = False) -> set[str]:
    compounds: set[str] = set()

    for path in _dataset_files(tahoe_path):
        adata = ad.read_h5ad(path, backed="r")
        try:
            obs = adata.obs.copy()
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()

        mask = _base_compound_mask(obs)
        if not bool(mask.any()):
            continue

        cids = obs.loc[mask, "pubchem_cid"].astype("string").str.strip()
        compounds.update(str(value) for value in cids.dropna().unique().tolist())

    if verbose:
        print(
            f"[filter-l1000-to-tahoe] Tahoe unique compounds={len(compounds)} path={tahoe_path}",
            flush=True,
        )
    return compounds


def _materialize_subset(adata: ad.AnnData, mask: pd.Series) -> ad.AnnData:
    return adata[mask.to_numpy(dtype=bool)].copy()


def _concat_subsets(subsets: list[ad.AnnData]) -> ad.AnnData:
    if not subsets:
        raise ValueError("Expected at least one subset to concatenate")
    if len(subsets) == 1:
        return subsets[0]
    return ad.concat(subsets, axis=0, join="inner", merge="same", uns_merge="same")


def _summarize_source(
    obs: pd.DataFrame,
    tahoe_compounds: set[str],
    min_shared_compounds: int,
) -> pd.DataFrame:
    if "cell_type" not in obs.columns:
        raise KeyError(f"Expected obs['cell_type']; available columns: {list(obs.columns)}")

    cell_types = obs["cell_type"].astype("string").fillna("").astype(str)
    base_mask = _base_compound_mask(obs)
    cids = obs["pubchem_cid"].astype("string").str.strip()
    shared_mask = base_mask & cids.isin(tahoe_compounds)

    source_rows = (
        cell_types.to_frame(name="cell_type")
        .groupby("cell_type", sort=True, dropna=False)
        .size()
        .rename("n_source_rows")
        .astype(int)
    )

    eligible_rows = (
        cell_types.loc[base_mask]
        .to_frame(name="cell_type")
        .groupby("cell_type", sort=True, dropna=False)
        .size()
        .rename("n_eligible_rows")
        .astype(int)
    )

    shared_rows = (
        cell_types.loc[shared_mask]
        .to_frame(name="cell_type")
        .groupby("cell_type", sort=True, dropna=False)
        .size()
        .rename("n_shared_rows")
        .astype(int)
    )

    shared_compounds = (
        pd.DataFrame(
            {
                "cell_type": cell_types.loc[shared_mask].to_numpy(dtype=str),
                "pubchem_cid": cids.loc[shared_mask].to_numpy(dtype=str),
            }
        )
        .drop_duplicates()
        .groupby("cell_type", sort=True)
        .size()
        .rename("n_shared_compounds")
        .astype(int)
    )

    summary = pd.concat([source_rows, eligible_rows, shared_rows, shared_compounds], axis=1).fillna(0)
    summary = summary.reset_index()
    numeric_columns = [
        "n_source_rows",
        "n_eligible_rows",
        "n_shared_rows",
        "n_shared_compounds",
    ]
    for column in numeric_columns:
        summary[column] = summary[column].astype(int)

    summary["kept"] = summary["n_shared_compounds"] >= int(min_shared_compounds)
    return summary


def filter_dataset(
    spec: DatasetSpec,
    output_root: Path,
    tahoe_compounds: set[str],
    min_shared_compounds: int,
    verbose: bool = False,
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    kept_subsets: list[ad.AnnData] = []
    dataset_output_path = output_root / f"{spec.name}.h5ad"

    for source_path in _dataset_files(spec.path):
        adata = ad.read_h5ad(source_path, backed="r")
        subset_masks: list[tuple[str, pd.Series]] = []
        try:
            obs = adata.obs.copy()
            summary = _summarize_source(
                obs=obs,
                tahoe_compounds=tahoe_compounds,
                min_shared_compounds=min_shared_compounds,
            )

            base_mask = _base_compound_mask(obs)
            cids = obs["pubchem_cid"].astype("string").str.strip()
            shared_mask = base_mask & cids.isin(tahoe_compounds)
            cell_types = obs["cell_type"].astype("string").fillna("").astype(str)

            for row in summary.itertuples(index=False):
                cell_type = str(row.cell_type)
                kept = bool(row.kept)
                if kept:
                    subset_masks.append((cell_type, shared_mask & cell_types.eq(cell_type)))

                summary_rows.append(
                    {
                        "dataset_name": spec.name,
                        "source_path": str(source_path),
                        "cell_type": cell_type,
                        "n_source_rows": int(row.n_source_rows),
                        "n_eligible_rows": int(row.n_eligible_rows),
                        "n_shared_rows": int(row.n_shared_rows),
                        "n_shared_compounds": int(row.n_shared_compounds),
                        "kept": kept,
                        "dataset_output_path": str(dataset_output_path) if kept else "",
                    }
                )
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()

        if subset_masks:
            # Backed boolean slicing with `.to_memory()` is unreliable on these large h5ad files.
            # Re-open in normal mode only after we know this source contributes kept rows.
            write_adata = ad.read_h5ad(source_path)
            try:
                for _, subset_mask in subset_masks:
                    kept_subsets.append(_materialize_subset(write_adata, subset_mask))
            finally:
                if getattr(write_adata, "isbacked", False):
                    write_adata.file.close()

    if kept_subsets:
        combined = _concat_subsets(kept_subsets)
        combined.write_h5ad(dataset_output_path)

    result = pd.DataFrame(summary_rows).sort_values(
        ["dataset_name", "kept", "n_shared_compounds", "cell_type"],
        ascending=[True, False, False, True],
        ignore_index=True,
    )

    if verbose:
        kept = int(result["kept"].sum()) if not result.empty else 0
        total = int(result.shape[0])
        print(
            f"[filter-l1000-to-tahoe] dataset={spec.name} kept_cell_types={kept}/{total} output={dataset_output_path}",
            flush=True,
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter L1000 phase1/phase2 DEG and level5 datasets to compounds observed in Tahoe, "
            "writing one filtered .h5ad per dataset after dropping cell lines below the Tahoe-overlap cutoff."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory where filtered dataset .h5ad files will be written.",
    )
    parser.add_argument(
        "--min-shared-compounds",
        type=int,
        default=10,
        help="Minimum number of unique Tahoe-shared compounds required to keep a cell type.",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=/path/to/data",
        help=(
            "Override an input path. Supported names: tahoe, l1000_phase1, l1000_phase2, "
            "l1000_phase1_level5, l1000_phase2_level5."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    return parser


def _ensure_new_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory must be new or empty to avoid stale files: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.min_shared_compounds < 1:
        raise ValueError("--min-shared-compounds must be >= 1")

    overrides = _parse_dataset_overrides(args.dataset_path)
    output_dir = _ensure_new_output_dir(args.output_dir)
    tahoe_path = resolve_tahoe_path(overrides)
    specs = resolve_input_specs(overrides)

    tahoe_compounds = collect_tahoe_compounds(tahoe_path=tahoe_path, verbose=args.verbose)
    if not tahoe_compounds:
        raise RuntimeError(f"No valid Tahoe compounds were found under {tahoe_path}")

    summaries = [
        filter_dataset(
            spec=spec,
            output_root=output_dir,
            tahoe_compounds=tahoe_compounds,
            min_shared_compounds=args.min_shared_compounds,
            verbose=args.verbose,
        )
        for spec in specs
    ]

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = output_dir / "filter_summary.csv"
    summary.to_csv(summary_path, index=False)

    if args.verbose:
        kept_total = int(summary["kept"].sum()) if not summary.empty else 0
        print(
            f"[filter-l1000-to-tahoe] wrote summary={summary_path} kept_cell_types={kept_total}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
