"""Source-by-cell-type per-gene population z-score standardization.

W4 asks whether heterogeneous source scales explain weak cross-source agreement.  The
analysis therefore fits one population mean and population standard deviation per gene
within each dataset/cell-type line file, using all eligible non-control grouped-condition
logFC signatures.  Cross-source match labels are never used while fitting.

Statistics are cached independently of notebook outputs so a line is scanned at most once
and the same transform is shared by the signature, DEG, and retrieval notebooks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd


ENGINE_VERSION = 1
DEFAULT_ROW_CHUNK_SIZE = 256
MIN_FINITE_OBSERVATIONS = 2
INVALID_STRING_VALUES = {"", "nan", "none", "<na>"}


@dataclass(frozen=True)
class PopulationGeneStats:
    dataset_name: str
    cell_type: str
    gene_keys: np.ndarray
    finite_counts: np.ndarray
    means: np.ndarray
    population_sds: np.ndarray
    valid_mask: np.ndarray
    population_row_count: int
    fingerprint: str
    cache_path: Path

    def __post_init__(self) -> None:
        gene_keys = np.asarray(self.gene_keys).astype(str)
        finite_counts = np.asarray(self.finite_counts, dtype=np.int64)
        means = np.asarray(self.means, dtype=np.float64)
        population_sds = np.asarray(self.population_sds, dtype=np.float64)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        lengths = {
            len(gene_keys),
            len(finite_counts),
            len(means),
            len(population_sds),
            len(valid_mask),
        }
        if len(lengths) != 1:
            raise ValueError("Population-gene statistic arrays have inconsistent lengths")
        if len(gene_keys) and len(set(gene_keys.tolist())) != len(gene_keys):
            raise ValueError("PopulationGeneStats.gene_keys must be unique")
        expected_valid = (
            (finite_counts >= MIN_FINITE_OBSERVATIONS)
            & np.isfinite(means)
            & np.isfinite(population_sds)
            & (population_sds > 0.0)
        )
        if not np.array_equal(valid_mask, expected_valid):
            raise ValueError("PopulationGeneStats.valid_mask is inconsistent with counts/SDs")
        object.__setattr__(self, "gene_keys", gene_keys)
        object.__setattr__(self, "finite_counts", finite_counts)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "population_sds", population_sds)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "cache_path", Path(self.cache_path))

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_keys))

    @property
    def n_valid_genes(self) -> int:
        return int(self.valid_mask.sum())


def _sanitized_strings(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("").astype(str).str.strip()
    normalized.loc[normalized.str.lower().isin(INVALID_STRING_VALUES)] = ""
    return normalized


def _control_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    normalized = values.astype("string").fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"}).to_numpy(dtype=bool)


def eligible_population_mask(obs: pd.DataFrame, cell_type: str) -> np.ndarray:
    """Match the grouped-condition eligibility used by the cross-source notebooks."""
    required = {"is_control", "pubchem_cid", "pert_time_h", "pert_dose_uM"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"Population source is missing obs columns: {sorted(missing)}")

    pubchem_cid = _sanitized_strings(obs["pubchem_cid"])
    time = pd.to_numeric(obs["pert_time_h"], errors="coerce").to_numpy(dtype=np.float64)
    dose = pd.to_numeric(obs["pert_dose_uM"], errors="coerce").to_numpy(dtype=np.float64)
    eligible = (
        ~_control_mask(obs["is_control"])
        & (pubchem_cid.to_numpy(dtype=str) != "")
        & np.isfinite(time)
        & np.isfinite(dose)
        & (dose > 0.0)
    )
    if "cell_type" in obs.columns:
        observed_cell_type = _sanitized_strings(obs["cell_type"]).to_numpy(dtype=str)
        eligible &= observed_cell_type == str(cell_type)
    return eligible


def unique_gene_index(var: pd.DataFrame, var_names: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return first-occurrence gene keys and positions, matching ``LineSource``."""
    if "symbol" in var.columns:
        gene_keys = _sanitized_strings(var["symbol"]).to_numpy(dtype=str)
    else:
        gene_keys = np.asarray(pd.Index(var_names).astype(str).str.strip(), dtype=str)
    keep = (gene_keys != "") & ~pd.Series(gene_keys).duplicated(keep="first").to_numpy()
    positions = np.flatnonzero(keep).astype(np.int64)
    return gene_keys[positions], positions


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    if value in {"", ".", ".."}:
        raise ValueError(f"Unsafe empty cache component derived from {value!r}")
    return value


def stats_cache_path(cache_root: Path, dataset_name: str, cell_type: str) -> Path:
    return (
        Path(cache_root)
        / _safe_component(dataset_name)
        / f"{_safe_component(cell_type)}.npz"
    )


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".cache.json")


def _source_fingerprint(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    shape: tuple[int, int],
    gene_keys: np.ndarray,
    eligible_mask: np.ndarray,
    layer_name: str,
) -> str:
    stat = source_path.stat()
    gene_hash = hashlib.sha256(
        "\0".join(np.asarray(gene_keys).astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    eligible_hash = hashlib.sha256(
        np.packbits(np.asarray(eligible_mask, dtype=np.uint8)).tobytes()
    ).hexdigest()
    payload = {
        "engine_version": ENGINE_VERSION,
        "dataset_name": str(dataset_name),
        "cell_type": str(cell_type),
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "shape": [int(shape[0]), int(shape[1])],
        "layer_name": str(layer_name),
        "gene_hash": gene_hash,
        "eligible_hash": eligible_hash,
        "ddof": 0,
        "minimum_finite_observations": MIN_FINITE_OBSERVATIONS,
        "eligibility_policy": (
            "non_control+valid_pubchem+finite_time+finite_positive_dose+matching_cell_type"
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _merge_batch_statistics(
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> None:
    """In-place Chan/Welford merge for a finite-aware row batch."""
    finite = np.isfinite(batch)
    batch_counts = finite.sum(axis=0, dtype=np.int64)
    if not np.any(batch_counts):
        return
    batch_sums = np.where(finite, batch, 0.0).sum(axis=0, dtype=np.float64)
    batch_means = np.zeros(batch.shape[1], dtype=np.float64)
    present = batch_counts > 0
    batch_means[present] = batch_sums[present] / batch_counts[present]
    centered = np.where(finite, batch - batch_means[None, :], 0.0)
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)

    previous_counts = counts.copy()
    combined_counts = previous_counts + batch_counts
    both = (previous_counts > 0) & present
    new_only = (previous_counts == 0) & present

    means[new_only] = batch_means[new_only]
    m2[new_only] = batch_m2[new_only]
    if np.any(both):
        delta = batch_means[both] - means[both]
        means[both] += delta * batch_counts[both] / combined_counts[both]
        m2[both] += (
            batch_m2[both]
            + delta
            * delta
            * previous_counts[both]
            * batch_counts[both]
            / combined_counts[both]
        )
    counts[:] = combined_counts


def fit_population_stats(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    cache_path: Path,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
) -> PopulationGeneStats:
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")

    adata = ad.read_h5ad(source_path, backed="r")
    try:
        if layer_name not in adata.layers:
            raise KeyError(f"{source_path} has no {layer_name!r} layer")
        gene_keys, gene_positions = unique_gene_index(adata.var, adata.var_names)
        eligible_mask = eligible_population_mask(adata.obs, cell_type)
        fingerprint = _source_fingerprint(
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            shape=adata.shape,
            gene_keys=gene_keys,
            eligible_mask=eligible_mask,
            layer_name=layer_name,
        )

        n_genes = len(gene_keys)
        counts = np.zeros(n_genes, dtype=np.int64)
        means = np.zeros(n_genes, dtype=np.float64)
        m2 = np.zeros(n_genes, dtype=np.float64)
        for start in range(0, adata.n_obs, row_chunk_size):
            stop = min(start + row_chunk_size, adata.n_obs)
            local_eligible = eligible_mask[start:stop]
            if not np.any(local_eligible):
                continue
            raw = np.asarray(
                adata.layers[layer_name][start:stop, :],
                dtype=np.float64,
            )
            batch = raw[local_eligible][:, gene_positions]
            _merge_batch_statistics(counts, means, m2, batch)
    finally:
        adata.file.close()

    population_sds = np.full(len(gene_keys), np.nan, dtype=np.float64)
    observed = counts > 0
    population_sds[observed] = np.sqrt(
        np.maximum(m2[observed], 0.0) / counts[observed]
    )
    means[~observed] = np.nan
    valid_mask = (
        (counts >= MIN_FINITE_OBSERVATIONS)
        & np.isfinite(means)
        & np.isfinite(population_sds)
        & (population_sds > 0.0)
    )
    return PopulationGeneStats(
        dataset_name=str(dataset_name),
        cell_type=str(cell_type),
        gene_keys=gene_keys,
        finite_counts=counts,
        means=means,
        population_sds=population_sds,
        valid_mask=valid_mask,
        population_row_count=int(eligible_mask.sum()),
        fingerprint=fingerprint,
        cache_path=cache_path,
    )


def _write_cache(stats: PopulationGeneStats) -> None:
    cache_path = stats.cache_path
    metadata_path = _metadata_path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}"
    )
    metadata = {
        "engine_version": ENGINE_VERSION,
        "dataset_name": stats.dataset_name,
        "cell_type": stats.cell_type,
        "population_row_count": stats.population_row_count,
        "n_genes": stats.n_genes,
        "n_valid_genes": stats.n_valid_genes,
        "fingerprint": stats.fingerprint,
    }
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                gene_keys=stats.gene_keys.astype(str),
                finite_counts=stats.finite_counts,
                means=stats.means,
                population_sds=stats.population_sds,
                valid_mask=stats.valid_mask,
            )
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_path, cache_path)
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        for temporary in (temporary_path, temporary_metadata_path):
            if temporary.exists():
                temporary.unlink()


def _load_cache(
    *,
    cache_path: Path,
    expected_fingerprint: str,
    dataset_name: str,
    cell_type: str,
) -> Optional[PopulationGeneStats]:
    metadata_path = _metadata_path(cache_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if metadata.get("fingerprint") != expected_fingerprint or not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as values:
            stats = PopulationGeneStats(
                dataset_name=str(dataset_name),
                cell_type=str(cell_type),
                gene_keys=values["gene_keys"].astype(str),
                finite_counts=values["finite_counts"],
                means=values["means"],
                population_sds=values["population_sds"],
                valid_mask=values["valid_mask"],
                population_row_count=int(metadata["population_row_count"]),
                fingerprint=str(expected_fingerprint),
                cache_path=cache_path,
            )
    except (KeyError, OSError, ValueError):
        return None
    return stats


def load_or_fit_population_stats(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    cache_root: Path,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
    force: bool = False,
    verbose: bool = True,
) -> PopulationGeneStats:
    """Load a compatible source-context cache or fit and publish it atomically."""
    source_path = Path(source_path)
    cache_path = stats_cache_path(cache_root, dataset_name, cell_type)

    adata = ad.read_h5ad(source_path, backed="r")
    try:
        if layer_name not in adata.layers:
            raise KeyError(f"{source_path} has no {layer_name!r} layer")
        gene_keys, _ = unique_gene_index(adata.var, adata.var_names)
        eligible_mask = eligible_population_mask(adata.obs, cell_type)
        fingerprint = _source_fingerprint(
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            shape=adata.shape,
            gene_keys=gene_keys,
            eligible_mask=eligible_mask,
            layer_name=layer_name,
        )
    finally:
        adata.file.close()

    if not force:
        cached = _load_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
            dataset_name=dataset_name,
            cell_type=cell_type,
        )
        if cached is not None:
            if verbose:
                print(
                    f"[w4_stats] reloaded {dataset_name}/{cell_type}: "
                    f"{cached.n_valid_genes:,}/{cached.n_genes:,} valid genes"
                )
            return cached

    stats = fit_population_stats(
        source_path=source_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        cache_path=cache_path,
        layer_name=layer_name,
        row_chunk_size=row_chunk_size,
    )
    if stats.fingerprint != fingerprint:
        raise AssertionError("Source changed while population statistics were being fitted")
    _write_cache(stats)
    if verbose:
        print(
            f"[w4_stats] computed {dataset_name}/{cell_type}: "
            f"{stats.population_row_count:,} rows, "
            f"{stats.n_valid_genes:,}/{stats.n_genes:,} valid genes"
        )
    return stats


def _validate_gene_alignment(gene_keys: np.ndarray, stats: PopulationGeneStats) -> None:
    observed = np.asarray(gene_keys).astype(str)
    if observed.shape != stats.gene_keys.shape or not np.array_equal(
        observed, stats.gene_keys
    ):
        raise ValueError(
            f"Gene keys do not match cached W4 statistics for "
            f"{stats.dataset_name}/{stats.cell_type}"
        )


def standardize_vector(
    values: np.ndarray,
    *,
    gene_keys: np.ndarray,
    stats: PopulationGeneStats,
) -> np.ndarray:
    _validate_gene_alignment(gene_keys, stats)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != stats.n_genes:
        raise ValueError("Vector length does not match population-gene statistics")
    standardized = np.full(values.size, np.nan, dtype=np.float64)
    valid = stats.valid_mask & np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized[valid] = (
            values[valid] - stats.means[valid]
        ) / stats.population_sds[valid]
    standardized[~np.isfinite(standardized)] = np.nan
    return standardized


def standardize_matrix(
    values: np.ndarray,
    *,
    gene_keys: np.ndarray,
    stats: PopulationGeneStats,
) -> np.ndarray:
    _validate_gene_alignment(gene_keys, stats)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != stats.n_genes:
        raise ValueError("Matrix shape does not match population-gene statistics")
    standardized = np.full(values.shape, np.nan, dtype=np.float64)
    valid_genes = stats.valid_mask
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized[:, valid_genes] = (
            values[:, valid_genes] - stats.means[None, valid_genes]
        ) / stats.population_sds[None, valid_genes]
    standardized[
        ~np.isfinite(values) | ~np.isfinite(standardized)
    ] = np.nan
    return standardized


def stats_qc_record(stats: PopulationGeneStats) -> dict[str, object]:
    finite_counts = stats.finite_counts[stats.valid_mask]
    return {
        "dataset_name": stats.dataset_name,
        "cell_type": stats.cell_type,
        "population_row_count": stats.population_row_count,
        "n_genes": stats.n_genes,
        "n_valid_genes": stats.n_valid_genes,
        "min_finite_count": int(finite_counts.min()) if finite_counts.size else 0,
        "median_finite_count": (
            float(np.median(finite_counts)) if finite_counts.size else float("nan")
        ),
        "max_finite_count": int(finite_counts.max()) if finite_counts.size else 0,
        "fingerprint": stats.fingerprint,
        "cache_path": str(stats.cache_path),
    }
