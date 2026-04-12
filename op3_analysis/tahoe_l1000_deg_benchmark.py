from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .retrieval.config import DEFAULT_DATASET_PATHS
from .retrieval.metrics import cosine_scores, mrrmse_scores, pearson_scores, rank_center_norm_rows, spearman_scores_precomputed

INVALID_CID_VALUES = frozenset({"", "nan", "none", "<na>"})
LABEL_VALUES = (-1, 0, 1)


@dataclass(frozen=True)
class LabelConfig:
    name: str
    fdr_threshold: float
    min_abs_logfc: float


DEFAULT_LABEL_CONFIGS = (
    LabelConfig(name="deg_fdr005_lfc000", fdr_threshold=0.05, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr005_lfc025", fdr_threshold=0.05, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr005_lfc050", fdr_threshold=0.05, min_abs_logfc=0.50),
    LabelConfig(name="deg_fdr010_lfc025", fdr_threshold=0.10, min_abs_logfc=0.25),
)


def _normalize_is_control(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"})


def normalize_pubchem_cids(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    invalid = text.isna() | text.str.lower().isin(INVALID_CID_VALUES)
    text = text.mask(invalid)
    return text


def dense_layer(adata: ad.AnnData, layer: str) -> np.ndarray:
    matrix = adata.layers[layer]
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def bh_adjust_1d(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        return adjusted

    local = values[valid]
    n_valid = local.size
    order = np.argsort(local)
    ranked = local[order]
    scaled = ranked * (n_valid / np.arange(1, n_valid + 1, dtype=np.float64))
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    restored = np.empty_like(monotone)
    restored[order] = monotone
    adjusted[valid] = restored
    return adjusted


def bh_adjust_rows(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D p-value matrix, got shape {values.shape}")
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        out[idx] = bh_adjust_1d(values[idx])
    return out


def signed_score_matrix(
    logfc: np.ndarray,
    padj: np.ndarray,
    p_floor: float = 1e-300,
) -> np.ndarray:
    clipped = np.clip(np.asarray(padj, dtype=np.float64), p_floor, 1.0)
    score = np.sign(np.asarray(logfc, dtype=np.float64)) * (-np.log10(clipped))
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def deg_label_matrix(
    logfc: np.ndarray,
    padj: np.ndarray,
    fdr_threshold: float,
    min_abs_logfc: float,
) -> np.ndarray:
    local_logfc = np.asarray(logfc, dtype=np.float64)
    local_padj = np.asarray(padj, dtype=np.float64)
    labels = np.zeros(local_logfc.shape, dtype=np.int8)
    significant = np.isfinite(local_padj) & (local_padj <= float(fdr_threshold))
    magnitude = np.abs(local_logfc) >= float(min_abs_logfc)
    active = significant & magnitude & np.isfinite(local_logfc)
    labels[active & (local_logfc > 0)] = 1
    labels[active & (local_logfc < 0)] = -1
    return labels


def predict_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.zeros(values.shape, dtype=np.int8)
    labels[values >= float(threshold)] = 1
    labels[values <= -float(threshold)] = -1
    return labels


def add_shared_padj_and_label_layers(
    adata: ad.AnnData,
    label_configs: Iterable[LabelConfig] = DEFAULT_LABEL_CONFIGS,
    pvalue_layer: str = "P.Value",
    logfc_layer: str = "logFC",
    padj_layer_name: str = "adj.P.Value.shared",
    signed_score_layer_name: str = "signed_score.shared",
) -> ad.AnnData:
    configs = tuple(label_configs)
    pvalues = dense_layer(adata, pvalue_layer)
    adata.layers[padj_layer_name] = bh_adjust_rows(pvalues).astype(np.float32)
    logfc = dense_layer(adata, logfc_layer)
    padj = dense_layer(adata, padj_layer_name)
    adata.layers[signed_score_layer_name] = signed_score_matrix(logfc=logfc, padj=padj)
    for config in configs:
        adata.layers[config.name] = deg_label_matrix(
            logfc=logfc,
            padj=padj,
            fdr_threshold=config.fdr_threshold,
            min_abs_logfc=config.min_abs_logfc,
        )
    adata.uns["label_configs"] = {
        "name": np.asarray([config.name for config in configs], dtype=object),
        "fdr_threshold": np.asarray([config.fdr_threshold for config in configs], dtype=np.float32),
        "min_abs_logfc": np.asarray([config.min_abs_logfc for config in configs], dtype=np.float32),
    }
    return adata


def _subset_from_h5ad(
    path: Path,
    row_idx: np.ndarray,
    var_names: np.ndarray,
) -> ad.AnnData:
    local_row_idx = np.asarray(row_idx, dtype=np.int64)
    if local_row_idx.size == 0:
        raise ValueError(f"Cannot subset empty row selection from {path}")

    backed = ad.read_h5ad(path, backed="r")
    try:
        var_idx = backed.var_names.get_indexer(var_names)
        if np.any(var_idx < 0):
            missing = [str(var_names[i]) for i in np.flatnonzero(var_idx < 0)[:10]]
            raise KeyError(f"{path} is missing requested genes, sample={missing}")
        view = backed[local_row_idx, var_idx]
        if hasattr(view, "to_memory"):
            subset = view.to_memory()
        else:
            subset = view.copy()
    except Exception:
        if getattr(backed, "isbacked", False):
            backed.file.close()
        full = ad.read_h5ad(path)
        var_idx = full.var_names.get_indexer(var_names)
        subset = full[local_row_idx, var_idx].copy()
        return subset
    finally:
        if getattr(backed, "isbacked", False):
            backed.file.close()
    return subset


def discover_filtered_l1000_paths(filtered_dir: Path | str) -> dict[str, Path]:
    root = Path(filtered_dir)
    paths = {path.stem: path.resolve() for path in sorted(root.glob("*.h5ad"))}
    if not paths:
        raise FileNotFoundError(f"No filtered L1000 .h5ad files found under {root}")
    return paths


def donor_compound_set(l1000_paths: Mapping[str, Path]) -> set[str]:
    compounds: set[str] = set()
    for path in l1000_paths.values():
        adata = ad.read_h5ad(path, backed="r")
        try:
            compounds.update(
                normalize_pubchem_cids(adata.obs["pubchem_cid"]).dropna().astype(str).tolist()
            )
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
    return compounds


def donor_cell_types(l1000_paths: Mapping[str, Path]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, path in l1000_paths.items():
        adata = ad.read_h5ad(path, backed="r")
        try:
            out[name] = sorted(pd.Index(adata.obs["cell_type"].astype(str)).unique().tolist())
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
    return out


def discover_tahoe_overlap_cell_types(
    l1000_paths: Mapping[str, Path],
    tahoe_dir: Path | str = DEFAULT_DATASET_PATHS["tahoe"],
) -> list[str]:
    tahoe_root = Path(tahoe_dir)
    tahoe_cell_types = {
        path.name.replace("_de.h5ad", "") for path in tahoe_root.glob("*_de.h5ad")
    }
    donors = donor_cell_types(l1000_paths)
    overlap = set()
    for cell_types in donors.values():
        overlap |= set(cell_types) & tahoe_cell_types
    if not overlap:
        raise RuntimeError("No overlapping Tahoe/L1000 cell types were found.")
    return sorted(overlap)


def common_gene_set(
    l1000_paths: Mapping[str, Path],
    tahoe_cell_types: Iterable[str],
    tahoe_dir: Path | str = DEFAULT_DATASET_PATHS["tahoe"],
) -> list[str]:
    shared: np.ndarray | None = None
    for path in l1000_paths.values():
        adata = ad.read_h5ad(path, backed="r")
        try:
            genes = adata.var_names.to_numpy(dtype=str)
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
        shared = genes if shared is None else np.intersect1d(shared, genes, assume_unique=False)

    tahoe_root = Path(tahoe_dir)
    for cell_type in tahoe_cell_types:
        path = tahoe_root / f"{cell_type}_de.h5ad"
        adata = ad.read_h5ad(path, backed="r")
        try:
            genes = adata.var_names.to_numpy(dtype=str)
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()
        shared = genes if shared is None else np.intersect1d(shared, genes, assume_unique=False)

    if shared is None or shared.size == 0:
        raise RuntimeError("No common genes found across Tahoe and filtered L1000 datasets.")
    return sorted(shared.tolist())


def _make_obs_names(
    dataset_name: str,
    cell_types: pd.Series,
    original_obs_names: pd.Index,
) -> pd.Index:
    return pd.Index(
        [
            f"{dataset_name}::{cell_type}::{idx}::{obs_name}"
            for idx, (cell_type, obs_name) in enumerate(zip(cell_types.astype(str), original_obs_names.astype(str)))
        ],
        dtype=object,
    )


def prepare_filtered_l1000_adata(
    path: Path | str,
    dataset_name: str,
    common_genes: Iterable[str],
    label_configs: Iterable[LabelConfig] = DEFAULT_LABEL_CONFIGS,
) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    genes = np.asarray(list(common_genes), dtype=object)
    var_idx = adata.var_names.get_indexer(genes)
    if np.any(var_idx < 0):
        raise KeyError(f"{path} is missing genes needed for the common benchmark space.")
    subset = adata[:, var_idx].copy()
    subset.obs = subset.obs.copy()
    subset.obs["dataset_name"] = dataset_name
    subset.obs["original_obs_name"] = subset.obs_names.astype(str)
    subset.obs_names = _make_obs_names(
        dataset_name=dataset_name,
        cell_types=subset.obs["cell_type"],
        original_obs_names=pd.Index(subset.obs["original_obs_name"].astype(str)),
    )
    return add_shared_padj_and_label_layers(subset, label_configs=label_configs)


def prepare_tahoe_overlap_adata(
    l1000_paths: Mapping[str, Path],
    output_cell_types: Iterable[str] | None = None,
    tahoe_dir: Path | str = DEFAULT_DATASET_PATHS["tahoe"],
    label_configs: Iterable[LabelConfig] = DEFAULT_LABEL_CONFIGS,
) -> ad.AnnData:
    tahoe_root = Path(tahoe_dir)
    donor_cids = donor_compound_set(l1000_paths)
    tahoe_cell_types = (
        list(output_cell_types)
        if output_cell_types is not None
        else discover_tahoe_overlap_cell_types(l1000_paths, tahoe_root)
    )
    genes = np.asarray(common_gene_set(l1000_paths, tahoe_cell_types, tahoe_root), dtype=object)

    blocks: list[ad.AnnData] = []
    for cell_type in tahoe_cell_types:
        path = tahoe_root / f"{cell_type}_de.h5ad"
        backed = ad.read_h5ad(path, backed="r")
        try:
            obs = backed.obs.copy()
        finally:
            if getattr(backed, "isbacked", False):
                backed.file.close()

        cids = normalize_pubchem_cids(obs["pubchem_cid"])
        keep = cids.isin(donor_cids)
        if "pert_type" in obs.columns:
            pert_type = obs["pert_type"].astype("string").str.strip().str.lower()
            compound_mask = pert_type.eq("compound")
            if bool(compound_mask.any()):
                keep &= compound_mask
        if "is_control" in obs.columns:
            keep &= ~_normalize_is_control(obs["is_control"])
        row_idx = np.flatnonzero(keep.to_numpy(dtype=bool))
        if row_idx.size == 0:
            continue

        block = _subset_from_h5ad(path=path, row_idx=row_idx, var_names=genes)
        block.obs = block.obs.copy()
        block.obs["dataset_name"] = "tahoe"
        block.obs["original_obs_name"] = block.obs_names.astype(str)
        block.obs_names = _make_obs_names(
            dataset_name="tahoe",
            cell_types=block.obs["cell_type"],
            original_obs_names=pd.Index(block.obs["original_obs_name"].astype(str)),
        )
        add_shared_padj_and_label_layers(block, label_configs=label_configs)
        blocks.append(block)

    if not blocks:
        raise RuntimeError("No Tahoe rows remained after restricting to the filtered L1000 compound set.")

    return ad.concat(blocks, axis=0, join="inner", merge="same", uns_merge="same")


def responsive_gene_table(
    dataset_adatas: Mapping[str, ad.AnnData],
    label_layer: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_name, adata in dataset_adatas.items():
        labels = np.asarray(adata.layers[label_layer], dtype=np.int8)
        frame = pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "gene_id": adata.var_names.to_numpy(dtype=str),
                "n_up": (labels > 0).sum(axis=0),
                "n_down": (labels < 0).sum(axis=0),
                "n_nonzero": (labels != 0).sum(axis=0),
            }
        )
        frame["responsive_fraction"] = frame["n_nonzero"] / float(adata.n_obs)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out


def _dose_distance(query_dose: float, donor_dose: float) -> float:
    if np.isfinite(query_dose) and np.isfinite(donor_dose) and query_dose > 0.0 and donor_dose > 0.0:
        return float(abs(np.log(query_dose) - np.log(donor_dose)))
    if np.isfinite(query_dose) and np.isfinite(donor_dose):
        return float(abs(query_dose - donor_dose))
    return float("inf")


def build_same_compound_candidate_table(
    tahoe_adata: ad.AnnData,
    donor_adatas: Mapping[str, ad.AnnData],
) -> pd.DataFrame:
    tahoe_obs = tahoe_adata.obs.copy().reset_index().rename(columns={"index": "tahoe_obs_name"})
    tahoe_obs["query_row"] = np.arange(tahoe_adata.n_obs, dtype=np.int64)
    tahoe_obs["pubchem_cid"] = normalize_pubchem_cids(tahoe_obs["pubchem_cid"])
    tahoe_obs = tahoe_obs.loc[tahoe_obs["pubchem_cid"].notna()].copy()
    tahoe_obs["pubchem_cid"] = tahoe_obs["pubchem_cid"].astype(str)

    donor_frames: list[pd.DataFrame] = []
    for dataset_name, adata in donor_adatas.items():
        obs = adata.obs.copy().reset_index().rename(columns={"index": "donor_obs_name"})
        obs["db_dataset"] = dataset_name
        obs["donor_row"] = np.arange(adata.n_obs, dtype=np.int64)
        obs["pubchem_cid"] = normalize_pubchem_cids(obs["pubchem_cid"])
        obs = obs.loc[obs["pubchem_cid"].notna()].copy()
        obs["pubchem_cid"] = obs["pubchem_cid"].astype(str)
        donor_frames.append(obs)
    donor_obs = pd.concat(donor_frames, ignore_index=True)

    merged = tahoe_obs.merge(
        donor_obs,
        on="pubchem_cid",
        how="inner",
        suffixes=("_query", "_donor"),
    )
    if merged.empty:
        return merged

    merged["query_cell_type"] = merged["cell_type_query"].astype(str)
    merged["donor_cell_type"] = merged["cell_type_donor"].astype(str)
    merged["time_distance"] = np.abs(
        pd.to_numeric(merged["pert_time_h_query"], errors="coerce")
        - pd.to_numeric(merged["pert_time_h_donor"], errors="coerce")
    )
    merged["dose_distance"] = [
        _dose_distance(float(q), float(d))
        for q, d in zip(
            pd.to_numeric(merged["pert_dose_uM_query"], errors="coerce"),
            pd.to_numeric(merged["pert_dose_uM_donor"], errors="coerce"),
        )
    ]
    merged["match_distance"] = merged["time_distance"].fillna(np.inf) + merged["dose_distance"]
    keep_columns = [
        "query_row",
        "tahoe_obs_name",
        "query_cell_type",
        "pubchem_cid",
        "pert_time_h_query",
        "pert_dose_uM_query",
        "db_dataset",
        "donor_row",
        "donor_obs_name",
        "donor_cell_type",
        "pert_time_h_donor",
        "pert_dose_uM_donor",
        "time_distance",
        "dose_distance",
        "match_distance",
    ]
    return merged[keep_columns].sort_values(
        ["query_row", "db_dataset", "donor_cell_type", "match_distance", "donor_row"],
        ignore_index=True,
    )


def pick_best_candidates_by_cell_type(candidate_table: pd.DataFrame) -> pd.DataFrame:
    if candidate_table.empty:
        return candidate_table.copy()
    order = candidate_table.sort_values(
        ["query_row", "db_dataset", "donor_cell_type", "match_distance", "donor_row"],
        ignore_index=True,
    )
    return order.groupby(["query_row", "db_dataset", "donor_cell_type"], as_index=False).first()


def macro_f1_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    local_true = np.asarray(y_true, dtype=np.int8).ravel()
    local_pred = np.asarray(y_pred, dtype=np.int8).ravel()
    scores = []
    for label in LABEL_VALUES:
        true_pos = np.sum((local_true == label) & (local_pred == label))
        false_pos = np.sum((local_true != label) & (local_pred == label))
        false_neg = np.sum((local_true == label) & (local_pred != label))
        denom = (2 * true_pos) + false_pos + false_neg
        scores.append(0.0 if denom == 0 else (2.0 * true_pos) / denom)
    return float(np.mean(scores))


def macro_f1_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    local_true = np.asarray(y_true, dtype=np.int8)
    local_pred = np.asarray(y_pred, dtype=np.int8)
    return np.asarray(
        [macro_f1_flat(local_true[idx], local_pred[idx]) for idx in range(local_true.shape[0])],
        dtype=np.float64,
    )


def signed_jaccard_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    local_true = np.asarray(y_true, dtype=np.int8)
    local_pred = np.asarray(y_pred, dtype=np.int8)
    out = np.zeros(local_true.shape[0], dtype=np.float64)
    for idx in range(local_true.shape[0]):
        row_scores = []
        for label in (-1, 1):
            true_mask = local_true[idx] == label
            pred_mask = local_pred[idx] == label
            union = int(np.sum(true_mask | pred_mask))
            row_scores.append(0.0 if union == 0 else int(np.sum(true_mask & pred_mask)) / union)
        out[idx] = float(np.mean(row_scores))
    return out


def paired_row_metric(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    metric: str,
) -> np.ndarray:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Expected paired matrices of the same shape, got {truth.shape} vs {pred.shape}")

    out = np.full(truth.shape[0], np.nan, dtype=np.float64)
    if metric == "pearson":
        for idx in range(truth.shape[0]):
            out[idx] = pearson_scores(truth[idx], pred[idx : idx + 1])[0]
        return out
    if metric == "cosine":
        for idx in range(truth.shape[0]):
            out[idx] = cosine_scores(truth[idx], pred[idx : idx + 1])[0]
        return out
    if metric == "mrrmse":
        for idx in range(truth.shape[0]):
            out[idx] = mrrmse_scores(truth[idx], pred[idx : idx + 1])[0]
        return out
    if metric == "spearman":
        truth_ranked, truth_norms, _ = rank_center_norm_rows(truth.astype(np.float32, copy=False))
        pred_ranked, pred_norms, _ = rank_center_norm_rows(pred.astype(np.float32, copy=False))
        for idx in range(truth.shape[0]):
            out[idx] = spearman_scores_precomputed(
                truth_ranked[idx],
                float(truth_norms[idx]),
                pred_ranked[idx : idx + 1],
                pred_norms[idx : idx + 1],
            )[0]
        return out
    raise ValueError(f"Unknown paired metric: {metric}")


def continuous_topk_signed_overlap(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Expected paired matrices of the same shape, got {truth.shape} vs {pred.shape}")

    out = np.zeros(truth.shape[0], dtype=np.float64)
    for idx in range(truth.shape[0]):
        local_k = int(min(k, truth.shape[1]))
        if local_k < 1:
            out[idx] = np.nan
            continue
        up_true = set(np.argpartition(-truth[idx], local_k - 1)[:local_k].tolist())
        up_pred = set(np.argpartition(-pred[idx], local_k - 1)[:local_k].tolist())
        down_true = set(np.argpartition(truth[idx], local_k - 1)[:local_k].tolist())
        down_pred = set(np.argpartition(pred[idx], local_k - 1)[:local_k].tolist())
        out[idx] = 0.5 * (
            len(up_true & up_pred) / local_k + len(down_true & down_pred) / local_k
        )
    return out


def summarize_continuous_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    top_k: int = 50,
) -> dict[str, float]:
    pearson = paired_row_metric(x_true, x_pred, metric="pearson")
    spearman = paired_row_metric(x_true, x_pred, metric="spearman")
    cosine = paired_row_metric(x_true, x_pred, metric="cosine")
    mrrmse = paired_row_metric(x_true, x_pred, metric="mrrmse")
    overlap = continuous_topk_signed_overlap(x_true, x_pred, k=top_k)
    return {
        "pearson_mean": float(np.nanmean(pearson)),
        "pearson_median": float(np.nanmedian(pearson)),
        "spearman_mean": float(np.nanmean(spearman)),
        "spearman_median": float(np.nanmedian(spearman)),
        "cosine_mean": float(np.nanmean(cosine)),
        "cosine_median": float(np.nanmedian(cosine)),
        "mrrmse_mean": float(np.nanmean(mrrmse)),
        "mrrmse_median": float(np.nanmedian(mrrmse)),
        "signed_overlap_mean": float(np.nanmean(overlap)),
        "signed_overlap_median": float(np.nanmedian(overlap)),
        "n_samples": int(np.asarray(x_true).shape[0]),
        "n_genes": int(np.asarray(x_true).shape[1]),
    }


def train_gene_majority_labels(y_train: np.ndarray) -> np.ndarray:
    local = np.asarray(y_train, dtype=np.int8)
    up = np.sum(local > 0, axis=0)
    down = np.sum(local < 0, axis=0)
    zero = np.sum(local == 0, axis=0)
    labels = np.zeros(local.shape[1], dtype=np.int8)
    labels[(up > zero) & (up >= down)] = 1
    labels[(down > zero) & (down > up)] = -1
    return labels


def tile_gene_labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    local = np.asarray(labels, dtype=np.int8)
    return np.broadcast_to(local, (int(n_rows), local.size)).copy()


def grouped_compound_folds(
    compounds: Iterable[str],
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    compound_array = np.asarray(list(compounds), dtype=object)
    unique = np.unique(compound_array)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    folds = [shuffled[idx::n_splits] for idx in range(n_splits)]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_compounds in folds:
        test_mask = np.isin(compound_array, fold_compounds)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        out.append((train_idx, test_idx))
    return out


def estimate_cellline_transfer_scores(
    candidate_table: pd.DataFrame,
    tahoe_labels: np.ndarray,
    donor_labels_by_dataset: Mapping[str, np.ndarray],
    train_query_rows: np.ndarray,
) -> pd.DataFrame:
    if candidate_table.empty or len(train_query_rows) == 0:
        return pd.DataFrame(
            columns=["query_cell_type", "db_dataset", "donor_cell_type", "n_queries", "median_macro_f1"]
        )

    train_candidates = candidate_table.loc[candidate_table["query_row"].isin(train_query_rows)].copy()
    if train_candidates.empty:
        return pd.DataFrame(
            columns=["query_cell_type", "db_dataset", "donor_cell_type", "n_queries", "median_macro_f1"]
        )
    best = pick_best_candidates_by_cell_type(train_candidates)

    rows: list[dict[str, object]] = []
    for item in best.itertuples(index=False):
        truth = tahoe_labels[int(item.query_row)]
        donor = donor_labels_by_dataset[str(item.db_dataset)][int(item.donor_row)]
        rows.append(
            {
                "query_cell_type": str(item.query_cell_type),
                "db_dataset": str(item.db_dataset),
                "donor_cell_type": str(item.donor_cell_type),
                "query_row": int(item.query_row),
                "macro_f1": macro_f1_flat(truth, donor),
            }
        )
    scored = pd.DataFrame(rows)
    return (
        scored.groupby(["query_cell_type", "db_dataset", "donor_cell_type"], as_index=False)
        .agg(
            n_queries=("query_row", "nunique"),
            median_macro_f1=("macro_f1", "median"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(
            ["query_cell_type", "median_macro_f1", "mean_macro_f1", "db_dataset", "donor_cell_type"],
            ascending=[True, False, False, True, True],
            ignore_index=True,
        )
    )


def estimate_cellline_transfer_scores_continuous(
    candidate_table: pd.DataFrame,
    tahoe_matrix: np.ndarray,
    donor_matrix_by_dataset: Mapping[str, np.ndarray],
    train_query_rows: np.ndarray,
    metric: str = "pearson",
) -> pd.DataFrame:
    if candidate_table.empty or len(train_query_rows) == 0:
        return pd.DataFrame(
            columns=["query_cell_type", "db_dataset", "donor_cell_type", "n_queries", f"median_{metric}"]
        )

    train_candidates = candidate_table.loc[candidate_table["query_row"].isin(train_query_rows)].copy()
    if train_candidates.empty:
        return pd.DataFrame(
            columns=["query_cell_type", "db_dataset", "donor_cell_type", "n_queries", f"median_{metric}"]
        )
    best = pick_best_candidates_by_cell_type(train_candidates)

    rows: list[dict[str, object]] = []
    for item in best.itertuples(index=False):
        truth = tahoe_matrix[int(item.query_row)][None, :]
        donor = donor_matrix_by_dataset[str(item.db_dataset)][int(item.donor_row)][None, :]
        rows.append(
            {
                "query_cell_type": str(item.query_cell_type),
                "db_dataset": str(item.db_dataset),
                "donor_cell_type": str(item.donor_cell_type),
                "query_row": int(item.query_row),
                metric: float(paired_row_metric(truth, donor, metric=metric)[0]),
            }
        )
    scored = pd.DataFrame(rows)
    metric_col = metric
    return (
        scored.groupby(["query_cell_type", "db_dataset", "donor_cell_type"], as_index=False)
        .agg(
            n_queries=("query_row", "nunique"),
            **{f"median_{metric_col}": (metric_col, "median"), f"mean_{metric_col}": (metric_col, "mean")},
        )
        .sort_values(
            ["query_cell_type", f"median_{metric_col}", f"mean_{metric_col}", "db_dataset", "donor_cell_type"],
            ascending=[True, False, False, True, True],
            ignore_index=True,
        )
    )


def build_prior_score_vector(y_train: np.ndarray) -> np.ndarray:
    local = np.asarray(y_train, dtype=np.int8)
    return local.mean(axis=0, dtype=np.float64).astype(np.float32)


def _rank_candidates_for_query(
    candidates: pd.DataFrame,
    query_cell_type: str,
    transfer_scores: pd.DataFrame | None,
    history_column: str = "median_macro_f1",
) -> pd.DataFrame:
    ranked = candidates.copy()
    ranked["history_score"] = 0.0
    if transfer_scores is not None and not transfer_scores.empty:
        merged = ranked.merge(
            transfer_scores[["query_cell_type", "db_dataset", "donor_cell_type", history_column]],
            on=["query_cell_type", "db_dataset", "donor_cell_type"],
            how="left",
        )
        ranked["history_score"] = merged[history_column].fillna(0.0).to_numpy(dtype=float)
    ranked["rank_score"] = ranked["history_score"] - 0.05 * ranked["match_distance"].astype(float)
    return ranked.sort_values(
        ["rank_score", "history_score", "match_distance", "db_dataset", "donor_row"],
        ascending=[False, False, True, True, True],
        ignore_index=True,
    )


def predict_donor_scores(
    query_rows: np.ndarray,
    tahoe_obs: pd.DataFrame,
    candidate_table: pd.DataFrame | Mapping[int, pd.DataFrame],
    donor_score_by_dataset: Mapping[str, np.ndarray],
    method: str,
    transfer_scores: pd.DataFrame | None = None,
    history_column: str = "median_macro_f1",
    top_k: int = 5,
    prior_score_vector: np.ndarray | None = None,
) -> np.ndarray:
    if prior_score_vector is None and method == "majority_prior":
        raise ValueError("majority_prior requires prior_score_vector")

    n_genes = (
        int(prior_score_vector.size)
        if prior_score_vector is not None
        else next(iter(donor_score_by_dataset.values())).shape[1]
    )
    predictions = np.zeros((len(query_rows), n_genes), dtype=np.float32)

    for out_idx, query_row in enumerate(query_rows):
        if isinstance(candidate_table, Mapping):
            local = candidate_table.get(int(query_row))
            local = pd.DataFrame() if local is None else local.copy()
        else:
            local = candidate_table.loc[candidate_table["query_row"] == int(query_row)].copy()
        if local.empty:
            if prior_score_vector is not None:
                predictions[out_idx] = prior_score_vector
            continue

        query_cell_type = str(tahoe_obs.iloc[int(query_row)]["cell_type"])
        ranked = _rank_candidates_for_query(
            local,
            query_cell_type,
            transfer_scores,
            history_column=history_column,
        )

        if method == "top1":
            chosen = ranked.head(1)
            dataset_name = str(chosen.iloc[0]["db_dataset"])
            predictions[out_idx] = donor_score_by_dataset[dataset_name][int(chosen.iloc[0]["donor_row"])]
            continue

        if method == "topk_then_prior":
            chosen = ranked.head(int(top_k))
            weights = np.maximum(chosen["rank_score"].to_numpy(dtype=np.float64), 0.0)
            if not np.any(weights > 0):
                weights = np.ones(chosen.shape[0], dtype=np.float64)
            scores = np.vstack(
                [
                    donor_score_by_dataset[str(row.db_dataset)][int(row.donor_row)]
                    for row in chosen.itertuples(index=False)
                ]
            )
            donor_score = np.average(scores, axis=0, weights=weights)
            if prior_score_vector is not None:
                donor_score = donor_score + prior_score_vector
            predictions[out_idx] = donor_score
            continue

        if method.startswith("topk"):
            chosen = ranked.head(int(top_k))
            weights = np.maximum(chosen["rank_score"].to_numpy(dtype=np.float64), 0.0)
            if not np.any(weights > 0):
                weights = np.ones(chosen.shape[0], dtype=np.float64)
            scores = []
            for row in chosen.itertuples(index=False):
                scores.append(donor_score_by_dataset[str(row.db_dataset)][int(row.donor_row)])
            stacked = np.vstack(scores)
            predictions[out_idx] = np.average(stacked, axis=0, weights=weights)
            continue

        if method == "best_cellline_mean":
            grouped = (
                ranked.groupby(["db_dataset", "donor_cell_type"], as_index=False)
                .agg(best_rank_score=("rank_score", "max"))
                .sort_values(
                    ["best_rank_score", "db_dataset", "donor_cell_type"],
                    ascending=[False, True, True],
                    ignore_index=True,
                )
            )
            best_dataset = str(grouped.iloc[0]["db_dataset"])
            best_cell_type = str(grouped.iloc[0]["donor_cell_type"])
            chosen = ranked.loc[
                (ranked["db_dataset"] == best_dataset) & (ranked["donor_cell_type"] == best_cell_type)
            ]
            scores = np.vstack(
                [donor_score_by_dataset[best_dataset][int(row.donor_row)] for row in chosen.itertuples(index=False)]
            )
            predictions[out_idx] = scores.mean(axis=0)
            continue

        if method == "all_donor_mean":
            scores = np.vstack(
                [
                    donor_score_by_dataset[str(row.db_dataset)][int(row.donor_row)]
                    for row in ranked.itertuples(index=False)
                ]
            )
            predictions[out_idx] = scores.mean(axis=0)
            continue

        if method == "majority_prior":
            predictions[out_idx] = prior_score_vector
            continue

        raise ValueError(f"Unknown donor prediction method: {method}")

    return predictions


def summarize_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    per_sample_macro = macro_f1_per_sample(y_true, y_pred)
    per_sample_jaccard = signed_jaccard_per_sample(y_true, y_pred)
    return {
        "macro_f1_global": macro_f1_flat(y_true, y_pred),
        "macro_f1_mean_per_sample": float(np.mean(per_sample_macro)),
        "macro_f1_median_per_sample": float(np.median(per_sample_macro)),
        "signed_jaccard_mean_per_sample": float(np.mean(per_sample_jaccard)),
        "signed_jaccard_median_per_sample": float(np.median(per_sample_jaccard)),
        "n_samples": int(y_true.shape[0]),
        "n_genes": int(y_true.shape[1]),
    }


def tune_threshold(
    score_method: str,
    threshold_grid: Iterable[float],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    tahoe_obs: pd.DataFrame,
    y_true: np.ndarray,
    candidate_table: pd.DataFrame,
    donor_score_by_dataset: Mapping[str, np.ndarray],
    transfer_scores_by_inner_fold: list[pd.DataFrame],
    history_column: str = "median_macro_f1",
    top_k: int = 5,
) -> float:
    candidates = list(threshold_grid)
    if score_method == "majority_prior":
        return 0.5

    if not inner_splits:
        return float(candidates[0])

    results = []
    for threshold in candidates:
        scores = []
        for split_idx, (_, val_idx) in enumerate(inner_splits):
            inner_transfer = transfer_scores_by_inner_fold[split_idx]
            prior_score = build_prior_score_vector(y_true[inner_splits[split_idx][0]])
            pred_scores = predict_donor_scores(
                query_rows=val_idx,
                tahoe_obs=tahoe_obs,
                candidate_table=candidate_table,
                donor_score_by_dataset=donor_score_by_dataset,
                method=score_method,
                transfer_scores=inner_transfer,
                history_column=history_column,
                top_k=top_k,
                prior_score_vector=prior_score,
            )
            pred_labels = predict_from_scores(pred_scores, threshold=threshold)
            if score_method == "topk_then_prior":
                fallback = tile_gene_labels(train_gene_majority_labels(y_true[inner_splits[split_idx][0]]), len(val_idx))
                undecided = pred_labels == 0
                pred_labels = np.where(undecided, fallback, pred_labels)
            scores.append(macro_f1_flat(y_true[val_idx], pred_labels))
        results.append((float(np.mean(scores)), float(threshold)))

    results.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return results[0][1]


def concordant_gene_table(
    tahoe_adata: ad.AnnData,
    donor_adatas: Mapping[str, ad.AnnData],
    label_layer: str,
    min_support: int = 10,
) -> pd.DataFrame:
    candidate_table = build_same_compound_candidate_table(tahoe_adata, donor_adatas)
    same_cell_type = candidate_table.loc[
        candidate_table["query_cell_type"] == candidate_table["donor_cell_type"]
    ].copy()
    if same_cell_type.empty:
        return pd.DataFrame(
            columns=["gene_id", "n_support", "n_sign_agree", "sign_agree_fraction", "n_nonzero_both"]
        )

    best = same_cell_type.sort_values(
        ["query_row", "db_dataset", "match_distance", "donor_row"],
        ignore_index=True,
    ).groupby(["query_row", "db_dataset"], as_index=False).first()

    tahoe_labels = np.asarray(tahoe_adata.layers[label_layer], dtype=np.int8)
    accum_support = np.zeros(tahoe_adata.n_vars, dtype=np.int64)
    accum_agree = np.zeros(tahoe_adata.n_vars, dtype=np.int64)
    accum_nonzero_both = np.zeros(tahoe_adata.n_vars, dtype=np.int64)

    for row in best.itertuples(index=False):
        truth = tahoe_labels[int(row.query_row)]
        donor = np.asarray(donor_adatas[str(row.db_dataset)].layers[label_layer], dtype=np.int8)[
            int(row.donor_row)
        ]
        support = (truth != 0) | (donor != 0)
        nonzero_both = (truth != 0) & (donor != 0)
        accum_support += support.astype(np.int64)
        accum_nonzero_both += nonzero_both.astype(np.int64)
        accum_agree += (support & (truth == donor)).astype(np.int64)

    table = pd.DataFrame(
        {
            "gene_id": tahoe_adata.var_names.to_numpy(dtype=str),
            "n_support": accum_support,
            "n_sign_agree": accum_agree,
            "n_nonzero_both": accum_nonzero_both,
        }
    )
    table["sign_agree_fraction"] = np.divide(
        table["n_sign_agree"],
        table["n_support"],
        out=np.zeros(table.shape[0], dtype=float),
        where=table["n_support"] > 0,
    )
    return table.loc[table["n_support"] >= int(min_support)].sort_values(
        ["sign_agree_fraction", "n_support", "n_nonzero_both"],
        ascending=[False, False, False],
        ignore_index=True,
    )
