from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .tahoe_l1000_benchmark_runner import load_prepared_inputs
from .tahoe_l1000_deg_benchmark import normalize_pubchem_cids, paired_row_metric


DEFAULT_SIGNATURE_LAYERS = ("logFC", "signed_score.shared")
DEFAULT_SIMILARITY_METRICS = ("pearson", "cosine")
DEFAULT_K_VALUES = (1, 3, 5, 10)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _dose_distance(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right) and left > 0.0 and right > 0.0:
        return float(abs(np.log(left) - np.log(right)))
    if np.isfinite(left) and np.isfinite(right):
        return float(abs(left - right))
    return float("inf")


def _vector_correlation(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    local_x = np.asarray(x, dtype=np.float64)
    local_y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(local_x) & np.isfinite(local_y)
    if int(mask.sum()) < 2:
        return float("nan")
    local_x = local_x[mask]
    local_y = local_y[mask]
    if metric == "spearman":
        local_x = pd.Series(local_x).rank(method="average").to_numpy(dtype=np.float64)
        local_y = pd.Series(local_y).rank(method="average").to_numpy(dtype=np.float64)
    elif metric != "pearson":
        raise ValueError(f"Unsupported vector correlation metric: {metric}")
    local_x = local_x - local_x.mean()
    local_y = local_y - local_y.mean()
    denom = float(np.linalg.norm(local_x) * np.linalg.norm(local_y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(local_x, local_y) / denom)


def _row_similarity_matrix(matrix: np.ndarray, metric: str) -> np.ndarray:
    local = np.asarray(matrix, dtype=np.float64)
    if metric == "pearson":
        local = local - local.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"Unsupported similarity metric: {metric}")
    norms = np.linalg.norm(local, axis=1, keepdims=True)
    normalized = np.divide(local, norms, out=np.zeros_like(local), where=norms > 0.0)
    sim = normalized @ normalized.T
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32, copy=False)


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return np.asarray(matrix[idx], dtype=np.float64)


def _topk_neighbors(similarity: np.ndarray, anchor: int, k: int) -> np.ndarray:
    n_items = int(similarity.shape[0])
    if n_items <= 1:
        return np.array([], dtype=np.int64)
    local_k = int(min(k, n_items - 1))
    if local_k < 1:
        return np.array([], dtype=np.int64)
    row = np.asarray(similarity[anchor], dtype=np.float64).copy()
    row[anchor] = -np.inf
    order = np.argsort(row, kind="stable")[::-1]
    return order[:local_k].astype(np.int64, copy=False)


def _directed_knn_edge_jaccard(similarity_left: np.ndarray, similarity_right: np.ndarray, k: int) -> float:
    n_items = int(similarity_left.shape[0])
    if n_items <= 1:
        return float("nan")
    left_edges: set[tuple[int, int]] = set()
    right_edges: set[tuple[int, int]] = set()
    for anchor in range(n_items):
        for neighbor in _topk_neighbors(similarity_left, anchor, k):
            left_edges.add((anchor, int(neighbor)))
        for neighbor in _topk_neighbors(similarity_right, anchor, k):
            right_edges.add((anchor, int(neighbor)))
    union = left_edges | right_edges
    if not union:
        return float("nan")
    return float(len(left_edges & right_edges) / len(union))


def aggregate_exact_contexts(
    adata,
    dataset_name: str,
    signature_layers: tuple[str, ...] = DEFAULT_SIGNATURE_LAYERS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    obs = adata.obs.copy().reset_index().rename(columns={"index": "original_obs_name"})
    obs["row_idx"] = np.arange(adata.n_obs, dtype=np.int64)
    obs["cell_type"] = obs["cell_type"].astype("string")
    obs["pubchem_cid"] = normalize_pubchem_cids(obs["pubchem_cid"]).astype("string")
    obs["pert_time_h"] = _numeric_series(obs["pert_time_h"])
    obs["pert_dose_uM"] = _numeric_series(obs["pert_dose_uM"])
    obs = obs.loc[obs["cell_type"].notna() & obs["pubchem_cid"].notna()].copy()

    group_cols = ["cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM"]
    grouped = (
        obs.groupby(group_cols, dropna=False, sort=True)["row_idx"]
        .apply(list)
        .reset_index(name="row_indices")
        .reset_index(names="context_idx")
    )
    grouped["dataset_name"] = dataset_name
    grouped["n_rows_aggregated"] = grouped["row_indices"].str.len().astype(int)

    aggregated_layers: dict[str, np.ndarray] = {}
    for layer_name in signature_layers:
        layer_matrix = np.asarray(adata.layers[layer_name], dtype=np.float32)
        out = np.empty((grouped.shape[0], adata.n_vars), dtype=np.float32)
        for idx, row_indices in enumerate(grouped["row_indices"]):
            out[idx] = layer_matrix[np.asarray(row_indices, dtype=np.int64)].mean(axis=0, dtype=np.float64)
        aggregated_layers[layer_name] = out
    return grouped, aggregated_layers


def build_closest_context_mapping(
    tahoe_contexts: pd.DataFrame,
    donor_contexts: pd.DataFrame,
    donor_dataset: str,
) -> pd.DataFrame:
    shared_cell_types = sorted(
        set(tahoe_contexts["cell_type"].astype(str)) & set(donor_contexts["cell_type"].astype(str))
    )
    rows: list[dict[str, object]] = []

    for cell_type in shared_cell_types:
        tahoe_cell = tahoe_contexts.loc[tahoe_contexts["cell_type"].astype(str).eq(cell_type)].copy()
        donor_cell = donor_contexts.loc[donor_contexts["cell_type"].astype(str).eq(cell_type)].copy()
        shared_cids = sorted(
            set(tahoe_cell["pubchem_cid"].astype(str)) & set(donor_cell["pubchem_cid"].astype(str))
        )
        for pubchem_cid in shared_cids:
            tahoe_cid = tahoe_cell.loc[tahoe_cell["pubchem_cid"].astype(str).eq(pubchem_cid)].copy()
            donor_cid = donor_cell.loc[donor_cell["pubchem_cid"].astype(str).eq(pubchem_cid)].copy()
            best: dict[str, object] | None = None
            best_key: tuple[float, float, int, int] | None = None
            for tahoe_item in tahoe_cid.itertuples(index=False):
                tahoe_time = float(tahoe_item.pert_time_h)
                tahoe_dose = float(tahoe_item.pert_dose_uM)
                for donor_item in donor_cid.itertuples(index=False):
                    donor_time = float(donor_item.pert_time_h)
                    donor_dose = float(donor_item.pert_dose_uM)
                    time_distance = float(abs(tahoe_time - donor_time))
                    log_dose_distance = _dose_distance(tahoe_dose, donor_dose)
                    key = (
                        time_distance,
                        log_dose_distance,
                        int(tahoe_item.context_idx),
                        int(donor_item.context_idx),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best = {
                            "donor_dataset": donor_dataset,
                            "cell_type": cell_type,
                            "pubchem_cid": pubchem_cid,
                            "tahoe_context_idx": int(tahoe_item.context_idx),
                            "donor_context_idx": int(donor_item.context_idx),
                            "tahoe_time_h": tahoe_time,
                            "donor_time_h": donor_time,
                            "tahoe_dose_uM": tahoe_dose,
                            "donor_dose_uM": donor_dose,
                            "time_distance_h": time_distance,
                            "log_dose_distance": log_dose_distance,
                            "combined_distance": float(time_distance + log_dose_distance),
                            "n_tahoe_candidate_contexts": int(tahoe_cid.shape[0]),
                            "n_donor_candidate_contexts": int(donor_cid.shape[0]),
                        }
            if best is not None:
                rows.append(best)
    return pd.DataFrame(rows).sort_values(
        ["donor_dataset", "cell_type", "pubchem_cid"],
        ignore_index=True,
    )


def summarize_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(
            columns=[
                "donor_dataset",
                "cell_type",
                "n_matched_compounds",
                "mean_time_distance_h",
                "median_time_distance_h",
                "mean_log_dose_distance",
                "median_log_dose_distance",
            ]
        )
    return (
        mapping.groupby(["donor_dataset", "cell_type"], as_index=False)
        .agg(
            n_matched_compounds=("pubchem_cid", "nunique"),
            mean_time_distance_h=("time_distance_h", "mean"),
            median_time_distance_h=("time_distance_h", "median"),
            mean_log_dose_distance=("log_dose_distance", "mean"),
            median_log_dose_distance=("log_dose_distance", "median"),
        )
        .sort_values(["donor_dataset", "n_matched_compounds", "cell_type"], ascending=[True, False, True], ignore_index=True)
    )


def headline_mapping_summary(mapping_summary: pd.DataFrame) -> pd.DataFrame:
    if mapping_summary.empty:
        return pd.DataFrame(
            columns=[
                "donor_dataset",
                "n_cell_types",
                "total_matched_compounds",
                "max_matched_compounds_in_cell_type",
                "mean_time_distance_h",
                "mean_log_dose_distance",
            ]
        )
    return (
        mapping_summary.groupby("donor_dataset", as_index=False)
        .agg(
            n_cell_types=("cell_type", "nunique"),
            total_matched_compounds=("n_matched_compounds", "sum"),
            max_matched_compounds_in_cell_type=("n_matched_compounds", "max"),
            mean_time_distance_h=("mean_time_distance_h", "mean"),
            mean_log_dose_distance=("mean_log_dose_distance", "mean"),
        )
        .sort_values("donor_dataset", ignore_index=True)
    )


def matched_signature_alignment(
    mapping: pd.DataFrame,
    tahoe_layers: Mapping[str, np.ndarray],
    donor_layers: Mapping[str, np.ndarray],
    signature_layers: tuple[str, ...] = DEFAULT_SIGNATURE_LAYERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for layer_name in signature_layers:
        tahoe_matrix = tahoe_layers[layer_name][mapping["tahoe_context_idx"].to_numpy(dtype=np.int64)]
        donor_matrix = donor_layers[layer_name][mapping["donor_context_idx"].to_numpy(dtype=np.int64)]
        pearson = paired_row_metric(tahoe_matrix, donor_matrix, metric="pearson")
        spearman = paired_row_metric(tahoe_matrix, donor_matrix, metric="spearman")
        cosine = paired_row_metric(tahoe_matrix, donor_matrix, metric="cosine")
        for idx, item in enumerate(mapping.itertuples(index=False)):
            rows.append(
                {
                    "donor_dataset": str(item.donor_dataset),
                    "cell_type": str(item.cell_type),
                    "pubchem_cid": str(item.pubchem_cid),
                    "signature_layer": layer_name,
                    "matched_signature_pearson": float(pearson[idx]),
                    "matched_signature_spearman": float(spearman[idx]),
                    "matched_signature_cosine": float(cosine[idx]),
                    "time_distance_h": float(item.time_distance_h),
                    "log_dose_distance": float(item.log_dose_distance),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["donor_dataset", "cell_type", "signature_layer"], as_index=False)
        .agg(
            n_matched_compounds=("pubchem_cid", "nunique"),
            mean_matched_signature_pearson=("matched_signature_pearson", "mean"),
            median_matched_signature_pearson=("matched_signature_pearson", "median"),
            mean_matched_signature_spearman=("matched_signature_spearman", "mean"),
            mean_matched_signature_cosine=("matched_signature_cosine", "mean"),
        )
        .sort_values(
            ["donor_dataset", "signature_layer", "n_matched_compounds", "cell_type"],
            ascending=[True, True, False, True],
            ignore_index=True,
        )
    )
    return detail, summary


def headline_alignment_summary(alignment_summary: pd.DataFrame) -> pd.DataFrame:
    if alignment_summary.empty:
        return pd.DataFrame(
            columns=[
                "donor_dataset",
                "signature_layer",
                "n_cell_types",
                "total_matched_compounds",
                "mean_matched_signature_pearson_weighted_mean",
                "median_matched_signature_pearson_weighted_mean",
                "mean_matched_signature_spearman_weighted_mean",
                "mean_matched_signature_cosine_weighted_mean",
            ]
        )

    rows: list[dict[str, object]] = []
    for (donor_dataset, signature_layer), group in alignment_summary.groupby(
        ["donor_dataset", "signature_layer"],
        sort=True,
    ):
        weights = group["n_matched_compounds"].to_numpy(dtype=np.float64)
        row: dict[str, object] = {
            "donor_dataset": str(donor_dataset),
            "signature_layer": str(signature_layer),
            "n_cell_types": int(group["cell_type"].nunique()),
            "total_matched_compounds": int(group["n_matched_compounds"].sum()),
        }
        for metric_col in (
            "mean_matched_signature_pearson",
            "median_matched_signature_pearson",
            "mean_matched_signature_spearman",
            "mean_matched_signature_cosine",
        ):
            values = group[metric_col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            row[f"{metric_col}_weighted_mean"] = (
                float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["donor_dataset", "signature_layer"],
        ignore_index=True,
    )


def _neighbor_metrics_for_celltype(
    tahoe_matrix: np.ndarray,
    donor_matrix: np.ndarray,
    similarity_metric: str,
    k_values: tuple[int, ...],
) -> dict[str, float]:
    n_compounds = int(tahoe_matrix.shape[0])
    if n_compounds != int(donor_matrix.shape[0]):
        raise ValueError("Tahoe and donor matrices must have the same number of compounds.")

    tahoe_similarity = _row_similarity_matrix(tahoe_matrix, metric=similarity_metric)
    donor_similarity = _row_similarity_matrix(donor_matrix, metric=similarity_metric)

    upper_tahoe = _upper_triangle_values(tahoe_similarity)
    upper_donor = _upper_triangle_values(donor_similarity)
    out = {
        "structure_pairwise_pearson": _vector_correlation(upper_tahoe, upper_donor, metric="pearson"),
        "structure_pairwise_spearman": _vector_correlation(upper_tahoe, upper_donor, metric="spearman"),
    }

    anchor_pearson: list[float] = []
    anchor_spearman: list[float] = []
    overlap_values: dict[int, list[float]] = {int(k): [] for k in k_values}
    jaccard_values: dict[int, list[float]] = {int(k): [] for k in k_values}
    for anchor in range(n_compounds):
        mask = np.ones(n_compounds, dtype=bool)
        mask[anchor] = False
        anchor_pearson.append(
            _vector_correlation(tahoe_similarity[anchor, mask], donor_similarity[anchor, mask], metric="pearson")
        )
        anchor_spearman.append(
            _vector_correlation(tahoe_similarity[anchor, mask], donor_similarity[anchor, mask], metric="spearman")
        )
        for k in k_values:
            left = set(_topk_neighbors(tahoe_similarity, anchor, int(k)).tolist())
            right = set(_topk_neighbors(donor_similarity, anchor, int(k)).tolist())
            if not left and not right:
                overlap_values[int(k)].append(float("nan"))
                jaccard_values[int(k)].append(float("nan"))
                continue
            local_k = max(len(left), len(right), 1)
            overlap_values[int(k)].append(float(len(left & right) / local_k))
            union = left | right
            jaccard_values[int(k)].append(float(len(left & right) / len(union)) if union else float("nan"))

    out["anchor_similarity_pearson_mean"] = float(np.nanmean(anchor_pearson)) if anchor_pearson else float("nan")
    out["anchor_similarity_spearman_mean"] = float(np.nanmean(anchor_spearman)) if anchor_spearman else float("nan")
    for k in k_values:
        out[f"neighbor_overlap_at_{k}"] = (
            float(np.nanmean(overlap_values[int(k)])) if overlap_values[int(k)] else float("nan")
        )
        out[f"neighbor_jaccard_at_{k}"] = (
            float(np.nanmean(jaccard_values[int(k)])) if jaccard_values[int(k)] else float("nan")
        )
        out[f"knn_edge_jaccard_at_{k}"] = _directed_knn_edge_jaccard(tahoe_similarity, donor_similarity, int(k))
    return out


def neighborhood_preservation_metrics(
    mapping: pd.DataFrame,
    tahoe_layers: Mapping[str, np.ndarray],
    donor_layers: Mapping[str, np.ndarray],
    signature_layers: tuple[str, ...] = DEFAULT_SIGNATURE_LAYERS,
    similarity_metrics: tuple[str, ...] = DEFAULT_SIMILARITY_METRICS,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (donor_dataset, cell_type), group in mapping.groupby(["donor_dataset", "cell_type"], sort=True):
        ordered = group.sort_values("pubchem_cid", kind="stable", ignore_index=True)
        n_compounds = int(ordered["pubchem_cid"].nunique())
        for layer_name in signature_layers:
            tahoe_matrix = tahoe_layers[layer_name][ordered["tahoe_context_idx"].to_numpy(dtype=np.int64)]
            donor_matrix = donor_layers[layer_name][ordered["donor_context_idx"].to_numpy(dtype=np.int64)]
            for similarity_metric in similarity_metrics:
                metrics = _neighbor_metrics_for_celltype(
                    tahoe_matrix=tahoe_matrix,
                    donor_matrix=donor_matrix,
                    similarity_metric=similarity_metric,
                    k_values=k_values,
                )
                rows.append(
                    {
                        "donor_dataset": str(donor_dataset),
                        "cell_type": str(cell_type),
                        "signature_layer": layer_name,
                        "similarity_metric": similarity_metric,
                        "n_compounds": n_compounds,
                        "n_compound_pairs": int(n_compounds * (n_compounds - 1) / 2),
                        **metrics,
                    }
                )

    detail = pd.DataFrame(rows).sort_values(
        ["donor_dataset", "signature_layer", "similarity_metric", "n_compounds", "cell_type"],
        ascending=[True, True, True, False, True],
        ignore_index=True,
    )
    metric_columns = [
        col
        for col in detail.columns
        if col not in {"donor_dataset", "cell_type", "signature_layer", "similarity_metric", "n_compounds", "n_compound_pairs"}
    ]
    summary_rows: list[dict[str, object]] = []
    for key, group in detail.groupby(["donor_dataset", "signature_layer", "similarity_metric"], sort=True):
        summary_row: dict[str, object] = {
            "donor_dataset": str(key[0]),
            "signature_layer": str(key[1]),
            "similarity_metric": str(key[2]),
            "n_cell_types": int(group["cell_type"].nunique()),
            "total_compounds": int(group["n_compounds"].sum()),
            "max_compounds_in_cell_type": int(group["n_compounds"].max()),
        }
        weights = group["n_compounds"].to_numpy(dtype=np.float64)
        for metric_col in metric_columns:
            values = group[metric_col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            summary_row[f"{metric_col}_weighted_mean"] = (
                float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else float("nan")
            )
            summary_row[f"{metric_col}_unweighted_mean"] = float(np.nanmean(values)) if np.any(finite) else float("nan")
        summary_rows.append(summary_row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["donor_dataset", "signature_layer", "similarity_metric"],
        ignore_index=True,
    )
    return detail, summary


def headline_neighborhood_summary(neighborhood_summary: pd.DataFrame) -> pd.DataFrame:
    if neighborhood_summary.empty:
        return neighborhood_summary.copy()
    columns = [
        "donor_dataset",
        "signature_layer",
        "similarity_metric",
        "n_cell_types",
        "total_compounds",
        "structure_pairwise_pearson_weighted_mean",
        "structure_pairwise_spearman_weighted_mean",
        "anchor_similarity_pearson_mean_weighted_mean",
        "anchor_similarity_spearman_mean_weighted_mean",
        "neighbor_overlap_at_1_weighted_mean",
        "neighbor_overlap_at_3_weighted_mean",
        "neighbor_overlap_at_5_weighted_mean",
        "neighbor_overlap_at_10_weighted_mean",
        "neighbor_jaccard_at_5_weighted_mean",
        "knn_edge_jaccard_at_5_weighted_mean",
    ]
    present = [col for col in columns if col in neighborhood_summary.columns]
    return neighborhood_summary[present].copy().sort_values(
        ["donor_dataset", "signature_layer", "similarity_metric"],
        ignore_index=True,
    )


def headline_best_neighborhood_by_dataset(neighborhood_summary: pd.DataFrame) -> pd.DataFrame:
    if neighborhood_summary.empty:
        return neighborhood_summary.copy()
    return (
        neighborhood_summary.sort_values(
            [
                "donor_dataset",
                "neighbor_overlap_at_5_weighted_mean",
                "neighbor_jaccard_at_5_weighted_mean",
                "structure_pairwise_spearman_weighted_mean",
            ],
            ascending=[True, False, False, False],
            ignore_index=True,
        )
        .groupby("donor_dataset", as_index=False)
        .first()
    )


def run_neighborhood_preservation_evaluation(
    prep_dir: Path | str,
    results_dir: Path | str,
    signature_layers: tuple[str, ...] = DEFAULT_SIGNATURE_LAYERS,
    similarity_metrics: tuple[str, ...] = DEFAULT_SIMILARITY_METRICS,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    start = time.time()
    prep_root = Path(prep_dir)
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    tahoe_adata, donor_adatas, _ = load_prepared_inputs(prep_root)
    tahoe_contexts, tahoe_layers = aggregate_exact_contexts(
        tahoe_adata,
        dataset_name="tahoe",
        signature_layers=signature_layers,
    )

    mapping_frames: list[pd.DataFrame] = []
    alignment_detail_frames: list[pd.DataFrame] = []
    alignment_summary_frames: list[pd.DataFrame] = []
    neighborhood_detail_frames: list[pd.DataFrame] = []
    neighborhood_summary_frames: list[pd.DataFrame] = []

    for donor_dataset, donor_adata in donor_adatas.items():
        if verbose:
            print("processing donor dataset", donor_dataset, flush=True)
        donor_contexts, donor_layers = aggregate_exact_contexts(
            donor_adata,
            dataset_name=donor_dataset,
            signature_layers=signature_layers,
        )
        mapping = build_closest_context_mapping(
            tahoe_contexts=tahoe_contexts,
            donor_contexts=donor_contexts,
            donor_dataset=donor_dataset,
        )
        mapping.to_csv(results_root / f"{donor_dataset}_closest_context_mapping.csv", index=False)
        mapping_frames.append(mapping)

        mapping_summary = summarize_mapping(mapping)
        mapping_summary.to_csv(results_root / f"{donor_dataset}_mapping_summary.csv", index=False)

        alignment_detail, alignment_summary = matched_signature_alignment(
            mapping=mapping,
            tahoe_layers=tahoe_layers,
            donor_layers=donor_layers,
            signature_layers=signature_layers,
        )
        alignment_detail.to_csv(results_root / f"{donor_dataset}_matched_signature_alignment.csv", index=False)
        alignment_summary.to_csv(results_root / f"{donor_dataset}_matched_signature_alignment_summary.csv", index=False)
        alignment_detail_frames.append(alignment_detail)
        alignment_summary_frames.append(alignment_summary)

        neighborhood_detail, neighborhood_summary = neighborhood_preservation_metrics(
            mapping=mapping,
            tahoe_layers=tahoe_layers,
            donor_layers=donor_layers,
            signature_layers=signature_layers,
            similarity_metrics=similarity_metrics,
            k_values=k_values,
        )
        neighborhood_detail.to_csv(results_root / f"{donor_dataset}_neighborhood_preservation_by_celltype.csv", index=False)
        neighborhood_summary.to_csv(results_root / f"{donor_dataset}_neighborhood_preservation_summary.csv", index=False)
        neighborhood_detail_frames.append(neighborhood_detail)
        neighborhood_summary_frames.append(neighborhood_summary)

    all_mapping = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    all_alignment_detail = pd.concat(alignment_detail_frames, ignore_index=True) if alignment_detail_frames else pd.DataFrame()
    all_alignment_summary = (
        pd.concat(alignment_summary_frames, ignore_index=True) if alignment_summary_frames else pd.DataFrame()
    )
    all_neighborhood_detail = (
        pd.concat(neighborhood_detail_frames, ignore_index=True) if neighborhood_detail_frames else pd.DataFrame()
    )
    all_neighborhood_summary = (
        pd.concat(neighborhood_summary_frames, ignore_index=True) if neighborhood_summary_frames else pd.DataFrame()
    )

    if not all_mapping.empty:
        all_mapping.to_csv(results_root / "all_datasets_closest_context_mapping.csv", index=False)
        all_mapping_summary = summarize_mapping(all_mapping)
        all_mapping_summary.to_csv(results_root / "all_datasets_mapping_summary.csv", index=False)
        headline_mapping_summary(all_mapping_summary).to_csv(results_root / "headline_mapping_summary.csv", index=False)
    if not all_alignment_detail.empty:
        all_alignment_detail.to_csv(results_root / "all_datasets_matched_signature_alignment.csv", index=False)
    if not all_alignment_summary.empty:
        all_alignment_summary.to_csv(results_root / "all_datasets_matched_signature_alignment_summary.csv", index=False)
        headline_alignment_summary(all_alignment_summary).to_csv(
            results_root / "headline_alignment_summary.csv",
            index=False,
        )
    if not all_neighborhood_detail.empty:
        all_neighborhood_detail.to_csv(results_root / "all_datasets_neighborhood_preservation_by_celltype.csv", index=False)
    if not all_neighborhood_summary.empty:
        all_neighborhood_summary.to_csv(results_root / "all_datasets_neighborhood_preservation_summary.csv", index=False)
        headline_neighborhood_summary(all_neighborhood_summary).to_csv(
            results_root / "headline_neighborhood_summary.csv",
            index=False,
        )
        headline_best_neighborhood_by_dataset(all_neighborhood_summary).to_csv(
            results_root / "headline_best_neighborhood_by_dataset.csv",
            index=False,
        )

    metadata = {
        "runtime_seconds": round(time.time() - start, 3),
        "prep_dir": str(prep_root),
        "signature_layers": list(signature_layers),
        "similarity_metrics": list(similarity_metrics),
        "k_values": [int(k) for k in k_values],
        "donor_datasets": sorted(donor_adatas),
    }
    (results_root / "neighborhood_preservation_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {
        "mapping": all_mapping,
        "alignment_detail": all_alignment_detail,
        "alignment_summary": all_alignment_summary,
        "neighborhood_detail": all_neighborhood_detail,
        "neighborhood_summary": all_neighborhood_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Tahoe/L1000 neighborhood preservation on shared compounds and shared cell types.",
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("data/tahoe_l1000_benchmark_prep"),
        help="Prepared shared-gene Tahoe/L1000 benchmark directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tahoe_l1000_neighborhood_preservation"),
        help="Directory to write neighborhood preservation outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while running the evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_neighborhood_preservation_evaluation(
        prep_dir=args.prep_dir,
        results_dir=args.results_dir,
        verbose=bool(args.verbose),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
