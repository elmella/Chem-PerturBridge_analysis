from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from op3_analysis.retrieval.cli import _parse_dataset_overrides, _split_csv
    from op3_analysis.retrieval.config import resolve_dataset_paths
    from op3_analysis.retrieval.data import DatasetStore, load_cell_type_data
    from op3_analysis.tahoe_l1000_deg_benchmark import LabelConfig, paired_row_metric
else:
    from .retrieval.cli import _parse_dataset_overrides, _split_csv
    from .retrieval.config import resolve_dataset_paths
    from .retrieval.data import DatasetStore, load_cell_type_data
    from .tahoe_l1000_deg_benchmark import LabelConfig, paired_row_metric


DEFAULT_NEIGHBORHOOD_METRICS = ("pearson", "cosine", "mse")
DEFAULT_K_VALUES = (1, 5, 10)
PAIR_MATCH_DEG_CONFIGS = (
    LabelConfig(name="deg_fdr001_lfc000", fdr_threshold=0.01, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr001_lfc025", fdr_threshold=0.01, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr001_lfc050", fdr_threshold=0.01, min_abs_logfc=0.50),
    LabelConfig(name="deg_fdr025_lfc000", fdr_threshold=0.025, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr025_lfc025", fdr_threshold=0.025, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr025_lfc050", fdr_threshold=0.025, min_abs_logfc=0.50),
    LabelConfig(name="deg_fdr005_lfc000", fdr_threshold=0.05, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr005_lfc025", fdr_threshold=0.05, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr005_lfc050", fdr_threshold=0.05, min_abs_logfc=0.50),
    LabelConfig(name="deg_fdr010_lfc000", fdr_threshold=0.10, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr010_lfc025", fdr_threshold=0.10, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr010_lfc050", fdr_threshold=0.10, min_abs_logfc=0.50),
    LabelConfig(name="deg_fdr020_lfc000", fdr_threshold=0.20, min_abs_logfc=0.0),
    LabelConfig(name="deg_fdr020_lfc025", fdr_threshold=0.20, min_abs_logfc=0.25),
    LabelConfig(name="deg_fdr020_lfc050", fdr_threshold=0.20, min_abs_logfc=0.50),
)
DE_SUBSET_SPECS = (
    {"name": "de_top20_query", "k": 20, "source": "query"},
    {"name": "de_top20_db", "k": 20, "source": "db"},
    {"name": "de_top20_union", "k": 20, "source": "union"},
    {"name": "de_top50_query", "k": 50, "source": "query"},
    {"name": "de_top50_db", "k": 50, "source": "db"},
    {"name": "de_top50_union", "k": 50, "source": "union"},
)
DE_SUBSET_METRICS = ("pearson", "mse", "wmse", "weighted_r2")
BASELINE_DE_SUBSET_NAMES = frozenset({"de_top20_union", "de_top50_union"})
DEFAULT_FOCUS_MATCH_SUBSET = "de_top50_union"
DEFAULT_FOCUS_MATCH_METRIC = "weighted_r2"
DEFAULT_FOCUS_DEG_SUBSET = "de_top50_union"
DEFAULT_FOCUS_NEIGHBORHOOD_SUBSET = "de_top50_union"
DEFAULT_FOCUS_NEIGHBORHOOD_METRIC = "pearson"
PAIR_MATCH_TRUTH_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "query_obs_id",
    "query_row",
    "pubchem_cid",
    "pert_time_h",
    "pert_dose_uM",
    "gt_db_cell_type",
    "gt_db_obs_id",
    "gt_db_row",
    "n_genes_gt_pair",
]
PAIR_MATCH_TRUTH_SUMMARY_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "n_query_obs",
    "n_shared_pubchem_cids",
    "n_truth_queries",
    "n_genes_gt_pair",
]
PAIR_MATCH_RULES = {
    "pairing_mode": "one_to_one",
    "require_same_time_h": True,
    "dose_matching": "closest_absolute_log_dose_difference",
    "drop_nonpositive_dose_rows": True,
}


def _log(verbose: bool, *parts: object) -> None:
    if verbose:
        print(*parts, flush=True)


def _matching_obs_table(cell_data) -> pd.DataFrame:
    obs = cell_data.obs.copy()
    dose_values = obs["pert_dose_uM"].to_numpy(dtype=np.float64)
    valid = (
        obs["pubchem_cid"].notna().to_numpy(dtype=bool)
        & np.isfinite(obs["pert_time_h"].to_numpy(dtype=np.float64))
        & np.isfinite(dose_values)
        & (dose_values > 0.0)
    )
    if not np.any(valid):
        return pd.DataFrame(columns=["_row", "pubchem_cid", "pert_time_h", "pert_dose_uM", "obs_id"])
    matched = obs.loc[valid, ["_row", "pubchem_cid", "pert_time_h", "pert_dose_uM"]].copy()
    matched["pubchem_cid"] = matched["pubchem_cid"].astype(str)
    matched["obs_id"] = cell_data.adata.obs_names[matched["_row"].to_numpy(dtype=np.int64)].astype(str)
    return matched.sort_values(["pubchem_cid", "pert_time_h", "_row"], kind="stable").reset_index(drop=True)


def _group_key_map(obs: pd.DataFrame) -> dict[tuple[str, float], pd.DataFrame]:
    if obs.empty:
        return {}
    return {
        (str(pubchem_cid), float(time_h)): group.sort_values(["pert_dose_uM", "_row"], kind="stable").reset_index(
            drop=True
        )
        for (pubchem_cid, time_h), group in obs.groupby(["pubchem_cid", "pert_time_h"], sort=True)
    }


def _identity_pairs(obs: pd.DataFrame) -> list[tuple[int, int, str, float]]:
    if obs.empty:
        return []
    pairs = [
        (int(row._row), int(row._row), str(row.pubchem_cid), float(row.pert_time_h))
        for row in obs.itertuples(index=False)
    ]
    pairs.sort(key=lambda item: (item[2], item[3], item[0], item[1]))
    return pairs


def _one_to_one_pairs(
    left_obs: pd.DataFrame,
    right_obs: pd.DataFrame,
) -> list[tuple[int, int, str, float]]:
    left_groups = _group_key_map(left_obs)
    right_groups = _group_key_map(right_obs)
    shared_keys = sorted(set(left_groups) & set(right_groups))
    matched_pairs: list[tuple[int, int, str, float]] = []

    for pubchem_cid, time_h in shared_keys:
        left_group = left_groups[(pubchem_cid, time_h)]
        right_group = right_groups[(pubchem_cid, time_h)]
        left_rows = left_group["_row"].to_numpy(dtype=np.int64)
        right_rows = right_group["_row"].to_numpy(dtype=np.int64)
        left_doses = left_group["pert_dose_uM"].to_numpy(dtype=np.float64)
        right_doses = right_group["pert_dose_uM"].to_numpy(dtype=np.float64)

        if left_rows.size == 0 or right_rows.size == 0:
            continue

        base_cost = np.abs(np.log(left_doses)[:, None] - np.log(right_doses)[None, :])
        row_tiebreak = np.arange(left_rows.size, dtype=np.float64)[:, None] * 1e-12
        col_tiebreak = np.arange(right_rows.size, dtype=np.float64)[None, :] * 1e-15
        row_ind, col_ind = linear_sum_assignment(base_cost + row_tiebreak + col_tiebreak)

        local_pairs = [
            (int(left_rows[left_idx]), int(right_rows[right_idx]), str(pubchem_cid), float(time_h))
            for left_idx, right_idx in zip(row_ind.tolist(), col_ind.tolist())
        ]
        local_pairs.sort(key=lambda item: (item[0], item[1]))
        matched_pairs.extend(local_pairs)

    return matched_pairs


def build_pair_match_truth_tables(
    dataset_paths: dict[str, Path],
    query_datasets: list[str],
    db_datasets: list[str],
    include_self_dataset: bool = False,
    cell_type_filter: Optional[set[str]] = None,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=Path(path), cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }
    cache: dict[tuple[str, str], object] = {}
    truth_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    requested_directions = {
        (query_dataset, db_dataset)
        for query_dataset in query_datasets
        for db_dataset in db_datasets
        if include_self_dataset or query_dataset != db_dataset
    }
    unordered_pairs = sorted(
        {
            (query_dataset, db_dataset)
            if query_dataset <= db_dataset
            else (db_dataset, query_dataset)
            for query_dataset, db_dataset in requested_directions
        }
    )

    for pair_idx, (dataset_a, dataset_b) in enumerate(unordered_pairs, start=1):
        if dataset_a not in stores:
            raise KeyError(f"Unknown query dataset: {dataset_a}")
        if dataset_b not in stores:
            raise KeyError(f"Unknown db dataset: {dataset_b}")

        pair_t0 = time.perf_counter()
        left_store = stores[dataset_a]
        right_store = stores[dataset_b]
        left_cell_types = left_store.list_cell_types()
        right_cell_types = right_store.list_cell_types()
        if cell_type_filter is not None:
            left_cell_types = [cell_type for cell_type in left_cell_types if cell_type in cell_type_filter]
            right_cell_types = [cell_type for cell_type in right_cell_types if cell_type in cell_type_filter]
        eligible_cell_types = sorted(set(left_cell_types) & set(right_cell_types))
        if not eligible_cell_types:
            _log(
                verbose,
                f"[pair-match-precompute] pair {dataset_a}<->{dataset_b}: no overlapping cell types; skipping",
            )
            continue

        requested_pair_directions = []
        if (dataset_a, dataset_b) in requested_directions:
            requested_pair_directions.append((dataset_a, dataset_b))
        if dataset_a != dataset_b and (dataset_b, dataset_a) in requested_directions:
            requested_pair_directions.append((dataset_b, dataset_a))

        _log(
            verbose,
            f"[pair-match-precompute] {pair_idx} {dataset_a}<->{dataset_b}: "
            f"eligible_cell_types={len(eligible_cell_types)} directions={requested_pair_directions}",
        )

        for cell_idx, query_cell_type in enumerate(eligible_cell_types, start=1):
            left_data = load_cell_type_data(left_store, query_cell_type, cache)
            right_data = load_cell_type_data(right_store, query_cell_type, cache)
            if left_data is None or left_data.adata.n_obs == 0:
                _log(
                    verbose,
                    f"[pair-match-precompute] pair {dataset_a}<->{dataset_b} "
                    f"{cell_idx}/{len(eligible_cell_types)} cell_type={query_cell_type}: "
                    f"{dataset_a} data missing or empty",
                )
                continue
            if right_data is None or right_data.adata.n_obs == 0:
                _log(
                    verbose,
                    f"[pair-match-precompute] pair {dataset_a}<->{dataset_b} "
                    f"{cell_idx}/{len(eligible_cell_types)} cell_type={query_cell_type}: "
                    f"{dataset_b} data missing or empty",
                )
                continue

            shared_genes = np.intersect1d(
                left_data.adata.var_names.values,
                right_data.adata.var_names.values,
                assume_unique=False,
            )
            n_genes = int(shared_genes.size)
            left_matching_obs = _matching_obs_table(left_data)
            right_matching_obs = _matching_obs_table(right_data)
            if dataset_a == dataset_b:
                pair_rows = _identity_pairs(left_matching_obs)
            else:
                pair_rows = _one_to_one_pairs(left_matching_obs, right_matching_obs)
            matched_pubchem_cids = {pubchem_cid for _, _, pubchem_cid, _ in pair_rows}

            left_obs_names = left_data.adata.obs_names.astype(str).to_numpy()
            right_obs_names = right_data.adata.obs_names.astype(str).to_numpy()
            left_obs_index = left_data.obs.set_index("_row")
            right_obs_index = right_data.obs.set_index("_row")

            directional_match_count = int(len(pair_rows))
            for query_dataset, db_dataset in requested_pair_directions:
                if query_dataset == dataset_a and db_dataset == dataset_b:
                    query_data = left_data
                    query_obs_names = left_obs_names
                    db_obs_names = right_obs_names
                    query_obs_index = left_obs_index
                    directional_pairs = pair_rows
                else:
                    query_data = right_data
                    query_obs_names = right_obs_names
                    db_obs_names = left_obs_names
                    query_obs_index = right_obs_index
                    directional_pairs = [
                        (int(right_row), int(left_row), str(pubchem_cid), float(time_h))
                        for left_row, right_row, pubchem_cid, time_h in pair_rows
                    ]

                for query_row, gt_db_row, pubchem_cid, time_h in directional_pairs:
                    query_obs = query_obs_index.loc[int(query_row)]
                    truth_rows.append(
                        {
                            "query_dataset": query_dataset,
                            "db_dataset": db_dataset,
                            "query_cell_type": query_cell_type,
                            "query_obs_id": str(query_obs_names[int(query_row)]),
                            "query_row": int(query_row),
                            "pubchem_cid": str(pubchem_cid),
                            "pert_time_h": float(time_h),
                            "pert_dose_uM": float(query_obs["pert_dose_uM"]),
                            "gt_db_cell_type": query_cell_type,
                            "gt_db_obs_id": str(db_obs_names[int(gt_db_row)]),
                            "gt_db_row": int(gt_db_row),
                            "n_genes_gt_pair": float(n_genes),
                        }
                    )

                summary_rows.append(
                    {
                        "query_dataset": query_dataset,
                        "db_dataset": db_dataset,
                        "query_cell_type": query_cell_type,
                        "n_query_obs": int(query_data.adata.n_obs),
                        "n_shared_pubchem_cids": int(len(matched_pubchem_cids)),
                        "n_truth_queries": directional_match_count,
                        "n_genes_gt_pair": float(n_genes),
                    }
                )

            _log(
                verbose,
                f"[pair-match-precompute] pair {dataset_a}<->{dataset_b} "
                f"{cell_idx}/{len(eligible_cell_types)} cell_type={query_cell_type}: "
                f"matched_pubchem_cids={len(matched_pubchem_cids)} matched_pairs={len(pair_rows)}",
            )

        _log(
            verbose,
            f"[pair-match-precompute] pair {dataset_a}<->{dataset_b}: "
            f"elapsed_s={time.perf_counter() - pair_t0:.1f}",
        )

    truth_df = pd.DataFrame(truth_rows, columns=PAIR_MATCH_TRUTH_COLUMNS)
    if not truth_df.empty:
        truth_df = truth_df.sort_values(
            ["query_dataset", "db_dataset", "query_cell_type", "query_obs_id"],
            ignore_index=True,
        )

    truth_summary_df = pd.DataFrame(summary_rows, columns=PAIR_MATCH_TRUTH_SUMMARY_COLUMNS)
    if not truth_summary_df.empty:
        truth_summary_df = truth_summary_df.sort_values(
            ["query_dataset", "db_dataset", "query_cell_type"],
            ignore_index=True,
        )
    return truth_df, truth_summary_df


def _matrix_for_representation(adata, representation: str, shared_genes: np.ndarray) -> np.ndarray:
    var_index = pd.Index(adata.var_names.astype(str))
    gene_idx = var_index.get_indexer(pd.Index(shared_genes.astype(str)))
    if np.any(gene_idx < 0):
        missing = int(np.sum(gene_idx < 0))
        raise KeyError(f"Representation matrix requested with {missing} missing shared genes.")

    if representation == "X":
        matrix = adata.X
    else:
        if representation not in adata.layers:
            raise KeyError(
                f"Representation '{representation}' not found. Available layers: {list(adata.layers.keys())}"
            )
        matrix = adata.layers[representation]

    subset = matrix[:, gene_idx]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float32)


def _mse_per_row(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    diff = np.asarray(x_true, dtype=np.float64) - np.asarray(x_pred, dtype=np.float64)
    return np.mean(diff * diff, axis=1)


def _de_score_matrix(logfc: np.ndarray, padj: np.ndarray, p_floor: float = 1e-300) -> np.ndarray:
    local_logfc = np.abs(np.asarray(logfc, dtype=np.float64))
    local_padj = np.asarray(padj, dtype=np.float64)
    clipped = np.clip(local_padj, p_floor, 1.0)
    score = local_logfc * (-np.log10(clipped))
    score[~np.isfinite(score)] = 0.0
    return score.astype(np.float32, copy=False)


def _normalize_positive_weight_rows(weights: np.ndarray) -> np.ndarray:
    local = np.asarray(weights, dtype=np.float64).copy()
    local[~np.isfinite(local)] = 0.0
    local[local < 0.0] = 0.0
    zero_rows = np.sum(local > 0.0, axis=1) == 0
    if np.any(zero_rows):
        local[zero_rows] = 1.0
    return local


def _weighted_mse_per_row(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    local_weights = _normalize_positive_weight_rows(weights)
    diff = truth - pred
    numer = np.sum(local_weights * diff * diff, axis=1, dtype=np.float64)
    denom = np.sum(local_weights, axis=1, dtype=np.float64)
    out = np.full(truth.shape[0], np.nan, dtype=np.float64)
    valid = denom > 0.0
    out[valid] = numer[valid] / denom[valid]
    return out


def _weighted_r2_per_row(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    local_weights = _normalize_positive_weight_rows(weights)
    weight_sums = np.sum(local_weights, axis=1, dtype=np.float64)
    out = np.full(truth.shape[0], np.nan, dtype=np.float64)
    valid = weight_sums > 0.0
    if not np.any(valid):
        return out
    truth_mean = np.divide(
        np.sum(local_weights * truth, axis=1, dtype=np.float64),
        weight_sums,
        out=np.full(truth.shape[0], np.nan, dtype=np.float64),
        where=valid,
    )
    pred_mean = np.divide(
        np.sum(local_weights * pred, axis=1, dtype=np.float64),
        weight_sums,
        out=np.full(truth.shape[0], np.nan, dtype=np.float64),
        where=valid,
    )
    ss_res = np.sum(local_weights * (truth - pred) * (truth - pred), axis=1, dtype=np.float64)
    ss_tot = 0.5 * (
        np.sum(local_weights * (truth - truth_mean[:, None]) ** 2, axis=1, dtype=np.float64)
        + np.sum(local_weights * (pred - pred_mean[:, None]) ** 2, axis=1, dtype=np.float64)
    )
    usable = valid & (ss_tot > 1e-12)
    out[usable] = 1.0 - (ss_res[usable] / ss_tot[usable])
    return out


def _topk_indices_1d(scores: np.ndarray, k: int) -> np.ndarray:
    local = np.asarray(scores, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(local))
    if valid_idx.size == 0:
        return np.array([], dtype=np.int64)
    local_k = int(min(int(k), valid_idx.size))
    if local_k < 1:
        return np.array([], dtype=np.int64)
    ranked = np.argsort(-local[valid_idx], kind="stable")
    return valid_idx[ranked[:local_k]].astype(np.int64, copy=False)


def _de_subset_indices(
    query_scores: np.ndarray,
    db_scores: np.ndarray,
    subset_spec: dict[str, object],
) -> np.ndarray:
    source = str(subset_spec["source"])
    local_k = int(subset_spec["k"])
    if source == "query":
        return _topk_indices_1d(query_scores, local_k)
    if source == "db":
        return _topk_indices_1d(db_scores, local_k)
    if source == "union":
        left = _topk_indices_1d(query_scores, local_k)
        right = _topk_indices_1d(db_scores, local_k)
        if left.size == 0:
            return right
        if right.size == 0:
            return left
        return np.unique(np.concatenate([left, right]).astype(np.int64, copy=False))
    raise ValueError(f"Unsupported DE subset source: {source}")


def _positive_weights_1d(weights: np.ndarray) -> np.ndarray:
    local = np.asarray(weights, dtype=np.float64).copy()
    local[~np.isfinite(local)] = 0.0
    local[local < 0.0] = 0.0
    if not np.any(local > 0.0):
        local[:] = 1.0
    return local


def _pearson_1d(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    truth_centered = truth - np.mean(truth)
    pred_centered = pred - np.mean(pred)
    denom = np.linalg.norm(truth_centered) * np.linalg.norm(pred_centered)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(truth_centered, pred_centered) / denom)


def _mse_1d(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    diff = np.asarray(x_true, dtype=np.float64) - np.asarray(x_pred, dtype=np.float64)
    return float(np.mean(diff * diff))


def _weighted_mse_1d(x_true: np.ndarray, x_pred: np.ndarray, weights: np.ndarray) -> float:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    local_weights = _positive_weights_1d(weights)
    diff = truth - pred
    return float(np.average(diff * diff, weights=local_weights))


def _weighted_r2_1d(x_true: np.ndarray, x_pred: np.ndarray, weights: np.ndarray) -> float:
    truth = np.asarray(x_true, dtype=np.float64)
    pred = np.asarray(x_pred, dtype=np.float64)
    local_weights = _positive_weights_1d(weights)
    truth_mean = float(np.average(truth, weights=local_weights))
    pred_mean = float(np.average(pred, weights=local_weights))
    ss_res = float(np.sum(local_weights * (truth - pred) ** 2))
    ss_tot = 0.5 * (
        float(np.sum(local_weights * (truth - truth_mean) ** 2))
        + float(np.sum(local_weights * (pred - pred_mean) ** 2))
    )
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _nir_per_row(model_error: np.ndarray, baseline_error: np.ndarray) -> np.ndarray:
    model = np.asarray(model_error, dtype=np.float64)
    baseline = np.asarray(baseline_error, dtype=np.float64)
    out = np.full(model.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(model) & np.isfinite(baseline) & (baseline > 1e-12)
    out[valid] = 1.0 - (model[valid] / baseline[valid])
    return out


def _subset_metric_bundle_1d(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    return {
        "pearson": _pearson_1d(x_true, x_pred),
        "mse": _mse_1d(x_true, x_pred),
        "wmse": _weighted_mse_1d(x_true, x_pred, weights),
        "weighted_r2": _weighted_r2_1d(x_true, x_pred, weights),
    }


def _leave_one_out_mean_predictions(matrix_all: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    local = np.asarray(matrix_all, dtype=np.float64)
    rows = np.asarray(row_indices, dtype=np.int64)
    if local.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={local.shape}")
    if rows.ndim != 1:
        raise ValueError(f"Expected 1D row indices, got shape={rows.shape}")
    if local.shape[0] <= 1:
        return np.broadcast_to(local.mean(axis=0, dtype=np.float64).astype(np.float32), (rows.size, local.shape[1])).copy()
    total_sum = local.sum(axis=0, dtype=np.float64)
    return ((total_sum[None, :] - local[rows]) / float(local.shape[0] - 1)).astype(np.float32)


def _broadcast_mean_predictions(matrix_all: np.ndarray, n_rows: int) -> np.ndarray:
    local = np.asarray(matrix_all, dtype=np.float64)
    if local.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={local.shape}")
    return np.broadcast_to(local.mean(axis=0, dtype=np.float64).astype(np.float32), (int(n_rows), local.shape[1])).copy()


def _bh_adjust_rows(pvalues: np.ndarray) -> np.ndarray:
    local = np.asarray(pvalues, dtype=np.float64)
    out = np.full(local.shape, np.nan, dtype=np.float64)
    for idx in range(local.shape[0]):
        row = local[idx]
        finite = np.isfinite(row)
        if int(finite.sum()) == 0:
            continue
        valid = np.clip(row[finite], 0.0, 1.0)
        order = np.argsort(valid, kind="stable")
        ranked = valid[order]
        m = ranked.size
        adjusted = ranked * float(m) / np.arange(1, m + 1, dtype=np.float64)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        target = np.full(row.shape, np.nan, dtype=np.float64)
        target[np.flatnonzero(finite)[order]] = adjusted
        out[idx] = target
    return out


def _binary_jaccard_per_row(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    union = np.sum(left_bool | right_bool, axis=1)
    inter = np.sum(left_bool & right_bool, axis=1)
    out = np.zeros(left_bool.shape[0], dtype=np.float64)
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def _deg_label_matrix_for_config(
    logfc: np.ndarray,
    padj: np.ndarray,
    config: LabelConfig,
) -> np.ndarray:
    local_logfc = np.asarray(logfc, dtype=np.float64)
    local_padj = np.asarray(padj, dtype=np.float64)
    labels = np.zeros(local_logfc.shape, dtype=np.int8)
    significant = np.isfinite(local_padj) & (local_padj <= float(config.fdr_threshold))
    magnitude = np.abs(local_logfc) >= float(config.min_abs_logfc)
    active = significant & magnitude & np.isfinite(local_logfc)
    labels[active & (local_logfc > 0.0)] = 1
    labels[active & (local_logfc < 0.0)] = -1
    return labels


def _label_overlap_metrics_per_row(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_left = np.asarray(left_labels, dtype=np.int8)
    local_right = np.asarray(right_labels, dtype=np.int8)
    if local_left.shape != local_right.shape:
        raise ValueError(f"Expected label matrices of the same shape, got {local_left.shape} vs {local_right.shape}")
    left_up = local_left > 0
    left_down = local_left < 0
    right_up = local_right > 0
    right_down = local_right < 0

    any_jaccard = _binary_jaccard_per_row(left_up | left_down, right_up | right_down)
    signed_jaccard = 0.5 * (
        _binary_jaccard_per_row(left_up, right_up) + _binary_jaccard_per_row(left_down, right_down)
    )
    return any_jaccard, signed_jaccard


def _score_rank_orders(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(local)
    desc_values = np.where(finite, local, -np.inf)
    asc_values = np.where(finite, local, np.inf)
    desc_order = np.argsort(-desc_values, axis=1, kind="stable").astype(np.int64, copy=False)
    asc_order = np.argsort(asc_values, axis=1, kind="stable").astype(np.int64, copy=False)
    finite_counts = np.sum(finite, axis=1, dtype=np.int64)
    return desc_order, asc_order, finite_counts


def _size_matched_prediction_jaccard_from_orders(
    truth_labels: np.ndarray,
    desc_order: np.ndarray,
    asc_order: np.ndarray,
    finite_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_truth = np.asarray(truth_labels, dtype=np.int8)
    n_rows, n_genes = local_truth.shape
    if desc_order.shape != (n_rows, n_genes) or asc_order.shape != (n_rows, n_genes):
        raise ValueError(
            f"Expected rank orders with shape {(n_rows, n_genes)}, got {desc_order.shape} and {asc_order.shape}"
        )
    local_finite_counts = np.asarray(finite_counts, dtype=np.int64)
    if local_finite_counts.shape != (n_rows,):
        raise ValueError(f"Expected finite_counts with shape {(n_rows,)}, got {local_finite_counts.shape}")

    any_jaccard = np.full(n_rows, np.nan, dtype=np.float64)
    signed_jaccard = np.full(n_rows, np.nan, dtype=np.float64)
    empty = np.array([], dtype=np.int64)

    for idx in range(n_rows):
        valid_count = int(local_finite_counts[idx])
        if valid_count <= 0:
            continue
        row_truth = local_truth[idx]
        true_up = np.flatnonzero(row_truth > 0)
        true_down = np.flatnonzero(row_truth < 0)
        true_any = np.flatnonzero(row_truth != 0)

        n_up = min(int(true_up.size), valid_count)
        pred_up = desc_order[idx, :n_up] if n_up > 0 else empty

        remaining = max(valid_count - n_up, 0)
        n_down = min(int(true_down.size), remaining)
        if n_down > 0:
            if n_up == 0:
                pred_down = asc_order[idx, :n_down]
            else:
                used = np.zeros(n_genes, dtype=bool)
                used[pred_up] = True
                asc_candidates = asc_order[idx]
                pred_down = asc_candidates[~used[asc_candidates]][:n_down]
        else:
            pred_down = empty

        pred_any = np.concatenate([pred_up, pred_down]) if (pred_up.size or pred_down.size) else empty

        any_inter = int(np.isin(pred_any, true_any).sum()) if pred_any.size else 0
        any_union = int(true_any.size + pred_any.size - any_inter)
        any_jaccard[idx] = 0.0 if any_union == 0 else float(any_inter / any_union)

        up_inter = int(np.isin(pred_up, true_up).sum()) if pred_up.size else 0
        down_inter = int(np.isin(pred_down, true_down).sum()) if pred_down.size else 0
        up_union = int(true_up.size + pred_up.size - up_inter)
        down_union = int(true_down.size + pred_down.size - down_inter)
        up_score = 0.0 if up_union == 0 else float(up_inter / up_union)
        down_score = 0.0 if down_union == 0 else float(down_inter / down_union)
        signed_jaccard[idx] = 0.5 * (up_score + down_score)

    return any_jaccard, signed_jaccard


def _size_matched_prediction_jaccard_from_scores(
    truth_labels: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    desc_order, asc_order, finite_counts = _score_rank_orders(pred_scores)
    return _size_matched_prediction_jaccard_from_orders(
        truth_labels=truth_labels,
        desc_order=desc_order,
        asc_order=asc_order,
        finite_counts=finite_counts,
    )


def _deg_overlap_metrics_per_row(
    query_logfc: np.ndarray,
    db_logfc: np.ndarray,
    query_padj: np.ndarray,
    db_padj: np.ndarray,
    config: LabelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    query_labels = _deg_label_matrix_for_config(query_logfc, query_padj, config)
    db_labels = _deg_label_matrix_for_config(db_logfc, db_padj, config)
    return _label_overlap_metrics_per_row(query_labels, db_labels)


def _row_similarity_matrix(matrix: np.ndarray, metric: str) -> np.ndarray:
    local = np.asarray(matrix, dtype=np.float64)
    if metric == "pearson":
        centered = local - local.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0.0)
        score = normalized @ normalized.T
        np.fill_diagonal(score, 1.0)
        return score.astype(np.float32, copy=False)
    if metric == "cosine":
        norms = np.linalg.norm(local, axis=1, keepdims=True)
        normalized = np.divide(local, norms, out=np.zeros_like(local), where=norms > 0.0)
        score = normalized @ normalized.T
        np.fill_diagonal(score, 1.0)
        return score.astype(np.float32, copy=False)
    if metric == "mse":
        sq_norm = np.sum(local * local, axis=1, keepdims=True)
        score = (sq_norm + sq_norm.T - (2.0 * (local @ local.T))) / float(local.shape[1])
        np.maximum(score, 0.0, out=score)
        np.fill_diagonal(score, 0.0)
        return score.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported neighborhood metric: {metric}")


def _anchor_similarity_vector(matrix: np.ndarray, anchor: int, metric: str) -> np.ndarray:
    local = np.asarray(matrix, dtype=np.float64)
    if local.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={local.shape}")
    if local.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if metric == "pearson":
        centered = local - local.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        anchor_vec = centered[int(anchor)]
        anchor_norm = float(norms[int(anchor)])
        if anchor_norm <= 0.0:
            score = np.zeros(local.shape[0], dtype=np.float64)
            score[int(anchor)] = 1.0
            return score.astype(np.float32, copy=False)
        numer = centered @ anchor_vec
        denom = norms * anchor_norm
        score = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
        score[int(anchor)] = 1.0
        return score.astype(np.float32, copy=False)
    if metric == "cosine":
        norms = np.linalg.norm(local, axis=1)
        anchor_vec = local[int(anchor)]
        anchor_norm = float(norms[int(anchor)])
        if anchor_norm <= 0.0:
            score = np.zeros(local.shape[0], dtype=np.float64)
            score[int(anchor)] = 1.0
            return score.astype(np.float32, copy=False)
        numer = local @ anchor_vec
        denom = norms * anchor_norm
        score = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)
        score[int(anchor)] = 1.0
        return score.astype(np.float32, copy=False)
    if metric == "mse":
        anchor_vec = local[int(anchor)]
        diff = local - anchor_vec[None, :]
        score = np.mean(diff * diff, axis=1, dtype=np.float64)
        score[int(anchor)] = 0.0
        return score.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported neighborhood metric: {metric}")


def _topk_neighbors(matrix: np.ndarray, anchor: int, k: int, larger_is_better: bool) -> np.ndarray:
    n_items = int(matrix.shape[0])
    if n_items <= 1:
        return np.array([], dtype=np.int64)
    local_k = int(min(k, n_items - 1))
    if local_k < 1:
        return np.array([], dtype=np.int64)
    row = np.asarray(matrix[anchor], dtype=np.float64).copy()
    row[anchor] = -np.inf if larger_is_better else np.inf
    order = np.argsort(row, kind="stable")
    if larger_is_better:
        order = order[::-1]
    return order[:local_k].astype(np.int64, copy=False)


def _topk_neighbors_from_scores(scores: np.ndarray, anchor: int, k: int, larger_is_better: bool) -> np.ndarray:
    local = np.asarray(scores, dtype=np.float64).copy()
    n_items = int(local.shape[0])
    if n_items <= 1:
        return np.array([], dtype=np.int64)
    local_k = int(min(k, n_items - 1))
    if local_k < 1:
        return np.array([], dtype=np.int64)
    local[int(anchor)] = -np.inf if larger_is_better else np.inf
    order = np.argsort(local, kind="stable")
    if larger_is_better:
        order = order[::-1]
    return order[:local_k].astype(np.int64, copy=False)


def _knn_edge_jaccard(left: np.ndarray, right: np.ndarray, k: int, larger_is_better: bool) -> float:
    n_items = int(left.shape[0])
    if n_items <= 1:
        return float("nan")
    left_edges: set[tuple[int, int]] = set()
    right_edges: set[tuple[int, int]] = set()
    for anchor in range(n_items):
        for neighbor in _topk_neighbors(left, anchor, k, larger_is_better=larger_is_better):
            left_edges.add((anchor, int(neighbor)))
        for neighbor in _topk_neighbors(right, anchor, k, larger_is_better=larger_is_better):
            right_edges.add((anchor, int(neighbor)))
    union = left_edges | right_edges
    if not union:
        return float("nan")
    return float(len(left_edges & right_edges) / len(union))


def _neighbor_overlap_detail(
    query_matrix: np.ndarray,
    db_matrix: np.ndarray,
    neighborhood_metric: str,
    k_values: tuple[int, ...],
    subset_indices_by_anchor: Optional[list[np.ndarray]] = None,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    larger_is_better = neighborhood_metric != "mse"

    overlap_by_k: dict[int, np.ndarray] = {}
    n_items = int(query_matrix.shape[0])
    for k in k_values:
        overlap_by_k[int(k)] = np.full(n_items, np.nan, dtype=np.float64)

    if subset_indices_by_anchor is None:
        query_scores = _row_similarity_matrix(query_matrix, metric=neighborhood_metric)
        db_scores = _row_similarity_matrix(db_matrix, metric=neighborhood_metric)
        for k in k_values:
            values = overlap_by_k[int(k)]
            for anchor in range(n_items):
                left = set(_topk_neighbors(query_scores, anchor, k, larger_is_better=larger_is_better).tolist())
                right = set(_topk_neighbors(db_scores, anchor, k, larger_is_better=larger_is_better).tolist())
                if not left and not right:
                    continue
                local_k = max(len(left), len(right), 1)
                values[anchor] = float(len(left & right) / local_k)
        edge_jaccard_by_k = {
            int(k): _knn_edge_jaccard(query_scores, db_scores, int(k), larger_is_better=larger_is_better)
            for k in k_values
        }
        return overlap_by_k, edge_jaccard_by_k

    left_edges_by_k = {int(k): set() for k in k_values}
    right_edges_by_k = {int(k): set() for k in k_values}
    for anchor in range(n_items):
        subset_idx = np.asarray(subset_indices_by_anchor[anchor], dtype=np.int64)
        if subset_idx.size == 0:
            continue
        query_scores = _anchor_similarity_vector(query_matrix[:, subset_idx], anchor, metric=neighborhood_metric)
        db_scores = _anchor_similarity_vector(db_matrix[:, subset_idx], anchor, metric=neighborhood_metric)
        for k in k_values:
            left_neighbors = _topk_neighbors_from_scores(query_scores, anchor, k, larger_is_better=larger_is_better)
            right_neighbors = _topk_neighbors_from_scores(db_scores, anchor, k, larger_is_better=larger_is_better)
            left = set(left_neighbors.tolist())
            right = set(right_neighbors.tolist())
            if left or right:
                local_k = max(len(left), len(right), 1)
                overlap_by_k[int(k)][anchor] = float(len(left & right) / local_k)
            for neighbor in left_neighbors:
                left_edges_by_k[int(k)].add((anchor, int(neighbor)))
            for neighbor in right_neighbors:
                right_edges_by_k[int(k)].add((anchor, int(neighbor)))

    edge_jaccard_by_k: dict[int, float] = {}
    for k in k_values:
        union = left_edges_by_k[int(k)] | right_edges_by_k[int(k)]
        edge_jaccard_by_k[int(k)] = float(len(left_edges_by_k[int(k)] & right_edges_by_k[int(k)]) / len(union)) if union else float("nan")
    return overlap_by_k, edge_jaccard_by_k


def summarize_pair_metrics(
    truth_df: pd.DataFrame,
    dataset_paths: dict[str, Path],
    representation: str,
    logfc_layer: str = "logFC",
    pvalue_layer: str = "P.Value",
    deg_configs: tuple[LabelConfig, ...] = PAIR_MATCH_DEG_CONFIGS,
    neighborhood_metrics: tuple[str, ...] = DEFAULT_NEIGHBORHOOD_METRICS,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if truth_df.empty:
        empty_detail = truth_df.copy()
        return empty_detail, pd.DataFrame()

    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=Path(path), cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }
    cache: dict[tuple[str, str], object] = {}

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    group_cols = ["query_dataset", "db_dataset", "query_cell_type"]

    for group_idx, (key, group) in enumerate(truth_df.groupby(group_cols, sort=True), start=1):
        query_dataset, db_dataset, query_cell_type = (str(key[0]), str(key[1]), str(key[2]))
        _log(
            verbose,
            f"[pair-match-neighborhoods] {group_idx} "
            f"{query_dataset}->{db_dataset} cell_type={query_cell_type} n_truth={len(group)}",
        )
        query_data = load_cell_type_data(stores[query_dataset], query_cell_type, cache)
        db_data = load_cell_type_data(stores[db_dataset], query_cell_type, cache)
        if query_data is None or db_data is None:
            continue

        shared_genes = np.intersect1d(
            query_data.adata.var_names.values,
            db_data.adata.var_names.values,
            assume_unique=False,
        )
        if shared_genes.size == 0:
            continue

        query_matrix_all = _matrix_for_representation(query_data.adata, representation, shared_genes)
        db_matrix_all = _matrix_for_representation(db_data.adata, representation, shared_genes)
        query_logfc_all = _matrix_for_representation(query_data.adata, logfc_layer, shared_genes)
        db_logfc_all = _matrix_for_representation(db_data.adata, logfc_layer, shared_genes)
        query_pvalue_all = _matrix_for_representation(query_data.adata, pvalue_layer, shared_genes)
        db_pvalue_all = _matrix_for_representation(db_data.adata, pvalue_layer, shared_genes)

        query_rows = group["query_row"].to_numpy(dtype=np.int64)
        db_rows = group["gt_db_row"].to_numpy(dtype=np.int64)
        query_matrix = query_matrix_all[query_rows]
        db_matrix = db_matrix_all[db_rows]
        query_logfc = query_logfc_all[query_rows]
        db_logfc = db_logfc_all[db_rows]
        query_padj = _bh_adjust_rows(query_pvalue_all[query_rows])
        db_padj = _bh_adjust_rows(db_pvalue_all[db_rows])
        query_de_score = _de_score_matrix(query_logfc, query_padj)
        db_de_score = _de_score_matrix(db_logfc, db_padj)
        gene_weights = np.maximum(query_de_score, db_de_score).astype(np.float64, copy=False)
        query_loo_mean_pred = _leave_one_out_mean_predictions(query_matrix_all, query_rows)
        db_mean_pred = _broadcast_mean_predictions(db_matrix_all, query_matrix.shape[0])
        query_loo_logfc_pred = _leave_one_out_mean_predictions(query_logfc_all, query_rows)
        db_mean_logfc_pred = _broadcast_mean_predictions(db_logfc_all, query_logfc.shape[0])
        db_logfc_desc_order, db_logfc_asc_order, db_logfc_finite_counts = _score_rank_orders(db_logfc)
        query_loo_logfc_desc_order, query_loo_logfc_asc_order, query_loo_logfc_finite_counts = _score_rank_orders(
            query_loo_logfc_pred
        )
        db_mean_logfc_desc_order, db_mean_logfc_asc_order, db_mean_logfc_finite_counts = _score_rank_orders(
            db_mean_logfc_pred
        )

        detail = group.copy().reset_index(drop=True)
        detail["n_shared_genes"] = int(shared_genes.size)
        detail["representation"] = representation
        detail["match_pearson"] = paired_row_metric(query_matrix, db_matrix, metric="pearson")
        detail["match_cosine"] = paired_row_metric(query_matrix, db_matrix, metric="cosine")
        detail["match_mse"] = _mse_per_row(query_matrix, db_matrix)
        detail["match_wmse"] = _weighted_mse_per_row(query_matrix, db_matrix, gene_weights)
        detail["match_weighted_r2"] = _weighted_r2_per_row(query_matrix, db_matrix, gene_weights)
        query_loo_mean_pearson = paired_row_metric(query_matrix, query_loo_mean_pred, metric="pearson")
        query_loo_mean_mse = _mse_per_row(query_matrix, query_loo_mean_pred)
        query_loo_mean_wmse = _weighted_mse_per_row(query_matrix, query_loo_mean_pred, gene_weights)
        query_loo_mean_weighted_r2 = _weighted_r2_per_row(query_matrix, query_loo_mean_pred, gene_weights)
        db_mean_pearson = paired_row_metric(query_matrix, db_mean_pred, metric="pearson")
        db_mean_mse = _mse_per_row(query_matrix, db_mean_pred)
        db_mean_wmse = _weighted_mse_per_row(query_matrix, db_mean_pred, gene_weights)
        db_mean_weighted_r2 = _weighted_r2_per_row(query_matrix, db_mean_pred, gene_weights)
        match_nir_vs_query_loo_mean = _nir_per_row(detail["match_wmse"].to_numpy(dtype=np.float64), query_loo_mean_wmse)
        match_nir_vs_db_mean = _nir_per_row(detail["match_wmse"].to_numpy(dtype=np.float64), db_mean_wmse)

        summary_row: dict[str, object] = {
            "query_dataset": query_dataset,
            "db_dataset": db_dataset,
            "query_cell_type": query_cell_type,
            "representation": representation,
            "n_truth_queries": int(detail.shape[0]),
            "n_shared_pubchem_cids": int(detail["pubchem_cid"].astype(str).nunique()),
            "n_shared_genes": int(shared_genes.size),
            "mean_match_pearson": float(np.nanmean(detail["match_pearson"])),
            "median_match_pearson": float(np.nanmedian(detail["match_pearson"])),
            "mean_match_cosine": float(np.nanmean(detail["match_cosine"])),
            "median_match_cosine": float(np.nanmedian(detail["match_cosine"])),
            "mean_match_mse": float(np.nanmean(detail["match_mse"])),
            "median_match_mse": float(np.nanmedian(detail["match_mse"])),
            "mean_match_wmse": float(np.nanmean(detail["match_wmse"])),
            "mean_match_weighted_r2": float(np.nanmean(detail["match_weighted_r2"])),
            "query_loo_mean_pearson": float(np.nanmean(query_loo_mean_pearson)),
            "query_loo_mean_mse": float(np.nanmean(query_loo_mean_mse)),
            "query_loo_mean_wmse": float(np.nanmean(query_loo_mean_wmse)),
            "query_loo_mean_weighted_r2": float(np.nanmean(query_loo_mean_weighted_r2)),
            "db_mean_pearson": float(np.nanmean(db_mean_pearson)),
            "db_mean_mse": float(np.nanmean(db_mean_mse)),
            "db_mean_wmse": float(np.nanmean(db_mean_wmse)),
            "db_mean_weighted_r2": float(np.nanmean(db_mean_weighted_r2)),
            "mean_match_nir_vs_query_loo_mean": float(np.nanmean(match_nir_vs_query_loo_mean)),
            "mean_match_nir_vs_db_mean": float(np.nanmean(match_nir_vs_db_mean)),
        }
        subset_indices_by_name: dict[str, list[np.ndarray]] = {}
        subset_sizes_by_name: dict[str, np.ndarray] = {}
        for subset_spec in DE_SUBSET_SPECS:
            subset_name = str(subset_spec["name"])
            subset_indices_by_anchor: list[np.ndarray] = []
            subset_sizes = np.zeros(query_matrix.shape[0], dtype=np.int64)
            for row_idx in range(query_matrix.shape[0]):
                subset_idx = _de_subset_indices(query_de_score[row_idx], db_de_score[row_idx], subset_spec)
                subset_indices_by_anchor.append(subset_idx.astype(np.int64, copy=False))
                subset_sizes[row_idx] = int(subset_idx.size)
            subset_indices_by_name[subset_name] = subset_indices_by_anchor
            subset_sizes_by_name[subset_name] = subset_sizes

        for subset_spec in DE_SUBSET_SPECS:
            subset_name = str(subset_spec["name"])
            subset_sizes = subset_sizes_by_name[subset_name]
            subset_indices_by_anchor = subset_indices_by_name[subset_name]
            metric_arrays = {
                metric_name: np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                for metric_name in DE_SUBSET_METRICS
            }
            query_baseline_metric_arrays = (
                {
                    metric_name: np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                    for metric_name in DE_SUBSET_METRICS
                }
                if subset_name in BASELINE_DE_SUBSET_NAMES
                else None
            )
            db_baseline_metric_arrays = (
                {
                    metric_name: np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                    for metric_name in DE_SUBSET_METRICS
                }
                if subset_name in BASELINE_DE_SUBSET_NAMES
                else None
            )
            for row_idx in range(query_matrix.shape[0]):
                subset_idx = subset_indices_by_anchor[row_idx]
                if subset_idx.size == 0:
                    continue
                subset_metrics = _subset_metric_bundle_1d(
                    query_matrix[row_idx, subset_idx],
                    db_matrix[row_idx, subset_idx],
                    gene_weights[row_idx, subset_idx],
                )
                for metric_name, metric_value in subset_metrics.items():
                    metric_arrays[metric_name][row_idx] = float(metric_value)
                if query_baseline_metric_arrays is not None:
                    query_baseline_metrics = _subset_metric_bundle_1d(
                        query_matrix[row_idx, subset_idx],
                        query_loo_mean_pred[row_idx, subset_idx],
                        gene_weights[row_idx, subset_idx],
                    )
                    db_baseline_metrics = _subset_metric_bundle_1d(
                        query_matrix[row_idx, subset_idx],
                        db_mean_pred[row_idx, subset_idx],
                        gene_weights[row_idx, subset_idx],
                    )
                    for metric_name, metric_value in query_baseline_metrics.items():
                        query_baseline_metric_arrays[metric_name][row_idx] = float(metric_value)
                    for metric_name, metric_value in db_baseline_metrics.items():
                        db_baseline_metric_arrays[metric_name][row_idx] = float(metric_value)
            detail[f"match_{subset_name}_n_genes"] = subset_sizes
            summary_row[f"mean_match_{subset_name}_n_genes"] = float(np.nanmean(subset_sizes.astype(np.float64)))
            for metric_name, metric_values in metric_arrays.items():
                col_name = f"match_{subset_name}_{metric_name}"
                detail[col_name] = metric_values
                summary_row[f"mean_{col_name}"] = float(np.nanmean(metric_values))
            if query_baseline_metric_arrays is not None:
                for metric_name, metric_values in query_baseline_metric_arrays.items():
                    summary_row[f"query_loo_mean_{subset_name}_{metric_name}"] = float(np.nanmean(metric_values))
                for metric_name, metric_values in db_baseline_metric_arrays.items():
                    summary_row[f"db_mean_{subset_name}_{metric_name}"] = float(np.nanmean(metric_values))
                summary_row[f"mean_match_{subset_name}_nir_vs_query_loo_mean"] = float(
                    np.nanmean(_nir_per_row(metric_arrays["wmse"], query_baseline_metric_arrays["wmse"]))
                )
                summary_row[f"mean_match_{subset_name}_nir_vs_db_mean"] = float(
                    np.nanmean(_nir_per_row(metric_arrays["wmse"], db_baseline_metric_arrays["wmse"]))
                )
        deg_detail_cols: dict[str, np.ndarray] = {}
        for config in deg_configs:
            query_labels = _deg_label_matrix_for_config(query_logfc, query_padj, config)
            db_labels = _deg_label_matrix_for_config(db_logfc, db_padj, config)

            any_jaccard, signed_jaccard = _label_overlap_metrics_per_row(query_labels, db_labels)
            deg_detail_cols[f"deg_any_jaccard_{config.name}"] = any_jaccard
            deg_detail_cols[f"deg_signed_jaccard_{config.name}"] = signed_jaccard
            summary_row[f"mean_deg_any_jaccard_{config.name}"] = float(np.nanmean(any_jaccard))
            summary_row[f"mean_deg_signed_jaccard_{config.name}"] = float(np.nanmean(signed_jaccard))

            pred_any_jaccard, pred_signed_jaccard = _size_matched_prediction_jaccard_from_orders(
                truth_labels=query_labels,
                desc_order=db_logfc_desc_order,
                asc_order=db_logfc_asc_order,
                finite_counts=db_logfc_finite_counts,
            )
            query_loo_pred_any_jaccard, query_loo_pred_signed_jaccard = _size_matched_prediction_jaccard_from_orders(
                truth_labels=query_labels,
                desc_order=query_loo_logfc_desc_order,
                asc_order=query_loo_logfc_asc_order,
                finite_counts=query_loo_logfc_finite_counts,
            )
            db_mean_pred_any_jaccard, db_mean_pred_signed_jaccard = _size_matched_prediction_jaccard_from_orders(
                truth_labels=query_labels,
                desc_order=db_mean_logfc_desc_order,
                asc_order=db_mean_logfc_asc_order,
                finite_counts=db_mean_logfc_finite_counts,
            )

            deg_detail_cols[f"deg_pred_any_jaccard_{config.name}"] = pred_any_jaccard
            deg_detail_cols[f"deg_pred_signed_jaccard_{config.name}"] = pred_signed_jaccard
            deg_detail_cols[f"query_loo_mean_deg_pred_any_jaccard_{config.name}"] = query_loo_pred_any_jaccard
            deg_detail_cols[f"query_loo_mean_deg_pred_signed_jaccard_{config.name}"] = query_loo_pred_signed_jaccard
            deg_detail_cols[f"db_mean_deg_pred_any_jaccard_{config.name}"] = db_mean_pred_any_jaccard
            deg_detail_cols[f"db_mean_deg_pred_signed_jaccard_{config.name}"] = db_mean_pred_signed_jaccard

            summary_row[f"mean_deg_pred_any_jaccard_{config.name}"] = float(np.nanmean(pred_any_jaccard))
            summary_row[f"mean_deg_pred_signed_jaccard_{config.name}"] = float(np.nanmean(pred_signed_jaccard))
            summary_row[f"query_loo_mean_deg_pred_any_jaccard_{config.name}"] = float(
                np.nanmean(query_loo_pred_any_jaccard)
            )
            summary_row[f"query_loo_mean_deg_pred_signed_jaccard_{config.name}"] = float(
                np.nanmean(query_loo_pred_signed_jaccard)
            )
            summary_row[f"db_mean_deg_pred_any_jaccard_{config.name}"] = float(np.nanmean(db_mean_pred_any_jaccard))
            summary_row[f"db_mean_deg_pred_signed_jaccard_{config.name}"] = float(
                np.nanmean(db_mean_pred_signed_jaccard)
            )

            for subset_spec in DE_SUBSET_SPECS:
                subset_name = str(subset_spec["name"])
                subset_indices_by_anchor = subset_indices_by_name[subset_name]
                subset_any_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_signed_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_pred_any_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_pred_signed_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_query_loo_pred_any_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_query_loo_pred_signed_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_db_mean_pred_any_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)
                subset_db_mean_pred_signed_jaccard = np.full(query_matrix.shape[0], np.nan, dtype=np.float64)

                for row_idx in range(query_matrix.shape[0]):
                    subset_idx = subset_indices_by_anchor[row_idx]
                    if subset_idx.size == 0:
                        continue
                    row_query_labels = query_labels[row_idx : row_idx + 1, subset_idx]
                    row_db_labels = db_labels[row_idx : row_idx + 1, subset_idx]
                    row_subset_any, row_subset_signed = _label_overlap_metrics_per_row(row_query_labels, row_db_labels)
                    subset_any_jaccard[row_idx] = row_subset_any[0]
                    subset_signed_jaccard[row_idx] = row_subset_signed[0]

                    row_pred_any, row_pred_signed = _size_matched_prediction_jaccard_from_scores(
                        truth_labels=row_query_labels,
                        pred_scores=db_logfc[row_idx : row_idx + 1, subset_idx],
                    )
                    subset_pred_any_jaccard[row_idx] = row_pred_any[0]
                    subset_pred_signed_jaccard[row_idx] = row_pred_signed[0]

                    row_query_loo_pred_any, row_query_loo_pred_signed = _size_matched_prediction_jaccard_from_scores(
                        truth_labels=row_query_labels,
                        pred_scores=query_loo_logfc_pred[row_idx : row_idx + 1, subset_idx],
                    )
                    subset_query_loo_pred_any_jaccard[row_idx] = row_query_loo_pred_any[0]
                    subset_query_loo_pred_signed_jaccard[row_idx] = row_query_loo_pred_signed[0]

                    row_db_mean_pred_any, row_db_mean_pred_signed = _size_matched_prediction_jaccard_from_scores(
                        truth_labels=row_query_labels,
                        pred_scores=db_mean_logfc_pred[row_idx : row_idx + 1, subset_idx],
                    )
                    subset_db_mean_pred_any_jaccard[row_idx] = row_db_mean_pred_any[0]
                    subset_db_mean_pred_signed_jaccard[row_idx] = row_db_mean_pred_signed[0]

                deg_detail_cols[f"deg_any_jaccard_{config.name}__{subset_name}"] = subset_any_jaccard
                deg_detail_cols[f"deg_signed_jaccard_{config.name}__{subset_name}"] = subset_signed_jaccard
                deg_detail_cols[f"deg_pred_any_jaccard_{config.name}__{subset_name}"] = subset_pred_any_jaccard
                deg_detail_cols[f"deg_pred_signed_jaccard_{config.name}__{subset_name}"] = subset_pred_signed_jaccard
                deg_detail_cols[
                    f"query_loo_mean_deg_pred_any_jaccard_{config.name}__{subset_name}"
                ] = subset_query_loo_pred_any_jaccard
                deg_detail_cols[
                    f"query_loo_mean_deg_pred_signed_jaccard_{config.name}__{subset_name}"
                ] = subset_query_loo_pred_signed_jaccard
                deg_detail_cols[f"db_mean_deg_pred_any_jaccard_{config.name}__{subset_name}"] = subset_db_mean_pred_any_jaccard
                deg_detail_cols[
                    f"db_mean_deg_pred_signed_jaccard_{config.name}__{subset_name}"
                ] = subset_db_mean_pred_signed_jaccard

                summary_row[f"mean_deg_any_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_any_jaccard)
                )
                summary_row[f"mean_deg_signed_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_signed_jaccard)
                )
                summary_row[f"mean_deg_pred_any_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_pred_any_jaccard)
                )
                summary_row[f"mean_deg_pred_signed_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_pred_signed_jaccard)
                )
                summary_row[f"query_loo_mean_deg_pred_any_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_query_loo_pred_any_jaccard)
                )
                summary_row[f"query_loo_mean_deg_pred_signed_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_query_loo_pred_signed_jaccard)
                )
                summary_row[f"db_mean_deg_pred_any_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_db_mean_pred_any_jaccard)
                )
                summary_row[f"db_mean_deg_pred_signed_jaccard_{config.name}__{subset_name}"] = float(
                    np.nanmean(subset_db_mean_pred_signed_jaccard)
                )

        if deg_detail_cols:
            detail = pd.concat([detail, pd.DataFrame(deg_detail_cols, index=detail.index)], axis=1)

        neighborhood_detail_cols: dict[str, np.ndarray] = {}
        for neighborhood_metric in neighborhood_metrics:
            overlap_by_k, edge_jaccard_by_k = _neighbor_overlap_detail(
                query_matrix=query_matrix,
                db_matrix=db_matrix,
                neighborhood_metric=neighborhood_metric,
                k_values=k_values,
            )
            for k in k_values:
                neighborhood_detail_cols[f"overlap_at_{int(k)}_{neighborhood_metric}"] = overlap_by_k[int(k)]
                summary_row[f"mean_overlap_at_{int(k)}_{neighborhood_metric}"] = float(
                    np.nanmean(overlap_by_k[int(k)])
                )
                summary_row[f"knn_edge_jaccard_at_{int(k)}_{neighborhood_metric}"] = float(
                    edge_jaccard_by_k[int(k)]
                )
            for subset_spec in DE_SUBSET_SPECS:
                subset_name = str(subset_spec["name"])
                subset_overlap_by_k, subset_edge_jaccard_by_k = _neighbor_overlap_detail(
                    query_matrix=query_matrix,
                    db_matrix=db_matrix,
                    neighborhood_metric=neighborhood_metric,
                    k_values=k_values,
                    subset_indices_by_anchor=subset_indices_by_name[subset_name],
                )
                for k in k_values:
                    neighborhood_detail_cols[
                        f"overlap_at_{int(k)}_{subset_name}_{neighborhood_metric}"
                    ] = subset_overlap_by_k[int(k)]
                    summary_row[f"mean_overlap_at_{int(k)}_{subset_name}_{neighborhood_metric}"] = float(
                        np.nanmean(subset_overlap_by_k[int(k)])
                    )
                    summary_row[f"knn_edge_jaccard_at_{int(k)}_{subset_name}_{neighborhood_metric}"] = float(
                        subset_edge_jaccard_by_k[int(k)]
                    )

        if neighborhood_detail_cols:
            detail = pd.concat([detail, pd.DataFrame(neighborhood_detail_cols, index=detail.index)], axis=1)
        detail_frames.append(detail)
        summary_rows.append(summary_row)

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_by_cell_type = pd.DataFrame(summary_rows).sort_values(
        ["query_dataset", "db_dataset", "query_cell_type"],
        ignore_index=True,
    )
    return detail_df, summary_by_cell_type


def summarize_overall(summary_by_cell_type: pd.DataFrame) -> pd.DataFrame:
    if summary_by_cell_type.empty:
        return pd.DataFrame()

    metric_columns = [
        col
        for col in summary_by_cell_type.columns
        if col
        not in {
            "query_dataset",
            "db_dataset",
            "query_cell_type",
            "representation",
            "n_truth_queries",
            "n_shared_pubchem_cids",
            "n_shared_genes",
        }
    ]

    rows: list[dict[str, object]] = []
    for (query_dataset, db_dataset, representation), group in summary_by_cell_type.groupby(
        ["query_dataset", "db_dataset", "representation"],
        sort=True,
    ):
        row: dict[str, object] = {
            "query_dataset": str(query_dataset),
            "db_dataset": str(db_dataset),
            "representation": str(representation),
            "n_query_cell_types": int(group["query_cell_type"].nunique()),
            "total_truth_queries": int(group["n_truth_queries"].sum()),
            "total_shared_pubchem_cids_across_cell_types": int(group["n_shared_pubchem_cids"].sum()),
            "mean_shared_genes": float(np.average(group["n_shared_genes"], weights=group["n_truth_queries"])),
        }
        weights = group["n_truth_queries"].to_numpy(dtype=np.float64)
        for metric_col in metric_columns:
            values = group[metric_col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            row[f"{metric_col}_weighted_mean"] = (
                float(np.average(values[finite], weights=weights[finite])) if np.any(finite) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["query_dataset", "db_dataset", "representation"],
        ignore_index=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched-pair gene agreement and neighborhood preservation for all dataset pairs "
            "using the same truth-match rule as the retrieval benchmark."
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
        "--representation",
        default="logFC",
        help="Layer or matrix representation to compare across shared genes. Default: logFC.",
    )
    parser.add_argument(
        "--logfc-layer",
        default="logFC",
        help="Layer to use for DEG thresholding overlap. Default: logFC.",
    )
    parser.add_argument(
        "--pvalue-layer",
        default="P.Value",
        help="Layer to use for shared-gene BH adjustment before DEG overlap. Default: P.Value.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pair_match_neighborhoods",
        help="Directory for detail and summary CSV outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="pair_match_neighborhoods",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable AnnData caching for cell-type slices.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_overrides = _parse_dataset_overrides(args.dataset_path)
    dataset_paths = resolve_dataset_paths(dataset_overrides)
    dataset_names = list(dataset_paths.keys())

    query_datasets = dataset_names if args.query_datasets.strip().lower() == "all" else _split_csv(args.query_datasets)
    db_datasets = dataset_names if args.db_datasets.strip().lower() == "all" else _split_csv(args.db_datasets)
    missing_query = sorted(set(query_datasets) - set(dataset_names))
    missing_db = sorted(set(db_datasets) - set(dataset_names))
    if missing_query or missing_db:
        raise KeyError(
            f"Unknown dataset names. query missing={missing_query}, db missing={missing_db}, available={dataset_names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix)
    cell_type_filter = set(_split_csv(args.cell_types)) if args.cell_types else None

    started = time.time()
    truth_df, truth_summary_df = build_pair_match_truth_tables(
        dataset_paths=dataset_paths,
        query_datasets=query_datasets,
        db_datasets=db_datasets,
        include_self_dataset=bool(args.include_self_dataset),
        cell_type_filter=cell_type_filter,
        cache_cell_types=not args.no_cache,
        verbose=bool(args.verbose),
    )

    detail_df, summary_by_cell_type = summarize_pair_metrics(
        truth_df=truth_df,
        dataset_paths=dataset_paths,
        representation=str(args.representation),
        logfc_layer=str(args.logfc_layer),
        pvalue_layer=str(args.pvalue_layer),
        cache_cell_types=not args.no_cache,
        verbose=bool(args.verbose),
    )
    summary_overall = summarize_overall(summary_by_cell_type)

    truth_path = output_dir / f"{output_prefix}_truth_matches.csv"
    truth_summary_path = output_dir / f"{output_prefix}_truth_summary.csv"
    detail_path = output_dir / f"{output_prefix}_detail.csv"
    summary_by_cell_type_path = output_dir / f"{output_prefix}_summary_by_cell_type.csv"
    summary_overall_path = output_dir / f"{output_prefix}_summary_overall.csv"
    metadata_path = output_dir / f"{output_prefix}_metadata.json"

    truth_df.to_csv(truth_path, index=False)
    truth_summary_df.to_csv(truth_summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    summary_by_cell_type.to_csv(summary_by_cell_type_path, index=False)
    summary_overall.to_csv(summary_overall_path, index=False)

    metadata = {
        "runtime_seconds": round(time.time() - started, 3),
        "query_datasets": list(query_datasets),
        "db_datasets": list(db_datasets),
        "include_self_dataset": bool(args.include_self_dataset),
        "truth_matching": PAIR_MATCH_RULES,
        "representation": str(args.representation),
        "logfc_layer": str(args.logfc_layer),
        "pvalue_layer": str(args.pvalue_layer),
        "de_subset_specs": list(DE_SUBSET_SPECS),
        "baseline_de_subset_names": sorted(BASELINE_DE_SUBSET_NAMES),
        "default_focus_match_subset": DEFAULT_FOCUS_MATCH_SUBSET,
        "default_focus_match_metric": DEFAULT_FOCUS_MATCH_METRIC,
        "default_focus_deg_subset": DEFAULT_FOCUS_DEG_SUBSET,
        "default_focus_neighborhood_subset": DEFAULT_FOCUS_NEIGHBORHOOD_SUBSET,
        "default_focus_neighborhood_metric": DEFAULT_FOCUS_NEIGHBORHOOD_METRIC,
        "de_subset_neighborhoods": {
            "enabled": True,
            "scope": "anchor_specific_gene_subsets_within_matched_rows",
        },
        "deg_configs": [
            {
                "name": config.name,
                "fdr_threshold": float(config.fdr_threshold),
                "min_abs_logfc": float(config.min_abs_logfc),
            }
            for config in PAIR_MATCH_DEG_CONFIGS
        ],
        "cell_type_filter": sorted(cell_type_filter) if cell_type_filter else None,
        "detail_rows": int(detail_df.shape[0]),
        "summary_by_cell_type_rows": int(summary_by_cell_type.shape[0]),
        "summary_overall_rows": int(summary_overall.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    _log(args.verbose, f"Saved truth matches -> {truth_path}")
    _log(args.verbose, f"Saved truth summary -> {truth_summary_path}")
    _log(args.verbose, f"Saved detail -> {detail_path}")
    _log(args.verbose, f"Saved summary by cell type -> {summary_by_cell_type_path}")
    _log(args.verbose, f"Saved summary overall -> {summary_overall_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
