from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .config import RetrievalSettings
from .data import (
    CellTypeData,
    DatasetStore,
    candidate_rows_for_queries,
    load_cell_type_data,
)
from .metrics import (
    compute_signed_and_z,
    cosine_scores,
    mrrmse_scores,
    pearson_scores,
    rank_center_norm_rows,
    spearman_scores_precomputed,
)

DERIVED_SIGNED_REP = "signed_-log10(p)*sign(logFC)"
DERIVED_Z_REP = "z=sign(logFC)*Phi^-1(1-p/2)"

ALL_METRICS = ("cosine", "pearson", "spearman", "mrrmse")
SIMILARITY_METRICS = {"cosine", "pearson", "spearman"}
DISTANCE_METRICS = {"mrrmse"}


@dataclass
class PairContext:
    db_cell_type: str
    db_data: CellTypeData
    representations: set[str]
    query_var_idx: np.ndarray
    db_var_idx: np.ndarray
    candidate_perturbagens: list[str]
    perturbagen_to_col: dict[str, int]
    candidate_rows: np.ndarray
    n_genes: int


def _dense_layer_subset(adata: ad.AnnData, layer: str, var_idx: np.ndarray) -> np.ndarray:
    matrix = adata.layers[layer]
    if issparse(matrix):
        dense = matrix[:, var_idx].toarray()
    else:
        dense = np.asarray(matrix[:, var_idx])
    return np.asarray(dense, dtype=np.float32, copy=False)


def _available_representations(
    query_adata: ad.AnnData, db_adata: ad.AnnData, settings: RetrievalSettings
) -> list[str]:
    common_layers = sorted(
        (set(query_adata.layers.keys()) & set(db_adata.layers.keys()))
        - set(settings.excluded_layers)
    )
    reps = list(common_layers)
    has_derived = all(
        [
            settings.logfc_layer in query_adata.layers,
            settings.pvalue_layer in query_adata.layers,
            settings.logfc_layer in db_adata.layers,
            settings.pvalue_layer in db_adata.layers,
        ]
    )
    if has_derived:
        reps.append(DERIVED_SIGNED_REP)
        reps.append(DERIVED_Z_REP)
    return reps


def _materialize_representation(
    query_data: CellTypeData,
    context: PairContext,
    representation: str,
    settings: RetrievalSettings,
) -> tuple[np.ndarray, np.ndarray]:
    query_adata = query_data.adata
    db_adata = context.db_data.adata
    q_idx = context.query_var_idx
    d_idx = context.db_var_idx

    if representation in query_adata.layers and representation in db_adata.layers:
        x_query = _dense_layer_subset(query_adata, representation, q_idx)
        x_db = _dense_layer_subset(db_adata, representation, d_idx)
        return x_query, x_db

    if representation in (DERIVED_SIGNED_REP, DERIVED_Z_REP):
        q_logfc = _dense_layer_subset(query_adata, settings.logfc_layer, q_idx)
        q_pvals = _dense_layer_subset(query_adata, settings.pvalue_layer, q_idx)
        d_logfc = _dense_layer_subset(db_adata, settings.logfc_layer, d_idx)
        d_pvals = _dense_layer_subset(db_adata, settings.pvalue_layer, d_idx)

        signed_query, z_query = compute_signed_and_z(
            q_logfc, q_pvals, p_floor=settings.p_floor, p_clip_high=settings.p_clip_high
        )
        signed_db, z_db = compute_signed_and_z(
            d_logfc, d_pvals, p_floor=settings.p_floor, p_clip_high=settings.p_clip_high
        )
        if representation == DERIVED_SIGNED_REP:
            return signed_query, signed_db
        return z_query, z_db

    raise ValueError(f"Unknown representation: {representation}")


def _compute_ranks(metric: str, all_scores: np.ndarray, gt_score: float) -> float:
    if metric in SIMILARITY_METRICS:
        return float(1 + int(np.sum(all_scores > gt_score)))
    if metric in DISTANCE_METRICS:
        return float(1 + int(np.sum(all_scores < gt_score)))
    raise ValueError(f"Unexpected metric: {metric}")


def _score_candidates_for_query(
    x: np.ndarray,
    y: np.ndarray,
    x_ranked: np.ndarray,
    x_rank_norm: float,
    y_ranked: np.ndarray,
    y_rank_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "cosine": cosine_scores(x, y),
        "pearson": pearson_scores(x, y),
        "spearman": spearman_scores_precomputed(x_ranked, x_rank_norm, y_ranked, y_rank_norm),
        "mrrmse": mrrmse_scores(x, y),
    }


def _build_pair_contexts(
    query_data: CellTypeData,
    db_store: DatasetStore,
    db_cell_types: Iterable[str],
    cache: dict[tuple[str, str], Optional[CellTypeData]],
    settings: RetrievalSettings,
) -> list[PairContext]:
    query_time = query_data.obs["pert_time_h"].to_numpy(dtype=np.float64)
    query_dose = query_data.obs["pert_dose_uM"].to_numpy(dtype=np.float64)
    query_var_names = query_data.adata.var_names.values

    contexts: list[PairContext] = []
    for db_cell_type in db_cell_types:
        db_data = load_cell_type_data(db_store, db_cell_type, cache)
        if db_data is None or db_data.adata.n_obs == 0:
            continue

        common_genes = np.intersect1d(
            query_var_names,
            db_data.adata.var_names.values,
            assume_unique=False,
        )
        if common_genes.size == 0:
            continue

        pair_reps = set(_available_representations(query_data.adata, db_data.adata, settings))
        if not pair_reps:
            continue

        perturbagens, candidate_rows = candidate_rows_for_queries(
            db_data.perturbagen_groups,
            query_times=query_time,
            query_doses=query_dose,
        )
        if candidate_rows.shape[1] == 0:
            continue

        contexts.append(
            PairContext(
                db_cell_type=db_cell_type,
                db_data=db_data,
                representations=pair_reps,
                query_var_idx=query_data.adata.var_names.get_indexer(common_genes),
                db_var_idx=db_data.adata.var_names.get_indexer(common_genes),
                candidate_perturbagens=perturbagens,
                perturbagen_to_col={pert: i for i, pert in enumerate(perturbagens)},
                candidate_rows=candidate_rows,
                n_genes=int(common_genes.size),
            )
        )
    return contexts


def evaluate_query_cell_type(
    query_data: CellTypeData,
    db_store: DatasetStore,
    db_cell_types: Iterable[str],
    cache: dict[tuple[str, str], Optional[CellTypeData]],
    settings: RetrievalSettings,
) -> list[dict]:
    contexts = _build_pair_contexts(query_data, db_store, db_cell_types, cache, settings)
    if not contexts:
        return []

    all_representations = sorted({rep for ctx in contexts for rep in ctx.representations})
    n_queries = query_data.adata.n_obs
    query_perts = query_data.obs["perturbagen"].to_numpy(dtype=object)
    query_pert_valid = query_data.obs["perturbagen"].notna().to_numpy(dtype=bool)
    query_times = query_data.obs["pert_time_h"].to_numpy(dtype=np.float64)
    query_doses = query_data.obs["pert_dose_uM"].to_numpy(dtype=np.float64)
    query_index = query_data.adata.obs_names.astype(str).to_numpy()

    out_rows: list[dict] = []
    for representation in all_representations:
        all_scores: dict[str, list[list[float]]] = {
            metric: [[] for _ in range(n_queries)] for metric in ALL_METRICS
        }
        gt_scores = {
            metric: np.full(n_queries, np.nan, dtype=np.float64) for metric in ALL_METRICS
        }
        gt_genes = np.full(n_queries, np.nan, dtype=np.float64)

        for context in contexts:
            if representation not in context.representations:
                continue

            x_query, x_db = _materialize_representation(query_data, context, representation, settings)
            q_finite = np.isfinite(x_query).all(axis=1)
            d_finite = np.isfinite(x_db).all(axis=1)

            q_ranked, q_rank_norm, q_rank_valid = rank_center_norm_rows(x_query)
            d_ranked, d_rank_norm, d_rank_valid = rank_center_norm_rows(x_db)

            for q_idx in range(n_queries):
                if not q_finite[q_idx]:
                    continue
                if not query_pert_valid[q_idx]:
                    continue
                if not (np.isfinite(query_times[q_idx]) and np.isfinite(query_doses[q_idx])):
                    continue

                candidate_rows = context.candidate_rows[q_idx]
                candidate_rows = candidate_rows[candidate_rows >= 0]
                if candidate_rows.size == 0:
                    continue

                finite_candidates = d_finite[candidate_rows]
                if not finite_candidates.any():
                    continue

                valid_rows = candidate_rows[finite_candidates]
                x = x_query[q_idx]
                y = x_db[valid_rows]

                metric_scores = _score_candidates_for_query(
                    x=x,
                    y=y,
                    x_ranked=q_ranked[q_idx],
                    x_rank_norm=float(q_rank_norm[q_idx]) if q_rank_valid[q_idx] else np.nan,
                    y_ranked=d_ranked[valid_rows],
                    y_rank_norm=d_rank_norm[valid_rows],
                )

                for metric_name, values in metric_scores.items():
                    finite_values = values[np.isfinite(values)]
                    if finite_values.size:
                        all_scores[metric_name][q_idx].extend(finite_values.tolist())

                if context.db_cell_type != query_data.cell_type:
                    continue

                gt_local_col = context.perturbagen_to_col.get(query_perts[q_idx])
                if gt_local_col is None:
                    continue

                gt_row = int(context.candidate_rows[q_idx, gt_local_col])
                if gt_row < 0 or not d_finite[gt_row]:
                    continue

                gt_y = x_db[[gt_row]]
                gt_metrics = _score_candidates_for_query(
                    x=x,
                    y=gt_y,
                    x_ranked=q_ranked[q_idx],
                    x_rank_norm=float(q_rank_norm[q_idx]) if q_rank_valid[q_idx] else np.nan,
                    y_ranked=d_ranked[[gt_row]],
                    y_rank_norm=d_rank_norm[[gt_row]],
                )

                for metric_name, value_arr in gt_metrics.items():
                    value = float(value_arr[0])
                    if np.isfinite(value):
                        gt_scores[metric_name][q_idx] = value
                        gt_genes[q_idx] = context.n_genes

        for metric_name in ALL_METRICS:
            for q_idx in range(n_queries):
                metric_scores = np.asarray(all_scores[metric_name][q_idx], dtype=np.float64)
                gt_score = float(gt_scores[metric_name][q_idx])
                if metric_scores.size == 0 or not np.isfinite(gt_score):
                    continue

                metric_scores = metric_scores[np.isfinite(metric_scores)]
                if metric_scores.size == 0:
                    continue

                rank = _compute_ranks(metric_name, metric_scores, gt_score)
                n_candidates = int(metric_scores.size)

                if n_candidates <= 1:
                    rank_normalized = 0.0
                    retrieval_score = 1.0
                else:
                    rank_normalized = float((rank - 1.0) / (n_candidates - 1.0))
                    retrieval_score = float(1.0 - rank_normalized)

                out_rows.append(
                    {
                        "query_dataset": query_data.dataset_name,
                        "db_dataset": db_store.dataset_name,
                        "query_cell_type": query_data.cell_type,
                        "query_obs_id": query_index[q_idx],
                        "perturbagen": str(query_perts[q_idx]),
                        "pert_time_h": float(query_times[q_idx]),
                        "pert_dose_uM": float(query_doses[q_idx]),
                        "representation": representation,
                        "metric": metric_name,
                        "n_candidates": n_candidates,
                        "rank": rank,
                        "rank_over_n": float(rank / n_candidates),
                        "rank_normalized": rank_normalized,
                        "retrieval_score": retrieval_score,
                        "n_genes_gt_pair": float(gt_genes[q_idx])
                    }
                )

    return out_rows


def evaluate_dataset_pair(
    query_store: DatasetStore,
    db_store: DatasetStore,
    settings: RetrievalSettings,
    cell_type_filter: Optional[set[str]] = None,
    cache: Optional[dict[tuple[str, str], Optional[CellTypeData]]] = None,
    verbose: bool = False,
) -> list[dict]:
    shared_cache: dict[tuple[str, str], Optional[CellTypeData]] = cache if cache is not None else {}
    query_cell_types = query_store.list_cell_types()
    if cell_type_filter is not None:
        query_cell_types = [ct for ct in query_cell_types if ct in cell_type_filter]
    db_cell_types = db_store.list_cell_types()
    if cell_type_filter is not None:
        db_cell_types = [ct for ct in db_cell_types if ct in cell_type_filter]

    detail_rows: list[dict] = []
    for idx, query_cell_type in enumerate(query_cell_types, start=1):
        query_data = load_cell_type_data(query_store, query_cell_type, shared_cache)
        if query_data is None or query_data.adata.n_obs == 0:
            continue

        rows = evaluate_query_cell_type(
            query_data=query_data,
            db_store=db_store,
            db_cell_types=db_cell_types,
            cache=shared_cache,
            settings=settings,
        )
        detail_rows.extend(rows)

        if verbose:
            print(
                f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                f"{idx}/{len(query_cell_types)} cell_type={query_cell_type} rows={len(rows)}"
            )
    return detail_rows


def summarize_retrieval(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    agg = {
        "n_queries": ("rank", "count"),
        "n_candidates_mean": ("n_candidates", "mean"),
        "n_candidates_median": ("n_candidates", "median"),
        "rank_mean": ("rank", "mean"),
        "rank_median": ("rank", "median"),
        "rank_over_n_mean": ("rank_over_n", "mean"),
        "rank_normalized_mean": ("rank_normalized", "mean"),
        "retrieval_score_mean": ("retrieval_score", "mean"),
        "retrieval_score_median": ("retrieval_score", "median"),
    }

    by_cell = (
        detail_df.groupby(
            ["query_dataset", "db_dataset", "query_cell_type", "representation", "metric"],
            dropna=False,
        )
        .agg(**agg)
        .reset_index()
        .sort_values(
            ["query_dataset", "db_dataset", "query_cell_type", "representation", "metric"]
        )
        .reset_index(drop=True)
    )

    overall = (
        detail_df.groupby(["query_dataset", "db_dataset", "representation", "metric"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["query_dataset", "db_dataset", "representation", "metric"])
        .reset_index(drop=True)
    )
    return by_cell, overall


def run_cross_dataset_retrieval(
    dataset_paths: dict[str, str | PathLike[str]],
    query_datasets: list[str],
    db_datasets: list[str],
    settings: Optional[RetrievalSettings] = None,
    include_self_dataset: bool = False,
    cell_type_filter: Optional[set[str]] = None,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or RetrievalSettings()
    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=path, cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }

    cache: dict[tuple[str, str], Optional[CellTypeData]] = {}
    detail_rows: list[dict] = []

    for query_dataset in query_datasets:
        for db_dataset in db_datasets:
            if not include_self_dataset and query_dataset == db_dataset:
                continue
            if query_dataset not in stores:
                raise KeyError(f"Unknown query dataset: {query_dataset}")
            if db_dataset not in stores:
                raise KeyError(f"Unknown db dataset: {db_dataset}")

            if verbose:
                print(f"Running pair: {query_dataset} -> {db_dataset}")

            pair_rows = evaluate_dataset_pair(
                query_store=stores[query_dataset],
                db_store=stores[db_dataset],
                settings=settings,
                cell_type_filter=cell_type_filter,
                cache=cache,
                verbose=verbose,
            )
            detail_rows.extend(pair_rows)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            [
                "query_dataset",
                "db_dataset",
                "query_cell_type",
                "representation",
                "metric",
                "query_obs_id",
            ]
        ).reset_index(drop=True)

    summary_by_cell, summary_overall = summarize_retrieval(detail_df)
    return detail_df, summary_by_cell, summary_overall
