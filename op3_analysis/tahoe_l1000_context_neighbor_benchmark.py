from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .tahoe_l1000_benchmark_runner import (
    _aggregate_deg_summary,
    _aggregate_lfc_summary,
    _candidate_table_for_eval,
    _log,
    _outer_folds_for_rows,
    load_gene_subsets,
    load_prepared_inputs,
)
from .tahoe_l1000_deg_benchmark import (
    build_prior_score_vector,
    macro_f1_flat,
    normalize_pubchem_cids,
    paired_row_metric,
    predict_from_scores,
    summarize_continuous_metrics,
    summarize_prediction_metrics,
    tile_gene_labels,
    train_gene_majority_labels,
)


DEFAULT_CONTEXT_DEG_METHODS = (
    {"name": "context_majority_prior", "top_k": 0},
    {"name": "context_top1", "top_k": 1},
    {"name": "context_top3", "top_k": 3},
    {"name": "context_top5", "top_k": 5},
    {"name": "context_top10", "top_k": 10},
)
DEFAULT_CONTEXT_LFC_METHODS = (
    {"name": "context_train_mean", "top_k": 0},
    {"name": "context_top1", "top_k": 1},
    {"name": "context_top3", "top_k": 3},
    {"name": "context_top5", "top_k": 5},
    {"name": "context_top10", "top_k": 10},
)


def _normalize_context_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    normalized = []
    for value in values.to_numpy(dtype=float):
        if np.isfinite(value):
            normalized.append(f"{float(value):.6g}")
        else:
            normalized.append("<NA>")
    return pd.Series(normalized, index=series.index, dtype="string")


def make_context_key(
    cell_type: pd.Series,
    pert_time_h: pd.Series,
    pert_dose_uM: pd.Series,
) -> pd.Series:
    return (
        cell_type.astype("string").fillna("<NA>")
        + "|t="
        + _normalize_context_numeric(pert_time_h)
        + "|d="
        + _normalize_context_numeric(pert_dose_uM)
    )


def add_context_keys_to_tahoe_obs(tahoe_obs: pd.DataFrame) -> pd.DataFrame:
    out = tahoe_obs.copy()
    out["context_key"] = make_context_key(
        out["cell_type"],
        out["pert_time_h"],
        out["pert_dose_uM"],
    )
    return out


def add_context_keys_to_candidate_table(candidate_table: pd.DataFrame) -> pd.DataFrame:
    out = candidate_table.copy()
    out["query_context_key"] = make_context_key(
        out["query_cell_type"],
        out["pert_time_h_query"],
        out["pert_dose_uM_query"],
    )
    out["donor_context_key"] = make_context_key(
        out["donor_cell_type"],
        out["pert_time_h_donor"],
        out["pert_dose_uM_donor"],
    )
    return out


def estimate_context_transfer_scores_deg(
    candidate_table: pd.DataFrame,
    tahoe_labels: np.ndarray,
    donor_labels_by_dataset: dict[str, np.ndarray],
    train_query_rows: np.ndarray,
) -> pd.DataFrame:
    train_candidates = candidate_table.loc[candidate_table["query_row"].isin(train_query_rows)].copy()
    if train_candidates.empty:
        return pd.DataFrame(
            columns=["query_context_key", "db_dataset", "donor_context_key", "n_queries", "median_macro_f1"]
        )

    best = (
        train_candidates.sort_values(
            ["query_row", "db_dataset", "donor_context_key", "match_distance", "donor_row"],
            ignore_index=True,
        )
        .groupby(["query_row", "db_dataset", "donor_context_key"], as_index=False)
        .first()
    )

    rows: list[dict[str, object]] = []
    for item in best.itertuples(index=False):
        truth = tahoe_labels[int(item.query_row)]
        donor = donor_labels_by_dataset[str(item.db_dataset)][int(item.donor_row)]
        rows.append(
            {
                "query_context_key": str(item.query_context_key),
                "db_dataset": str(item.db_dataset),
                "donor_context_key": str(item.donor_context_key),
                "query_row": int(item.query_row),
                "macro_f1": macro_f1_flat(truth, donor),
            }
        )
    scored = pd.DataFrame(rows)
    return (
        scored.groupby(["query_context_key", "db_dataset", "donor_context_key"], as_index=False)
        .agg(
            n_queries=("query_row", "nunique"),
            median_macro_f1=("macro_f1", "median"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .sort_values(
            ["query_context_key", "median_macro_f1", "mean_macro_f1", "db_dataset", "donor_context_key"],
            ascending=[True, False, False, True, True],
            ignore_index=True,
        )
    )


def estimate_context_transfer_scores_lfc(
    candidate_table: pd.DataFrame,
    tahoe_matrix: np.ndarray,
    donor_matrix_by_dataset: dict[str, np.ndarray],
    train_query_rows: np.ndarray,
    metric: str = "pearson",
) -> pd.DataFrame:
    train_candidates = candidate_table.loc[candidate_table["query_row"].isin(train_query_rows)].copy()
    if train_candidates.empty:
        return pd.DataFrame(
            columns=["query_context_key", "db_dataset", "donor_context_key", "n_queries", f"median_{metric}"]
        )

    best = (
        train_candidates.sort_values(
            ["query_row", "db_dataset", "donor_context_key", "match_distance", "donor_row"],
            ignore_index=True,
        )
        .groupby(["query_row", "db_dataset", "donor_context_key"], as_index=False)
        .first()
    )

    rows: list[dict[str, object]] = []
    for item in best.itertuples(index=False):
        truth = tahoe_matrix[int(item.query_row)][None, :]
        donor = donor_matrix_by_dataset[str(item.db_dataset)][int(item.donor_row)][None, :]
        rows.append(
            {
                "query_context_key": str(item.query_context_key),
                "db_dataset": str(item.db_dataset),
                "donor_context_key": str(item.donor_context_key),
                "query_row": int(item.query_row),
                metric: float(paired_row_metric(truth, donor, metric=metric)[0]),
            }
        )
    scored = pd.DataFrame(rows)
    return (
        scored.groupby(["query_context_key", "db_dataset", "donor_context_key"], as_index=False)
        .agg(
            n_queries=("query_row", "nunique"),
            **{f"median_{metric}": (metric, "median"), f"mean_{metric}": (metric, "mean")},
        )
        .sort_values(
            ["query_context_key", f"median_{metric}", f"mean_{metric}", "db_dataset", "donor_context_key"],
            ascending=[True, False, False, True, True],
            ignore_index=True,
        )
    )


def _aggregate_donor_vectors(
    frame: pd.DataFrame,
    donor_matrix_by_dataset: dict[str, np.ndarray],
) -> np.ndarray:
    vectors = []
    for row in frame.itertuples(index=False):
        vectors.append(donor_matrix_by_dataset[str(row.db_dataset)][int(row.donor_row)])
    return np.vstack(vectors).mean(axis=0)


def _build_train_neighbor_pool(
    train_candidate_table: pd.DataFrame,
    donor_matrix_by_dataset: dict[str, np.ndarray],
    tahoe_target_matrix: np.ndarray,
) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    pools: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    group_cols = ["query_context_key", "db_dataset", "donor_context_key"]
    for key, frame in train_candidate_table.groupby(group_cols, sort=False):
        query_rows = []
        donor_vectors = []
        for query_row, row_frame in frame.groupby("query_row", sort=False):
            query_rows.append(int(query_row))
            donor_vectors.append(_aggregate_donor_vectors(row_frame, donor_matrix_by_dataset))
        rows_array = np.asarray(query_rows, dtype=np.int64)
        pools[(str(key[0]), str(key[1]), str(key[2]))] = {
            "query_rows": rows_array,
            "donor_matrix": np.vstack(donor_vectors).astype(np.float32),
            "tahoe_targets": tahoe_target_matrix[rows_array],
        }
    return pools


def _vector_similarity(
    query_vector: np.ndarray,
    candidate_matrix: np.ndarray,
    metric: str = "pearson",
) -> np.ndarray:
    query = np.asarray(query_vector, dtype=np.float64)
    matrix = np.asarray(candidate_matrix, dtype=np.float64)
    if metric == "cosine":
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(matrix, axis=1)
        denom = query_norm * candidate_norms
        scores = np.divide(
            matrix @ query,
            denom,
            out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
            where=denom > 0,
        )
        return scores
    if metric == "pearson":
        centered_query = query - query.mean()
        centered_matrix = matrix - matrix.mean(axis=1, keepdims=True)
        query_norm = np.linalg.norm(centered_query)
        candidate_norms = np.linalg.norm(centered_matrix, axis=1)
        denom = query_norm * candidate_norms
        scores = np.divide(
            centered_matrix @ centered_query,
            denom,
            out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
            where=denom > 0,
        )
        return scores
    raise ValueError(f"Unsupported similarity metric: {metric}")


def _context_train_rows_for_query(
    tahoe_obs: pd.DataFrame,
    train_rows: np.ndarray,
    query_context_key: str,
) -> np.ndarray:
    mask = tahoe_obs.iloc[train_rows]["context_key"].astype(str).eq(str(query_context_key)).to_numpy(dtype=bool)
    return train_rows[mask]


def _context_baseline_rows(
    tahoe_obs: pd.DataFrame,
    train_rows: np.ndarray,
    query_context_key: str,
) -> np.ndarray:
    context_rows = _context_train_rows_for_query(tahoe_obs, train_rows, query_context_key)
    return context_rows if context_rows.size > 0 else train_rows


def _history_lookup(
    context_transfer_scores: pd.DataFrame,
    query_context_key: str,
    history_column: str,
) -> dict[tuple[str, str], float]:
    if context_transfer_scores.empty:
        return {}
    subset = context_transfer_scores.loc[
        context_transfer_scores["query_context_key"].astype(str).eq(str(query_context_key))
    ]
    return {
        (str(row.db_dataset), str(row.donor_context_key)): float(getattr(row, history_column))
        for row in subset.itertuples(index=False)
    }


def _select_best_available_context(
    query_candidates: pd.DataFrame,
    query_context_key: str,
    history_scores: dict[tuple[str, str], float],
    train_neighbor_pools: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> tuple[str, str] | None:
    options = (
        query_candidates.groupby(["db_dataset", "donor_context_key"], as_index=False)
        .agg(min_match_distance=("match_distance", "min"))
        .sort_values(["db_dataset", "donor_context_key"], ignore_index=True)
    )
    rows = []
    for item in options.itertuples(index=False):
        db_dataset = str(item.db_dataset)
        donor_context_key = str(item.donor_context_key)
        pool_key = (str(query_context_key), db_dataset, donor_context_key)
        if pool_key not in train_neighbor_pools:
            continue
        rows.append(
            {
                "db_dataset": db_dataset,
                "donor_context_key": donor_context_key,
                "history_score": float(history_scores.get((db_dataset, donor_context_key), 0.0)),
                "min_match_distance": float(item.min_match_distance),
                "n_train_rows": int(train_neighbor_pools[pool_key]["query_rows"].size),
            }
        )
    if not rows:
        return None
    ranked = pd.DataFrame(rows).sort_values(
        ["history_score", "n_train_rows", "min_match_distance", "db_dataset", "donor_context_key"],
        ascending=[False, False, True, True, True],
        ignore_index=True,
    )
    return str(ranked.iloc[0]["db_dataset"]), str(ranked.iloc[0]["donor_context_key"])


def _query_donor_vector_for_context(
    query_candidates: pd.DataFrame,
    chosen_dataset: str,
    chosen_donor_context_key: str,
    donor_matrix_by_dataset: dict[str, np.ndarray],
) -> np.ndarray:
    subset = query_candidates.loc[
        query_candidates["db_dataset"].astype(str).eq(str(chosen_dataset))
        & query_candidates["donor_context_key"].astype(str).eq(str(chosen_donor_context_key))
    ]
    if subset.empty:
        raise RuntimeError("Expected at least one donor row in the chosen context.")
    return _aggregate_donor_vectors(subset, donor_matrix_by_dataset).astype(np.float32)


def _weighted_neighbor_average(
    candidate_targets: np.ndarray,
    order: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> np.ndarray:
    chosen = order[: int(min(top_k, order.size))]
    weights = np.nan_to_num(scores[chosen], nan=0.0)
    weights = np.maximum(weights, 0.0)
    if not np.any(weights > 0):
        weights = np.ones(chosen.size, dtype=np.float64)
    return np.average(candidate_targets[chosen], axis=0, weights=weights).astype(np.float32)


def run_context_neighbor_deg_benchmarks(
    prep_dir: Path | str,
    results_dir: Path | str,
    label_layers: Iterable[str] | None = None,
    subset_names: Iterable[str] | None = None,
    fixed_threshold: float = 1.0,
    n_outer_splits: int = 5,
    seed: int = 0,
    similarity_metric: str = "pearson",
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    start = time.time()
    prep_root = Path(prep_dir)
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    tahoe_adata, donor_adatas, tahoe_obs = load_prepared_inputs(prep_root)
    tahoe_obs = add_context_keys_to_tahoe_obs(tahoe_obs)
    candidate_table, candidate_lookup, valid_rows = _candidate_table_for_eval(tahoe_adata, donor_adatas)
    candidate_table = add_context_keys_to_candidate_table(candidate_table)
    candidate_lookup = {
        int(query_row): frame.reset_index(drop=True)
        for query_row, frame in candidate_table.groupby("query_row", sort=False)
    }
    outer_folds = _outer_folds_for_rows(
        compounds=tahoe_obs.iloc[valid_rows]["pubchem_cid"],
        valid_rows=valid_rows,
        n_splits=n_outer_splits,
        seed=seed,
    )
    subsets = load_gene_subsets(prep_root, tahoe_adata.var_names, subset_names=subset_names)
    label_config_df = pd.read_csv(prep_root / "label_summary.csv")[["label_layer"]]
    available_label_layers = label_config_df["label_layer"].drop_duplicates().tolist()
    selected_label_layers = list(label_layers) if label_layers is not None else available_label_layers

    fold_rows: list[dict[str, object]] = []
    transfer_rows: list[pd.DataFrame] = []
    _log(verbose, "context-deg outer folds", len(outer_folds), "valid queries", len(valid_rows))

    for label_layer in selected_label_layers:
        tahoe_labels_all = np.asarray(tahoe_adata.layers[label_layer], dtype=np.int8)
        tahoe_scores_all = np.asarray(tahoe_adata.layers["signed_score.shared"], dtype=np.float32)
        donor_labels_all = {name: np.asarray(adata.layers[label_layer], dtype=np.int8) for name, adata in donor_adatas.items()}
        donor_scores_all = {
            name: np.asarray(adata.layers["signed_score.shared"], dtype=np.float32)
            for name, adata in donor_adatas.items()
        }

        for subset_name, gene_idx in subsets.items():
            gene_idx = np.asarray(gene_idx, dtype=np.int64)
            if gene_idx.size == 0:
                continue
            tahoe_labels = tahoe_labels_all[:, gene_idx]
            tahoe_scores = tahoe_scores_all[:, gene_idx]
            donor_labels = {name: matrix[:, gene_idx] for name, matrix in donor_labels_all.items()}
            donor_scores = {name: matrix[:, gene_idx] for name, matrix in donor_scores_all.items()}
            _log(verbose, "context-deg label", label_layer, "subset", subset_name, "genes", gene_idx.size)

            for fold_idx, (train_rows, test_rows) in enumerate(outer_folds):
                _log(verbose, "context-deg fold", fold_idx, "train", len(train_rows), "test", len(test_rows))
                train_candidate_table = candidate_table.loc[candidate_table["query_row"].isin(train_rows)].copy()
                context_transfer = estimate_context_transfer_scores_deg(
                    candidate_table=candidate_table,
                    tahoe_labels=tahoe_labels,
                    donor_labels_by_dataset=donor_labels,
                    train_query_rows=train_rows,
                )
                if not context_transfer.empty:
                    context_transfer = context_transfer.copy()
                    context_transfer["fold"] = fold_idx
                    context_transfer["label_layer"] = label_layer
                    context_transfer["subset_name"] = subset_name
                    transfer_rows.append(context_transfer)
                train_neighbor_pools = _build_train_neighbor_pool(
                    train_candidate_table=train_candidate_table,
                    donor_matrix_by_dataset=donor_scores,
                    tahoe_target_matrix=tahoe_scores,
                )
                fold_predictions = []
                for query_row in test_rows:
                    query_context_key = str(tahoe_obs.iloc[int(query_row)]["context_key"])
                    context_train_rows = _context_baseline_rows(tahoe_obs, train_rows, query_context_key)
                    majority_labels = train_gene_majority_labels(tahoe_labels[context_train_rows])
                    history_scores = _history_lookup(
                        context_transfer_scores=context_transfer,
                        query_context_key=query_context_key,
                        history_column="median_macro_f1",
                    )
                    query_candidates = candidate_lookup.get(int(query_row), pd.DataFrame())
                    chosen_context = _select_best_available_context(
                        query_candidates=query_candidates,
                        query_context_key=query_context_key,
                        history_scores=history_scores,
                        train_neighbor_pools=train_neighbor_pools,
                    )

                    method_outputs: dict[str, np.ndarray] = {
                        "context_majority_prior": tile_gene_labels(majority_labels, 1)[0],
                    }
                    if chosen_context is not None:
                        chosen_dataset, chosen_donor_context_key = chosen_context
                        query_vector = _query_donor_vector_for_context(
                            query_candidates=query_candidates,
                            chosen_dataset=chosen_dataset,
                            chosen_donor_context_key=chosen_donor_context_key,
                            donor_matrix_by_dataset=donor_scores,
                        )
                        pool = train_neighbor_pools[(query_context_key, chosen_dataset, chosen_donor_context_key)]
                        similarity_scores = _vector_similarity(
                            query_vector,
                            pool["donor_matrix"],
                            metric=similarity_metric,
                        )
                        similarity_order = np.argsort(np.nan_to_num(similarity_scores, nan=-np.inf))[::-1]
                        for method_spec in DEFAULT_CONTEXT_DEG_METHODS:
                            top_k = int(method_spec["top_k"])
                            method_name = str(method_spec["name"])
                            if top_k < 1:
                                continue
                            predicted_score = _weighted_neighbor_average(
                                candidate_targets=pool["tahoe_targets"],
                                order=similarity_order,
                                scores=similarity_scores,
                                top_k=top_k,
                            )
                            method_outputs[method_name] = predict_from_scores(
                                predicted_score[None, :],
                                threshold=float(fixed_threshold),
                            )[0]
                    for method_spec in DEFAULT_CONTEXT_DEG_METHODS:
                        method_name = str(method_spec["name"])
                        if method_name not in method_outputs:
                            method_outputs[method_name] = tile_gene_labels(majority_labels, 1)[0]
                        fold_predictions.append(
                            {
                                "query_row": int(query_row),
                                "method_name": method_name,
                                "y_pred": method_outputs[method_name],
                            }
                        )

                for method_spec in DEFAULT_CONTEXT_DEG_METHODS:
                    method_name = str(method_spec["name"])
                    method_rows = [row for row in fold_predictions if row["method_name"] == method_name]
                    ordered_rows = np.asarray([row["query_row"] for row in method_rows], dtype=np.int64)
                    y_true = tahoe_labels[ordered_rows]
                    y_pred = np.vstack([row["y_pred"] for row in method_rows]).astype(np.int8)
                    metrics = summarize_prediction_metrics(y_true, y_pred)
                    metrics.update(
                        {
                            "fold": fold_idx,
                            "label_layer": label_layer,
                            "subset_name": subset_name,
                            "method_name": method_name,
                            "tuned_threshold": float(fixed_threshold) if method_name != "context_majority_prior" else np.nan,
                            "train_samples": int(len(train_rows)),
                            "test_samples": int(len(test_rows)),
                            "train_compounds": int(
                                normalize_pubchem_cids(tahoe_obs.iloc[train_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                            ),
                            "test_compounds": int(
                                normalize_pubchem_cids(tahoe_obs.iloc[test_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                            ),
                            "similarity_metric": similarity_metric,
                        }
                    )
                    fold_rows.append(metrics)
                _log(verbose, "context-deg fold complete", fold_idx)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _aggregate_deg_summary(fold_metrics)
    transfer_scores = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame()
    fold_metrics.to_csv(results_root / "deg_context_neighbor_fold_metrics.csv", index=False)
    summary.to_csv(results_root / "deg_context_neighbor_summary.csv", index=False)
    if not transfer_scores.empty:
        transfer_scores.to_csv(results_root / "deg_context_transfer_scores.csv", index=False)
    metadata = {
        "runtime_seconds": round(time.time() - start, 3),
        "n_outer_folds": len(outer_folds),
        "fixed_threshold": float(fixed_threshold),
        "label_layers": selected_label_layers,
        "subset_names": list(subsets),
        "methods": [row["name"] for row in DEFAULT_CONTEXT_DEG_METHODS],
        "similarity_metric": similarity_metric,
    }
    (results_root / "deg_context_neighbor_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(verbose, "finished context-deg benchmark in seconds", metadata["runtime_seconds"])
    return {"fold_metrics": fold_metrics, "summary": summary, "transfer_scores": transfer_scores}


def run_context_neighbor_lfc_benchmarks(
    prep_dir: Path | str,
    results_dir: Path | str,
    subset_names: Iterable[str] | None = None,
    n_outer_splits: int = 5,
    seed: int = 0,
    similarity_metric: str = "pearson",
    top_k_overlap: int = 50,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    start = time.time()
    prep_root = Path(prep_dir)
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    tahoe_adata, donor_adatas, tahoe_obs = load_prepared_inputs(prep_root)
    tahoe_obs = add_context_keys_to_tahoe_obs(tahoe_obs)
    candidate_table, candidate_lookup, valid_rows = _candidate_table_for_eval(tahoe_adata, donor_adatas)
    candidate_table = add_context_keys_to_candidate_table(candidate_table)
    candidate_lookup = {
        int(query_row): frame.reset_index(drop=True)
        for query_row, frame in candidate_table.groupby("query_row", sort=False)
    }
    outer_folds = _outer_folds_for_rows(
        compounds=tahoe_obs.iloc[valid_rows]["pubchem_cid"],
        valid_rows=valid_rows,
        n_splits=n_outer_splits,
        seed=seed,
    )
    subsets = load_gene_subsets(prep_root, tahoe_adata.var_names, subset_names=subset_names)
    tahoe_logfc_all = np.asarray(tahoe_adata.layers["logFC"], dtype=np.float32)
    donor_logfc_all = {name: np.asarray(adata.layers["logFC"], dtype=np.float32) for name, adata in donor_adatas.items()}

    fold_rows: list[dict[str, object]] = []
    transfer_rows: list[pd.DataFrame] = []
    _log(verbose, "context-lfc outer folds", len(outer_folds), "valid queries", len(valid_rows))

    for subset_name, gene_idx in subsets.items():
        gene_idx = np.asarray(gene_idx, dtype=np.int64)
        if gene_idx.size == 0:
            continue
        tahoe_logfc = tahoe_logfc_all[:, gene_idx]
        donor_logfc = {name: matrix[:, gene_idx] for name, matrix in donor_logfc_all.items()}
        _log(verbose, "context-lfc subset", subset_name, "genes", gene_idx.size)

        for fold_idx, (train_rows, test_rows) in enumerate(outer_folds):
            _log(verbose, "context-lfc fold", fold_idx, "train", len(train_rows), "test", len(test_rows))
            train_candidate_table = candidate_table.loc[candidate_table["query_row"].isin(train_rows)].copy()
            context_transfer = estimate_context_transfer_scores_lfc(
                candidate_table=candidate_table,
                tahoe_matrix=tahoe_logfc,
                donor_matrix_by_dataset=donor_logfc,
                train_query_rows=train_rows,
                metric="pearson",
            )
            if not context_transfer.empty:
                context_transfer = context_transfer.copy()
                context_transfer["fold"] = fold_idx
                context_transfer["subset_name"] = subset_name
                transfer_rows.append(context_transfer)
            train_neighbor_pools = _build_train_neighbor_pool(
                train_candidate_table=train_candidate_table,
                donor_matrix_by_dataset=donor_logfc,
                tahoe_target_matrix=tahoe_logfc,
            )
            fold_predictions = []
            for query_row in test_rows:
                query_context_key = str(tahoe_obs.iloc[int(query_row)]["context_key"])
                context_train_rows = _context_baseline_rows(tahoe_obs, train_rows, query_context_key)
                train_mean_vector = tahoe_logfc[context_train_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
                history_scores = _history_lookup(
                    context_transfer_scores=context_transfer,
                    query_context_key=query_context_key,
                    history_column="median_pearson",
                )
                query_candidates = candidate_lookup.get(int(query_row), pd.DataFrame())
                chosen_context = _select_best_available_context(
                    query_candidates=query_candidates,
                    query_context_key=query_context_key,
                    history_scores=history_scores,
                    train_neighbor_pools=train_neighbor_pools,
                )

                method_outputs: dict[str, np.ndarray] = {
                    "context_train_mean": train_mean_vector,
                }
                if chosen_context is not None:
                    chosen_dataset, chosen_donor_context_key = chosen_context
                    query_vector = _query_donor_vector_for_context(
                        query_candidates=query_candidates,
                        chosen_dataset=chosen_dataset,
                        chosen_donor_context_key=chosen_donor_context_key,
                        donor_matrix_by_dataset=donor_logfc,
                    )
                    pool = train_neighbor_pools[(query_context_key, chosen_dataset, chosen_donor_context_key)]
                    similarity_scores = _vector_similarity(
                        query_vector,
                        pool["donor_matrix"],
                        metric=similarity_metric,
                    )
                    similarity_order = np.argsort(np.nan_to_num(similarity_scores, nan=-np.inf))[::-1]
                    for method_spec in DEFAULT_CONTEXT_LFC_METHODS:
                        top_k = int(method_spec["top_k"])
                        method_name = str(method_spec["name"])
                        if top_k < 1:
                            continue
                        method_outputs[method_name] = _weighted_neighbor_average(
                            candidate_targets=pool["tahoe_targets"],
                            order=similarity_order,
                            scores=similarity_scores,
                            top_k=top_k,
                        )
                for method_spec in DEFAULT_CONTEXT_LFC_METHODS:
                    method_name = str(method_spec["name"])
                    if method_name not in method_outputs:
                        method_outputs[method_name] = train_mean_vector
                    fold_predictions.append(
                        {
                            "query_row": int(query_row),
                            "method_name": method_name,
                            "y_pred": method_outputs[method_name],
                        }
                    )

            for method_spec in DEFAULT_CONTEXT_LFC_METHODS:
                method_name = str(method_spec["name"])
                method_rows = [row for row in fold_predictions if row["method_name"] == method_name]
                ordered_rows = np.asarray([row["query_row"] for row in method_rows], dtype=np.int64)
                y_true = tahoe_logfc[ordered_rows]
                y_pred = np.vstack([row["y_pred"] for row in method_rows]).astype(np.float32)
                metrics = summarize_continuous_metrics(y_true, y_pred, top_k=top_k_overlap)
                metrics.update(
                    {
                        "fold": fold_idx,
                        "subset_name": subset_name,
                        "method_name": method_name,
                        "train_samples": int(len(train_rows)),
                        "test_samples": int(len(test_rows)),
                        "train_compounds": int(
                            normalize_pubchem_cids(tahoe_obs.iloc[train_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                        ),
                        "test_compounds": int(
                            normalize_pubchem_cids(tahoe_obs.iloc[test_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                        ),
                        "similarity_metric": similarity_metric,
                    }
                )
                fold_rows.append(metrics)
            _log(verbose, "context-lfc fold complete", fold_idx)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _aggregate_lfc_summary(fold_metrics)
    transfer_scores = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame()
    fold_metrics.to_csv(results_root / "lfc_context_neighbor_fold_metrics.csv", index=False)
    summary.to_csv(results_root / "lfc_context_neighbor_summary.csv", index=False)
    if not transfer_scores.empty:
        transfer_scores.to_csv(results_root / "lfc_context_transfer_scores.csv", index=False)
    metadata = {
        "runtime_seconds": round(time.time() - start, 3),
        "n_outer_folds": len(outer_folds),
        "subset_names": list(subsets),
        "methods": [row["name"] for row in DEFAULT_CONTEXT_LFC_METHODS],
        "similarity_metric": similarity_metric,
        "top_k_overlap": int(top_k_overlap),
    }
    (results_root / "lfc_context_neighbor_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(verbose, "finished context-lfc benchmark in seconds", metadata["runtime_seconds"])
    return {"fold_metrics": fold_metrics, "summary": summary, "transfer_scores": transfer_scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Tahoe-neighbor benchmarks where L1000 is used only to select same-context Tahoe training neighbors."
    )
    parser.add_argument(
        "--mode",
        choices=("deg", "lfc", "both"),
        default="both",
        help="Which benchmark family to run.",
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("data/tahoe_l1000_benchmark_prep"),
        help="Prepared shared-gene benchmark directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tahoe_l1000_context_neighbor_benchmarks"),
        help="Output directory for the context-neighbor benchmark tables.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for compound-held-out folds.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args(argv)

    if args.mode in {"deg", "both"}:
        run_context_neighbor_deg_benchmarks(
            prep_dir=args.prep_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            verbose=args.verbose,
        )
    if args.mode in {"lfc", "both"}:
        run_context_neighbor_lfc_benchmarks(
            prep_dir=args.prep_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
