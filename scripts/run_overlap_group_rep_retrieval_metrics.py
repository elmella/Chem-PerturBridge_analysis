#!/usr/bin/env python3
"""Parallel, resumable cross-source retrieval scoring for Table 9."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cross_source_core
from scripts.cross_source_parallel import (
    TaskSpec,
    add_computation_arguments,
    add_common_arguments,
    diagnostic_frame,
    file_record,
    make_worker_catalog,
    make_worker_w4_catalog,
    print_computations,
    read_overlap_metadata,
    read_task_input,
    report_task_progress,
    resolve_computations,
    run_analysis,
    selected_w4_scale_variants,
)
from scripts.cross_source_scoring import (
    ADJ_PVALUE_LAYER_PREFERENCES,
    RETRIEVAL_SIMILARITIES,
    RETRIEVAL_VARIANTS,
    build_logfc_pool_matrices,
    build_pool_matrices,
    dose_aware_positive_mask,
    exact_cross_assay_null,
    exact_null_mid_p,
    exact_null_rank_calibration,
    expected_single_signature_metrics,
    pool_frame_for_context,
    primary_positive_mask,
    raw_dose_fold_differences,
    retrieval_peer_summary,
    score_vector_against_matrix,
    score_vector_pair,
    select_peer_indices,
    signed_significance,
    similarity_matrix,
    strict_positive_maps,
    strict_positive_mask,
    summarize_retrieval_scores,
)
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
)


ANALYSIS = "retrieval"
FINAL_METRICS_NAME = "retrieval_scored_metrics.tsv"
CONTEXT_COLUMNS = ("dataset_a", "dataset_b", "cell_type", "time_key")
MIN_TARGET_CANDIDATES = 5
MIN_UNIQUE_COMPOUNDS_PER_SIDE = 2
RAW_SCALE_VARIANT = "raw"
FULL_WORKLOAD = "full"
REVIEWER_MINIMAL_WORKLOAD = "reviewer-minimal"
PEER_ONLY_WORKLOAD = "peer-only"
SELECTED_WORKLOAD = "selected"
REVIEWER_MINIMAL_SIMILARITIES = ("cosine", "spearman")
REVIEWER_MINIMAL_SCALE_VARIANTS = (PER_GENE_DATASET_VARIANT,)
PEER_ONLY_FINAL_METRICS_NAME = "retrieval_peer_sensitivity_metrics.tsv"
COMPUTATIONS = {
    "legacy-l2": (
        "Original raw L2 bundle across logFC, moderated-t, and signed-"
        "significance representations and all retrieval variants."
    ),
    "raw-l2": (
        "Raw strict-logFC L2 retrieval, Recall@1, AUROC, and baselines."
    ),
    "raw-cosine": (
        "Raw strict-logFC cosine retrieval, Recall@1, AUROC, and baselines."
    ),
    "raw-spearman": (
        "Raw strict-logFC Spearman retrieval, Recall@1, AUROC, and baselines."
    ),
    "w4-dataset-l2": (
        "Dataset-wide W4 strict-logFC L2 retrieval and baselines."
    ),
    "w4-dataset-cosine": (
        "Dataset-wide W4 strict-logFC cosine retrieval and baselines."
    ),
    "w4-dataset-spearman": (
        "Dataset-wide W4 strict-logFC Spearman retrieval and baselines."
    ),
    "w4-dataset-cell-type-l2": (
        "Dataset-by-cell-type W4 strict-logFC L2 retrieval and baselines."
    ),
    "w4-dataset-cell-type-cosine": (
        "Dataset-by-cell-type W4 strict-logFC cosine retrieval and baselines."
    ),
    "w4-dataset-cell-type-spearman": (
        "Dataset-by-cell-type W4 strict-logFC Spearman retrieval and baselines."
    ),
}
COMPUTATION_SPEC = {
    "raw-l2": (RAW_SCALE_VARIANT, "negative_l2"),
    "raw-cosine": (RAW_SCALE_VARIANT, "cosine"),
    "raw-spearman": (RAW_SCALE_VARIANT, "spearman"),
    "w4-dataset-l2": (PER_GENE_DATASET_VARIANT, "negative_l2"),
    "w4-dataset-cosine": (PER_GENE_DATASET_VARIANT, "cosine"),
    "w4-dataset-spearman": (PER_GENE_DATASET_VARIANT, "spearman"),
    "w4-dataset-cell-type-l2": (
        PER_GENE_DATASET_CELL_TYPE_VARIANT,
        "negative_l2",
    ),
    "w4-dataset-cell-type-cosine": (
        PER_GENE_DATASET_CELL_TYPE_VARIANT,
        "cosine",
    ),
    "w4-dataset-cell-type-spearman": (
        PER_GENE_DATASET_CELL_TYPE_VARIANT,
        "spearman",
    ),
}

IDENTITY_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "direction",
    "query_dataset",
    "target_dataset",
    "cell_type",
    "time_key",
    "query_obs_id",
    "query_pubchem_cid",
    "query_dose_key",
    "query_dose_uM",
    "representation",
    "retrieval_variant",
    "similarity_metric",
    "scale_variant",
]
PEER_ONLY_REQUIRED_COLUMNS = (
    *IDENTITY_COLUMNS,
    "best_target_dose_key",
    "observed_best_positive_similarity",
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
)
PEER_ONLY_OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    "peer_cap_order",
    "peer_cap_label",
    "peer_cap",
    "peer_reference_cap",
    "peer_sampling_seed",
    "source_individual_total_count",
    "source_individual_scored_count",
    "target_individual_total_count",
    "target_individual_scored_count",
    "best_target_dose_key",
    "observed_best_positive_similarity",
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
    *[
        f"{side}_individual_{suffix}"
        for side in ("source", "target")
        for suffix in (
            "n_peers",
            "mean_similarity",
            "sd_similarity",
            "n_below_observed",
            "fraction_below_observed",
            "corrected_percentile",
            "n_at_least_observed",
            "empirical_p_upper",
        )
    ],
    *[
        f"{side}_peer_{field}"
        for side in ("source", "target")
        for field in (
            "best_rank",
            "normalized_rank",
            "recall_at_1",
            "auroc",
        )
    ],
    *[
        f"delta_vs_{side}_peer_{field}"
        for side in ("source", "target")
        for field in ("normalized_rank", "recall_at_1", "auroc")
    ],
]
COUNT_COLUMNS = [
    "target_pool_size",
    "target_pool_size_before_similarity_filter",
    "n_query_side_conditions",
    "n_target_side_conditions",
    "n_query_side_unique_compounds",
    "n_target_side_unique_compounds",
    "n_shared_genes",
    "n_raw_shared_genes",
    "n_positive_candidates",
    "n_negative_candidates",
    "n_zero_norm_target_candidates",
    "query_baseline_peer_count",
    "centroid_baseline_peer_count",
    "single_signature_baseline_peer_count",
    "n_excluded_single_signature_peers",
    "source_centroid_peer_count",
    "target_centroid_peer_count",
    "source_individual_total_count",
    "source_individual_scored_count",
    "target_individual_total_count",
    "target_individual_scored_count",
    "left_population_row_count",
    "right_population_row_count",
]
METRIC_COLUMNS = [
    "observed_best_positive_rank",
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
    "best_positive_rank",
    "normalized_best_positive_rank",
    "recall_at_1",
    "auroc",
    "observed_best_positive_similarity",
    "observed_mean_positive_similarity",
    "observed_sd_positive_similarity",
    "best_target_obs_id",
    "best_target_dose_key",
    "best_target_dose_uM",
    "best_target_dose_fold_difference",
    "source_centroid_similarity",
    "baseline_candidate_similarity",
    "baseline_available",
    "baseline_best_positive_rank",
    "baseline_normalized_best_positive_rank",
    "baseline_recall_at_1",
    "baseline_auroc",
    "centroid_baseline_candidate_similarity",
    "centroid_baseline_best_positive_rank",
    "centroid_baseline_normalized_best_positive_rank",
    "centroid_baseline_recall_at_1",
    "centroid_baseline_auroc",
    "single_signature_baseline_best_positive_rank",
    "single_signature_baseline_normalized_best_positive_rank",
    "single_signature_baseline_recall_at_1",
    "single_signature_baseline_auroc",
    "delta_vs_baseline_normalized_best_positive_rank",
    "delta_vs_baseline_recall_at_1",
    "delta_vs_baseline_auroc",
    "random_normalized_best_positive_rank",
    "random_recall_at_1",
    "random_auroc",
    "target_centroid_similarity",
    "target_decoy_null_normalized_best_positive_rank",
    "target_decoy_null_best_positive_rank",
    "target_decoy_null_recall_at_1",
    "target_decoy_null_auroc",
    "exact_null_mid_p",
    "null_pit",
    "null_p_value",
    "min_achievable_null_p_value",
    "null_pit_below_half",
    "observed_auroc_minus_half",
    "within_source_best_positive_rank",
    "within_source_normalized_best_positive_rank",
    "within_source_recall_at_1",
    "within_source_auroc",
    "source_centroid_best_rank",
    "source_centroid_normalized_rank",
    "source_centroid_recall_at_1",
    "source_centroid_auroc",
    "target_centroid_best_rank",
    "target_centroid_normalized_rank",
    "target_centroid_recall_at_1",
    "target_centroid_auroc",
    "source_peer_best_rank",
    "source_peer_normalized_rank",
    "source_peer_recall_at_1",
    "source_peer_auroc",
    "target_peer_best_rank",
    "target_peer_normalized_rank",
    "target_peer_recall_at_1",
    "target_peer_auroc",
    "within_dataset_positive_control_best_positive_rank",
    "within_dataset_positive_control_normalized_best_positive_rank",
    "within_dataset_positive_control_recall_at_1",
    "within_dataset_positive_control_auroc",
    "delta_observed_vs_source_centroid",
    "delta_observed_vs_target_centroid",
    "delta_observed_vs_target_decoy_null_rank",
    "delta_observed_vs_target_decoy_null_recall_at_1",
    "delta_observed_vs_target_decoy_null_auroc",
    "delta_vs_target_decoy_null_normalized_best_positive_rank",
    "delta_vs_target_decoy_null_recall_at_1",
    "delta_vs_target_decoy_null_auroc",
    "delta_vs_random_normalized_best_positive_rank",
    "delta_vs_centroid_baseline_normalized_best_positive_rank",
    "delta_vs_centroid_baseline_recall_at_1",
    "delta_vs_centroid_baseline_auroc",
    "delta_vs_single_signature_baseline_normalized_best_positive_rank",
    "delta_vs_single_signature_baseline_recall_at_1",
    "delta_vs_single_signature_baseline_auroc",
    "delta_vs_within_dataset_positive_control_normalized_best_positive_rank",
    *[
        f"delta_vs_{variant}_{field}"
        for variant in (
            "source_centroid",
            "target_centroid",
            "source_peer",
            "target_peer",
            "within_dataset_positive_control",
        )
        for field in ("normalized_rank", "recall_at_1", "auroc")
    ],
]
PEER_COLUMNS = [
    f"{prefix}_{field}"
    for prefix in ("source_individual", "target_individual")
    for field in (
        "n_peers",
        "mean_similarity",
        "sd_similarity",
        "n_below_observed",
        "fraction_below_observed",
        "corrected_percentile",
        "n_at_least_observed",
        "empirical_p_upper",
    )
]
DELTA_PEER_COLUMNS = [
    "delta_observed_vs_source_individual_mean",
    "delta_observed_vs_target_individual_mean",
]
RETRIEVAL_OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    *COUNT_COLUMNS,
    *METRIC_COLUMNS,
    *PEER_COLUMNS,
    *DELTA_PEER_COLUMNS,
]


def _adj_layer(source: cross_source_core.LineSource) -> str:
    return cross_source_core.first_available_layer(
        source,
        ADJ_PVALUE_LAYER_PREFERENCES,
    )


def _record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record[column])
        for column in (
            "dataset_a",
            "dataset_b",
            "direction",
            "cell_type",
            "time_key",
            "query_obs_id",
            "representation",
            "retrieval_variant",
            "similarity_metric",
            "scale_variant",
        )
    )


def parse_peer_caps(value: str) -> tuple[Optional[int], ...]:
    caps: list[Optional[int]] = []
    seen: set[str] = set()
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "all":
            cap = None
            label = "all"
        else:
            try:
                cap = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid peer cap {raw!r}; use positive integers or 'all'"
                ) from exc
            if cap < 1:
                raise ValueError(
                    f"Invalid peer cap {raw!r}; use positive integers or 'all'"
                )
            label = str(cap)
        if label in seen:
            continue
        seen.add(label)
        caps.append(cap)
    if not caps:
        raise ValueError("--peer-caps must contain at least one cap")
    return tuple(caps)


def parse_peer_scales(value: str) -> tuple[str, ...]:
    mapping = {
        "raw": RAW_SCALE_VARIANT,
        "dataset": PER_GENE_DATASET_VARIANT,
    }
    scales: list[str] = []
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        try:
            scale = mapping[token]
        except KeyError as exc:
            raise ValueError(
                "--peer-scales supports only 'raw' and 'dataset'"
            ) from exc
        if scale not in scales:
            scales.append(scale)
    if not scales:
        raise ValueError("--peer-scales must select at least one scale")
    return tuple(scales)


def nested_peer_indices(
    n_peers: int,
    caps: Sequence[Optional[int]],
    *,
    reference_cap: int,
    seed_key: str,
    sampling_seed: int,
) -> dict[str, np.ndarray]:
    """Return nested selections while exactly retaining the production cap."""
    n_peers = int(n_peers)
    reference_cap = int(reference_cap)
    if reference_cap < 1:
        raise ValueError("reference_cap must be positive")
    numeric_caps = [int(cap) for cap in caps if cap is not None]
    if any(cap < reference_cap for cap in numeric_caps):
        raise ValueError(
            "Every numeric peer sensitivity cap must be at least "
            f"--peer-reference-cap={reference_cap}"
        )
    anchor = select_peer_indices(
        n_peers,
        reference_cap,
        seed_key,
        sampling_seed=sampling_seed,
    )
    if n_peers <= len(anchor):
        ordered = anchor
    else:
        remaining = np.setdiff1d(
            np.arange(n_peers, dtype=np.int64),
            anchor,
            assume_unique=True,
        )
        digest = hashlib.blake2b(
            (
                f"{int(sampling_seed)}|{seed_key}|"
                "peer-sensitivity-extension"
            ).encode("utf-8"),
            digest_size=8,
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        ordered = np.concatenate([anchor, rng.permutation(remaining)])
    selections: dict[str, np.ndarray] = {}
    for cap in caps:
        label = "all" if cap is None else str(int(cap))
        count = n_peers if cap is None else min(n_peers, int(cap))
        selections[label] = np.sort(ordered[:count]).astype(np.int64)
    return selections


def _load_peer_only_task_frame(
    scope,
    *,
    observed_metrics_path: Path,
    scales: Sequence[str],
) -> pd.DataFrame:
    path = Path(observed_metrics_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Peer-only retrieval input does not exist: {path}"
        )
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    missing = sorted(set(PEER_ONLY_REQUIRED_COLUMNS) - set(header))
    if missing:
        raise KeyError(
            f"Peer-only retrieval input is missing columns: {missing}"
        )
    string_columns = {
        column: "string"
        for column in PEER_ONLY_REQUIRED_COLUMNS
        if column in {
            *IDENTITY_COLUMNS,
            "best_target_dose_key",
        }
    }
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=list(PEER_ONLY_REQUIRED_COLUMNS),
        dtype=string_columns,
        low_memory=False,
    )
    frame = frame.loc[
        (frame["representation"].astype(str) == "logFC")
        & (
            frame["retrieval_variant"].astype(str)
            == "strict_matched_condition"
        )
        & frame["similarity_metric"].astype(str).isin(
            REVIEWER_MINIMAL_SIMILARITIES
        )
        & frame["scale_variant"].astype(str).isin(scales)
    ].copy()
    allowed_pairs = {
        (str(row.dataset_a), str(row.dataset_b))
        for row in scope.matched_pairs[
            ["dataset_a", "dataset_b"]
        ].drop_duplicates().itertuples(index=False)
    }
    frame = frame.loc[
        [
            (str(a), str(b)) in allowed_pairs
            for a, b in frame[["dataset_a", "dataset_b"]].itertuples(
                index=False,
                name=None,
            )
        ]
    ].copy()
    if frame.empty:
        raise ValueError(
            "No strict-logFC cosine/Spearman records matched the selected "
            "datasets and peer scales"
        )
    missing_scales = sorted(set(scales) - set(frame["scale_variant"].astype(str)))
    if missing_scales:
        raise ValueError(
            f"Peer-only input has no records for scales: {missing_scales}"
        )
    key_columns = [
        "dataset_a",
        "dataset_b",
        "direction",
        "cell_type",
        "time_key",
        "query_obs_id",
        "representation",
        "retrieval_variant",
        "similarity_metric",
        "scale_variant",
    ]
    if frame.duplicated(key_columns).any():
        raise ValueError("Peer-only retrieval input contains duplicate identities")
    return frame.sort_values(key_columns).reset_index(drop=True)


def _base_record(
    *,
    dataset_a: str,
    dataset_b: str,
    direction: str,
    query_dataset: str,
    target_dataset: str,
    cell_type: str,
    time_key: str,
    query_pool: pd.DataFrame,
    query_idx: int,
    target_pool: pd.DataFrame,
    representation: str,
    retrieval_variant: str,
    similarity_metric: str,
    scale_variant: str,
    n_shared_genes: int,
) -> dict[str, Any]:
    return {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "direction": direction,
        "query_dataset": query_dataset,
        "target_dataset": target_dataset,
        "cell_type": cell_type,
        "time_key": time_key,
        "query_obs_id": str(query_pool.loc[query_idx, "obs_id"]),
        "query_pubchem_cid": str(query_pool.loc[query_idx, "pubchem_cid"]),
        "query_dose_key": str(query_pool.loc[query_idx, "dose_key"]),
        "query_dose_uM": float(query_pool.loc[query_idx, "pert_dose_uM"]),
        "representation": representation,
        "retrieval_variant": retrieval_variant,
        "similarity_metric": similarity_metric,
        "scale_variant": scale_variant,
        "target_pool_size": int(len(target_pool)),
        "n_query_side_conditions": int(len(query_pool)),
        "n_target_side_conditions": int(len(target_pool)),
        "n_query_side_unique_compounds": int(
            query_pool["pubchem_cid"].astype(str).nunique()
        ),
        "n_target_side_unique_compounds": int(
            target_pool["pubchem_cid"].astype(str).nunique()
        ),
        "n_shared_genes": int(n_shared_genes),
        "n_raw_shared_genes": int(n_shared_genes),
    }


def _retrieval_fields(
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "n_positive_candidates": int(metrics["n_positives"]),
        "n_negative_candidates": int(metrics["n_negatives"]),
        "observed_best_positive_rank": metrics["best_rank"],
        "observed_normalized_best_positive_rank": metrics["normalized_rank"],
        "observed_recall_at_1": metrics["recall_at_1"],
        "observed_auroc": metrics["auroc"],
    }


def _injected_candidate_metrics(
    query_scores: np.ndarray,
    candidate_similarity: float,
) -> dict[str, float]:
    if not np.isfinite(candidate_similarity):
        return {
            "best_rank": float("nan"),
            "normalized_rank": float("nan"),
            "recall_at_1": float("nan"),
            "auroc": float("nan"),
        }
    scores = np.concatenate(
        [
            np.asarray(query_scores, dtype=np.float64),
            np.asarray([candidate_similarity], dtype=np.float64),
        ]
    )
    positive = np.zeros(len(scores), dtype=bool)
    positive[-1] = True
    return summarize_retrieval_scores(scores, positive)


def _primary_rows_for_direction(
    *,
    records: dict[tuple[str, ...], dict[str, Any]],
    dataset_a: str,
    dataset_b: str,
    direction: str,
    query_dataset: str,
    target_dataset: str,
    cell_type: str,
    time_key: str,
    query_pool: pd.DataFrame,
    target_pool: pd.DataFrame,
    strict_map: Mapping[str, set[str]],
    representations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    query_centroids: Mapping[str, np.ndarray],
    query_centroid_peer_counts: np.ndarray,
    max_dose_fold: float,
    n_shared_genes: int,
) -> None:
    query_compounds = query_pool["pubchem_cid"].astype(str).to_numpy()
    query_obs_ids = query_pool["obs_id"].astype(str).to_numpy()
    query_doses = query_pool["pert_dose_uM"].to_numpy(dtype=np.float64)
    target_compounds = target_pool["pubchem_cid"].astype(str).to_numpy()
    target_obs_ids = target_pool["obs_id"].astype(str).to_numpy()
    target_doses = target_pool["pert_dose_uM"].to_numpy(dtype=np.float64)

    for representation, (query_matrix, target_matrix) in representations.items():
        scores, valid_queries, valid_targets = similarity_matrix(
            query_matrix,
            target_matrix,
            "negative_l2",
        )
        target_indices = np.flatnonzero(valid_targets)
        if len(target_indices) < MIN_TARGET_CANDIDATES:
            continue
        kept_target_compounds = target_compounds[target_indices]
        kept_target_obs_ids = target_obs_ids[target_indices]
        kept_target_doses = target_doses[target_indices]
        centroid_matrix = np.asarray(query_centroids[representation])
        for query_idx in range(len(query_pool)):
            if not valid_queries[query_idx]:
                continue
            query_scores = scores[query_idx, target_indices]
            masks = {
                "compound_across_doses": primary_positive_mask(
                    query_compounds[query_idx],
                    kept_target_compounds,
                ),
                "dose_aware_compound": dose_aware_positive_mask(
                    query_compounds[query_idx],
                    query_doses[query_idx],
                    kept_target_compounds,
                    kept_target_doses,
                    max_fold=max_dose_fold,
                ),
                "strict_matched_condition": strict_positive_mask(
                    query_obs_ids[query_idx],
                    kept_target_obs_ids,
                    strict_map,
                ),
            }
            for variant, positive_mask in masks.items():
                observed = summarize_retrieval_scores(
                    query_scores,
                    positive_mask,
                )
                if (
                    observed["n_positives"] == 0
                    or observed["n_negatives"] == 0
                ):
                    continue
                record = _base_record(
                    dataset_a=dataset_a,
                    dataset_b=dataset_b,
                    direction=direction,
                    query_dataset=query_dataset,
                    target_dataset=target_dataset,
                    cell_type=cell_type,
                    time_key=time_key,
                    query_pool=query_pool,
                    query_idx=query_idx,
                    target_pool=target_pool.iloc[target_indices].reset_index(
                        drop=True
                    ),
                    representation=representation,
                    retrieval_variant=variant,
                    similarity_metric="negative_l2",
                    scale_variant=RAW_SCALE_VARIANT,
                    n_shared_genes=n_shared_genes,
                )
                record.update(_retrieval_fields(observed))
                centroid_similarity = score_vector_pair(
                    query_matrix[query_idx],
                    centroid_matrix[query_idx],
                    "negative_l2",
                )
                centroid_metrics = _injected_candidate_metrics(
                    query_scores,
                    centroid_similarity,
                )
                null = exact_cross_assay_null(
                    n_candidates=len(query_scores),
                    n_positives=int(observed["n_positives"]),
                )
                record.update(
                    {
                        "query_baseline_peer_count": int(
                            query_centroid_peer_counts[query_idx]
                        ),
                        "baseline_candidate_similarity": centroid_similarity,
                        "baseline_available": bool(
                            np.isfinite(centroid_metrics["normalized_rank"])
                        ),
                        "best_positive_rank": observed["best_rank"],
                        "normalized_best_positive_rank": (
                            observed["normalized_rank"]
                        ),
                        "recall_at_1": observed["recall_at_1"],
                        "auroc": observed["auroc"],
                        "baseline_best_positive_rank": (
                            centroid_metrics["best_rank"]
                        ),
                        "baseline_normalized_best_positive_rank": (
                            centroid_metrics["normalized_rank"]
                        ),
                        "baseline_recall_at_1": (
                            centroid_metrics["recall_at_1"]
                        ),
                        "baseline_auroc": centroid_metrics["auroc"],
                        "delta_vs_baseline_normalized_best_positive_rank": (
                            cross_source_core.difference_if_both_defined(
                                observed["normalized_rank"],
                                centroid_metrics["normalized_rank"],
                            )
                        ),
                        "delta_vs_baseline_recall_at_1": (
                            cross_source_core.difference_if_both_defined(
                                observed["recall_at_1"],
                                centroid_metrics["recall_at_1"],
                            )
                        ),
                        "delta_vs_baseline_auroc": (
                            cross_source_core.difference_if_both_defined(
                                observed["auroc"],
                                centroid_metrics["auroc"],
                            )
                        ),
                        "random_normalized_best_positive_rank": (
                            null["normalized_rank"]
                        ),
                        "random_recall_at_1": null["recall_at_1"],
                        "random_auroc": null["auroc"],
                    }
                )
                records[_record_key(record)] = record


def _strict_enriched_rows_for_direction(
    *,
    records: dict[tuple[str, ...], dict[str, Any]],
    dataset_a: str,
    dataset_b: str,
    direction: str,
    query_dataset: str,
    target_dataset: str,
    cell_type: str,
    time_key: str,
    query_pool: pd.DataFrame,
    target_pool: pd.DataFrame,
    query_matrix: np.ndarray,
    target_matrix: np.ndarray,
    strict_map: Mapping[str, set[str]],
    scale_variant: str,
    n_shared_genes: int,
    max_dose_fold: float,
    max_peers: int,
    sampling_seed: int,
    similarity_metrics: tuple[str, ...] = RETRIEVAL_SIMILARITIES,
    include_within_control: bool = True,
    precomputed_score_matrix: Optional[np.ndarray] = None,
    precomputed_valid_queries: Optional[np.ndarray] = None,
    precomputed_valid_targets: Optional[np.ndarray] = None,
    precomputed_source_score_matrix: Optional[np.ndarray] = None,
    eligibility_valid_queries: Optional[np.ndarray] = None,
    eligibility_valid_targets: Optional[np.ndarray] = None,
    population_counts: tuple[Optional[int], Optional[int]] = (None, None),
    eligibility_query_matrix: Optional[np.ndarray] = None,
    eligibility_target_matrix: Optional[np.ndarray] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> None:
    query_obs_ids = query_pool["obs_id"].astype(str).to_numpy()
    query_compounds = query_pool["pubchem_cid"].astype(str).to_numpy()
    query_dose_keys = query_pool["dose_key"].astype(str).to_numpy()
    query_doses = query_pool["pert_dose_uM"].to_numpy(dtype=np.float64)
    target_obs_ids = target_pool["obs_id"].astype(str).to_numpy()
    target_compounds = target_pool["pubchem_cid"].astype(str).to_numpy()
    target_dose_keys = target_pool["dose_key"].astype(str).to_numpy()
    target_doses = target_pool["pert_dose_uM"].to_numpy(dtype=np.float64)

    for metric in similarity_metrics:
        if precomputed_score_matrix is None:
            score_matrix, valid_queries, valid_targets = similarity_matrix(
                query_matrix,
                target_matrix,
                metric,
            )
        else:
            if len(similarity_metrics) != 1:
                raise ValueError(
                    "Precomputed retrieval scores require exactly one metric"
                )
            if (
                precomputed_valid_queries is None
                or precomputed_valid_targets is None
            ):
                raise ValueError(
                    "Precomputed retrieval scores require validity masks"
                )
            score_matrix = np.asarray(
                precomputed_score_matrix,
                dtype=np.float64,
            )
            valid_queries = np.asarray(
                precomputed_valid_queries,
                dtype=bool,
            )
            valid_targets = np.asarray(
                precomputed_valid_targets,
                dtype=bool,
            )

        if (
            eligibility_valid_queries is not None
            and eligibility_valid_targets is not None
        ):
            eligibility_queries = np.asarray(
                eligibility_valid_queries,
                dtype=bool,
            )
            eligibility_targets = np.asarray(
                eligibility_valid_targets,
                dtype=bool,
            )
        elif (
            eligibility_query_matrix is not None
            and eligibility_target_matrix is not None
        ):
            _, eligibility_queries, eligibility_targets = similarity_matrix(
                eligibility_query_matrix,
                eligibility_target_matrix,
                metric,
            )
        else:
            eligibility_queries = None
            eligibility_targets = None

        if (
            eligibility_queries is not None
            and eligibility_targets is not None
        ):
            if (
                not bool(np.all(valid_queries[eligibility_queries]))
                or not bool(np.all(valid_targets[eligibility_targets]))
            ):
                raise AssertionError(
                    f"W4 {scale_variant} created an invalid signature in "
                    f"raw-valid {metric} retrieval geometry"
                )
            valid_queries = eligibility_queries
            target_indices = np.flatnonzero(eligibility_targets)
        else:
            target_indices = np.flatnonzero(valid_targets)
        if len(target_indices) < MIN_TARGET_CANDIDATES:
            continue
        filtered_obs = target_obs_ids[target_indices]
        filtered_compounds = target_compounds[target_indices]
        filtered_dose_keys = target_dose_keys[target_indices]
        filtered_doses = target_doses[target_indices]
        filtered_matrix = target_matrix[target_indices]
        query_count = len(query_pool)
        progress_interval = max(1, (query_count + 9) // 10)
        for query_idx in range(len(query_pool)):
            if (
                progress is not None
                and query_idx % progress_interval == 0
            ):
                progress(
                    "query_progress",
                    (
                        f"scale={scale_variant} metric={metric} "
                        f"direction={direction} "
                        f"processed={query_idx}/{query_count}"
                    ),
                )
            if not valid_queries[query_idx]:
                continue
            query_scores = score_matrix[query_idx, target_indices]
            positive = strict_positive_mask(
                query_obs_ids[query_idx],
                filtered_obs,
                strict_map,
            )
            observed = summarize_retrieval_scores(query_scores, positive)
            if (
                observed["n_positives"] == 0
                or observed["n_negatives"] == 0
            ):
                continue
            positive_indices = np.flatnonzero(positive)
            positive_scores = query_scores[positive_indices]
            best_target_idx = int(
                positive_indices[int(np.argmax(positive_scores))]
            )
            observed_best_score = float(query_scores[best_target_idx])
            best_target_dose = float(filtered_doses[best_target_idx])
            best_target_dose_key = str(
                filtered_dose_keys[best_target_idx]
            )

            source_mask = (
                (query_dose_keys == query_dose_keys[query_idx])
                & (query_compounds != query_compounds[query_idx])
                & valid_queries
            )
            source_all_rows = np.flatnonzero(source_mask)
            source_offsets = select_peer_indices(
                len(source_all_rows),
                max_peers,
                "|".join(
                    [
                        str(query_dataset),
                        str(target_dataset),
                        str(cell_type),
                        str(time_key),
                        str(query_obs_ids[query_idx]),
                        str(query_dose_keys[query_idx]),
                        str(query_compounds[query_idx]),
                        "source",
                    ]
                ),
                sampling_seed=sampling_seed,
            )
            source_rows = source_all_rows[source_offsets]
            source_scores = (
                (
                    np.asarray(
                        precomputed_source_score_matrix[
                            query_idx,
                            source_rows,
                        ],
                        dtype=np.float64,
                    )
                    if precomputed_source_score_matrix is not None
                    else score_vector_against_matrix(
                        query_matrix[query_idx],
                        query_matrix[source_rows],
                        metric,
                    )
                )
                if len(source_rows)
                else np.empty(0, dtype=np.float64)
            )
            source_summary = retrieval_peer_summary(
                observed_best_score,
                source_scores,
                "source_individual",
            )
            source_centroid_similarity = float("nan")
            if len(source_all_rows):
                source_centroid_similarity = score_vector_pair(
                    query_matrix[query_idx],
                    query_matrix[source_all_rows].mean(
                        axis=0,
                        dtype=np.float64,
                    ),
                    metric,
                )

            target_mask = (
                (filtered_dose_keys == best_target_dose_key)
                & (filtered_compounds != query_compounds[query_idx])
            )
            target_all_rows = np.flatnonzero(target_mask)
            target_offsets = select_peer_indices(
                len(target_all_rows),
                max_peers,
                "|".join(
                    [
                        str(query_dataset),
                        str(target_dataset),
                        str(cell_type),
                        str(time_key),
                        str(query_obs_ids[query_idx]),
                        str(best_target_dose_key),
                        str(query_compounds[query_idx]),
                        "target",
                    ]
                ),
                sampling_seed=sampling_seed,
            )
            target_rows = target_all_rows[target_offsets]
            target_scores = query_scores[target_rows]
            target_summary = retrieval_peer_summary(
                observed_best_score,
                target_scores,
                "target_individual",
            )
            target_centroid_similarity = float("nan")
            if len(target_all_rows):
                target_centroid_similarity = score_vector_pair(
                    query_matrix[query_idx],
                    filtered_matrix[target_all_rows].mean(
                        axis=0,
                        dtype=np.float64,
                    ),
                    metric,
                )
            source_centroid_metrics = _injected_candidate_metrics(
                query_scores,
                source_centroid_similarity,
            )
            target_centroid_metrics = _injected_candidate_metrics(
                query_scores,
                target_centroid_similarity,
            )
            source_peer_metrics = expected_single_signature_metrics(
                query_scores,
                source_scores,
            )
            target_peer_metrics = expected_single_signature_metrics(
                query_scores,
                target_scores,
            )

            within_metrics = {
                "best_rank": float("nan"),
                "normalized_rank": float("nan"),
                "recall_at_1": float("nan"),
                "auroc": float("nan"),
            }
            if include_within_control:
                within_mask = np.arange(len(query_pool)) != query_idx
                within_mask &= valid_queries
                within_matrix = query_matrix[within_mask]
                within_compounds = query_compounds[within_mask]
                within_doses = query_doses[within_mask]
                if len(within_matrix) >= 2:
                    within_scores = score_vector_against_matrix(
                        query_matrix[query_idx],
                        within_matrix,
                        metric,
                    )
                    within_positive = (
                        within_compounds == query_compounds[query_idx]
                    ) & (
                        raw_dose_fold_differences(
                            query_doses[query_idx],
                            within_doses,
                        )
                        <= max_dose_fold + 1e-12
                    )
                    within_metrics = summarize_retrieval_scores(
                        within_scores,
                        within_positive,
                    )

            null = exact_cross_assay_null(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
            )
            calibration = exact_null_rank_calibration(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
                best_rank=observed["best_rank"],
            )
            minimum_calibration = exact_null_rank_calibration(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
                best_rank=1.0,
            )
            record = _base_record(
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                direction=direction,
                query_dataset=query_dataset,
                target_dataset=target_dataset,
                cell_type=cell_type,
                time_key=time_key,
                query_pool=query_pool,
                query_idx=query_idx,
                target_pool=target_pool.iloc[target_indices].reset_index(
                    drop=True
                ),
                representation="logFC",
                retrieval_variant="strict_matched_condition",
                similarity_metric=metric,
                scale_variant=scale_variant,
                n_shared_genes=n_shared_genes,
            )
            key = _record_key(record)
            if key in records:
                records[key].update(record)
                record = records[key]
            else:
                records[key] = record
            record.update(_retrieval_fields(observed))
            record.update(source_summary)
            record.update(target_summary)
            record.update(
                {
                    "target_pool_size_before_similarity_filter": int(
                        len(target_pool)
                    ),
                    "n_zero_norm_target_candidates": int(
                        len(target_pool) - len(target_indices)
                    ),
                    "centroid_baseline_peer_count": int(
                        len(source_all_rows)
                    ),
                    "single_signature_baseline_peer_count": int(
                        source_peer_metrics["n_peers"]
                    ),
                    "n_excluded_single_signature_peers": int(
                        len(source_all_rows)
                        - int(source_peer_metrics["n_peers"])
                    ),
                    "source_centroid_peer_count": int(len(source_all_rows)),
                    "target_centroid_peer_count": int(len(target_all_rows)),
                    "source_individual_total_count": int(
                        len(source_all_rows)
                    ),
                    "source_individual_scored_count": int(len(source_rows)),
                    "target_individual_total_count": int(
                        len(target_all_rows)
                    ),
                    "target_individual_scored_count": int(len(target_rows)),
                    "left_population_row_count": population_counts[0],
                    "right_population_row_count": population_counts[1],
                    "observed_best_positive_similarity": observed_best_score,
                    "observed_mean_positive_similarity": float(
                        np.mean(positive_scores)
                    ),
                    "observed_sd_positive_similarity": (
                        float(np.std(positive_scores, ddof=1))
                        if len(positive_scores) >= 2
                        else float("nan")
                    ),
                    "best_target_obs_id": str(
                        filtered_obs[best_target_idx]
                    ),
                    "best_target_dose_key": best_target_dose_key,
                    "best_target_dose_uM": best_target_dose,
                    "best_target_dose_fold_difference": float(
                        max(
                            query_doses[query_idx] / best_target_dose,
                            best_target_dose / query_doses[query_idx],
                        )
                    ),
                    "source_centroid_similarity": source_centroid_similarity,
                    "target_centroid_similarity": target_centroid_similarity,
                    "centroid_baseline_candidate_similarity": (
                        source_centroid_similarity
                    ),
                    "target_decoy_null_normalized_best_positive_rank": (
                        null["normalized_rank"]
                    ),
                    "target_decoy_null_best_positive_rank": (
                        null["best_rank"]
                    ),
                    "target_decoy_null_recall_at_1": null["recall_at_1"],
                    "target_decoy_null_auroc": null["auroc"],
                    "exact_null_mid_p": exact_null_mid_p(
                        n_candidates=len(query_scores),
                        n_positives=int(observed["n_positives"]),
                        best_rank=int(observed["best_rank"]),
                    ),
                    "null_pit": calibration["null_pit"],
                    "null_p_value": calibration["null_p_value"],
                    "min_achievable_null_p_value": (
                        minimum_calibration["null_p_value"]
                    ),
                    "null_pit_below_half": float(
                        calibration["null_pit"] < 0.5
                    ),
                    "observed_auroc_minus_half": (
                        float(observed["auroc"]) - 0.5
                    ),
                    "within_source_best_positive_rank": (
                        within_metrics["best_rank"]
                    ),
                    "within_source_normalized_best_positive_rank": (
                        within_metrics["normalized_rank"]
                    ),
                    "within_source_recall_at_1": (
                        within_metrics["recall_at_1"]
                    ),
                    "within_source_auroc": within_metrics["auroc"],
                    "centroid_baseline_best_positive_rank": (
                        source_centroid_metrics["best_rank"]
                    ),
                    "centroid_baseline_normalized_best_positive_rank": (
                        source_centroid_metrics["normalized_rank"]
                    ),
                    "centroid_baseline_recall_at_1": (
                        source_centroid_metrics["recall_at_1"]
                    ),
                    "centroid_baseline_auroc": (
                        source_centroid_metrics["auroc"]
                    ),
                    "single_signature_baseline_best_positive_rank": (
                        source_peer_metrics["best_rank"]
                    ),
                    "single_signature_baseline_normalized_best_positive_rank": (
                        source_peer_metrics["normalized_rank"]
                    ),
                    "single_signature_baseline_recall_at_1": (
                        source_peer_metrics["recall_at_1"]
                    ),
                    "single_signature_baseline_auroc": (
                        source_peer_metrics["auroc"]
                    ),
                    "within_dataset_positive_control_best_positive_rank": (
                        within_metrics["best_rank"]
                    ),
                    "within_dataset_positive_control_normalized_best_positive_rank": (
                        within_metrics["normalized_rank"]
                    ),
                    "within_dataset_positive_control_recall_at_1": (
                        within_metrics["recall_at_1"]
                    ),
                    "within_dataset_positive_control_auroc": (
                        within_metrics["auroc"]
                    ),
                    "delta_observed_vs_source_centroid": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            source_centroid_similarity,
                        )
                    ),
                    "delta_observed_vs_target_centroid": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            target_centroid_similarity,
                        )
                    ),
                    "delta_observed_vs_source_individual_mean": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            source_summary[
                                "source_individual_mean_similarity"
                            ],
                        )
                    ),
                    "delta_observed_vs_target_individual_mean": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            target_summary[
                                "target_individual_mean_similarity"
                            ],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_recall_at_1": (
                        cross_source_core.difference_if_both_defined(
                            observed["recall_at_1"],
                            null["recall_at_1"],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_auroc": (
                        cross_source_core.difference_if_both_defined(
                            observed["auroc"],
                            null["auroc"],
                        )
                    ),
                    "delta_vs_target_decoy_null_normalized_best_positive_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                    "delta_vs_target_decoy_null_recall_at_1": (
                        cross_source_core.difference_if_both_defined(
                            observed["recall_at_1"],
                            null["recall_at_1"],
                        )
                    ),
                    "delta_vs_target_decoy_null_auroc": (
                        cross_source_core.difference_if_both_defined(
                            observed["auroc"],
                            null["auroc"],
                        )
                    ),
                    "delta_vs_random_normalized_best_positive_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                }
            )
            metric_variants = {
                "source_centroid": source_centroid_metrics,
                "target_centroid": target_centroid_metrics,
                "source_peer": source_peer_metrics,
                "target_peer": target_peer_metrics,
                "within_dataset_positive_control": within_metrics,
            }
            for variant, values in metric_variants.items():
                for field in (
                    "best_rank",
                    "normalized_rank",
                    "recall_at_1",
                    "auroc",
                ):
                    record[f"{variant}_{field}"] = values.get(
                        field,
                        float("nan"),
                    )
                for field in ("normalized_rank", "recall_at_1", "auroc"):
                    record[f"delta_vs_{variant}_{field}"] = (
                        cross_source_core.difference_if_both_defined(
                            observed[field],
                            values.get(field, float("nan")),
                        )
                    )
            for alias, values in (
                ("centroid_baseline", source_centroid_metrics),
                ("single_signature_baseline", source_peer_metrics),
                ("within_dataset_positive_control", within_metrics),
            ):
                for field, output_field in (
                    ("normalized_rank", "normalized_best_positive_rank"),
                    ("recall_at_1", "recall_at_1"),
                    ("auroc", "auroc"),
                ):
                    record[f"delta_vs_{alias}_{output_field}"] = (
                        cross_source_core.difference_if_both_defined(
                            observed[field],
                            values.get(field, float("nan")),
                        )
                    )
        if progress is not None:
            progress(
                "query_progress",
                (
                    f"scale={scale_variant} metric={metric} "
                    f"direction={direction} "
                    f"processed={query_count}/{query_count}"
                ),
            )


def _score_selected_scale(
    *,
    records: dict[tuple[str, ...], dict[str, Any]],
    dataset_a: str,
    dataset_b: str,
    cell_type: str,
    time_key: str,
    left_pool: pd.DataFrame,
    right_pool: pd.DataFrame,
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    strict_maps: Mapping[str, Mapping[str, set[str]]],
    scale_variant: str,
    n_shared_genes: int,
    max_dose_fold: float,
    max_peers: int,
    sampling_seed: int,
    similarity_metrics: tuple[str, ...],
    population_counts: tuple[Optional[int], Optional[int]] = (None, None),
    eligibility_masks: Optional[
        Mapping[str, tuple[np.ndarray, np.ndarray]]
    ] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw_validity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in similarity_metrics:
        if progress is not None:
            progress(
                "similarity",
                f"scale={scale_variant} metric={metric}",
            )
        cross_scores, valid_left, valid_right = similarity_matrix(
            left_matrix,
            right_matrix,
            metric,
        )
        left_scores, left_self_valid, _ = similarity_matrix(
            left_matrix,
            left_matrix,
            metric,
        )
        right_scores, right_self_valid, _ = similarity_matrix(
            right_matrix,
            right_matrix,
            metric,
        )
        if not np.array_equal(valid_left, left_self_valid):
            raise AssertionError(
                f"Left {metric} validity differs between cross- and "
                "within-source geometry"
            )
        if not np.array_equal(valid_right, right_self_valid):
            raise AssertionError(
                f"Right {metric} validity differs between cross- and "
                "within-source geometry"
            )
        raw_validity[metric] = (valid_left.copy(), valid_right.copy())
        eligibility_left = None
        eligibility_right = None
        if eligibility_masks is not None:
            eligibility_left, eligibility_right = eligibility_masks[metric]

        _strict_enriched_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="A_to_B",
            query_dataset=dataset_a,
            target_dataset=dataset_b,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=left_pool,
            target_pool=right_pool,
            query_matrix=left_matrix,
            target_matrix=right_matrix,
            strict_map=strict_maps["A_to_B"],
            scale_variant=scale_variant,
            n_shared_genes=n_shared_genes,
            max_dose_fold=max_dose_fold,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
            similarity_metrics=(metric,),
            include_within_control=False,
            precomputed_score_matrix=cross_scores,
            precomputed_valid_queries=valid_left,
            precomputed_valid_targets=valid_right,
            precomputed_source_score_matrix=left_scores,
            eligibility_valid_queries=eligibility_left,
            eligibility_valid_targets=eligibility_right,
            population_counts=population_counts,
            progress=progress,
        )
        _strict_enriched_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="B_to_A",
            query_dataset=dataset_b,
            target_dataset=dataset_a,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=right_pool,
            target_pool=left_pool,
            query_matrix=right_matrix,
            target_matrix=left_matrix,
            strict_map=strict_maps["B_to_A"],
            scale_variant=scale_variant,
            n_shared_genes=n_shared_genes,
            max_dose_fold=max_dose_fold,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
            similarity_metrics=(metric,),
            include_within_control=False,
            precomputed_score_matrix=cross_scores.T,
            precomputed_valid_queries=valid_right,
            precomputed_valid_targets=valid_left,
            precomputed_source_score_matrix=right_scores,
            eligibility_valid_queries=eligibility_right,
            eligibility_valid_targets=eligibility_left,
            population_counts=population_counts,
            progress=progress,
        )
    return raw_validity


def _peer_only_direction_records(
    *,
    observed_rows: pd.DataFrame,
    direction: str,
    query_pool: pd.DataFrame,
    target_pool: pd.DataFrame,
    query_matrix: np.ndarray,
    target_matrix: np.ndarray,
    cross_scores: np.ndarray,
    query_self_scores: np.ndarray,
    valid_queries: np.ndarray,
    valid_targets: np.ndarray,
    caps: Sequence[Optional[int]],
    reference_cap: int,
    sampling_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = observed_rows.loc[
        observed_rows["direction"].astype(str) == str(direction)
    ]
    if rows.empty:
        return [], []
    query_obs = query_pool["obs_id"].astype(str).to_numpy()
    query_compounds = query_pool["pubchem_cid"].astype(str).to_numpy()
    query_doses = query_pool["dose_key"].astype(str).to_numpy()
    target_compounds = target_pool["pubchem_cid"].astype(str).to_numpy()
    target_doses = target_pool["dose_key"].astype(str).to_numpy()
    query_lookup = {
        str(obs_id): index for index, obs_id in enumerate(query_obs)
    }
    target_indices = np.flatnonzero(valid_targets)
    filtered_target_compounds = target_compounds[target_indices]
    filtered_target_doses = target_doses[target_indices]
    output: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        row_values = row._asdict()
        query_obs_id = str(row_values["query_obs_id"])
        query_idx = query_lookup.get(query_obs_id)
        if query_idx is None or not bool(valid_queries[query_idx]):
            diagnostics.append(
                {
                    "analysis": ANALYSIS,
                    "dataset_a": row_values["dataset_a"],
                    "dataset_b": row_values["dataset_b"],
                    "cell_type": row_values["cell_type"],
                    "time_key": row_values["time_key"],
                    "query_obs_id": query_obs_id,
                    "stage": "retrieval_peer_only",
                    "reason": "unresolved_or_invalid_query",
                }
            )
            continue
        observed_similarity = float(
            row_values["observed_best_positive_similarity"]
        )
        query_scores = np.asarray(
            cross_scores[query_idx, target_indices],
            dtype=np.float64,
        )
        query_compound = str(query_compounds[query_idx])
        query_dose = str(query_doses[query_idx])
        best_target_dose = cross_source_core.format_numeric(
            float(row_values["best_target_dose_key"])
        )

        source_all_rows = np.flatnonzero(
            (query_doses == query_dose)
            & (query_compounds != query_compound)
            & valid_queries
        )
        target_all_rows = np.flatnonzero(
            (filtered_target_doses == best_target_dose)
            & (filtered_target_compounds != query_compound)
        )
        query_dataset = str(row_values["query_dataset"])
        target_dataset = str(row_values["target_dataset"])
        common_key = [
            query_dataset,
            target_dataset,
            str(row_values["cell_type"]),
            str(row_values["time_key"]),
            query_obs_id,
        ]
        source_key = "|".join(
            [*common_key, query_dose, query_compound, "source"]
        )
        target_key = "|".join(
            [*common_key, best_target_dose, query_compound, "target"]
        )
        source_selections = nested_peer_indices(
            len(source_all_rows),
            caps,
            reference_cap=reference_cap,
            seed_key=source_key,
            sampling_seed=sampling_seed,
        )
        target_selections = nested_peer_indices(
            len(target_all_rows),
            caps,
            reference_cap=reference_cap,
            seed_key=target_key,
            sampling_seed=sampling_seed,
        )
        source_all_scores = np.asarray(
            query_self_scores[query_idx, source_all_rows],
            dtype=np.float64,
        )
        target_all_scores = np.asarray(
            query_scores[target_all_rows],
            dtype=np.float64,
        )
        for cap_order, cap in enumerate(caps):
            label = "all" if cap is None else str(int(cap))
            source_offsets = source_selections[label]
            target_offsets = target_selections[label]
            source_scores = source_all_scores[source_offsets]
            target_scores = target_all_scores[target_offsets]
            source_summary = retrieval_peer_summary(
                observed_similarity,
                source_scores,
                "source_individual",
            )
            target_summary = retrieval_peer_summary(
                observed_similarity,
                target_scores,
                "target_individual",
            )
            source_metrics = expected_single_signature_metrics(
                query_scores,
                source_scores,
            )
            target_metrics = expected_single_signature_metrics(
                query_scores,
                target_scores,
            )
            record = {
                column: row_values[column] for column in IDENTITY_COLUMNS
            }
            record.update(
                {
                    "peer_cap_order": cap_order,
                    "peer_cap_label": label,
                    "peer_cap": 0 if cap is None else int(cap),
                    "peer_reference_cap": int(reference_cap),
                    "peer_sampling_seed": int(sampling_seed),
                    "source_individual_total_count": int(
                        len(source_all_rows)
                    ),
                    "source_individual_scored_count": int(
                        len(source_offsets)
                    ),
                    "target_individual_total_count": int(
                        len(target_all_rows)
                    ),
                    "target_individual_scored_count": int(
                        len(target_offsets)
                    ),
                    "best_target_dose_key": best_target_dose,
                    "observed_best_positive_similarity": (
                        observed_similarity
                    ),
                    "observed_normalized_best_positive_rank": float(
                        row_values[
                            "observed_normalized_best_positive_rank"
                        ]
                    ),
                    "observed_recall_at_1": float(
                        row_values["observed_recall_at_1"]
                    ),
                    "observed_auroc": float(
                        row_values["observed_auroc"]
                    ),
                    **source_summary,
                    **target_summary,
                }
            )
            for side, metrics in (
                ("source", source_metrics),
                ("target", target_metrics),
            ):
                for field in (
                    "best_rank",
                    "normalized_rank",
                    "recall_at_1",
                    "auroc",
                ):
                    record[f"{side}_peer_{field}"] = metrics[field]
                for field, observed_field in (
                    (
                        "normalized_rank",
                        "observed_normalized_best_positive_rank",
                    ),
                    ("recall_at_1", "observed_recall_at_1"),
                    ("auroc", "observed_auroc"),
                ):
                    record[f"delta_vs_{side}_peer_{field}"] = (
                        cross_source_core.difference_if_both_defined(
                            record[observed_field],
                            metrics[field],
                        )
                    )
            output.append(record)
    return output, diagnostics


def _peer_only_context_records(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    catalog: cross_source_core.LineSourceCatalog,
    w4_catalog,
    progress: Optional[Callable[[str, str], None]] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    settings = config["settings"]
    caps = tuple(
        None if value == "all" else int(value)
        for value in settings["peer_caps"]
    )
    scales = tuple(str(value) for value in settings["peer_scales"])
    dataset_a = str(frame.iloc[0]["dataset_a"])
    dataset_b = str(frame.iloc[0]["dataset_b"])
    cell_type = str(frame.iloc[0]["cell_type"])
    time_key = str(frame.iloc[0]["time_key"])
    left_pool_frame = pool_frame_for_context(
        read_overlap_metadata(config, dataset_a),
        cell_type,
        time_key,
    )
    right_pool_frame = pool_frame_for_context(
        read_overlap_metadata(config, dataset_b),
        cell_type,
        time_key,
    )
    left_source = catalog.get_line_source(dataset_a, cell_type)
    right_source = catalog.get_line_source(dataset_b, cell_type)
    shared_genes, left_positions, right_positions = (
        catalog.shared_gene_positions(left_source, right_source)
    )
    left = build_logfc_pool_matrices(
        left_pool_frame,
        left_source,
        left_positions,
    )
    right = build_logfc_pool_matrices(
        right_pool_frame,
        right_source,
        right_positions,
    )
    diagnostics: list[dict[str, Any]] = []
    for side, values in (("left", left.diagnostics), ("right", right.diagnostics)):
        for diagnostic in values:
            diagnostics.append(
                {
                    **diagnostic,
                    "analysis": ANALYSIS,
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "cell_type": cell_type,
                    "time_key": time_key,
                    "stage": "retrieval_peer_only",
                    "reason": f"{side}_pool_{diagnostic.get('reason', '')}",
                }
            )
    finite = np.isfinite(left.logfc).all(axis=0) & np.isfinite(
        right.logfc
    ).all(axis=0)
    if int(finite.sum()) < 2:
        raise ValueError(
            f"Fewer than two finite shared genes for {dataset_a} vs "
            f"{dataset_b} / {cell_type} / {time_key}"
        )
    left_raw = left.logfc[:, finite]
    right_raw = right.logfc[:, finite]
    scale_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {
        RAW_SCALE_VARIANT: (left_raw, right_raw)
    }
    if PER_GENE_DATASET_VARIANT in scales:
        if w4_catalog is None:
            raise AssertionError("Dataset W4 peer scoring needs a W4 catalog")
        left_w4 = w4_catalog.standardize_aligned_matrix(
            left_source,
            left.logfc,
            left_positions,
            scale_variant=PER_GENE_DATASET_VARIANT,
        )
        right_w4 = w4_catalog.standardize_aligned_matrix(
            right_source,
            right.logfc,
            right_positions,
            scale_variant=PER_GENE_DATASET_VARIANT,
        )
        scale_matrices[PER_GENE_DATASET_VARIANT] = (
            left_w4[:, finite],
            right_w4[:, finite],
        )

    output: list[dict[str, Any]] = []
    raw_geometry: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for metric in REVIEWER_MINIMAL_SIMILARITIES:
        raw_scores, valid_left, valid_right = similarity_matrix(
            left_raw,
            right_raw,
            metric,
        )
        raw_geometry[metric] = (raw_scores, valid_left, valid_right)
    for scale in scales:
        left_matrix, right_matrix = scale_matrices[scale]
        for metric in REVIEWER_MINIMAL_SIMILARITIES:
            selected = frame.loc[
                (frame["scale_variant"].astype(str) == scale)
                & (frame["similarity_metric"].astype(str) == metric)
            ]
            if selected.empty:
                continue
            if progress is not None:
                progress(
                    "peer_similarity",
                    f"scale={scale} metric={metric}",
                )
            if scale == RAW_SCALE_VARIANT:
                (
                    cross_scores,
                    scale_valid_left,
                    scale_valid_right,
                ) = raw_geometry[metric]
            else:
                cross_scores, scale_valid_left, scale_valid_right = (
                    similarity_matrix(left_matrix, right_matrix, metric)
                )
            left_self, _, _ = similarity_matrix(
                left_matrix,
                left_matrix,
                metric,
            )
            right_self, _, _ = similarity_matrix(
                right_matrix,
                right_matrix,
                metric,
            )
            _, valid_left, valid_right = raw_geometry[metric]
            if (
                not np.all(scale_valid_left[valid_left])
                or not np.all(scale_valid_right[valid_right])
            ):
                raise AssertionError(
                    f"{scale} created invalid raw-valid {metric} signatures"
                )
            records, record_diagnostics = _peer_only_direction_records(
                observed_rows=selected,
                direction="A_to_B",
                query_pool=left.frame,
                target_pool=right.frame,
                query_matrix=left_matrix,
                target_matrix=right_matrix,
                cross_scores=cross_scores,
                query_self_scores=left_self,
                valid_queries=valid_left,
                valid_targets=valid_right,
                caps=caps,
                reference_cap=int(settings["peer_reference_cap"]),
                sampling_seed=int(settings["peer_sampling_seed"]),
            )
            output.extend(records)
            diagnostics.extend(record_diagnostics)
            records, record_diagnostics = _peer_only_direction_records(
                observed_rows=selected,
                direction="B_to_A",
                query_pool=right.frame,
                target_pool=left.frame,
                query_matrix=right_matrix,
                target_matrix=left_matrix,
                cross_scores=cross_scores.T,
                query_self_scores=right_self,
                valid_queries=valid_right,
                valid_targets=valid_left,
                caps=caps,
                reference_cap=int(settings["peer_reference_cap"]),
                sampling_seed=int(settings["peer_sampling_seed"]),
            )
            output.extend(records)
            diagnostics.extend(record_diagnostics)
    result = pd.DataFrame(output)
    if result.empty:
        return pd.DataFrame(columns=PEER_ONLY_OUTPUT_COLUMNS), diagnostics
    for column in PEER_ONLY_OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    result = result.loc[:, PEER_ONLY_OUTPUT_COLUMNS]
    sort_columns = [
        "dataset_a",
        "dataset_b",
        "direction",
        "cell_type",
        "time_key",
        "query_obs_id",
        "similarity_metric",
        "scale_variant",
        "peer_cap_order",
    ]
    return result.sort_values(sort_columns).reset_index(drop=True), diagnostics


def _context_records(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    catalog: cross_source_core.LineSourceCatalog,
    w4_catalog,
    progress: Optional[Callable[[str, str], None]] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    settings = config["settings"]
    workload = str(settings.get("workload", FULL_WORKLOAD))
    similarity_metrics = tuple(
        str(metric) for metric in settings["similarity_metrics"]
    )
    scale_similarity_metrics = {
        str(scale): tuple(str(metric) for metric in metrics)
        for scale, metrics in settings.get(
            "scale_similarity_metrics",
            {},
        ).items()
    }
    include_legacy_retrieval = bool(
        settings.get(
            "include_legacy_retrieval",
            workload == FULL_WORKLOAD,
        )
    )
    w4_scale_variants = tuple(
        str(variant) for variant in settings["w4_scale_variants"]
    )
    dataset_a = str(frame.iloc[0]["dataset_a"])
    dataset_b = str(frame.iloc[0]["dataset_b"])
    cell_type = str(frame.iloc[0]["cell_type"])
    time_key = str(frame.iloc[0]["time_key"])
    if progress is not None:
        progress(
            "context_loaded",
            (
                f"pair={dataset_a}__{dataset_b} cell_type={cell_type} "
                f"time={time_key} matched_rows={len(frame)}"
            ),
        )
    left_pool_frame = pool_frame_for_context(
        read_overlap_metadata(config, dataset_a),
        cell_type,
        time_key,
    )
    right_pool_frame = pool_frame_for_context(
        read_overlap_metadata(config, dataset_b),
        cell_type,
        time_key,
    )
    diagnostics: list[dict[str, Any]] = []
    if (
        len(left_pool_frame) < MIN_TARGET_CANDIDATES
        or len(right_pool_frame) < MIN_TARGET_CANDIDATES
        or left_pool_frame["pubchem_cid"].astype(str).nunique()
        < MIN_UNIQUE_COMPOUNDS_PER_SIDE
        or right_pool_frame["pubchem_cid"].astype(str).nunique()
        < MIN_UNIQUE_COMPOUNDS_PER_SIDE
    ):
        diagnostics.append(
            {
                "analysis": ANALYSIS,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "cell_type": cell_type,
                "time_key": time_key,
                "stage": "retrieval_context",
                "reason": "insufficient_candidates_or_unique_compounds",
            }
        )
        return pd.DataFrame(columns=RETRIEVAL_OUTPUT_COLUMNS), diagnostics

    left_source = catalog.get_line_source(dataset_a, cell_type)
    right_source = catalog.get_line_source(dataset_b, cell_type)
    shared_genes, left_positions, right_positions = (
        catalog.shared_gene_positions(left_source, right_source)
    )
    if shared_genes.size < 2:
        raise ValueError(
            f"Fewer than two shared genes for {dataset_a} vs "
            f"{dataset_b} / {cell_type}"
        )
    if (
        workload in (REVIEWER_MINIMAL_WORKLOAD, SELECTED_WORKLOAD)
        and not include_legacy_retrieval
    ):
        left = build_logfc_pool_matrices(
            left_pool_frame,
            left_source,
            left_positions,
        )
        right = build_logfc_pool_matrices(
            right_pool_frame,
            right_source,
            right_positions,
        )
    else:
        left = build_pool_matrices(
            left_pool_frame,
            left_source,
            left_positions,
            adj_layer_name=_adj_layer(left_source),
        )
        right = build_pool_matrices(
            right_pool_frame,
            right_source,
            right_positions,
            adj_layer_name=_adj_layer(right_source),
        )
    for side, pool_diagnostics in (
        ("left", left.diagnostics),
        ("right", right.diagnostics),
    ):
        for diagnostic in pool_diagnostics:
            diagnostics.append(
                {
                    **diagnostic,
                    "analysis": ANALYSIS,
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "cell_type": cell_type,
                    "time_key": time_key,
                    "query_obs_id": diagnostic.get("obs_id", ""),
                    "reason": (
                        f"{side}_pool_"
                        f"{diagnostic.get('reason', 'unresolved_source_row')}"
                    ),
                }
            )
    if (
        len(left.frame) < MIN_TARGET_CANDIDATES
        or len(right.frame) < MIN_TARGET_CANDIDATES
    ):
        diagnostics.append(
            {
                "analysis": ANALYSIS,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "cell_type": cell_type,
                "time_key": time_key,
                "stage": "retrieval_context",
                "reason": "insufficient_resolved_candidates",
            }
        )
        return pd.DataFrame(columns=RETRIEVAL_OUTPUT_COLUMNS), diagnostics

    finite = np.isfinite(left.logfc).all(axis=0) & np.isfinite(
        right.logfc
    ).all(axis=0)
    if include_legacy_retrieval:
        finite &= (
            np.isfinite(left.t).all(axis=0)
            & np.isfinite(right.t).all(axis=0)
            & np.isfinite(left.adj_p).all(axis=0)
            & np.isfinite(right.adj_p).all(axis=0)
        )
    if int(finite.sum()) < 2:
        raise ValueError(
            f"Fewer than two fully finite shared genes for {dataset_a} vs "
            f"{dataset_b} / {cell_type} / {time_key}"
        )
    n_shared = int(finite.sum())
    left_logfc = left.logfc[:, finite]
    right_logfc = right.logfc[:, finite]
    maps = strict_positive_maps(frame)
    max_fold = float(cross_source_core.DEFAULT_MATCH_SETTINGS.max_dose_fold_difference)
    records: dict[tuple[str, ...], dict[str, Any]] = {}

    if include_legacy_retrieval:
        left_t = left.t[:, finite]
        right_t = right.t[:, finite]
        left_adj = left.adj_p[:, finite]
        right_adj = right.adj_p[:, finite]
        left_centroid_logfc = left.centroid_logfc[:, finite]
        right_centroid_logfc = right.centroid_logfc[:, finite]
        left_centroid_t = left.centroid_t[:, finite]
        right_centroid_t = right.centroid_t[:, finite]
        left_centroid_adj = left.centroid_adj_p[:, finite]
        right_centroid_adj = right.centroid_adj_p[:, finite]
        left_representations = {
            "logFC": left_logfc,
            "moderated_t": left_t,
            "signed_significance": signed_significance(left_logfc, left_adj),
        }
        right_representations = {
            "logFC": right_logfc,
            "moderated_t": right_t,
            "signed_significance": signed_significance(right_logfc, right_adj),
        }
        left_centroids = {
            "logFC": left_centroid_logfc,
            "moderated_t": left_centroid_t,
            "signed_significance": signed_significance(
                left_centroid_logfc,
                left_centroid_adj,
            ),
        }
        right_centroids = {
            "logFC": right_centroid_logfc,
            "moderated_t": right_centroid_t,
            "signed_significance": signed_significance(
                right_centroid_logfc,
                right_centroid_adj,
            ),
        }

        _primary_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="A_to_B",
            query_dataset=dataset_a,
            target_dataset=dataset_b,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=left.frame,
            target_pool=right.frame,
            strict_map=maps["A_to_B"],
            representations={
                name: (matrix, right_representations[name])
                for name, matrix in left_representations.items()
            },
            query_centroids=left_centroids,
            query_centroid_peer_counts=left.centroid_peer_counts,
            max_dose_fold=max_fold,
            n_shared_genes=n_shared,
        )
        _primary_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="B_to_A",
            query_dataset=dataset_b,
            target_dataset=dataset_a,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=right.frame,
            target_pool=left.frame,
            strict_map=maps["B_to_A"],
            representations={
                name: (matrix, left_representations[name])
                for name, matrix in right_representations.items()
            },
            query_centroids=right_centroids,
            query_centroid_peer_counts=right.centroid_peer_counts,
            max_dose_fold=max_fold,
            n_shared_genes=n_shared,
        )
    raw_validity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if workload in (REVIEWER_MINIMAL_WORKLOAD, SELECTED_WORKLOAD):
        raw_metrics = scale_similarity_metrics.get(RAW_SCALE_VARIANT, ())
        if raw_metrics:
            raw_validity.update(
                _score_selected_scale(
                    records=records,
                    dataset_a=dataset_a,
                    dataset_b=dataset_b,
                    cell_type=cell_type,
                    time_key=time_key,
                    left_pool=left.frame,
                    right_pool=right.frame,
                    left_matrix=left_logfc,
                    right_matrix=right_logfc,
                    strict_maps=maps,
                    scale_variant=RAW_SCALE_VARIANT,
                    n_shared_genes=n_shared,
                    max_dose_fold=max_fold,
                    max_peers=int(settings["max_baseline_peers"]),
                    sampling_seed=int(settings["peer_sampling_seed"]),
                    similarity_metrics=raw_metrics,
                    progress=progress,
                )
            )
        required_validity_metrics = {
            metric
            for scale_variant in w4_scale_variants
            for metric in scale_similarity_metrics.get(scale_variant, ())
        }
        for metric in sorted(required_validity_metrics - set(raw_validity)):
            _, valid_left, valid_right = similarity_matrix(
                left_logfc,
                right_logfc,
                metric,
            )
            raw_validity[metric] = (valid_left, valid_right)
    else:
        for direction, query_dataset, target_dataset, query_pool, target_pool, query_matrix, target_matrix, strict_map in (
            (
                "A_to_B",
                dataset_a,
                dataset_b,
                left.frame,
                right.frame,
                left_logfc,
                right_logfc,
                maps["A_to_B"],
            ),
            (
                "B_to_A",
                dataset_b,
                dataset_a,
                right.frame,
                left.frame,
                right_logfc,
                left_logfc,
                maps["B_to_A"],
            ),
        ):
            _strict_enriched_rows_for_direction(
                records=records,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                direction=direction,
                query_dataset=query_dataset,
                target_dataset=target_dataset,
                cell_type=cell_type,
                time_key=time_key,
                query_pool=query_pool,
                target_pool=target_pool,
                query_matrix=query_matrix,
                target_matrix=target_matrix,
                strict_map=strict_map,
                scale_variant=RAW_SCALE_VARIANT,
                n_shared_genes=n_shared,
                max_dose_fold=max_fold,
                max_peers=int(settings["max_baseline_peers"]),
                sampling_seed=int(settings["peer_sampling_seed"]),
                similarity_metrics=similarity_metrics,
                include_within_control=True,
            )

    for scale_variant in w4_scale_variants:
        left_w4 = w4_catalog.standardize_aligned_matrix(
            left_source,
            left.logfc,
            left_positions,
            scale_variant=scale_variant,
        )
        right_w4 = w4_catalog.standardize_aligned_matrix(
            right_source,
            right.logfc,
            right_positions,
            scale_variant=scale_variant,
        )
        w4_finite = (
            np.isfinite(left_w4).all(axis=0)
            & np.isfinite(right_w4).all(axis=0)
        )
        invalid_raw_genes = finite & ~w4_finite
        if invalid_raw_genes.any():
            raise AssertionError(
                f"W4 {scale_variant} would remove "
                f"{int(invalid_raw_genes.sum())} raw-valid shared genes for "
                f"{dataset_a} vs {dataset_b} / {cell_type} / {time_key}"
            )
        left_matrix = left_w4[:, finite]
        right_matrix = right_w4[:, finite]
        population_counts = (
            w4_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).population_row_count,
            w4_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).population_row_count,
        )
        if workload in (REVIEWER_MINIMAL_WORKLOAD, SELECTED_WORKLOAD):
            selected_metrics = scale_similarity_metrics.get(
                scale_variant,
                (),
            )
            _score_selected_scale(
                records=records,
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                cell_type=cell_type,
                time_key=time_key,
                left_pool=left.frame,
                right_pool=right.frame,
                left_matrix=left_matrix,
                right_matrix=right_matrix,
                strict_maps=maps,
                scale_variant=scale_variant,
                n_shared_genes=n_shared,
                max_dose_fold=max_fold,
                max_peers=int(settings["max_baseline_peers"]),
                sampling_seed=int(settings["peer_sampling_seed"]),
                similarity_metrics=selected_metrics,
                population_counts=population_counts,
                eligibility_masks=raw_validity,
                progress=progress,
            )
        else:
            for direction, query_dataset, target_dataset, query_pool, target_pool, query_matrix, target_matrix, strict_map in (
                (
                    "A_to_B",
                    dataset_a,
                    dataset_b,
                    left.frame,
                    right.frame,
                    left_matrix,
                    right_matrix,
                    maps["A_to_B"],
                ),
                (
                    "B_to_A",
                    dataset_b,
                    dataset_a,
                    right.frame,
                    left.frame,
                    right_matrix,
                    left_matrix,
                    maps["B_to_A"],
                ),
            ):
                _strict_enriched_rows_for_direction(
                    records=records,
                    dataset_a=dataset_a,
                    dataset_b=dataset_b,
                    direction=direction,
                    query_dataset=query_dataset,
                    target_dataset=target_dataset,
                    cell_type=cell_type,
                    time_key=time_key,
                    query_pool=query_pool,
                    target_pool=target_pool,
                    query_matrix=query_matrix,
                    target_matrix=target_matrix,
                    strict_map=strict_map,
                    scale_variant=scale_variant,
                    n_shared_genes=n_shared,
                    max_dose_fold=max_fold,
                    max_peers=int(settings["max_baseline_peers"]),
                    sampling_seed=int(settings["peer_sampling_seed"]),
                    similarity_metrics=similarity_metrics,
                    include_within_control=True,
                    population_counts=population_counts,
                    eligibility_query_matrix=(
                        left_logfc
                        if direction == "A_to_B"
                        else right_logfc
                    ),
                    eligibility_target_matrix=(
                        right_logfc
                        if direction == "A_to_B"
                        else left_logfc
                    ),
                )

    result = pd.DataFrame(list(records.values()))
    for column in RETRIEVAL_OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    result = result.loc[:, RETRIEVAL_OUTPUT_COLUMNS]
    return result.sort_values(IDENTITY_COLUMNS).reset_index(drop=True), diagnostics


def score_task(
    task: TaskSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_task_input(config, task)
    catalog = make_worker_catalog(config)

    def progress(phase: str, detail: str = "") -> None:
        report_task_progress(
            config,
            task,
            phase=phase,
            detail=detail,
        )

    try:
        workload = str(config["settings"].get("workload", FULL_WORKLOAD))
        w4_catalog = (
            make_worker_w4_catalog(config, catalog)
            if config["settings"].get("w4_scale_variants")
            else None
        )
        if workload == PEER_ONLY_WORKLOAD:
            metrics, records = _peer_only_context_records(
                frame,
                config=config,
                catalog=catalog,
                w4_catalog=w4_catalog,
                progress=progress,
            )
        else:
            metrics, records = _context_records(
                frame,
                config=config,
                catalog=catalog,
                w4_catalog=w4_catalog,
                progress=progress,
            )
        for record in records:
            record["task_id"] = task.task_id
        progress(
            "finished",
            f"rows={len(metrics)} diagnostics={len(records)}",
        )
        return metrics, diagnostic_frame(records)
    finally:
        catalog.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable cross-source retrieval scoring."
    )
    add_common_arguments(parser, analysis=ANALYSIS)
    add_computation_arguments(parser, computations=COMPUTATIONS)
    parser.add_argument(
        "--workload",
        choices=(
            REVIEWER_MINIMAL_WORKLOAD,
            PEER_ONLY_WORKLOAD,
            FULL_WORKLOAD,
        ),
        default=FULL_WORKLOAD,
        help=(
            "Scoring workload. reviewer-minimal runs strict logFC cosine and "
            "Spearman on raw and dataset-wide W4 scales; peer-only recomputes "
            "only individual-peer baselines from an existing retrieval TSV; "
            "full preserves the complete legacy bundle."
        ),
    )
    parser.add_argument(
        "--peer-only-from",
        type=Path,
        default=None,
        help="Existing reviewer-minimal retrieval TSV used by peer-only mode.",
    )
    parser.add_argument(
        "--peer-caps",
        default="256,512,1024,all",
        help="Comma-separated nested sensitivity caps; 'all' uses every peer.",
    )
    parser.add_argument(
        "--peer-scales",
        default="raw",
        help=(
            "Peer-only scales: raw or raw,dataset. Raw does not validate, "
            "load, or calculate W4."
        ),
    )
    parser.add_argument(
        "--peer-reference-cap",
        type=int,
        default=256,
        help=(
            "Production cap whose deterministic sample must be retained "
            "exactly as the first nested selection."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_computations:
        print_computations(COMPUTATIONS)
        return 0
    peer_only = args.workload == PEER_ONLY_WORKLOAD
    if args.compute and args.workload != FULL_WORKLOAD:
        raise ValueError(
            "--compute replaces the scoring bundle and cannot be combined "
            "with --workload reviewer-minimal or peer-only"
        )
    if peer_only and args.peer_only_from is None:
        raise ValueError("--peer-only-from is required for peer-only workload")
    if not peer_only and args.peer_only_from is not None:
        raise ValueError("--peer-only-from requires --workload peer-only")
    if peer_only and not str(args.run_tag).strip():
        args.run_tag = "peer_sensitivity"
    peer_caps = parse_peer_caps(args.peer_caps) if peer_only else ()
    peer_scales = parse_peer_scales(args.peer_scales) if peer_only else ()
    if peer_only:
        numeric_caps = [cap for cap in peer_caps if cap is not None]
        if args.peer_reference_cap < 1:
            raise ValueError("--peer-reference-cap must be positive")
        if any(cap < args.peer_reference_cap for cap in numeric_caps):
            raise ValueError(
                "Every numeric --peer-caps value must be at least "
                "--peer-reference-cap"
            )
    reviewer_minimal = args.workload == REVIEWER_MINIMAL_WORKLOAD
    selected_mode = bool(args.compute)
    effective_workload = (
        SELECTED_WORKLOAD if selected_mode else args.workload
    )
    if selected_mode:
        selected_computations = resolve_computations(
            args.compute,
            computations=COMPUTATIONS,
        )
    elif reviewer_minimal:
        selected_computations = (
            "raw-cosine",
            "raw-spearman",
            "w4-dataset-cosine",
            "w4-dataset-spearman",
        )
    elif peer_only:
        selected_computations = ()
    else:
        selected_scales = selected_w4_scale_variants(args.w4_scales)
        selected_computations = (
            "legacy-l2",
            "raw-l2",
            "raw-cosine",
            "raw-spearman",
            *(
                name
                for name, (scale, _) in COMPUTATION_SPEC.items()
                if scale in selected_scales
            ),
        )
    scale_similarity_metrics: dict[str, list[str]] = {}
    for computation in selected_computations:
        if computation not in COMPUTATION_SPEC:
            continue
        scale, metric = COMPUTATION_SPEC[computation]
        scale_similarity_metrics.setdefault(scale, []).append(metric)
    include_legacy_retrieval = "legacy-l2" in selected_computations
    if selected_mode:
        args.required_layers_override = (
            ("logFC", "t")
            if include_legacy_retrieval
            else ("logFC",)
        )
        args.require_adj_p_override = include_legacy_retrieval
    similarity_metrics = (
        REVIEWER_MINIMAL_SIMILARITIES
        if peer_only
        else tuple(
            metric
            for metric in RETRIEVAL_SIMILARITIES
            if any(
                metric in metrics
                for metrics in scale_similarity_metrics.values()
            )
            or (include_legacy_retrieval and metric == "negative_l2")
        )
    )
    if peer_only:
        w4_scale_variants = tuple(
            scale for scale in peer_scales if scale != RAW_SCALE_VARIANT
        )
    elif reviewer_minimal:
        w4_scale_variants = REVIEWER_MINIMAL_SCALE_VARIANTS
    elif selected_mode:
        w4_scale_variants = tuple(
            scale
            for scale in POPULATION_SCALE_VARIANTS
            if scale in scale_similarity_metrics
        )
    else:
        w4_scale_variants = selected_w4_scale_variants(args.w4_scales)
    settings = {
        "workload": effective_workload,
        "min_target_candidates": MIN_TARGET_CANDIDATES,
        "min_unique_compounds_per_side": MIN_UNIQUE_COMPOUNDS_PER_SIDE,
        "max_dose_fold_difference": (
            cross_source_core.DEFAULT_MATCH_SETTINGS.max_dose_fold_difference
        ),
        "retrieval_variants": (
            ["strict_matched_condition"]
            if (
                reviewer_minimal
                or peer_only
                or not include_legacy_retrieval
            )
            else list(RETRIEVAL_VARIANTS)
        ),
        "computations": list(selected_computations),
        "include_legacy_retrieval": include_legacy_retrieval,
        "scale_similarity_metrics": scale_similarity_metrics,
        "similarity_metrics": list(similarity_metrics),
        "max_baseline_peers": (
            args.peer_reference_cap
            if peer_only
            else args.max_baseline_peers
        ),
        "peer_sampling_seed": args.peer_sampling_seed,
        "w4_scale_variants": list(w4_scale_variants),
        "scorer_version": "parallel-retrieval-v3",
    }
    task_frame_factory = None
    final_metrics_name = FINAL_METRICS_NAME
    if peer_only:
        observed_path = Path(args.peer_only_from).resolve()
        settings.update(
            {
                "peer_caps": [
                    "all" if cap is None else int(cap)
                    for cap in peer_caps
                ],
                "peer_scales": list(peer_scales),
                "peer_reference_cap": int(args.peer_reference_cap),
                "observed_metrics_input": file_record(observed_path),
                "scorer_version": "parallel-retrieval-peer-only-v1",
            }
        )
        task_frame_factory = lambda scope: _load_peer_only_task_frame(
            scope,
            observed_metrics_path=observed_path,
            scales=peer_scales,
        )
        final_metrics_name = PEER_ONLY_FINAL_METRICS_NAME
    run_analysis(
        analysis=ANALYSIS,
        scorer_module="scripts.run_overlap_group_rep_retrieval_metrics",
        args=args,
        context_columns=CONTEXT_COLUMNS,
        rows_per_shard=None,
        final_metrics_name=final_metrics_name,
        settings=settings,
        code_paths=[
            Path(__file__),
            REPO_ROOT / "scripts" / "cross_source_parallel.py",
            REPO_ROOT / "scripts" / "cross_source_scoring.py",
            REPO_ROOT / "scripts" / "cross_source_core.py",
            REPO_ROOT / "scripts" / "peer_baselines.py",
            REPO_ROOT / "scripts" / "population_zscore.py",
            REPO_ROOT / "scripts" / "notebook_cache.py",
        ],
        task_frame_factory=task_frame_factory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
