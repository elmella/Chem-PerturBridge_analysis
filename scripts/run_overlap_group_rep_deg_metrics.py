#!/usr/bin/env python3
"""Parallel, resumable matched-pair DEG scoring for Tables 4 and 5."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

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
    make_worker_catalog,
    make_worker_w4_catalog,
    print_computations,
    read_task_input,
    resolve_computations,
    run_analysis,
    selected_w4_scale_variants,
)
from scripts.cross_source_scoring import (
    ADJ_PVALUE_LAYER_PREFERENCES,
    DEG_DEFINITIONS,
    DE_OVERLAP_K_VALUES,
    PEER_BASELINE_VARIANTS,
    deg_mask,
    deg_restricted_spearman,
    direction_agreement_with_masks,
    finite_values_mask,
    observed_deg_metrics,
    sample_baseline_deg_metrics,
    score_deg_peer_distribution,
    select_source_peers,
    strict_symmetric_mean,
)
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
)


ANALYSIS = "deg"
FINAL_METRICS_NAME = "deg_scored_metrics.tsv"
CONTEXT_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "left_dose_key",
    "right_dose_key",
)
PEER_DEG_DEFINITION = "p05"
PEER_METRICS = ("deg_lfc_spearman", "direction_agreement")
COMPUTATIONS = {
    "raw": "Raw matched-pair DEG metrics and individual-peer baselines.",
    "w4-dataset": "Dataset-wide per-gene population-z-score DEG metrics.",
    "w4-dataset-cell-type": (
        "Dataset-by-cell-type per-gene population-z-score DEG metrics."
    ),
}
COMPUTATION_TO_SCALE = {
    "w4-dataset": PER_GENE_DATASET_VARIANT,
    "w4-dataset-cell-type": PER_GENE_DATASET_CELL_TYPE_VARIANT,
}


def _lookup(row: Mapping[str, Any], side: str) -> dict[str, str]:
    return {
        "pubchem_cid": str(row["pubchem_cid"]),
        "dose_key": str(row[f"{side}_dose_key"]),
        "time_key": str(row["time_key"]),
    }


def _adj_layer(source: cross_source_core.LineSource) -> str:
    return cross_source_core.first_available_layer(
        source,
        ADJ_PVALUE_LAYER_PREFERENCES,
    )


def _centroid(
    source: cross_source_core.LineSource,
    obs_id: str,
    lookup: Mapping[str, str],
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    value = source.get_baseline_vector(obs_id, "logFC", **lookup)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)[positions]


def _pair_baseline_fields(
    metrics: dict[str, Any],
    *,
    definition_key: str,
    suffix: str = "",
) -> None:
    observed_lookup = {
        "deg_lfc_spearman": (
            f"observed_deg_lfc_spearman_sym_{definition_key}{suffix}"
        ),
        "de_overlap_refn": (
            f"observed_de_overlap_refn_sym_{definition_key}{suffix}"
        ),
        "direction_agreement": (
            f"observed_direction_agreement_{definition_key}{suffix}"
        ),
        **{
            f"de_overlap_k{k}": (
                f"observed_de_overlap_k{k}_{definition_key}{suffix}"
            )
            for k in DE_OVERLAP_K_VALUES
        },
    }
    for metric_name, observed_key in observed_lookup.items():
        left_key = (
            f"left_baseline_{metric_name}_{definition_key}{suffix}"
        )
        right_key = (
            f"right_baseline_{metric_name}_{definition_key}{suffix}"
        )
        pair_key = (
            f"baseline_pair_{metric_name}_{definition_key}{suffix}"
        )
        delta_key = (
            f"delta_vs_baseline_pair_{metric_name}_{definition_key}{suffix}"
        )
        pair_value = cross_source_core.mean_available(
            [metrics[left_key], metrics[right_key]]
        )
        metrics[pair_key] = pair_value
        metrics[delta_key] = cross_source_core.difference_if_both_defined(
            metrics[observed_key],
            pair_value,
        )


def _l1000_restricted_positions(
    *,
    catalog: cross_source_core.LineSourceCatalog,
    left_source: cross_source_core.LineSource,
    right_source: cross_source_core.LineSource,
    retained_lines: Mapping[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {left_source.dataset_name, right_source.dataset_name} != {
        "sciplex",
        "tahoe",
    }:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=object), empty, empty
    cell_type = left_source.cell_type
    l1000_names = [
        dataset
        for dataset in ("l1000_phase1", "l1000_phase2")
        if cell_type in retained_lines.get(dataset, [])
    ]
    if not l1000_names:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=object), empty, empty
    sources = [
        left_source,
        right_source,
        *[
            catalog.get_line_source(dataset, cell_type)
            for dataset in l1000_names
        ],
    ]
    shared = sorted(
        set.intersection(
            *[
                {str(gene) for gene in source.unique_gene_keys.tolist()}
                for source in sources
            ]
        )
    )
    genes = np.asarray(shared, dtype=object)
    left_positions = np.fromiter(
        (left_source.gene_to_pos[str(gene)] for gene in shared),
        dtype=np.int64,
        count=len(shared),
    )
    right_positions = np.fromiter(
        (right_source.gene_to_pos[str(gene)] for gene in shared),
        dtype=np.int64,
        count=len(shared),
    )
    return genes, left_positions, right_positions


def _metric_value(
    *,
    metric: str,
    sample: np.ndarray,
    comparison: Optional[np.ndarray],
    sample_adj_p: np.ndarray,
) -> float:
    if comparison is None:
        return float("nan")
    mask = deg_mask(sample, sample_adj_p, PEER_DEG_DEFINITION)
    if metric == "deg_lfc_spearman":
        return deg_restricted_spearman(sample, comparison, mask)
    if metric == "direction_agreement":
        return direction_agreement_with_masks(
            sample,
            comparison,
            mask,
            finite_values_mask(comparison),
        )
    raise ValueError(f"Unsupported DEG metric: {metric}")


def _score_deg_baselines(
    *,
    left_sample: np.ndarray,
    right_sample: np.ndarray,
    left_adj_p: np.ndarray,
    right_adj_p: np.ndarray,
    left_centroid: Optional[np.ndarray],
    right_centroid: Optional[np.ndarray],
    left_peers: np.ndarray,
    right_peers: np.ndarray,
    prefix: str,
    include_direction_agreement: bool = True,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    observed_by_metric = {
        "deg_lfc_spearman": (
            deg_restricted_spearman(
                left_sample,
                right_sample,
                deg_mask(
                    left_sample,
                    left_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
            deg_restricted_spearman(
                left_sample,
                right_sample,
                deg_mask(
                    right_sample,
                    right_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
        ),
        "direction_agreement": (
            direction_agreement_with_masks(
                left_sample,
                right_sample,
                deg_mask(
                    left_sample,
                    left_adj_p,
                    PEER_DEG_DEFINITION,
                ),
                deg_mask(
                    right_sample,
                    right_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
        )
        * 2,
    }
    sides = {
        "left": {
            "sample": left_sample,
            "adj_p": left_adj_p,
            "source_centroid": left_centroid,
            "target_centroid": right_centroid,
            "source_peer": left_peers,
            "target_peer": right_peers,
            "observed_index": 0,
        },
        "right": {
            "sample": right_sample,
            "adj_p": right_adj_p,
            "source_centroid": right_centroid,
            "target_centroid": left_centroid,
            "source_peer": right_peers,
            "target_peer": left_peers,
            "observed_index": 1,
        },
    }
    metrics = (
        PEER_METRICS
        if include_direction_agreement
        else ("deg_lfc_spearman",)
    )
    for metric in metrics:
        observed_left, observed_right = observed_by_metric[metric]
        observed_pair = (
            strict_symmetric_mean(observed_left, observed_right)
            if metric == "deg_lfc_spearman"
            else observed_left
        )
        if prefix == "w4" and metric == "deg_lfc_spearman":
            record[
                "w4_observed_deg_lfc_spearman_left_ref_p05"
            ] = observed_left
            record[
                "w4_observed_deg_lfc_spearman_right_ref_p05"
            ] = observed_right
            record["w4_observed_deg_lfc_spearman_sym_p05"] = observed_pair
        else:
            record[
                f"{prefix}_observed_{metric}_{PEER_DEG_DEFINITION}"
            ] = observed_pair
        for variant in PEER_BASELINE_VARIANTS:
            means: list[float] = []
            sds: list[float] = []
            fractions: list[float] = []
            percentiles: list[float] = []
            for side, spec in sides.items():
                observed_side = (observed_left, observed_right)[
                    spec["observed_index"]
                ]
                stem = (
                    f"{prefix}_{variant}_{metric}_{side}_"
                    f"{PEER_DEG_DEFINITION}"
                )
                if variant.endswith("_centroid"):
                    value = _metric_value(
                        metric=metric,
                        sample=spec["sample"],
                        comparison=spec[variant],
                        sample_adj_p=spec["adj_p"],
                    )
                    record[stem] = value
                    means.append(value)
                    continue
                summary = score_deg_peer_distribution(
                    observed=observed_side,
                    sample=spec["sample"],
                    comparison_matrix=spec[variant],
                    sample_adj_p=spec["adj_p"],
                    metric=metric,
                    stem=stem,
                )
                record.update(summary)
                means.append(float(summary[f"{stem}_mean_score"]))
                sds.append(float(summary[f"{stem}_sd_score"]))
                fractions.append(
                    float(summary[f"{stem}_fraction_below_observed"])
                )
                percentiles.append(
                    float(summary[f"{stem}_corrected_percentile"])
                )
            pair_stem = (
                f"{prefix}_{variant}_{metric}_pair_"
                f"{PEER_DEG_DEFINITION}"
            )
            pair_value = cross_source_core.mean_available(means)
            record[pair_stem] = pair_value
            record[
                f"{prefix}_delta_vs_{variant}_{metric}_"
                f"{PEER_DEG_DEFINITION}"
            ] = (
                cross_source_core.difference_if_both_defined(
                    observed_pair,
                    pair_value,
                )
            )
            if not variant.endswith("_centroid"):
                record[f"{pair_stem}_sd_score"] = (
                    cross_source_core.mean_available(sds)
                )
                record[f"{pair_stem}_fraction_below_observed"] = (
                    cross_source_core.mean_available(fractions)
                )
                record[f"{pair_stem}_corrected_percentile"] = (
                    cross_source_core.mean_available(percentiles)
                )
    return record


def _raw_deg_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    retained_lines: Mapping[str, list[str]],
    max_peers: int,
    sampling_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_a = str(row["dataset_a"])
    dataset_b = str(row["dataset_b"])
    cell_type = str(row["cell_type"])
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
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_adj_layer = _adj_layer(left_source)
    right_adj_layer = _adj_layer(right_source)
    left_logfc_full = left_source.get_vector(
        left_obs_id,
        "logFC",
        **left_lookup,
    )
    right_logfc_full = right_source.get_vector(
        right_obs_id,
        "logFC",
        **right_lookup,
    )
    left_adj_full = left_source.get_vector(
        left_obs_id,
        left_adj_layer,
        **left_lookup,
    )
    right_adj_full = right_source.get_vector(
        right_obs_id,
        right_adj_layer,
        **right_lookup,
    )
    left_logfc = left_logfc_full[left_positions]
    right_logfc = right_logfc_full[right_positions]
    left_adj_p = left_adj_full[left_positions]
    right_adj_p = right_adj_full[right_positions]
    left_centroid = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        left_positions,
    )
    right_centroid = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        right_positions,
    )

    values: dict[str, Any] = {}
    for definition in DEG_DEFINITIONS:
        values.update(
            observed_deg_metrics(
                gene_keys=shared_genes,
                logfc_left=left_logfc,
                logfc_right=right_logfc,
                adj_p_left=left_adj_p,
                adj_p_right=right_adj_p,
                definition_key=definition,
            )
        )
        values.update(
            sample_baseline_deg_metrics(
                prefix="left",
                gene_keys=shared_genes,
                sample_logfc=left_logfc,
                baseline_logfc=left_centroid,
                sample_adj_p=left_adj_p,
                definition_key=definition,
            )
        )
        values.update(
            sample_baseline_deg_metrics(
                prefix="right",
                gene_keys=shared_genes,
                sample_logfc=right_logfc,
                baseline_logfc=right_centroid,
                sample_adj_p=right_adj_p,
                definition_key=definition,
            )
        )
        _pair_baseline_fields(values, definition_key=definition)

    restricted_genes, left_restricted, right_restricted = (
        _l1000_restricted_positions(
            catalog=catalog,
            left_source=left_source,
            right_source=right_source,
            retained_lines=retained_lines,
        )
    )
    suffix = "_l1000_restricted"
    if restricted_genes.size >= 2:
        left_restricted_centroid = (
            None
            if left_source.get_baseline_vector(
                left_obs_id,
                "logFC",
                **left_lookup,
            )
            is None
            else np.asarray(
                left_source.get_baseline_vector(
                    left_obs_id,
                    "logFC",
                    **left_lookup,
                )
            )[left_restricted]
        )
        right_restricted_centroid = (
            None
            if right_source.get_baseline_vector(
                right_obs_id,
                "logFC",
                **right_lookup,
            )
            is None
            else np.asarray(
                right_source.get_baseline_vector(
                    right_obs_id,
                    "logFC",
                    **right_lookup,
                )
            )[right_restricted]
        )
        for definition in DEG_DEFINITIONS:
            observed = observed_deg_metrics(
                gene_keys=restricted_genes,
                logfc_left=left_logfc_full[left_restricted],
                logfc_right=right_logfc_full[right_restricted],
                adj_p_left=left_adj_full[left_restricted],
                adj_p_right=right_adj_full[right_restricted],
                definition_key=definition,
            )
            left_baseline = sample_baseline_deg_metrics(
                prefix="left",
                gene_keys=restricted_genes,
                sample_logfc=left_logfc_full[left_restricted],
                baseline_logfc=left_restricted_centroid,
                sample_adj_p=left_adj_full[left_restricted],
                definition_key=definition,
            )
            right_baseline = sample_baseline_deg_metrics(
                prefix="right",
                gene_keys=restricted_genes,
                sample_logfc=right_logfc_full[right_restricted],
                baseline_logfc=right_restricted_centroid,
                sample_adj_p=right_adj_full[right_restricted],
                definition_key=definition,
            )
            for mapping in (observed, left_baseline, right_baseline):
                values.update(
                    {f"{key}{suffix}": value for key, value in mapping.items()}
                )
            _pair_baseline_fields(
                values,
                definition_key=definition,
                suffix=suffix,
            )

    left_selected = select_source_peers(
        left_source,
        left_obs_id,
        left_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    right_selected = select_source_peers(
        right_source,
        right_obs_id,
        right_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    if (
        left_selected.total_count
        != left_source.baseline_peer_count(left_obs_id, **left_lookup)
    ):
        raise AssertionError("Left peer membership disagrees with centroid")
    if (
        right_selected.total_count
        != right_source.baseline_peer_count(right_obs_id, **right_lookup)
    ):
        raise AssertionError("Right peer membership disagrees with centroid")
    left_peers = left_selected.stratum.values[
        left_selected.row_indices
    ][:, left_positions]
    right_peers = right_selected.stratum.values[
        right_selected.row_indices
    ][:, right_positions]
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "n_l1000_restricted_common_genes": int(restricted_genes.size),
        "left_adj_pvalue_layer": left_adj_layer,
        "right_adj_pvalue_layer": right_adj_layer,
        "left_baseline_peer_count": left_selected.total_count,
        "right_baseline_peer_count": right_selected.total_count,
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
        **values,
    }
    record.update(
        _score_deg_baselines(
            left_sample=left_logfc,
            right_sample=right_logfc,
            left_adj_p=left_adj_p,
            right_adj_p=right_adj_p,
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peers=left_peers,
            right_peers=right_peers,
            prefix="pb",
        )
    )
    state = {
        "pubchem_cid": str(row["pubchem_cid"]),
        "left_source": left_source,
        "right_source": right_source,
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_logfc_full": left_logfc_full,
        "right_logfc_full": right_logfc_full,
        "left_adj_p": left_adj_p,
        "right_adj_p": right_adj_p,
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _deg_w4_base_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: int,
    sampling_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare only the shared inputs required by selected W4 kernels."""
    dataset_a = str(row["dataset_a"])
    dataset_b = str(row["dataset_b"])
    cell_type = str(row["cell_type"])
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
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_adj_layer = _adj_layer(left_source)
    right_adj_layer = _adj_layer(right_source)
    left_logfc_full = left_source.get_vector(
        left_obs_id,
        "logFC",
        **left_lookup,
    )
    right_logfc_full = right_source.get_vector(
        right_obs_id,
        "logFC",
        **right_lookup,
    )
    left_adj_p = left_source.get_vector(
        left_obs_id,
        left_adj_layer,
        **left_lookup,
    )[left_positions]
    right_adj_p = right_source.get_vector(
        right_obs_id,
        right_adj_layer,
        **right_lookup,
    )[right_positions]
    left_selected = select_source_peers(
        left_source,
        left_obs_id,
        left_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    right_selected = select_source_peers(
        right_source,
        right_obs_id,
        right_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    if (
        left_selected.total_count
        != left_source.baseline_peer_count(left_obs_id, **left_lookup)
    ):
        raise AssertionError("Left peer membership disagrees with centroid")
    if (
        right_selected.total_count
        != right_source.baseline_peer_count(right_obs_id, **right_lookup)
    ):
        raise AssertionError("Right peer membership disagrees with centroid")
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "left_adj_pvalue_layer": left_adj_layer,
        "right_adj_pvalue_layer": right_adj_layer,
        "left_baseline_peer_count": left_selected.total_count,
        "right_baseline_peer_count": right_selected.total_count,
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
    }
    state = {
        "pubchem_cid": str(row["pubchem_cid"]),
        "left_source": left_source,
        "right_source": right_source,
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_logfc_full": left_logfc_full,
        "right_logfc_full": right_logfc_full,
        "left_adj_p": left_adj_p,
        "right_adj_p": right_adj_p,
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _w4_deg_columns(
    *,
    state: Mapping[str, Any],
    w4_catalog,
    scale_variant: str,
) -> dict[str, Any]:
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_selected = state["left_selected"]
    right_selected = state["right_selected"]
    pubchem_cid = str(state["pubchem_cid"])
    left_sample = w4_catalog.standardize_vector(
        left_source,
        state["left_logfc_full"],
        scale_variant=scale_variant,
    )[left_positions]
    right_sample = w4_catalog.standardize_vector(
        right_source,
        state["right_logfc_full"],
        scale_variant=scale_variant,
    )[right_positions]
    left_matrix = w4_catalog.standardize_matrix(
        left_source,
        left_selected.stratum.values,
        scale_variant=scale_variant,
    )[:, left_positions]
    right_matrix = w4_catalog.standardize_matrix(
        right_source,
        right_selected.stratum.values,
        scale_variant=scale_variant,
    )[:, right_positions]
    left_all_rows = left_selected.stratum.different_compound_peer_indices(
        pubchem_cid
    )
    right_all_rows = right_selected.stratum.different_compound_peer_indices(
        pubchem_cid
    )
    left_centroid = (
        None
        if not len(left_all_rows)
        else left_matrix[left_all_rows].mean(axis=0, dtype=np.float64)
    )
    right_centroid = (
        None
        if not len(right_all_rows)
        else right_matrix[right_all_rows].mean(axis=0, dtype=np.float64)
    )
    values: dict[str, Any] = {
        "n_common_genes": int(len(left_positions)),
        "n_w4_common_genes": int(
            np.sum(np.isfinite(left_sample) & np.isfinite(right_sample))
        ),
        "left_population_row_count": (
            w4_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        "right_population_row_count": (
            w4_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        "left_population_valid_genes": (
            w4_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        "right_population_valid_genes": (
            w4_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "n_deg_left_p05": int(
            deg_mask(
                state["left_logfc_full"][left_positions],
                state["left_adj_p"],
                "p05",
            ).sum()
        ),
        "n_deg_right_p05": int(
            deg_mask(
                state["right_logfc_full"][right_positions],
                state["right_adj_p"],
                "p05",
            ).sum()
        ),
    }
    values.update(
        _score_deg_baselines(
            left_sample=left_sample,
            right_sample=right_sample,
            left_adj_p=state["left_adj_p"],
            right_adj_p=state["right_adj_p"],
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peers=left_matrix[left_selected.row_indices],
            right_peers=right_matrix[right_selected.row_indices],
            prefix="w4",
            include_direction_agreement=False,
        )
    )
    return {
        f"{scale_variant}__{key}": value
        for key, value in values.items()
    }


def score_deg_row(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    w4_catalog,
    retained_lines: Mapping[str, list[str]],
    max_peers: int,
    sampling_seed: int,
    include_raw_metrics: bool = True,
    w4_scale_variants: tuple[str, ...] = POPULATION_SCALE_VARIANTS,
) -> dict[str, Any]:
    if include_raw_metrics:
        record, state = _raw_deg_record(
            row,
            catalog=catalog,
            retained_lines=retained_lines,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    else:
        record, state = _deg_w4_base_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    for scale_variant in w4_scale_variants:
        record.update(
            _w4_deg_columns(
                state=state,
                w4_catalog=w4_catalog,
                scale_variant=scale_variant,
            )
        )
    return record


def score_task(
    task: TaskSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_task_input(config, task)
    settings = config["settings"]
    catalog = make_worker_catalog(config)
    diagnostics: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        w4_catalog = (
            make_worker_w4_catalog(config, catalog)
            if settings["w4_scale_variants"]
            else None
        )
        for _, row in frame.iterrows():
            try:
                records.append(
                    score_deg_row(
                        row,
                        catalog=catalog,
                        w4_catalog=w4_catalog,
                        retained_lines=config["retained_lines"],
                        max_peers=int(settings["max_baseline_peers"]),
                        sampling_seed=int(settings["peer_sampling_seed"]),
                        include_raw_metrics=bool(
                            settings["include_raw_metrics"]
                        ),
                        w4_scale_variants=tuple(
                            settings["w4_scale_variants"]
                        ),
                    )
                )
            except KeyError as exc:
                if "Could not resolve obs_id=" not in str(exc):
                    raise
                diagnostics.append(
                    {
                        "task_id": task.task_id,
                        "analysis": ANALYSIS,
                        "dataset_a": row.get("dataset_a", ""),
                        "dataset_b": row.get("dataset_b", ""),
                        "cell_type": row.get("cell_type", ""),
                        "time_key": row.get("time_key", ""),
                        "left_dose_key": row.get("left_dose_key", ""),
                        "right_dose_key": row.get("right_dose_key", ""),
                        "left_obs_id": row.get("left_obs_id", ""),
                        "right_obs_id": row.get("right_obs_id", ""),
                        "stage": "deg_scoring",
                        "reason": "unresolved_source_row",
                        "error": str(exc),
                    }
                )
    finally:
        catalog.close()
    return pd.DataFrame(records), diagnostic_frame(diagnostics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable matched-pair DEG metric scoring."
    )
    add_common_arguments(parser, analysis=ANALYSIS)
    add_computation_arguments(parser, computations=COMPUTATIONS)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_computations:
        print_computations(COMPUTATIONS)
        return 0
    if args.compute:
        selected_computations = resolve_computations(
            args.compute,
            computations=COMPUTATIONS,
        )
        args.required_layers_override = ("logFC",)
        args.require_adj_p_override = True
        w4_scale_variants = tuple(
            COMPUTATION_TO_SCALE[name]
            for name in selected_computations
            if name in COMPUTATION_TO_SCALE
        )
    else:
        w4_scale_variants = selected_w4_scale_variants(args.w4_scales)
        selected_computations = (
            "raw",
            *(
                name
                for name in COMPUTATIONS
                if COMPUTATION_TO_SCALE.get(name) in w4_scale_variants
            ),
        )
    settings = {
        "deg_p_threshold": 0.05,
        "deg_abs_logfc_threshold": 0.2,
        "deg_definitions": list(DEG_DEFINITIONS),
        "de_overlap_k": list(DE_OVERLAP_K_VALUES),
        "max_baseline_peers": args.max_baseline_peers,
        "peer_sampling_seed": args.peer_sampling_seed,
        "computations": list(selected_computations),
        "include_raw_metrics": "raw" in selected_computations,
        "w4_scale_variants": list(w4_scale_variants),
        "scorer_version": "parallel-deg-v3",
    }
    run_analysis(
        analysis=ANALYSIS,
        scorer_module="scripts.run_overlap_group_rep_deg_metrics",
        args=args,
        context_columns=CONTEXT_COLUMNS,
        rows_per_shard=args.rows_per_shard,
        final_metrics_name=FINAL_METRICS_NAME,
        settings=settings,
        code_paths=[
            Path(__file__),
            REPO_ROOT / "scripts" / "cross_source_parallel.py",
            REPO_ROOT / "scripts" / "cross_source_scoring.py",
            REPO_ROOT / "scripts" / "cross_source_core.py",
            REPO_ROOT / "scripts" / "peer_baselines.py",
            REPO_ROOT / "scripts" / "population_zscore.py",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
