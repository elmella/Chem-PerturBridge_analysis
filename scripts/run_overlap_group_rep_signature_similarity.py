#!/usr/bin/env python3
"""Parallel, resumable Table 6 matched-signature scoring.

This command performs the expensive row-level work from
``overlap_group_rep_signature_similarity.ipynb``.  Cheap summaries, confidence
intervals, and plotting intentionally remain in the notebook.
"""

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
    score_signature_baselines,
    select_source_peers,
)
from scripts.cross_source_strata import SignatureStratum
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
)


ANALYSIS = "signature"
FINAL_METRICS_NAME = "signature_scored_metrics.tsv"
CONTEXT_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "left_dose_key",
    "right_dose_key",
)
COMPUTATIONS = {
    "raw": "Raw matched-signature metrics and individual-peer baselines.",
    "w4-dataset": (
        "Dataset-wide per-gene population-z-score signature metrics."
    ),
    "w4-dataset-cell-type": (
        "Dataset-by-cell-type per-gene population-z-score signature metrics."
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


def _centroid(
    source: cross_source_core.LineSource,
    obs_id: str,
    lookup: Mapping[str, str],
    layer: str,
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    value = source.get_baseline_vector(obs_id, layer, **lookup)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)[positions]


def _raw_signature_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: Optional[int],
    sampling_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
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
    global_genes, left_global_positions = catalog.global_gene_positions(
        left_source
    )
    _, right_global_positions = catalog.global_gene_positions(right_source)

    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
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
    left_t_full = left_source.get_vector(left_obs_id, "t", **left_lookup)
    right_t_full = right_source.get_vector(right_obs_id, "t", **right_lookup)
    left_logfc = left_logfc_full[left_positions]
    right_logfc = right_logfc_full[right_positions]
    left_t = left_t_full[left_positions]
    right_t = right_t_full[right_positions]
    observed = cross_source_core.score_signature_pair(
        left_logfc=left_logfc,
        right_logfc=right_logfc,
        left_t=left_t,
        right_t=right_t,
    )
    global_observed = cross_source_core.empty_score_dict()
    if global_genes.size >= 2:
        global_observed = cross_source_core.score_signature_pair(
            left_logfc=left_logfc_full[left_global_positions],
            right_logfc=right_logfc_full[right_global_positions],
            left_t=left_t_full[left_global_positions],
            right_t=right_t_full[right_global_positions],
        )

    left_centroid_logfc = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "logFC",
        left_positions,
    )
    right_centroid_logfc = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "logFC",
        right_positions,
    )
    left_centroid_t = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "t",
        left_positions,
    )
    right_centroid_t = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "t",
        right_positions,
    )
    left_centroid_logfc_global = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "logFC",
        left_global_positions,
    )
    right_centroid_logfc_global = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "logFC",
        right_global_positions,
    )
    left_centroid_t_global = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "t",
        left_global_positions,
    )
    right_centroid_t_global = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "t",
        right_global_positions,
    )
    left_baseline = cross_source_core.empty_score_dict()
    if left_centroid_logfc is not None and left_centroid_t is not None:
        left_baseline = cross_source_core.score_signature_pair(
            left_logfc=left_logfc,
            right_logfc=left_centroid_logfc,
            left_t=left_t,
            right_t=left_centroid_t,
        )
    right_baseline = cross_source_core.empty_score_dict()
    if right_centroid_logfc is not None and right_centroid_t is not None:
        right_baseline = cross_source_core.score_signature_pair(
            left_logfc=right_logfc,
            right_logfc=right_centroid_logfc,
            left_t=right_t,
            right_t=right_centroid_t,
        )
    left_baseline_global = cross_source_core.empty_score_dict()
    if (
        global_genes.size >= 2
        and left_centroid_logfc_global is not None
        and left_centroid_t_global is not None
    ):
        left_baseline_global = cross_source_core.score_signature_pair(
            left_logfc=left_logfc_full[left_global_positions],
            right_logfc=left_centroid_logfc_global,
            left_t=left_t_full[left_global_positions],
            right_t=left_centroid_t_global,
        )
    right_baseline_global = cross_source_core.empty_score_dict()
    if (
        global_genes.size >= 2
        and right_centroid_logfc_global is not None
        and right_centroid_t_global is not None
    ):
        right_baseline_global = cross_source_core.score_signature_pair(
            left_logfc=right_logfc_full[right_global_positions],
            right_logfc=right_centroid_logfc_global,
            left_t=right_t_full[right_global_positions],
            right_t=right_centroid_t_global,
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
    expected_left = left_source.baseline_peer_count(
        left_obs_id,
        **left_lookup,
    )
    expected_right = right_source.baseline_peer_count(
        right_obs_id,
        **right_lookup,
    )
    if left_selected.total_count != expected_left:
        raise AssertionError(
            f"Left peer set has {left_selected.total_count} rows; "
            f"centroid used {expected_left}"
        )
    if right_selected.total_count != expected_right:
        raise AssertionError(
            f"Right peer set has {right_selected.total_count} rows; "
            f"centroid used {expected_right}"
        )
    left_peers = left_selected.stratum.values[
        left_selected.row_indices
    ][:, left_positions]
    right_peers = right_selected.stratum.values[
        right_selected.row_indices
    ][:, right_positions]
    overlap_name = f"signed_overlap_t_top{cross_source_core.TOP_K}"
    left_mean_abs_t = cross_source_core.mean_available(np.abs(left_t).tolist())
    right_mean_abs_t = cross_source_core.mean_available(np.abs(right_t).tolist())
    left_mean_abs_t_global = (
        cross_source_core.mean_available(
            np.abs(left_t_full[left_global_positions]).tolist()
        )
        if global_genes.size >= 2
        else float("nan")
    )
    right_mean_abs_t_global = (
        cross_source_core.mean_available(
            np.abs(right_t_full[right_global_positions]).tolist()
        )
        if global_genes.size >= 2
        else float("nan")
    )
    record: dict[str, Any] = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "n_global_common_genes": int(global_genes.size),
        "left_mean_abs_t": left_mean_abs_t,
        "right_mean_abs_t": right_mean_abs_t,
        "pair_mean_abs_t": cross_source_core.mean_available(
            [left_mean_abs_t, right_mean_abs_t]
        ),
        "left_mean_abs_t_global": left_mean_abs_t_global,
        "right_mean_abs_t_global": right_mean_abs_t_global,
        "pair_mean_abs_t_global": cross_source_core.mean_available(
            [left_mean_abs_t_global, right_mean_abs_t_global]
        ),
        "observed_spearman_logfc": observed["spearman_logfc"],
        "observed_spearman_logfc_global": global_observed["spearman_logfc"],
        "observed_spearman_t": observed["spearman_t"],
        "observed_spearman_t_global": global_observed["spearman_t"],
        f"observed_{overlap_name}": observed[overlap_name],
        f"observed_{overlap_name}_global": global_observed[overlap_name],
        "left_baseline_peer_count": expected_left,
        "right_baseline_peer_count": expected_right,
        "left_baseline_spearman_logfc": left_baseline["spearman_logfc"],
        "right_baseline_spearman_logfc": right_baseline["spearman_logfc"],
        "left_baseline_spearman_logfc_global": (
            left_baseline_global["spearman_logfc"]
        ),
        "right_baseline_spearman_logfc_global": (
            right_baseline_global["spearman_logfc"]
        ),
        "left_baseline_spearman_t": left_baseline["spearman_t"],
        "right_baseline_spearman_t": right_baseline["spearman_t"],
        "left_baseline_spearman_t_global": left_baseline_global["spearman_t"],
        "right_baseline_spearman_t_global": (
            right_baseline_global["spearman_t"]
        ),
        f"left_baseline_{overlap_name}": left_baseline[overlap_name],
        f"right_baseline_{overlap_name}": right_baseline[overlap_name],
        f"left_baseline_{overlap_name}_global": (
            left_baseline_global[overlap_name]
        ),
        f"right_baseline_{overlap_name}_global": (
            right_baseline_global[overlap_name]
        ),
        "baseline_pair_mean_spearman_logfc": (
            cross_source_core.mean_available(
                [
                    left_baseline["spearman_logfc"],
                    right_baseline["spearman_logfc"],
                ]
            )
        ),
        "baseline_pair_mean_spearman_t": cross_source_core.mean_available(
            [
                left_baseline["spearman_t"],
                right_baseline["spearman_t"],
            ]
        ),
        "baseline_pair_mean_spearman_logfc_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global["spearman_logfc"],
                    right_baseline_global["spearman_logfc"],
                ]
            )
        ),
        "baseline_pair_mean_spearman_t_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global["spearman_t"],
                    right_baseline_global["spearman_t"],
                ]
            )
        ),
        f"baseline_pair_mean_{overlap_name}": (
            cross_source_core.mean_available(
                [
                    left_baseline[overlap_name],
                    right_baseline[overlap_name],
                ]
            )
        ),
        f"baseline_pair_mean_{overlap_name}_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global[overlap_name],
                    right_baseline_global[overlap_name],
                ]
            )
        ),
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
        "pb_observed_spearman_logfc": observed["spearman_logfc"],
    }
    record.update(
        score_signature_baselines(
            observed=observed["spearman_logfc"],
            left_sample=left_logfc,
            right_sample=right_logfc,
            left_centroid=left_centroid_logfc,
            right_centroid=right_centroid_logfc,
            left_peer_matrix=left_peers,
            right_peer_matrix=right_peers,
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
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _signature_w4_base_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: Optional[int],
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
    expected_left = left_source.baseline_peer_count(
        left_obs_id,
        **left_lookup,
    )
    expected_right = right_source.baseline_peer_count(
        right_obs_id,
        **right_lookup,
    )
    if left_selected.total_count != expected_left:
        raise AssertionError(
            f"Left peer set has {left_selected.total_count} rows; "
            f"centroid used {expected_left}"
        )
    if right_selected.total_count != expected_right:
        raise AssertionError(
            f"Right peer set has {right_selected.total_count} rows; "
            f"centroid used {expected_right}"
        )
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "left_baseline_peer_count": expected_left,
        "right_baseline_peer_count": expected_right,
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
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _w4_signature_columns(
    *,
    state: Mapping[str, Any],
    w4_catalog,
    scale_variant: str,
    prepared_strata: dict[tuple[Any, ...], SignatureStratum],
) -> dict[str, Any]:
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_selected = state["left_selected"]
    right_selected = state["right_selected"]
    left_full = state["left_logfc_full"]
    right_full = state["right_logfc_full"]
    pubchem_cid = str(state["pubchem_cid"])

    left_sample = w4_catalog.standardize_aligned_vector(
        left_source,
        np.asarray(left_full, dtype=np.float64)[left_positions],
        left_positions,
        scale_variant=scale_variant,
    )
    right_sample = w4_catalog.standardize_aligned_vector(
        right_source,
        np.asarray(right_full, dtype=np.float64)[right_positions],
        right_positions,
        scale_variant=scale_variant,
    )
    observed = cross_source_core.signed_spearman(left_sample, right_sample)

    def prepared_stratum(source, selected, positions) -> SignatureStratum:
        raw_stratum = selected.stratum
        cache_key = (
            id(raw_stratum),
            str(scale_variant),
            np.asarray(positions, dtype=np.int64).tobytes(),
        )
        prepared = prepared_strata.get(cache_key)
        if prepared is None:
            selected_values = np.asarray(
                raw_stratum.values[:, positions],
                dtype=np.float64,
            )
            standardized = w4_catalog.standardize_aligned_matrix(
                source,
                selected_values,
                np.asarray(positions, dtype=np.int64),
                scale_variant=scale_variant,
            )
            prepared = SignatureStratum(
                raw_stratum.compounds,
                standardized,
                gene_keys=np.asarray(raw_stratum.gene_keys)[positions],
                max_prepared_cache_bytes=0,
            )
            prepared_strata[cache_key] = prepared
        return prepared

    left_prepared = prepared_stratum(
        left_source,
        left_selected,
        left_positions,
    )
    right_prepared = prepared_stratum(
        right_source,
        right_selected,
        right_positions,
    )
    left_centroid = left_prepared.different_compound_centroid(
        pubchem_cid,
        require_all_finite=True,
    )
    right_centroid = right_prepared.different_compound_centroid(
        pubchem_cid,
        require_all_finite=True,
    )
    left_peers = left_prepared.values[left_selected.row_indices]
    right_peers = right_prepared.values[right_selected.row_indices]
    prefix = f"{scale_variant}__w4"
    values: dict[str, Any] = {
        f"{scale_variant}__n_common_genes": int(len(left_positions)),
        f"{scale_variant}__n_w4_common_genes": int(
            np.sum(np.isfinite(left_sample) & np.isfinite(right_sample))
        ),
        f"{scale_variant}__left_population_row_count": (
            w4_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        f"{scale_variant}__right_population_row_count": (
            w4_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        f"{scale_variant}__left_population_valid_genes": (
            w4_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        f"{scale_variant}__right_population_valid_genes": (
            w4_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        f"{scale_variant}__left_peer_total_count": left_selected.total_count,
        f"{scale_variant}__right_peer_total_count": right_selected.total_count,
        f"{scale_variant}__left_peer_scored_count": left_selected.scored_count,
        f"{scale_variant}__right_peer_scored_count": right_selected.scored_count,
        f"{scale_variant}__w4_observed_spearman_logfc": observed,
    }
    values.update(
        score_signature_baselines(
            observed=observed,
            left_sample=left_sample,
            right_sample=right_sample,
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peer_matrix=left_peers,
            right_peer_matrix=right_peers,
            prefix=prefix,
        )
    )
    return values


def score_signature_row(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    w4_catalog,
    max_peers: Optional[int],
    sampling_seed: int,
    include_raw_metrics: bool = True,
    w4_scale_variants: tuple[str, ...] = POPULATION_SCALE_VARIANTS,
    prepared_strata: Optional[
        dict[tuple[Any, ...], SignatureStratum]
    ] = None,
) -> dict[str, Any]:
    if prepared_strata is None:
        prepared_strata = {}
    if include_raw_metrics:
        record, state = _raw_signature_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    else:
        record, state = _signature_w4_base_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    for scale_variant in w4_scale_variants:
        record.update(
            _w4_signature_columns(
                state=state,
                w4_catalog=w4_catalog,
                scale_variant=scale_variant,
                prepared_strata=prepared_strata,
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
    prepared_strata: dict[tuple[Any, ...], SignatureStratum] = {}
    try:
        w4_catalog = (
            make_worker_w4_catalog(config, catalog)
            if settings["w4_scale_variants"]
            else None
        )
        for _, row in frame.iterrows():
            try:
                records.append(
                    score_signature_row(
                        row,
                        catalog=catalog,
                        w4_catalog=w4_catalog,
                        max_peers=int(settings["max_baseline_peers"]),
                        sampling_seed=int(settings["peer_sampling_seed"]),
                        include_raw_metrics=bool(
                            settings["include_raw_metrics"]
                        ),
                        w4_scale_variants=tuple(
                            settings["w4_scale_variants"]
                        ),
                        prepared_strata=prepared_strata,
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
                        "stage": "signature_scoring",
                        "reason": "unresolved_source_row",
                        "error": str(exc),
                    }
                )
    finally:
        catalog.close()
    return pd.DataFrame(records), diagnostic_frame(diagnostics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel, resumable matched-pair signature-similarity scoring."
        )
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
        args.required_layers_override = (
            ("logFC", "t")
            if "raw" in selected_computations
            else ("logFC",)
        )
        args.require_adj_p_override = False
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
        "top_k": cross_source_core.TOP_K,
        "max_baseline_peers": args.max_baseline_peers,
        "peer_sampling_seed": args.peer_sampling_seed,
        "computations": list(selected_computations),
        "include_raw_metrics": "raw" in selected_computations,
        "w4_scale_variants": list(w4_scale_variants),
        "scorer_version": "parallel-signature-v4",
    }
    run_analysis(
        analysis=ANALYSIS,
        scorer_module="scripts.run_overlap_group_rep_signature_similarity",
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
