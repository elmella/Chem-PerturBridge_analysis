#!/usr/bin/env python3
"""Build the small reviewer-facing W1, W3, and W4 summary tables."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cross_source_core
from scripts.cluster_bootstrap_ci import (
    cluster_bca_nested_mean_ci_table,
    summarize_ci_half_width_ranges,
)
from scripts.cross_source_parallel import (
    append_progress_log,
    atomic_write_frame,
    atomic_write_json,
    file_record,
    output_directory_lock,
    sha256_file,
)
from scripts.notebook_cache import stable_json_fingerprint
from scripts.population_zscore import PER_GENE_DATASET_VARIANT


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_RANDOM_SEED = 20260505
DRUG_LINE_TIME_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "pubchem_cid",
)
DOSE_CONTEXT_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
)
DOSE_THRESHOLD_SPECS = (
    ("exact", 0, 1.0),
    ("2x", 1, 2.0),
    ("3x", 2, 3.0),
    ("10x_reference", 3, 10.0),
)
DOSE_METRICS = (
    "observed_deg_lfc_spearman_sym_p05",
    "baseline_pair_deg_lfc_spearman_p05",
    "delta_vs_baseline_pair_deg_lfc_spearman_p05",
    "observed_direction_agreement_p05",
    "baseline_pair_direction_agreement_p05",
    "delta_vs_baseline_pair_direction_agreement_p05",
)
RETRIEVAL_CI_METRICS = (
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
    "observed_best_positive_similarity",
    "random_normalized_best_positive_rank",
    "random_recall_at_1",
    "random_auroc",
    "delta_vs_random_normalized_best_positive_rank",
    "source_individual_n_peers",
    "source_individual_mean_similarity",
    "source_individual_sd_similarity",
    "source_individual_n_below_observed",
    "source_individual_fraction_below_observed",
    "source_individual_corrected_percentile",
    "target_individual_n_peers",
    "target_individual_mean_similarity",
    "target_individual_sd_similarity",
    "target_individual_n_below_observed",
    "target_individual_fraction_below_observed",
    "target_individual_corrected_percentile",
    "source_centroid_similarity",
    "source_centroid_normalized_rank",
    "source_centroid_recall_at_1",
    "source_centroid_auroc",
    "target_centroid_similarity",
    "target_centroid_normalized_rank",
    "target_centroid_recall_at_1",
    "target_centroid_auroc",
    "source_peer_normalized_rank",
    "target_peer_normalized_rank",
    "delta_vs_source_peer_normalized_rank",
    "delta_vs_target_peer_normalized_rank",
    "target_decoy_null_normalized_best_positive_rank",
    "delta_vs_target_decoy_null_normalized_best_positive_rank",
)
W4_DEG_METRICS = (
    "w4_observed_deg_lfc_spearman_sym_p05",
    "w4_source_centroid_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_source_centroid_deg_lfc_spearman_p05",
    "w4_target_centroid_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_target_centroid_deg_lfc_spearman_p05",
    "w4_source_peer_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_source_peer_deg_lfc_spearman_p05",
    "w4_source_peer_deg_lfc_spearman_pair_p05_sd_score",
    "w4_source_peer_deg_lfc_spearman_pair_p05_fraction_below_observed",
    "w4_source_peer_deg_lfc_spearman_pair_p05_corrected_percentile",
    "w4_target_peer_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_target_peer_deg_lfc_spearman_p05",
    "w4_target_peer_deg_lfc_spearman_pair_p05_sd_score",
    "w4_target_peer_deg_lfc_spearman_pair_p05_fraction_below_observed",
    "w4_target_peer_deg_lfc_spearman_pair_p05_corrected_percentile",
)
W4_SIGNATURE_METRICS = (
    "w4_observed_spearman_logfc",
    "w4_source_centroid_spearman_logfc_pair",
    "w4_delta_vs_source_centroid_spearman_logfc",
    "w4_target_centroid_spearman_logfc_pair",
    "w4_delta_vs_target_centroid_spearman_logfc",
    "w4_source_peer_spearman_logfc_pair",
    "w4_delta_vs_source_peer_spearman_logfc",
    "w4_source_peer_spearman_logfc_pair_sd_score",
    "w4_source_peer_spearman_logfc_pair_fraction_below_observed",
    "w4_source_peer_spearman_logfc_pair_corrected_percentile",
    "w4_target_peer_spearman_logfc_pair",
    "w4_delta_vs_target_peer_spearman_logfc",
    "w4_target_peer_spearman_logfc_pair_sd_score",
    "w4_target_peer_spearman_logfc_pair_fraction_below_observed",
    "w4_target_peer_spearman_logfc_pair_corrected_percentile",
)
W4_IDENTITY_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "pubchem_cid",
    "matched_condition_key",
    "left_obs_id",
    "right_obs_id",
)
RETRIEVAL_IDENTITY_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "direction",
    "cell_type",
    "time_key",
    "query_pubchem_cid",
    "representation",
    "retrieval_variant",
    "similarity_metric",
    "scale_variant",
)


def _summary_input_columns(label: str) -> set[str]:
    prefix = f"{PER_GENE_DATASET_VARIANT}__"
    if label == "DEG":
        return {
            *cross_source_core.MATCH_PAIR_IDENTITY_COLUMNS,
            *W4_IDENTITY_COLUMNS,
            *DOSE_METRICS,
            *(f"{prefix}{metric}" for metric in W4_DEG_METRICS),
        }
    if label == "signature":
        return {
            *W4_IDENTITY_COLUMNS,
            *(f"{prefix}{metric}" for metric in W4_SIGNATURE_METRICS),
        }
    if label == "retrieval":
        return {
            *RETRIEVAL_IDENTITY_COLUMNS,
            *RETRIEVAL_CI_METRICS,
        }
    raise ValueError(f"Unknown summary input label: {label!r}")


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _read_metrics(
    path: Path,
    *,
    label: str,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} metrics do not exist: {path}")
    usecols = None
    if columns is not None:
        requested = set(columns)
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        usecols = [column for column in header if column in requested]
    else:
        header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    identity_columns = (
        set(cross_source_core.MATCH_PAIR_IDENTITY_COLUMNS)
        | set(W4_IDENTITY_COLUMNS)
        | set(RETRIEVAL_IDENTITY_COLUMNS)
    )
    identity_dtypes = {
        column: "string"
        for column in header
        if column in identity_columns
    }
    frame = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        usecols=usecols,
        dtype=identity_dtypes,
    )
    if frame.empty:
        raise ValueError(f"{label} metrics are empty: {path}")
    return frame


def _normalize_identity_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = (
                frame[column].astype("string").fillna("").astype(str)
            )
    return frame


def _present_metrics(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: Sequence[str],
    label: str,
) -> list[str]:
    _require_columns(frame, required, label=label)
    return [column for column in candidates if column in frame.columns]


def _w4_long_frame(
    frame: pd.DataFrame,
    *,
    metric_candidates: Sequence[str],
    required_metrics: Sequence[str],
    label: str,
    scale_variant: str = PER_GENE_DATASET_VARIANT,
) -> tuple[pd.DataFrame, list[str]]:
    prefix = f"{scale_variant}__"
    identity_columns = [
        column
        for column in (
            "dataset_a",
            "dataset_b",
            "cell_type",
            "time_key",
            "pubchem_cid",
            "matched_condition_key",
            "left_obs_id",
            "right_obs_id",
        )
        if column in frame.columns
    ]
    prefixed_candidates = [
        f"{prefix}{column}" for column in metric_candidates
    ]
    prefixed_required = [
        f"{prefix}{column}" for column in required_metrics
    ]
    metric_columns = _present_metrics(
        frame,
        prefixed_candidates,
        required=prefixed_required,
        label=label,
    )
    result = frame[[*identity_columns, *metric_columns]].copy()
    result = result.rename(
        columns={
            column: column[len(prefix) :]
            for column in metric_columns
        }
    )
    result.insert(2, "scale_variant", scale_variant)
    return result, [column[len(prefix) :] for column in metric_columns]


def _drug_summary(
    frame: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    _require_columns(
        frame,
        [*DRUG_LINE_TIME_COLUMNS, "matched_condition_key", *metrics],
        label="W4 scoring metrics",
    )
    aggregation: dict[str, tuple[str, str]] = {
        "n_matched_sample_pairs": ("left_obs_id", "size"),
        "n_matching_conditions": ("matched_condition_key", "nunique"),
    }
    for metric in metrics:
        aggregation[f"mean_{metric}"] = (metric, "mean")
    return (
        frame.groupby(
            ["scale_variant", *DRUG_LINE_TIME_COLUMNS],
            as_index=False,
        )
        .agg(**aggregation)
        .sort_values(["scale_variant", *DRUG_LINE_TIME_COLUMNS])
        .reset_index(drop=True)
    )


def _w4_ci(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    n_boot: int,
    seed: int,
    summary_level: str,
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> pd.DataFrame:
    drug = _drug_summary(frame, metrics)
    return cluster_bca_nested_mean_ci_table(
        drug,
        group_cols=["dataset_a", "dataset_b", "scale_variant"],
        metric_cols={
            metric: f"mean_{metric}" for metric in metrics
        },
        cluster_col="pubchem_cid",
        n_boot=n_boot,
        seed=seed,
        summary_level=summary_level,
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
        progress_desc=summary_level,
    )


def build_retrieval_ci(
    retrieval: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    similarity_metrics: Sequence[str] = ("cosine", "spearman"),
    scale_variants: Sequence[str] = (
        "raw",
        PER_GENE_DATASET_VARIANT,
    ),
    summary_level: str = "reviewer_minimal_retrieval_dataset_pair",
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> pd.DataFrame:
    required = (
        "dataset_a",
        "dataset_b",
        "direction",
        "cell_type",
        "time_key",
        "query_pubchem_cid",
        "representation",
        "retrieval_variant",
        "similarity_metric",
        "scale_variant",
        "observed_normalized_best_positive_rank",
    )
    _require_columns(retrieval, required, label="retrieval metrics")
    focused = retrieval.loc[
        (retrieval["representation"].astype(str) == "logFC")
        & (
            retrieval["retrieval_variant"].astype(str)
            == "strict_matched_condition"
        )
        & retrieval["similarity_metric"].astype(str).isin(
            tuple(str(value) for value in similarity_metrics)
        )
        & retrieval["scale_variant"].astype(str).isin(
            tuple(str(value) for value in scale_variants)
        )
    ].copy()
    if focused.empty:
        raise ValueError(
            "No strict-logFC retrieval rows found for similarities "
            f"{tuple(similarity_metrics)!r} and scales "
            f"{tuple(scale_variants)!r}"
        )
    metrics = _present_metrics(
        focused,
        RETRIEVAL_CI_METRICS,
        required=(
            "observed_normalized_best_positive_rank",
            "source_individual_corrected_percentile",
            "target_individual_corrected_percentile",
        ),
        label="reviewer-minimal retrieval metrics",
    )
    return cluster_bca_nested_mean_ci_table(
        focused,
        group_cols=[
            "dataset_a",
            "dataset_b",
            "representation",
            "retrieval_variant",
            "similarity_metric",
            "scale_variant",
        ],
        metric_cols=metrics,
        cluster_col="query_pubchem_cid",
        inner_cols=["direction", "cell_type", "time_key"],
        outer_cols=["direction"],
        n_boot=n_boot,
        seed=seed,
        summary_level=summary_level,
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
        progress_desc="retrieval BCa",
    )


def _dataset_order(
    value: str,
    frames: Sequence[pd.DataFrame],
) -> list[str]:
    if value.strip().lower() != "all":
        selected = [
            item.strip() for item in value.split(",") if item.strip()
        ]
        if len(selected) < 2:
            raise ValueError("--datasets must select at least two datasets")
        return list(dict.fromkeys(selected))
    observed = {
        str(dataset)
        for frame in frames
        for column in ("dataset_a", "dataset_b")
        if column in frame.columns
        for dataset in frame[column].dropna().tolist()
    }
    production = cross_source_core.production_dataset_order("deg")
    ordered = [dataset for dataset in production if dataset in observed]
    ordered.extend(sorted(observed - set(ordered)))
    if len(ordered) < 2:
        raise ValueError("Could not infer at least two datasets from inputs")
    return ordered


def _independent_matches(
    *,
    dataset_indices: Mapping[str, Mapping[str, object]],
    datasets: Sequence[str],
    max_fold: float,
) -> pd.DataFrame:
    try:
        return cross_source_core.build_matched_pairs(
            dataset_indices,
            datasets,
            settings=cross_source_core.MatchSettings(
                max_dose_fold_difference=max_fold,
                min_context_shared_drugs=(
                    cross_source_core.DEFAULT_MATCH_SETTINGS
                    .min_context_shared_drugs
                ),
            ),
        )
    except ValueError as exc:
        if "No matched grouped-replicate sample pairs" not in str(exc):
            raise
        return pd.DataFrame(columns=cross_source_core.MATCH_PAIR_COLUMNS)


def _dose_coverage(
    matches: pd.DataFrame,
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_keys = ["dataset_a", "dataset_b"]
    line_keys = [*pair_keys, "cell_type"]

    def coverage(keys: list[str]) -> pd.DataFrame:
        if matches.empty:
            return pd.DataFrame(
                columns=[
                    *keys,
                    "n_matched_sample_pairs",
                    "n_matching_conditions",
                    "n_matching_drugs",
                    "n_eligible_contexts",
                ]
            )
        result = (
            matches.groupby(keys, as_index=False)
            .agg(
                n_matched_sample_pairs=("left_obs_id", "size"),
                n_matching_conditions=("matched_condition_key", "nunique"),
                n_matching_drugs=("pubchem_cid", "nunique"),
            )
        )
        contexts = (
            matches[list(DOSE_CONTEXT_COLUMNS)]
            .drop_duplicates()
            .groupby(keys, as_index=False)
            .size()
            .rename(columns={"size": "n_eligible_contexts"})
        )
        return result.merge(
            contexts,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    pair = coverage(pair_keys)
    line = coverage(line_keys)
    if scored.empty:
        pair["n_scored_sample_pairs"] = 0
        line["n_scored_sample_pairs"] = 0
        return pair, line
    for keys, result in ((pair_keys, pair), (line_keys, line)):
        counts = (
            scored.groupby(keys, as_index=False)
            .size()
            .rename(columns={"size": "n_scored_sample_pairs"})
        )
        merged = result.merge(counts, on=keys, how="left")
        merged["n_scored_sample_pairs"] = (
            merged["n_scored_sample_pairs"].fillna(0).astype(int)
        )
        if keys == pair_keys:
            pair = merged
        else:
            line = merged
    return pair, line


def _dose_metric_summaries(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        mean_columns = [f"mean_{metric}" for metric in DOSE_METRICS]
        drug = pd.DataFrame(
            columns=[
                *DRUG_LINE_TIME_COLUMNS,
                "n_scored_sample_pairs",
                "n_matching_conditions_scored",
                *mean_columns,
            ]
        )
        pair = pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "n_scored_sample_pairs",
                "n_scored_drug_line_times",
                *mean_columns,
            ]
        )
        line = pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "cell_type",
                "n_scored_sample_pairs",
                "n_scored_drug_line_times",
                *mean_columns,
            ]
        )
        return drug, pair, line
    aggregation: dict[str, tuple[str, str]] = {
        "n_scored_sample_pairs": ("left_obs_id", "size"),
        "n_matching_conditions_scored": (
            "matched_condition_key",
            "nunique",
        ),
    }
    for metric in DOSE_METRICS:
        aggregation[f"mean_{metric}"] = (metric, "mean")
    drug = (
        scored.groupby(list(DRUG_LINE_TIME_COLUMNS), as_index=False)
        .agg(**aggregation)
        .sort_values(list(DRUG_LINE_TIME_COLUMNS))
        .reset_index(drop=True)
    )
    mean_columns = [
        column for column in drug.columns if column.startswith("mean_")
    ]

    def summarize(keys: list[str]) -> pd.DataFrame:
        aggregate: dict[str, tuple[str, str]] = {
            "n_scored_sample_pairs": ("n_scored_sample_pairs", "sum"),
            "n_scored_drug_line_times": ("pubchem_cid", "size"),
        }
        for column in mean_columns:
            aggregate[column] = (column, "mean")
        return (
            drug.groupby(keys, as_index=False)
            .agg(**aggregate)
            .sort_values(keys)
            .reset_index(drop=True)
        )

    return (
        drug,
        summarize(["dataset_a", "dataset_b"]),
        summarize(["dataset_a", "dataset_b", "cell_type"]),
    )


def _add_threshold_columns(
    frame: pd.DataFrame,
    *,
    label: str,
    order: int,
    max_fold: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "is_reference", label == "10x_reference")
    frame.insert(0, "max_abs_delta_log10_dose", float(np.log10(max_fold)))
    frame.insert(0, "max_fold_difference", float(max_fold))
    frame.insert(0, "threshold_order", int(order))
    frame.insert(0, "dose_threshold", label)
    return frame


def build_dose_outputs(
    deg: pd.DataFrame,
    *,
    overlap_dir: Path,
    datasets: Sequence[str],
    n_boot: int,
    seed: int,
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    identity = list(cross_source_core.MATCH_PAIR_IDENTITY_COLUMNS)
    _require_columns(
        deg,
        [*identity, *DOSE_METRICS],
        label="DEG metrics",
    )
    normalized_deg = _normalize_identity_columns(deg, identity)
    if normalized_deg.duplicated(identity).any():
        raise ValueError("DEG metrics contain duplicate matched-pair identities")

    dataset_indices = cross_source_core.build_dataset_indices(
        datasets,
        Path(overlap_dir),
    )
    pair_frames: list[pd.DataFrame] = []
    line_frames: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    matched_by_threshold: dict[str, pd.DataFrame] = {}
    for label, order, max_fold in DOSE_THRESHOLD_SPECS:
        matches = _independent_matches(
            dataset_indices=dataset_indices,
            datasets=datasets,
            max_fold=max_fold,
        )
        matches = _normalize_identity_columns(matches, identity)
        matched_by_threshold[label] = matches
        merged = matches.merge(
            normalized_deg[[*identity, *DOSE_METRICS]],
            on=identity,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        scored = merged.loc[merged["_merge"] == "both"].drop(
            columns="_merge"
        )
        drug, metric_pair, metric_line = _dose_metric_summaries(scored)
        pair_coverage, line_coverage = _dose_coverage(matches, scored)
        pair = pair_coverage.merge(
            metric_pair.drop(
                columns="n_scored_sample_pairs",
                errors="ignore",
            ),
            on=["dataset_a", "dataset_b"],
            how="left",
            validate="one_to_one",
        )
        line = line_coverage.merge(
            metric_line.drop(
                columns="n_scored_sample_pairs",
                errors="ignore",
            ),
            on=["dataset_a", "dataset_b", "cell_type"],
            how="left",
            validate="one_to_one",
        )
        pair_frames.append(
            _add_threshold_columns(
                pair,
                label=label,
                order=order,
                max_fold=max_fold,
            )
        )
        line_frames.append(
            _add_threshold_columns(
                line,
                label=label,
                order=order,
                max_fold=max_fold,
            )
        )
        if not drug.empty:
            ci = pd.concat(
                [
                    cluster_bca_nested_mean_ci_table(
                        drug,
                        group_cols=["dataset_a", "dataset_b"],
                        metric_cols={
                            metric: f"mean_{metric}"
                            for metric in DOSE_METRICS
                        },
                        cluster_col="pubchem_cid",
                        n_boot=n_boot,
                        seed=seed,
                        summary_level="dose_threshold_dataset_pair",
                        workers=workers,
                        bootstrap_batch_size=bootstrap_batch_size,
                        progress=progress,
                        progress_desc=f"dose {label} pair BCa",
                    ),
                    cluster_bca_nested_mean_ci_table(
                        drug,
                        group_cols=[
                            "dataset_a",
                            "dataset_b",
                            "cell_type",
                        ],
                        metric_cols={
                            metric: f"mean_{metric}"
                            for metric in DOSE_METRICS
                        },
                        cluster_col="pubchem_cid",
                        n_boot=n_boot,
                        seed=seed,
                        summary_level=(
                            "dose_threshold_dataset_pair_line"
                        ),
                        workers=workers,
                        bootstrap_batch_size=bootstrap_batch_size,
                        progress=progress,
                        progress_desc=f"dose {label} line BCa",
                    ),
                ],
                ignore_index=True,
            )
            ci_frames.append(
                _add_threshold_columns(
                    ci,
                    label=label,
                    order=order,
                    max_fold=max_fold,
                )
            )

    previous: set[tuple[str, ...]] = set()
    for label, _, _ in DOSE_THRESHOLD_SPECS:
        current = {
            tuple(row)
            for row in matched_by_threshold[label][identity].itertuples(
                index=False,
                name=None,
            )
        }
        if not previous.issubset(current):
            raise AssertionError(
                "Dose-threshold matching is not nested at "
                f"{label}: tighter matches disappeared"
            )
        previous = current

    pair_summary = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["threshold_order", "dataset_a", "dataset_b"]
    )
    line_summary = pd.concat(line_frames, ignore_index=True).sort_values(
        ["threshold_order", "dataset_a", "dataset_b", "cell_type"]
    )
    observed_totals = (
        pair_summary.groupby(
            ["threshold_order", "dose_threshold", "max_fold_difference"],
            as_index=False,
        )["n_matched_sample_pairs"]
        .sum()
        .sort_values("threshold_order")
        .reset_index(drop=True)
    )
    totals = pd.DataFrame(
        [
            {
                "threshold_order": order,
                "dose_threshold": label,
                "max_fold_difference": max_fold,
            }
            for label, order, max_fold in DOSE_THRESHOLD_SPECS
        ]
    ).merge(
        observed_totals,
        on=[
            "threshold_order",
            "dose_threshold",
            "max_fold_difference",
        ],
        how="left",
        validate="one_to_one",
    )
    totals["n_matched_sample_pairs"] = (
        totals["n_matched_sample_pairs"].fillna(0).astype(int)
    )
    if np.any(
        np.diff(totals["n_matched_sample_pairs"].to_numpy(dtype=float)) < 0
    ):
        raise AssertionError(
            "Dose-threshold coverage is not nested: expected "
            "exact <= 2x <= 3x <= 10x"
        )

    reference = matched_by_threshold["10x_reference"].copy()
    reference["dose_fold_difference"] = np.power(
        10.0,
        pd.to_numeric(
            reference["abs_delta_log10_dose"],
            errors="coerce",
        ),
    )
    conditions = reference.drop_duplicates(
        ["dataset_a", "dataset_b", "matched_condition_key"]
    )

    def mismatch_quantiles(
        frame: pd.DataFrame,
        unit: str,
    ) -> pd.DataFrame:
        summary = (
            frame.groupby(["dataset_a", "dataset_b"], as_index=False)
            .agg(
                n=("dose_fold_difference", "size"),
                min_fold_difference=("dose_fold_difference", "min"),
                q25_fold_difference=(
                    "dose_fold_difference",
                    lambda values: values.quantile(0.25),
                ),
                median_fold_difference=("dose_fold_difference", "median"),
                q75_fold_difference=(
                    "dose_fold_difference",
                    lambda values: values.quantile(0.75),
                ),
                q90_fold_difference=(
                    "dose_fold_difference",
                    lambda values: values.quantile(0.90),
                ),
                q95_fold_difference=(
                    "dose_fold_difference",
                    lambda values: values.quantile(0.95),
                ),
                max_fold_difference=("dose_fold_difference", "max"),
            )
        )
        summary.insert(0, "unit", unit)
        return summary

    mismatch = pd.concat(
        [
            mismatch_quantiles(reference, "matched_sample_pair"),
            mismatch_quantiles(conditions, "unique_matched_condition"),
        ],
        ignore_index=True,
    )
    ci = (
        pd.concat(ci_frames, ignore_index=True)
        if ci_frames
        else pd.DataFrame(
            columns=[
                "dose_threshold",
                "threshold_order",
                "max_fold_difference",
                "metric",
                "mean",
                "ci_low",
                "ci_high",
            ]
        )
    )
    return {
        "dose_threshold_deg_metric_summary.tsv": (
            pair_summary.reset_index(drop=True)
        ),
        "dose_threshold_deg_metric_line_summary.tsv": (
            line_summary.reset_index(drop=True)
        ),
        "dose_threshold_deg_cluster_bca_ci.tsv": ci,
        "dose_threshold_overall_matched_pair_counts.tsv": totals,
        "dose_mismatch_distribution.tsv": mismatch,
    }


def _summary_run_fingerprint(
    *,
    args: argparse.Namespace,
    datasets: Sequence[str],
) -> str:
    input_paths = (
        Path(args.deg_metrics).resolve(),
        Path(args.signature_metrics).resolve(),
        Path(args.retrieval_metrics).resolve(),
    )
    overlap_paths = [
        (
            Path(args.overlap_dir)
            / f"{dataset}_overlap_filtered.h5ad"
        ).resolve()
        for dataset in datasets
    ]
    code_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "cluster_bootstrap_ci.py",
        REPO_ROOT / "scripts" / "cross_source_core.py",
    )
    return stable_json_fingerprint(
        {
            "analysis": "reviewer-minimal-summary-v2",
            "datasets": list(datasets),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_batch_size": int(args.bootstrap_batch_size),
            "inputs": [file_record(path) for path in input_paths],
            "overlap_inputs": [
                file_record(path) for path in overlap_paths
            ],
            "code": [
                file_record(path, content_hash=True)
                for path in code_paths
            ],
        }
    )


def _stage_marker_path(
    checkpoint_dir: Path,
    stage: str,
) -> Path:
    return Path(checkpoint_dir) / f"{stage}.complete.json"


def _validate_summary_stage(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    stage: str,
    fingerprint: str,
    filenames: Sequence[str],
) -> bool:
    marker_path = _stage_marker_path(checkpoint_dir, stage)
    try:
        marker = json.loads(marker_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if (
        marker.get("stage") != stage
        or marker.get("fingerprint") != fingerprint
        or marker.get("filenames") != list(filenames)
    ):
        return False
    records = marker.get("outputs")
    if not isinstance(records, list) or len(records) != len(filenames):
        return False
    by_name = {
        str(record.get("filename")): record
        for record in records
        if isinstance(record, dict)
    }
    for filename in filenames:
        path = Path(output_dir) / filename
        record = by_name.get(filename)
        if record is None or not path.is_file():
            return False
        if int(path.stat().st_size) != int(record.get("size", -1)):
            return False
        if sha256_file(path) != str(record.get("sha256", "")):
            return False
    return True


def _publish_summary_stage(
    *,
    output_dir: Path,
    checkpoint_dir: Path,
    stage: str,
    fingerprint: str,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    for filename, frame in frames.items():
        atomic_write_frame(Path(output_dir) / filename, frame)
    records = []
    for filename, frame in frames.items():
        path = Path(output_dir) / filename
        records.append(
            {
                "filename": filename,
                "rows": int(len(frame)),
                "columns": frame.columns.tolist(),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    atomic_write_json(
        _stage_marker_path(checkpoint_dir, stage),
        {
            "stage": stage,
            "fingerprint": fingerprint,
            "filenames": list(frames),
            "outputs": records,
            "completed_at_unix": time.time(),
        },
    )


def _observed_dataset_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=["dataset_a", "dataset_b"],
        low_memory=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize reviewer-minimal W1/W3/W4 scoring outputs."
        )
    )
    parser.add_argument("--deg-metrics", type=Path, required=True)
    parser.add_argument("--signature-metrics", type=Path, required=True)
    parser.add_argument("--retrieval-metrics", type=Path, required=True)
    parser.add_argument("--overlap-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names, or infer them with 'all'.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_RANDOM_SEED,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Threads used across independent BCa metrics.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=64,
        help="Number of bootstrap draws prepared per deterministic batch.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "off"),
        default="auto",
        help="Show per-metric tqdm progress; stage events are always logged.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed summary stages.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations < 20:
        raise ValueError("--bootstrap-iterations must be at least 20")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive")
    output_dir = Path(args.output_dir).resolve()
    progress_log = output_dir / "progress.log"
    progress_enabled = (
        args.progress == "always"
        or (args.progress == "auto" and sys.stderr.isatty())
    )

    def report(message: str) -> None:
        append_progress_log(
            progress_log,
            analysis="reviewer-summary",
            message=message,
        )
        print(f"[reviewer-summary] {message}", flush=True)

    retrieval_filenames = (
        "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
        "reviewer_minimal_retrieval_ci_ranges.tsv",
    )
    deg_filenames = (
        "w4_deg_cluster_bca_ci.tsv",
        "dose_threshold_deg_metric_summary.tsv",
        "dose_threshold_deg_metric_line_summary.tsv",
        "dose_threshold_deg_cluster_bca_ci.tsv",
        "dose_threshold_overall_matched_pair_counts.tsv",
        "dose_mismatch_distribution.tsv",
    )
    signature_filenames = ("w4_signature_cluster_bca_ci.tsv",)
    output_filenames = (
        *retrieval_filenames,
        *deg_filenames,
        *signature_filenames,
    )

    with output_directory_lock(output_dir):
        report(
            f"starting bootstrap_iterations={args.bootstrap_iterations:,} "
            f"workers={args.workers} batch_size={args.bootstrap_batch_size}"
        )
        if str(args.datasets).strip().lower() == "all":
            report("inferring datasets from slim identity columns")
            dataset_frames = [
                _observed_dataset_frame(Path(path))
                for path in (
                    args.deg_metrics,
                    args.signature_metrics,
                    args.retrieval_metrics,
                )
            ]
        else:
            dataset_frames = []
        datasets = _dataset_order(args.datasets, dataset_frames)
        del dataset_frames
        fingerprint = _summary_run_fingerprint(
            args=args,
            datasets=datasets,
        )
        checkpoint_dir = output_dir / "checkpoints" / fingerprint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        report(f"fingerprint={fingerprint} datasets={','.join(datasets)}")

        def run_stage(
            stage: str,
            filenames: Sequence[str],
            compute,
        ) -> None:
            if (
                not args.force
                and _validate_summary_stage(
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    stage=stage,
                    fingerprint=fingerprint,
                    filenames=filenames,
                )
            ):
                report(f"stage={stage} cached files={len(filenames)}")
                return
            started = time.monotonic()
            report(f"stage={stage} started")
            try:
                frames = compute()
                if list(frames) != list(filenames):
                    raise RuntimeError(
                        f"Stage {stage} returned files {list(frames)!r}; "
                        f"expected {list(filenames)!r}"
                    )
                _publish_summary_stage(
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    stage=stage,
                    fingerprint=fingerprint,
                    frames=frames,
                )
            except BaseException as exc:
                report(
                    f"stage={stage} failed error={type(exc).__name__}: {exc}"
                )
                raise
            report(
                f"stage={stage} completed files={len(filenames)} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )

        def compute_retrieval() -> dict[str, pd.DataFrame]:
            started = time.monotonic()
            retrieval = _read_metrics(
                args.retrieval_metrics,
                label="retrieval",
                columns=_summary_input_columns("retrieval"),
            )
            report(
                f"stage=retrieval loaded rows={len(retrieval):,} "
                f"columns={len(retrieval.columns)} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
            ci = build_retrieval_ci(
                retrieval,
                n_boot=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
                workers=args.workers,
                bootstrap_batch_size=args.bootstrap_batch_size,
                progress=progress_enabled,
            )
            return {
                retrieval_filenames[0]: ci,
                retrieval_filenames[1]: summarize_ci_half_width_ranges(ci),
            }

        def compute_deg() -> dict[str, pd.DataFrame]:
            started = time.monotonic()
            deg = _read_metrics(
                args.deg_metrics,
                label="DEG",
                columns=_summary_input_columns("DEG"),
            )
            report(
                f"stage=deg loaded rows={len(deg):,} "
                f"columns={len(deg.columns)} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
            deg_w4, deg_metrics = _w4_long_frame(
                deg,
                metric_candidates=W4_DEG_METRICS,
                required_metrics=(
                    "w4_observed_deg_lfc_spearman_sym_p05",
                ),
                label="DEG W4 metrics",
            )
            frames = {
                deg_filenames[0]: _w4_ci(
                    deg_w4,
                    metrics=deg_metrics,
                    n_boot=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                    summary_level="w4_deg_dataset_pair",
                    workers=args.workers,
                    bootstrap_batch_size=args.bootstrap_batch_size,
                    progress=progress_enabled,
                )
            }
            frames.update(
                build_dose_outputs(
                    deg,
                    overlap_dir=args.overlap_dir,
                    datasets=datasets,
                    n_boot=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                    workers=args.workers,
                    bootstrap_batch_size=args.bootstrap_batch_size,
                    progress=progress_enabled,
                )
            )
            return frames

        def compute_signature() -> dict[str, pd.DataFrame]:
            started = time.monotonic()
            signature = _read_metrics(
                args.signature_metrics,
                label="signature",
                columns=_summary_input_columns("signature"),
            )
            report(
                f"stage=signature loaded rows={len(signature):,} "
                f"columns={len(signature.columns)} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )
            signature_w4, signature_metrics = _w4_long_frame(
                signature,
                metric_candidates=W4_SIGNATURE_METRICS,
                required_metrics=("w4_observed_spearman_logfc",),
                label="signature W4 metrics",
            )
            return {
                signature_filenames[0]: _w4_ci(
                    signature_w4,
                    metrics=signature_metrics,
                    n_boot=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                    summary_level="w4_signature_dataset_pair",
                    workers=args.workers,
                    bootstrap_batch_size=args.bootstrap_batch_size,
                    progress=progress_enabled,
                )
            }

        run_stage("retrieval", retrieval_filenames, compute_retrieval)
        run_stage("deg", deg_filenames, compute_deg)
        run_stage("signature", signature_filenames, compute_signature)
        atomic_write_json(
            output_dir / "run_metadata.json",
            {
                "analysis": "reviewer-minimal-summary",
                "fingerprint": fingerprint,
                "datasets": datasets,
                "bootstrap_iterations": args.bootstrap_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_batch_size": args.bootstrap_batch_size,
                "workers": args.workers,
                "w4_scale_variant": PER_GENE_DATASET_VARIANT,
                "inputs": [
                    file_record(Path(path).resolve())
                    for path in (
                        args.deg_metrics,
                        args.signature_metrics,
                        args.retrieval_metrics,
                    )
                ],
                "overlap_inputs": [
                    file_record(
                        (
                            Path(args.overlap_dir)
                            / f"{dataset}_overlap_filtered.h5ad"
                        ).resolve(),
                    )
                    for dataset in datasets
                ],
                "outputs": sorted(output_filenames),
            },
        )
        report(
            f"completed files={len(output_filenames)} output_dir={output_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
