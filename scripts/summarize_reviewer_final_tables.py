#!/usr/bin/env python3
"""Finalize reviewer retrieval and dose tables from completed local results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_source_parallel import (
    atomic_write_frame,
    atomic_write_json,
    file_record,
    output_directory_lock,
)
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
)
from scripts.summarize_reviewer_minimal_metrics import (
    DOSE_METRICS,
    RETRIEVAL_CI_METRICS,
    W4_DEG_METRICS,
    W4_IDENTITY_COLUMNS,
    W4_SIGNATURE_METRICS,
    _w4_ci,
    _w4_long_frame,
    _read_metrics,
    _summary_input_columns,
    build_retrieval_ci,
)


DEFAULT_SUMMARY_DIR = Path(
    "results/parallel_cross_source/reviewer_minimal_summary"
)
DEFAULT_L2_METRICS = Path(
    "results/parallel_cross_source/retrieval/l2_raw_w4_both/"
    "retrieval_scored_metrics.tsv"
)
DEFAULT_PEER_SUMMARY = Path(
    "results/parallel_cross_source/retrieval/"
    "peer_sensitivity_raw_summary/"
    "retrieval_peer_sensitivity_cluster_bca_ci.tsv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/parallel_cross_source/reviewer_final_summary"
)
DEFAULT_RAW_DEG_METRICS = Path(
    "results/parallel_cross_source/deg/production/deg_scored_metrics.tsv"
)
DEFAULT_DATASET_CELL_TYPE_DEG_METRICS = Path(
    "results/parallel_cross_source/deg/normalized_dataset_cell_type/"
    "deg_scored_metrics.tsv"
)
DEFAULT_DATASET_CELL_TYPE_SIGNATURE_METRICS = Path(
    "results/parallel_cross_source/signature/normalized_dataset_cell_type/"
    "signature_scored_metrics.tsv"
)
DEFAULT_DATASET_CELL_TYPE_RETRIEVAL_METRICS = Path(
    "results/parallel_cross_source/retrieval/normalized_dataset_cell_type/"
    "retrieval_scored_metrics.tsv"
)
L2_SCALE_VARIANTS = (
    "raw",
    PER_GENE_DATASET_VARIANT,
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
)
TABLE9_METRICS = (
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
    "source_peer_normalized_rank",
    "delta_vs_source_peer_normalized_rank",
    "source_individual_corrected_percentile",
    "target_peer_normalized_rank",
    "delta_vs_target_peer_normalized_rank",
    "target_individual_corrected_percentile",
)
TABLE9_COLUMNS = (
    ("observed_normalized_best_positive_rank", "Observed rank [95% CI]"),
    ("observed_recall_at_1", "Recall@1 [95% CI]"),
    ("observed_auroc", "AUROC [95% CI]"),
    ("source_peer_normalized_rank", "Source baseline rank [95% CI]"),
    ("delta_vs_source_peer_normalized_rank", "Delta source [95% CI]"),
    (
        "source_individual_corrected_percentile",
        "Source corrected percentile [95% CI]",
    ),
    ("target_peer_normalized_rank", "Target baseline rank [95% CI]"),
    ("delta_vs_target_peer_normalized_rank", "Delta target [95% CI]"),
    (
        "target_individual_corrected_percentile",
        "Target corrected percentile [95% CI]",
    ),
)
COUNT_COLUMNS = (
    "n_matched_sample_pairs",
    "n_matching_conditions",
    "n_matching_drugs",
    "n_eligible_contexts",
    "n_scored_sample_pairs",
    "n_scored_drug_line_times",
)
PAIR_COLUMNS = ("dataset_a", "dataset_b")
CI_IDENTITY_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "representation",
    "retrieval_variant",
    "similarity_metric",
    "scale_variant",
    "metric",
)
TABLE5_METRICS = (
    "observed_direction_agreement_p05",
    "baseline_pair_direction_agreement_p05",
    "delta_vs_baseline_pair_direction_agreement_p05",
    "raw_source_centroid_direction_agreement_pair_p05",
    "raw_delta_vs_source_centroid_direction_agreement_p05",
    "raw_target_centroid_direction_agreement_pair_p05",
    "raw_delta_vs_target_centroid_direction_agreement_p05",
    "raw_source_peer_direction_agreement_pair_p05",
    "raw_delta_vs_source_peer_direction_agreement_p05",
    "raw_source_peer_direction_agreement_pair_p05_sd_score",
    "raw_source_peer_direction_agreement_pair_p05_fraction_below_observed",
    "raw_source_peer_direction_agreement_pair_p05_corrected_percentile",
    "raw_target_peer_direction_agreement_pair_p05",
    "raw_delta_vs_target_peer_direction_agreement_p05",
    "raw_target_peer_direction_agreement_pair_p05_sd_score",
    "raw_target_peer_direction_agreement_pair_p05_fraction_below_observed",
    "raw_target_peer_direction_agreement_pair_p05_corrected_percentile",
)
TABLE5_SCORER_ALIASES = {
    metric: metric.replace("raw_", "pb_", 1)
    for metric in TABLE5_METRICS
    if metric.startswith("raw_")
}


def build_table5_ci(
    deg: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> pd.DataFrame:
    """Summarize raw cross-source direction agreement for reviewer Table 5."""
    deg = deg.copy()
    for presentation_name, scorer_name in TABLE5_SCORER_ALIASES.items():
        if presentation_name not in deg.columns and scorer_name in deg.columns:
            deg[presentation_name] = deg[scorer_name]
    required = (
        "dataset_a",
        "dataset_b",
        "cell_type",
        "time_key",
        "pubchem_cid",
        "matched_condition_key",
        "left_obs_id",
        *TABLE5_METRICS,
    )
    _require_columns(deg, required, label="Table 5 DEG metrics")
    focused = deg.loc[:, required].copy()
    focused.insert(2, "scale_variant", "raw")
    return _w4_ci(
        focused,
        metrics=TABLE5_METRICS,
        n_boot=n_boot,
        seed=seed,
        summary_level="table_5_dataset_pair",
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
    )


def build_population_ci(
    frame: pd.DataFrame,
    *,
    metric_candidates: Sequence[str],
    required_metrics: Sequence[str],
    scale_variant: str,
    summary_level: str,
    n_boot: int,
    seed: int,
    workers: int = 1,
    bootstrap_batch_size: int = 64,
    progress: bool = False,
) -> pd.DataFrame:
    normalized, metrics = _w4_long_frame(
        frame,
        metric_candidates=metric_candidates,
        required_metrics=required_metrics,
        label=f"{summary_level} metrics",
        scale_variant=scale_variant,
    )
    return _w4_ci(
        normalized,
        metrics=metrics,
        n_boot=n_boot,
        seed=seed,
        summary_level=summary_level,
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=progress,
    )


def _population_input_columns(
    metrics: Sequence[str],
    *,
    scale_variant: str,
) -> set[str]:
    prefix = f"{scale_variant}__"
    return {
        *W4_IDENTITY_COLUMNS,
        *(f"{prefix}{metric}" for metric in metrics),
    }


def _ci_presentation(
    ci: pd.DataFrame,
    *,
    pair_labels: dict[tuple[str, str], str],
) -> pd.DataFrame:
    required = (
        "dataset_a",
        "dataset_b",
        "scale_variant",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "n_rows",
        "n_finite_rows",
        "n_compounds",
        "ci_status",
    )
    _require_columns(ci, required, label="confidence intervals")
    result = ci.loc[:, required].copy()
    result.insert(
        0,
        "dataset_pair",
        [
            pair_labels.get(
                (str(dataset_a), str(dataset_b)),
                _fallback_pair_label(dataset_a, dataset_b),
            )
            for dataset_a, dataset_b in zip(
                result["dataset_a"], result["dataset_b"]
            )
        ],
    )
    result["estimate_95_ci"] = [
        _format_ci(
            mean,
            low,
            high,
            signed=(
                str(metric).startswith("delta")
                or "_delta_" in str(metric)
            ),
        )
        for mean, low, high, metric in zip(
            result["mean"],
            result["ci_low"],
            result["ci_high"],
            result["metric"],
        )
    ]
    return result.sort_values(
        ["scale_variant", "dataset_a", "dataset_b", "metric"],
        kind="stable",
    ).reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def _read_tsv(path: Path, *, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def _dataset_pair_labels(summary_dir: Path) -> dict[tuple[str, str], str]:
    path = (
        Path(summary_dir)
        / "shareable_tables"
        / "table9_retrieval_numeric_long.tsv"
    )
    if not path.is_file():
        return {}
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["dataset_a", "dataset_b", "dataset_pair"],
    )
    return {
        (str(row.dataset_a), str(row.dataset_b)): str(row.dataset_pair)
        for row in frame.drop_duplicates(
            ["dataset_a", "dataset_b"]
        ).itertuples(index=False)
    }


def _fallback_pair_label(dataset_a: str, dataset_b: str) -> str:
    def label(value: str) -> str:
        return str(value).replace("_", " ").title()

    return f"{label(dataset_a)}–{label(dataset_b)}"


def _table9_numeric_long(
    retrieval_ci: pd.DataFrame,
    *,
    pair_labels: dict[tuple[str, str], str],
) -> pd.DataFrame:
    selected = retrieval_ci.loc[
        retrieval_ci["metric"].astype(str).isin(TABLE9_METRICS)
    ].copy()
    selected.insert(
        0,
        "dataset_pair",
        [
            pair_labels.get(
                (str(dataset_a), str(dataset_b)),
                _fallback_pair_label(dataset_a, dataset_b),
            )
            for dataset_a, dataset_b in zip(
                selected["dataset_a"],
                selected["dataset_b"],
            )
        ],
    )
    columns = [
        "dataset_pair",
        "dataset_a",
        "dataset_b",
        "similarity_metric",
        "scale_variant",
        "metric",
        "mean",
        "ci_low",
        "ci_high",
        "ci_method",
        "ci_status",
        "ci_level",
        "n_bootstrap_iterations",
        "n_rows",
        "n_finite_rows",
        "n_compounds",
        "uncertainty_scope",
    ]
    return selected.loc[:, columns].sort_values(
        [
            "similarity_metric",
            "scale_variant",
            "dataset_a",
            "dataset_b",
            "metric",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _format_ci(
    mean: float,
    ci_low: float,
    ci_high: float,
    *,
    signed: bool,
) -> str:
    if not np.isfinite(mean):
        return "NA"
    mean_text = f"{mean:+.3f}" if signed else f"{mean:.3f}"
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return mean_text
    return f"{mean_text} [{ci_low:.3f}, {ci_high:.3f}]"


def _table9_panel(
    numeric: pd.DataFrame,
    *,
    similarity_metric: str,
    scale_variant: str,
) -> pd.DataFrame:
    panel = numeric.loc[
        (
            numeric["similarity_metric"].astype(str)
            == str(similarity_metric)
        )
        & (numeric["scale_variant"].astype(str) == str(scale_variant))
    ].copy()
    if panel.empty:
        raise ValueError(
            "No Table 9 rows for "
            f"{similarity_metric}/{scale_variant}"
        )
    rows: list[dict[str, object]] = []
    for (
        dataset_pair,
        dataset_a,
        dataset_b,
    ), group in panel.groupby(
        ["dataset_pair", "dataset_a", "dataset_b"],
        sort=False,
        dropna=False,
    ):
        values = group.set_index("metric")
        row: dict[str, object] = {
            "Dataset pair": dataset_pair,
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "Similarity": (
                "L2"
                if similarity_metric == "negative_l2"
                else similarity_metric.title()
            ),
            "Scale": scale_variant,
            "Baseline peer scope": (
                "all eligible"
                if similarity_metric == "negative_l2"
                else "up to 256 deterministic peers"
            ),
            "n_cmpd": int(values["n_compounds"].max()),
            "n_rows": int(values["n_rows"].max()),
        }
        for metric, column in TABLE9_COLUMNS:
            if metric not in values.index:
                row[column] = "NA"
                continue
            value = values.loc[metric]
            if isinstance(value, pd.DataFrame):
                raise ValueError(
                    "Duplicate Table 9 metric for "
                    f"{dataset_a}/{dataset_b}/{similarity_metric}/"
                    f"{scale_variant}/{metric}"
                )
            row[column] = _format_ci(
                float(value["mean"]),
                float(value["ci_low"]),
                float(value["ci_high"]),
                signed=metric.startswith("delta_"),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _complete_dose_pair_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        summary,
        [*PAIR_COLUMNS, "dose_threshold", "threshold_order"],
        label="dose metric summary",
    )
    pairs = (
        summary[list(PAIR_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(PAIR_COLUMNS))
    )
    exact = summary.loc[
        summary["dose_threshold"].astype(str) == "exact"
    ]
    observed = {
        tuple(row)
        for row in exact[list(PAIR_COLUMNS)].itertuples(
            index=False,
            name=None,
        )
    }
    missing = [
        tuple(row)
        for row in pairs.itertuples(index=False, name=None)
        if tuple(row) not in observed
    ]
    rows: list[dict[str, object]] = []
    for dataset_a, dataset_b in missing:
        row = {column: np.nan for column in summary.columns}
        row.update(
            {
                "dose_threshold": "exact",
                "threshold_order": 0,
                "max_fold_difference": 1.0,
                "max_abs_delta_log10_dose": 0.0,
                "is_reference": False,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
            }
        )
        for column in COUNT_COLUMNS:
            if column in row:
                row[column] = 0
        rows.append(row)
    result = pd.concat(
        [summary, pd.DataFrame(rows, columns=summary.columns)],
        ignore_index=True,
    )
    result = result.sort_values(
        ["threshold_order", *PAIR_COLUMNS],
        kind="stable",
    ).reset_index(drop=True)
    expected = len(pairs)
    coverage = result.groupby("dose_threshold")[
        list(PAIR_COLUMNS)
    ].apply(lambda frame: len(frame.drop_duplicates()))
    if not (coverage == expected).all():
        raise AssertionError(
            "Completed dose summary does not contain every dataset pair "
            f"at each threshold: {coverage.to_dict()}"
        )
    return result


def _empty_dose_ci_row(
    template: pd.DataFrame,
    *,
    dataset_a: str,
    dataset_b: str,
    metric: str,
) -> dict[str, object]:
    row = {column: np.nan for column in template.columns}
    row.update(
        {
            "dose_threshold": "exact",
            "threshold_order": 0,
            "max_fold_difference": 1.0,
            "max_abs_delta_log10_dose": 0.0,
            "is_reference": False,
            "dataset_a": dataset_a,
            "dataset_b": dataset_b,
            "summary_level": "dose_threshold_dataset_pair",
            "metric": metric,
            "value_col": f"mean_{metric}",
            "ci_method": "not_estimable",
            "ci_status": "no_matched_pairs",
            "ci_level": 0.95,
            "n_bootstrap_iterations": 0,
            "n_bootstrap_valid": 0,
            "n_rows": 0,
            "n_finite_rows": 0,
            "n_compounds": 0,
            "cluster_col": "pubchem_cid",
            "inner_strata": "",
            "outer_strata": "",
            "uncertainty_scope": (
                "No exact-dose matched sample pairs; effect and interval "
                "are not estimable."
            ),
            "cell_type": "",
        }
    )
    return row


def _complete_dose_ci(
    ci: pd.DataFrame,
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    pairs = pair_summary[list(PAIR_COLUMNS)].drop_duplicates()
    exact_pair_ci = ci.loc[
        (ci["dose_threshold"].astype(str) == "exact")
        & (
            ci["summary_level"].astype(str)
            == "dose_threshold_dataset_pair"
        )
    ]
    observed = {
        tuple(row)
        for row in exact_pair_ci[list(PAIR_COLUMNS)]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    rows = [
        _empty_dose_ci_row(
            ci,
            dataset_a=str(dataset_a),
            dataset_b=str(dataset_b),
            metric=metric,
        )
        for dataset_a, dataset_b in pairs.itertuples(
            index=False,
            name=None,
        )
        if (str(dataset_a), str(dataset_b)) not in observed
        for metric in DOSE_METRICS
    ]
    result = pd.concat(
        [ci, pd.DataFrame(rows, columns=ci.columns)],
        ignore_index=True,
    )
    return result.sort_values(
        [
            "threshold_order",
            "dataset_a",
            "dataset_b",
            "summary_level",
            "cell_type",
            "metric",
        ],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def _validate_retrieval_ci(ci: pd.DataFrame) -> None:
    if ci.duplicated(list(CI_IDENTITY_COLUMNS)).any():
        raise ValueError("Combined retrieval CI contains duplicate keys")
    observed = ci.loc[
        ci["metric"].astype(str)
        == "observed_normalized_best_positive_rank"
    ]
    coverage = observed.groupby(
        ["similarity_metric", "scale_variant"]
    )[list(PAIR_COLUMNS)].apply(lambda frame: len(frame.drop_duplicates()))
    if not (coverage == 9).all():
        raise AssertionError(
            "Retrieval CI does not cover all nine dataset pairs: "
            f"{coverage.to_dict()}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build final reviewer L2, combined Table 9, and complete "
            "dose-threshold summaries from existing local results."
        )
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--l2-metrics", type=Path, default=DEFAULT_L2_METRICS)
    parser.add_argument(
        "--peer-summary",
        type=Path,
        default=DEFAULT_PEER_SUMMARY,
    )
    parser.add_argument(
        "--include-additional-tables",
        action="store_true",
        help=(
            "Also build raw direction-agreement Table 5 and dataset-by-cell-type "
            "normalization comparisons for Tables 4, 6, and 9."
        ),
    )
    parser.add_argument(
        "--raw-deg-metrics",
        type=Path,
        default=DEFAULT_RAW_DEG_METRICS,
    )
    parser.add_argument(
        "--dataset-cell-type-deg-metrics",
        type=Path,
        default=DEFAULT_DATASET_CELL_TYPE_DEG_METRICS,
    )
    parser.add_argument(
        "--dataset-cell-type-signature-metrics",
        type=Path,
        default=DEFAULT_DATASET_CELL_TYPE_SIGNATURE_METRICS,
    )
    parser.add_argument(
        "--dataset-cell-type-retrieval-metrics",
        type=Path,
        default=DEFAULT_DATASET_CELL_TYPE_RETRIEVAL_METRICS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260505)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-batch-size", type=int, default=64)
    parser.add_argument(
        "--progress",
        choices=("auto", "always", "off"),
        default="auto",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_iterations < 20:
        raise ValueError("--bootstrap-iterations must be at least 20")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive")
    summary_dir = Path(args.summary_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    progress = args.progress == "always" or (
        args.progress == "auto" and sys.stderr.isatty()
    )

    l2 = _read_metrics(
        args.l2_metrics,
        label="L2 retrieval metrics",
        columns=_summary_input_columns("retrieval"),
    )
    l2_ci = build_retrieval_ci(
        l2,
        n_boot=int(args.bootstrap_iterations),
        seed=int(args.bootstrap_seed),
        similarity_metrics=("negative_l2",),
        scale_variants=L2_SCALE_VARIANTS,
        summary_level="reviewer_l2_retrieval_dataset_pair",
        workers=int(args.workers),
        bootstrap_batch_size=int(args.bootstrap_batch_size),
        progress=progress,
    )
    minimal_ci = _read_tsv(
        summary_dir / "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
        label="reviewer-minimal retrieval CI",
    )
    retrieval_ci_parts = [minimal_ci, l2_ci]
    dataset_cell_type_retrieval_ci = None
    if args.include_additional_tables:
        dataset_cell_type_retrieval = _read_metrics(
            args.dataset_cell_type_retrieval_metrics,
            label="dataset-by-cell-type retrieval metrics",
            columns=_summary_input_columns("retrieval"),
        )
        dataset_cell_type_retrieval_ci = build_retrieval_ci(
            dataset_cell_type_retrieval,
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            similarity_metrics=("cosine", "spearman"),
            scale_variants=(PER_GENE_DATASET_CELL_TYPE_VARIANT,),
            summary_level=(
                "reviewer_dataset_cell_type_retrieval_dataset_pair"
            ),
            workers=int(args.workers),
            bootstrap_batch_size=int(args.bootstrap_batch_size),
            progress=progress,
        )
        retrieval_ci_parts.append(dataset_cell_type_retrieval_ci)
    combined_ci = pd.concat(retrieval_ci_parts, ignore_index=True)
    _validate_retrieval_ci(combined_ci)
    pair_labels = _dataset_pair_labels(summary_dir)
    numeric = _table9_numeric_long(
        combined_ci,
        pair_labels=pair_labels,
    )
    cosine_spearman_scales = ["raw", PER_GENE_DATASET_VARIANT]
    if args.include_additional_tables:
        cosine_spearman_scales.append(
            PER_GENE_DATASET_CELL_TYPE_VARIANT
        )
    panels = {
        (
            f"table9_{metric}_{scale}.tsv"
            .replace("negative_l2", "l2")
            .replace(PER_GENE_DATASET_VARIANT, "w4_dataset")
            .replace(
                PER_GENE_DATASET_CELL_TYPE_VARIANT,
                "w4_dataset_cell_type",
            )
        ): _table9_panel(
            numeric,
            similarity_metric=metric,
            scale_variant=scale,
        )
        for metric, scales in (
            ("cosine", tuple(cosine_spearman_scales)),
            ("spearman", tuple(cosine_spearman_scales)),
            ("negative_l2", L2_SCALE_VARIANTS),
        )
        for scale in scales
    }
    combined_table9 = pd.concat(
        list(panels.values()),
        ignore_index=True,
    )

    dose_pair = _read_tsv(
        summary_dir / "dose_threshold_deg_metric_summary.tsv",
        label="dose-threshold metric summary",
    )
    dose_ci = _read_tsv(
        summary_dir / "dose_threshold_deg_cluster_bca_ci.tsv",
        label="dose-threshold CI",
    )
    complete_dose_pair = _complete_dose_pair_summary(dose_pair)
    complete_dose_ci = _complete_dose_ci(dose_ci, complete_dose_pair)
    peer_summary = _read_tsv(
        args.peer_summary,
        label="all-peer sensitivity summary",
    )
    if set(peer_summary["ci_status"].astype(str)) != {"ok"}:
        raise ValueError(
            "All-peer sensitivity summary contains non-OK intervals"
        )

    additional_outputs: dict[str, pd.DataFrame] = {}
    additional_input_paths: list[Path] = []
    if args.include_additional_tables:
        raw_deg = _read_metrics(
            args.raw_deg_metrics,
            label="raw DEG metrics for Table 5",
            columns={
                *W4_IDENTITY_COLUMNS,
                *TABLE5_METRICS,
                *TABLE5_SCORER_ALIASES.values(),
            },
        )
        table5_ci = build_table5_ci(
            raw_deg,
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            workers=int(args.workers),
            bootstrap_batch_size=int(args.bootstrap_batch_size),
            progress=progress,
        )

        dataset_cell_type_deg = _read_metrics(
            args.dataset_cell_type_deg_metrics,
            label="dataset-by-cell-type DEG metrics",
            columns=_population_input_columns(
                W4_DEG_METRICS,
                scale_variant=PER_GENE_DATASET_CELL_TYPE_VARIANT,
            ),
        )
        table4_cell_type_ci = build_population_ci(
            dataset_cell_type_deg,
            metric_candidates=W4_DEG_METRICS,
            required_metrics=(
                "w4_observed_deg_lfc_spearman_sym_p05",
            ),
            scale_variant=PER_GENE_DATASET_CELL_TYPE_VARIANT,
            summary_level="table_4_dataset_cell_type_dataset_pair",
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            workers=int(args.workers),
            bootstrap_batch_size=int(args.bootstrap_batch_size),
            progress=progress,
        )
        dataset_table4_ci = _read_tsv(
            summary_dir / "w4_deg_cluster_bca_ci.tsv",
            label="dataset-wide Table 4 confidence intervals",
        )
        table4_comparison_ci = pd.concat(
            [dataset_table4_ci, table4_cell_type_ci],
            ignore_index=True,
        )

        dataset_cell_type_signature = _read_metrics(
            args.dataset_cell_type_signature_metrics,
            label="dataset-by-cell-type signature metrics",
            columns=_population_input_columns(
                W4_SIGNATURE_METRICS,
                scale_variant=PER_GENE_DATASET_CELL_TYPE_VARIANT,
            ),
        )
        table6_cell_type_ci = build_population_ci(
            dataset_cell_type_signature,
            metric_candidates=W4_SIGNATURE_METRICS,
            required_metrics=("w4_observed_spearman_logfc",),
            scale_variant=PER_GENE_DATASET_CELL_TYPE_VARIANT,
            summary_level="table_6_dataset_cell_type_dataset_pair",
            n_boot=int(args.bootstrap_iterations),
            seed=int(args.bootstrap_seed),
            workers=int(args.workers),
            bootstrap_batch_size=int(args.bootstrap_batch_size),
            progress=progress,
        )
        dataset_table6_ci = _read_tsv(
            summary_dir / "w4_signature_cluster_bca_ci.tsv",
            label="dataset-wide Table 6 confidence intervals",
        )
        table6_comparison_ci = pd.concat(
            [dataset_table6_ci, table6_cell_type_ci],
            ignore_index=True,
        )

        additional_outputs = {
            "table5_direction_agreement_cluster_bca_ci.tsv": table5_ci,
            "table5_direction_agreement.tsv": _ci_presentation(
                table5_ci,
                pair_labels=pair_labels,
            ),
            "table4_dataset_cell_type_cluster_bca_ci.tsv": (
                table4_cell_type_ci
            ),
            "table4_normalization_comparison_cluster_bca_ci.tsv": (
                table4_comparison_ci
            ),
            "table4_normalization_comparison.tsv": _ci_presentation(
                table4_comparison_ci,
                pair_labels=pair_labels,
            ),
            "table6_dataset_cell_type_cluster_bca_ci.tsv": (
                table6_cell_type_ci
            ),
            "table6_normalization_comparison_cluster_bca_ci.tsv": (
                table6_comparison_ci
            ),
            "table6_normalization_comparison.tsv": _ci_presentation(
                table6_comparison_ci,
                pair_labels=pair_labels,
            ),
        }
        if dataset_cell_type_retrieval_ci is not None:
            additional_outputs[
                "retrieval_dataset_cell_type_cluster_bca_ci.tsv"
            ] = dataset_cell_type_retrieval_ci
        additional_input_paths = [
            Path(args.raw_deg_metrics),
            Path(args.dataset_cell_type_deg_metrics),
            Path(args.dataset_cell_type_signature_metrics),
            Path(args.dataset_cell_type_retrieval_metrics),
            summary_dir / "w4_deg_cluster_bca_ci.tsv",
            summary_dir / "w4_signature_cluster_bca_ci.tsv",
        ]

    outputs = {
        "retrieval_l2_cluster_bca_ci.tsv": l2_ci,
        "retrieval_combined_cluster_bca_ci.tsv": combined_ci,
        "table9_retrieval_combined_numeric_long.tsv": numeric,
        "table9_retrieval_combined.tsv": combined_table9,
        "retrieval_peer_sensitivity_all.tsv": peer_summary,
        "dose_threshold_deg_metric_summary_complete.tsv": (
            complete_dose_pair
        ),
        "dose_threshold_deg_cluster_bca_ci_complete.tsv": complete_dose_ci,
        **panels,
        **additional_outputs,
    }
    with output_directory_lock(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, frame in outputs.items():
            atomic_write_frame(output_dir / filename, frame)
        atomic_write_json(
            output_dir / "run_metadata.json",
            {
                "analysis": "reviewer-final-summary",
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "bootstrap_seed": int(args.bootstrap_seed),
                "workers": int(args.workers),
                "inputs": [
                    file_record(Path(path).resolve())
                    for path in (
                        args.l2_metrics,
                        args.peer_summary,
                        summary_dir
                        / "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
                        summary_dir
                        / "dose_threshold_deg_metric_summary.tsv",
                        summary_dir
                        / "dose_threshold_deg_cluster_bca_ci.tsv",
                        *additional_input_paths,
                    )
                ],
                "outputs": sorted([*outputs, "run_metadata.json"]),
            },
        )
    print(
        f"[reviewer-final-summary] wrote {len(outputs) + 1} files -> "
        f"{output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
