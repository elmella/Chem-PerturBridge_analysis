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
    combined_ci = pd.concat(
        [minimal_ci, l2_ci],
        ignore_index=True,
    )
    _validate_retrieval_ci(combined_ci)
    pair_labels = _dataset_pair_labels(summary_dir)
    numeric = _table9_numeric_long(
        combined_ci,
        pair_labels=pair_labels,
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
            ("cosine", ("raw", PER_GENE_DATASET_VARIANT)),
            ("spearman", ("raw", PER_GENE_DATASET_VARIANT)),
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
