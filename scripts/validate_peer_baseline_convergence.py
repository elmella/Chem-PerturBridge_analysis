#!/usr/bin/env python3
"""Compare capped peer-baseline runs with an uncapped production reference.

Example:
    python scripts/validate_peer_baseline_convergence.py \
      --run 128=results/peers_128 --run 256=results/peers_256 \
      --run 512=results/peers_512 --run 1024=results/peers_1024 \
      --run full=results/peers_full --reference-label full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table


METRIC_SPECS = {
    "condition_metric_summary.tsv": {
        "mean_peer_baseline_spearman_logfc": 0.01,
        "mean_replicate_minus_peer_baseline_spearman_logfc": 0.01,
        "mean_peer_baseline_fraction_below_observed_spearman_logfc": 0.02,
        "mean_peer_baseline_corrected_percentile_spearman_logfc": 0.02,
    },
    "condition_deg_metric_summary.tsv": {
        "mean_peer_baseline_deg_lfc_spearman_sym_p05": 0.01,
        "mean_delta_vs_peer_baseline_deg_lfc_spearman_sym_p05": 0.01,
        "mean_peer_baseline_deg_lfc_spearman_sym_fraction_below_observed_p05": 0.02,
        "mean_peer_baseline_deg_lfc_spearman_sym_corrected_percentile_p05": 0.02,
        "mean_peer_baseline_direction_agreement_p05": 0.01,
        "mean_delta_vs_peer_baseline_direction_agreement_p05": 0.01,
        "mean_peer_baseline_direction_agreement_fraction_below_observed_p05": 0.02,
        "mean_peer_baseline_direction_agreement_corrected_percentile_p05": 0.02,
    },
}
DELTA_METRICS = {
    "mean_replicate_minus_peer_baseline_spearman_logfc",
    "mean_delta_vs_peer_baseline_deg_lfc_spearman_sym_p05",
    "mean_delta_vs_peer_baseline_direction_agreement_p05",
}
KEY_COLUMNS = ["dataset_name", "condition_key"]


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must have LABEL=RESULTS_DIR form")
    return label, Path(raw_path)


def interval_sign(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "not_estimable"
    if ci_low > 0.0:
        return "positive"
    if ci_high < 0.0:
        return "negative"
    return "crosses_zero"


def dataset_metric_means(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return (
        frame.groupby("dataset_name", as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values("dataset_name")
        .reset_index(drop=True)
    )


def delta_interval_signs(
    frame: pd.DataFrame,
    metrics: list[str],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    delta_metrics = [metric for metric in metrics if metric in DELTA_METRICS]
    if not delta_metrics:
        return pd.DataFrame(columns=["dataset_name", "metric", "interval_sign"])
    ci = cluster_bca_nested_mean_ci_table(
        frame,
        group_cols=["dataset_name"],
        metric_cols={metric: metric for metric in delta_metrics},
        cluster_col="pubchem_cid",
        n_boot=n_boot,
        seed=seed,
        summary_level="convergence",
    )
    ci["interval_sign"] = [
        interval_sign(low, high)
        for low, high in zip(ci["ci_low"].to_numpy(), ci["ci_high"].to_numpy())
    ]
    return ci[["dataset_name", "metric", "interval_sign"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--reference-label", default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("peer_baseline_convergence.tsv"),
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260505)
    args = parser.parse_args()

    run_dirs = dict(args.run)
    if args.reference_label not in run_dirs:
        raise SystemExit(
            f"Reference label {args.reference_label!r} is not among {sorted(run_dirs)!r}"
        )

    records: list[dict[str, object]] = []
    for filename, metric_tolerances in METRIC_SPECS.items():
        frames: dict[str, pd.DataFrame] = {}
        for label, results_dir in run_dirs.items():
            path = results_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path, sep="\t")
            missing = sorted(
                set([*KEY_COLUMNS, "pubchem_cid", *metric_tolerances]) - set(frame.columns)
            )
            if missing:
                raise KeyError(f"{path} is missing columns: {missing}")
            frames[label] = frame

        reference = frames[args.reference_label]
        reference_means = dataset_metric_means(
            reference,
            list(metric_tolerances),
        ).set_index("dataset_name")
        reference_signs = delta_interval_signs(
            reference,
            list(metric_tolerances),
            n_boot=args.n_boot,
            seed=args.seed,
        ).rename(columns={"interval_sign": "reference_interval_sign"})

        for label, frame in frames.items():
            if label == args.reference_label:
                continue
            candidate_means = dataset_metric_means(
                frame,
                list(metric_tolerances),
            ).set_index("dataset_name")
            shared_datasets = reference_means.index.intersection(candidate_means.index)
            candidate_signs = delta_interval_signs(
                frame,
                list(metric_tolerances),
                n_boot=args.n_boot,
                seed=args.seed,
            )
            sign_comparison = candidate_signs.merge(
                reference_signs,
                on=["dataset_name", "metric"],
                how="outer",
            )
            sign_lookup = {
                (str(row.dataset_name), str(row.metric)): (
                    str(row.interval_sign),
                    str(row.reference_interval_sign),
                )
                for row in sign_comparison.itertuples(index=False)
            }
            for metric, tolerance in metric_tolerances.items():
                differences = (
                    candidate_means.loc[shared_datasets, metric]
                    - reference_means.loc[shared_datasets, metric]
                ).abs()
                max_difference = float(differences.max()) if len(differences) else np.nan
                mean_difference = float(differences.mean()) if len(differences) else np.nan
                sign_mismatches = 0
                if metric in DELTA_METRICS:
                    sign_mismatches = sum(
                        candidate_sign != reference_sign
                        for (dataset, compared_metric), (
                            candidate_sign,
                            reference_sign,
                        ) in sign_lookup.items()
                        if compared_metric == metric
                    )
                passed = (
                    np.isfinite(max_difference)
                    and max_difference <= float(tolerance)
                    and sign_mismatches == 0
                )
                records.append(
                    {
                        "result_file": filename,
                        "run_label": label,
                        "reference_label": args.reference_label,
                        "metric": metric,
                        "n_datasets": int(len(shared_datasets)),
                        "mean_abs_dataset_difference": mean_difference,
                        "max_abs_dataset_difference": max_difference,
                        "tolerance": float(tolerance),
                        "n_interval_sign_mismatches": int(sign_mismatches),
                        "passed": bool(passed),
                    }
                )

    report = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, sep="\t", index=False)
    print(report.to_string(index=False))
    print(f"Saved convergence report to {args.output}")
    if report.empty or not bool(report["passed"].all()):
        raise SystemExit("Peer-baseline cap convergence failed.")


if __name__ == "__main__":
    main()
