from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


OBSERVED_COMPOUND_PANEL_SCOPE = (
    "95% BCa bootstrap CI clustered by PubChem compound; uncertainty is over the observed "
    "compound panel, not broader chemical space."
)


def _as_list(value: Sequence[str] | None) -> list[str]:
    return list(value) if value is not None else []


def _metric_items(metric_cols: Mapping[str, str] | Sequence[str]) -> list[tuple[str, str]]:
    if isinstance(metric_cols, Mapping):
        return [(str(label), str(column_name)) for label, column_name in metric_cols.items()]
    return [(str(column_name), str(column_name)) for column_name in metric_cols]


def _stable_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int((int(base_seed) + int.from_bytes(digest, "little")) % (2**32 - 1))


def _string_key_frame(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[columns].astype("string").fillna("").agg("\x1f".join, axis=1)


def _nested_statistic_from_totals(
    total_sum: np.ndarray,
    total_count: np.ndarray,
    *,
    outer_codes: np.ndarray | None,
) -> float:
    valid = total_count > 0
    if not np.any(valid):
        return float("nan")

    stratum_means = np.full(total_sum.shape, np.nan, dtype=np.float64)
    stratum_means[valid] = total_sum[valid] / total_count[valid]
    if outer_codes is None:
        return float(np.nanmean(stratum_means))

    outer_values = []
    for outer_code in np.unique(outer_codes):
        mask = outer_codes == outer_code
        if np.isfinite(stratum_means[mask]).any():
            outer_values.append(float(np.nanmean(stratum_means[mask])))
    if not outer_values:
        return float("nan")
    return float(np.nanmean(np.asarray(outer_values, dtype=np.float64)))


def _cluster_stratum_matrices(
    frame: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    inner_cols: list[str],
    outer_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    needed_cols = list(dict.fromkeys([cluster_col, value_col, *inner_cols, *outer_cols]))
    working = frame[needed_cols].copy()
    working[cluster_col] = working[cluster_col].astype("string").fillna("").astype(str).str.strip()
    working["_metric_value"] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.loc[(working[cluster_col] != "") & np.isfinite(working["_metric_value"])].copy()
    if working.empty:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty, None, 0, 0

    if inner_cols:
        working["_stratum_key"] = _string_key_frame(working, inner_cols)
        stratum_keys = pd.Index(pd.unique(working["_stratum_key"]))
        stratum_codes = stratum_keys.get_indexer(working["_stratum_key"])
        stratum_frame = working.drop_duplicates("_stratum_key").set_index("_stratum_key").loc[stratum_keys]
        if outer_cols:
            missing_outer_cols = [column_name for column_name in outer_cols if column_name not in inner_cols]
            if missing_outer_cols:
                raise ValueError(
                    "outer_cols must be a subset of inner_cols for nested bootstrap weighting: "
                    f"{missing_outer_cols}"
                )
            outer_key = _string_key_frame(stratum_frame.reset_index(drop=True), outer_cols)
            outer_codes = pd.Categorical(outer_key).codes.astype(np.int64)
        else:
            outer_codes = None
    else:
        working["_stratum_key"] = "__all__"
        stratum_codes = np.zeros(len(working), dtype=np.int64)
        outer_codes = None

    cluster_keys = pd.Index(pd.unique(working[cluster_col]))
    cluster_codes = cluster_keys.get_indexer(working[cluster_col])
    n_clusters = int(len(cluster_keys))
    n_strata = int(np.max(stratum_codes) + 1) if len(stratum_codes) else 0

    sum_matrix = np.zeros((n_clusters, n_strata), dtype=np.float64)
    count_matrix = np.zeros((n_clusters, n_strata), dtype=np.float64)
    values = working["_metric_value"].to_numpy(dtype=np.float64)
    np.add.at(sum_matrix, (cluster_codes, stratum_codes), values)
    np.add.at(count_matrix, (cluster_codes, stratum_codes), 1.0)
    return sum_matrix, count_matrix, outer_codes, int(len(working)), n_clusters


def _bca_interval(
    observed: float,
    bootstrap_values: np.ndarray,
    jackknife_values: np.ndarray,
    *,
    ci_level: float,
) -> tuple[float, float, str, str, int]:
    bootstrap_values = bootstrap_values[np.isfinite(bootstrap_values)]
    jackknife_values = jackknife_values[np.isfinite(jackknife_values)]
    n_bootstrap_valid = int(bootstrap_values.size)
    if not np.isfinite(observed):
        return float("nan"), float("nan"), "not_estimable", "nonfinite_observed", n_bootstrap_valid
    if n_bootstrap_valid < 20:
        return float("nan"), float("nan"), "not_estimable", "too_few_valid_bootstraps", n_bootstrap_valid
    if np.nanmax(bootstrap_values) == np.nanmin(bootstrap_values):
        return float(observed), float(observed), "degenerate", "zero_bootstrap_variance", n_bootstrap_valid

    alpha_low = (1.0 - float(ci_level)) / 2.0
    alpha_high = 1.0 - alpha_low
    percentile_low, percentile_high = np.quantile(bootstrap_values, [alpha_low, alpha_high])

    if jackknife_values.size < 3:
        return (
            float(percentile_low),
            float(percentile_high),
            "percentile_fallback",
            "too_few_jackknife_values_for_bca",
            n_bootstrap_valid,
        )

    proportion_less = float(np.mean(bootstrap_values < observed))
    proportion_less = min(max(proportion_less, 1.0 / (2.0 * n_bootstrap_valid)), 1.0 - 1.0 / (2.0 * n_bootstrap_valid))
    z0 = float(norm.ppf(proportion_less))

    jackknife_mean = float(np.nanmean(jackknife_values))
    jackknife_delta = jackknife_mean - jackknife_values
    denominator = 6.0 * float(np.sum(jackknife_delta**2) ** 1.5)
    if denominator == 0.0 or not np.isfinite(denominator):
        return (
            float(percentile_low),
            float(percentile_high),
            "percentile_fallback",
            "zero_or_nonfinite_bca_acceleration",
            n_bootstrap_valid,
        )
    acceleration = float(np.sum(jackknife_delta**3) / denominator)
    if not np.isfinite(acceleration):
        return (
            float(percentile_low),
            float(percentile_high),
            "percentile_fallback",
            "nonfinite_bca_acceleration",
            n_bootstrap_valid,
        )

    adjusted_quantiles = []
    for alpha in [alpha_low, alpha_high]:
        z_alpha = float(norm.ppf(alpha))
        adjustment_denominator = 1.0 - acceleration * (z0 + z_alpha)
        if adjustment_denominator == 0.0 or not np.isfinite(adjustment_denominator):
            return (
                float(percentile_low),
                float(percentile_high),
                "percentile_fallback",
                "invalid_bca_adjusted_quantile",
                n_bootstrap_valid,
            )
        adjusted = float(norm.cdf(z0 + ((z0 + z_alpha) / adjustment_denominator)))
        adjusted_quantiles.append(min(max(adjusted, 0.0), 1.0))

    ci_low, ci_high = np.quantile(bootstrap_values, adjusted_quantiles)
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low
    return float(ci_low), float(ci_high), "bca", "ok", n_bootstrap_valid


def cluster_bca_nested_mean_ci_table(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    metric_cols: Mapping[str, str] | Sequence[str],
    cluster_col: str = "pubchem_cid",
    inner_cols: Sequence[str] | None = None,
    outer_cols: Sequence[str] | None = None,
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 20260505,
    summary_level: str,
    uncertainty_scope: str = OBSERVED_COMPOUND_PANEL_SCOPE,
) -> pd.DataFrame:
    group_cols = _as_list(group_cols)
    inner_cols = _as_list(inner_cols)
    outer_cols = _as_list(outer_cols)
    metric_items = _metric_items(metric_cols)

    missing = sorted(set([*group_cols, cluster_col, *inner_cols, *outer_cols, *[column for _, column in metric_items]]) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns for cluster bootstrap CI table: {missing}")

    records: list[dict[str, Any]] = []
    grouped = frame.groupby(group_cols, dropna=False, sort=False) if group_cols else [((), frame)]
    for group_values, group_frame in grouped:
        if len(group_cols) == 1:
            group_values = (group_values[0],) if isinstance(group_values, tuple) else (group_values,)
        elif not isinstance(group_values, tuple):
            group_values = tuple(group_values)
        group_record = {column_name: value for column_name, value in zip(group_cols, group_values)}

        for metric_label, value_col in metric_items:
            sum_matrix, count_matrix, outer_codes, n_finite_rows, n_clusters = _cluster_stratum_matrices(
                group_frame,
                value_col=value_col,
                cluster_col=cluster_col,
                inner_cols=inner_cols,
                outer_cols=outer_cols,
            )
            if n_clusters < 2 or n_finite_rows == 0:
                records.append(
                    {
                        **group_record,
                        "summary_level": summary_level,
                        "metric": metric_label,
                        "value_col": value_col,
                        "mean": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "ci_half_width": float("nan"),
                        "ci_method": "not_estimable",
                        "ci_status": "too_few_clusters_or_values",
                        "ci_level": float(ci_level),
                        "n_bootstrap_iterations": int(n_boot),
                        "n_bootstrap_valid": 0,
                        "n_rows": int(len(group_frame)),
                        "n_finite_rows": int(n_finite_rows),
                        "n_compounds": int(n_clusters),
                        "cluster_col": cluster_col,
                        "inner_strata": ",".join(inner_cols),
                        "outer_strata": ",".join(outer_cols),
                        "uncertainty_scope": uncertainty_scope,
                    }
                )
                continue

            total_sum = sum_matrix.sum(axis=0)
            total_count = count_matrix.sum(axis=0)
            observed = _nested_statistic_from_totals(total_sum, total_count, outer_codes=outer_codes)

            local_rng = np.random.default_rng(_stable_seed(seed, summary_level, group_values, metric_label, value_col))
            bootstrap_values = np.empty(int(n_boot), dtype=np.float64)
            for bootstrap_idx in range(int(n_boot)):
                sampled_clusters = local_rng.integers(0, n_clusters, n_clusters)
                sampled_sum = sum_matrix[sampled_clusters].sum(axis=0)
                sampled_count = count_matrix[sampled_clusters].sum(axis=0)
                bootstrap_values[bootstrap_idx] = _nested_statistic_from_totals(
                    sampled_sum,
                    sampled_count,
                    outer_codes=outer_codes,
                )

            jackknife_values = np.empty(n_clusters, dtype=np.float64)
            for cluster_idx in range(n_clusters):
                jackknife_values[cluster_idx] = _nested_statistic_from_totals(
                    total_sum - sum_matrix[cluster_idx],
                    total_count - count_matrix[cluster_idx],
                    outer_codes=outer_codes,
                )

            ci_low, ci_high, ci_method, ci_status, n_bootstrap_valid = _bca_interval(
                observed,
                bootstrap_values,
                jackknife_values,
                ci_level=float(ci_level),
            )
            ci_half_width = (
                float((ci_high - ci_low) / 2.0)
                if np.isfinite(ci_low) and np.isfinite(ci_high)
                else float("nan")
            )
            records.append(
                {
                    **group_record,
                    "summary_level": summary_level,
                    "metric": metric_label,
                    "value_col": value_col,
                    "mean": float(observed),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_half_width": ci_half_width,
                    "ci_method": ci_method,
                    "ci_status": ci_status,
                    "ci_level": float(ci_level),
                    "n_bootstrap_iterations": int(n_boot),
                    "n_bootstrap_valid": int(n_bootstrap_valid),
                    "n_rows": int(len(group_frame)),
                    "n_finite_rows": int(n_finite_rows),
                    "n_compounds": int(n_clusters),
                    "cluster_col": cluster_col,
                    "inner_strata": ",".join(inner_cols),
                    "outer_strata": ",".join(outer_cols),
                    "uncertainty_scope": uncertainty_scope,
                }
            )

    return pd.DataFrame(records)


def summarize_ci_half_width_ranges(
    ci_table: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("summary_level", "metric"),
) -> pd.DataFrame:
    group_cols = _as_list(group_cols)
    required = set(group_cols) | {"ci_half_width", "ci_method", "ci_status", "n_compounds"}
    missing = sorted(required - set(ci_table.columns))
    if missing:
        raise KeyError(f"Missing columns for CI range summary: {missing}")

    finite = ci_table.loc[np.isfinite(pd.to_numeric(ci_table["ci_half_width"], errors="coerce"))].copy()
    if finite.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "n_intervals",
                "min_ci_half_width",
                "median_ci_half_width",
                "max_ci_half_width",
                "min_n_compounds",
                "max_n_compounds",
                "ci_methods",
                "ci_statuses",
            ]
        )

    return (
        finite.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_intervals=("ci_half_width", "size"),
            min_ci_half_width=("ci_half_width", "min"),
            median_ci_half_width=("ci_half_width", "median"),
            max_ci_half_width=("ci_half_width", "max"),
            min_n_compounds=("n_compounds", "min"),
            max_n_compounds=("n_compounds", "max"),
            ci_methods=("ci_method", lambda values: ",".join(sorted(set(map(str, values))))),
            ci_statuses=("ci_status", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .reset_index(drop=True)
    )
