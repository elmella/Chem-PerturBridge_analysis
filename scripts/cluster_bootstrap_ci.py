from __future__ import annotations

import hashlib
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm


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


def _nested_statistics_from_totals(
    total_sum: np.ndarray,
    total_count: np.ndarray,
    *,
    outer_codes: np.ndarray | None,
) -> np.ndarray:
    """Vectorized form of ``_nested_statistic_from_totals``."""
    total_sum = np.asarray(total_sum, dtype=np.float64)
    total_count = np.asarray(total_count, dtype=np.float64)
    if total_sum.ndim != 2 or total_count.shape != total_sum.shape:
        raise ValueError("Batched totals must be matching two-dimensional arrays")
    valid = total_count > 0
    stratum_means = np.full(total_sum.shape, np.nan, dtype=np.float64)
    np.divide(
        total_sum,
        total_count,
        out=stratum_means,
        where=valid,
    )

    if outer_codes is None:
        finite_count = valid.sum(axis=1)
        result = np.full(total_sum.shape[0], np.nan, dtype=np.float64)
        np.divide(
            np.where(valid, stratum_means, 0.0).sum(axis=1),
            finite_count,
            out=result,
            where=finite_count > 0,
        )
        return result

    outer_codes = np.asarray(outer_codes, dtype=np.int64)
    unique_outer = np.unique(outer_codes)
    outer_means = np.full(
        (total_sum.shape[0], unique_outer.size),
        np.nan,
        dtype=np.float64,
    )
    for output_index, outer_code in enumerate(unique_outer):
        mask = outer_codes == outer_code
        outer_valid = valid[:, mask]
        outer_count = outer_valid.sum(axis=1)
        np.divide(
            np.where(
                outer_valid,
                stratum_means[:, mask],
                0.0,
            ).sum(axis=1),
            outer_count,
            out=outer_means[:, output_index],
            where=outer_count > 0,
        )
    outer_valid = np.isfinite(outer_means)
    outer_count = outer_valid.sum(axis=1)
    result = np.full(total_sum.shape[0], np.nan, dtype=np.float64)
    np.divide(
        np.where(outer_valid, outer_means, 0.0).sum(axis=1),
        outer_count,
        out=result,
        where=outer_count > 0,
    )
    return result


@dataclass(frozen=True)
class _ClusterStratumLayout:
    cluster_values: np.ndarray
    stratum_codes: np.ndarray
    outer_codes: np.ndarray | None
    n_strata: int


def _prepare_cluster_stratum_layout(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    inner_cols: list[str],
    outer_cols: list[str],
) -> _ClusterStratumLayout:
    cluster_values = (
        frame[cluster_col]
        .astype("string")
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy(dtype=object)
    )
    if not inner_cols:
        return _ClusterStratumLayout(
            cluster_values=cluster_values,
            stratum_codes=np.zeros(len(frame), dtype=np.int64),
            outer_codes=None,
            n_strata=1,
        )

    normalized = frame[inner_cols].astype("string").fillna("").astype(str)
    stratum_index = pd.MultiIndex.from_frame(normalized)
    stratum_codes, unique_strata = pd.factorize(
        stratum_index,
        sort=False,
    )
    unique_frame = unique_strata.to_frame(index=False)
    unique_frame.columns = inner_cols
    if outer_cols:
        missing_outer_cols = [
            column_name
            for column_name in outer_cols
            if column_name not in inner_cols
        ]
        if missing_outer_cols:
            raise ValueError(
                "outer_cols must be a subset of inner_cols for nested "
                f"bootstrap weighting: {missing_outer_cols}"
            )
        outer_index = pd.MultiIndex.from_frame(unique_frame[outer_cols])
        outer_codes = pd.factorize(outer_index, sort=False)[0].astype(
            np.int64
        )
    else:
        outer_codes = None
    return _ClusterStratumLayout(
        cluster_values=cluster_values,
        stratum_codes=np.asarray(stratum_codes, dtype=np.int64),
        outer_codes=outer_codes,
        n_strata=int(len(unique_strata)),
    )


def _cluster_stratum_matrices(
    frame: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    inner_cols: list[str],
    outer_cols: list[str],
    layout: _ClusterStratumLayout | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    if layout is None:
        layout = _prepare_cluster_stratum_layout(
            frame,
            cluster_col=cluster_col,
            inner_cols=inner_cols,
            outer_cols=outer_cols,
        )
    values = pd.to_numeric(frame[value_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid = (layout.cluster_values != "") & np.isfinite(values)
    n_finite_rows = int(valid.sum())
    if n_finite_rows == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty, None, 0, 0
    cluster_codes, cluster_keys = pd.factorize(
        layout.cluster_values[valid],
        sort=False,
    )
    cluster_codes = np.asarray(cluster_codes, dtype=np.int64)
    raw_stratum_codes = layout.stratum_codes[valid]
    active_strata = pd.unique(raw_stratum_codes)
    stratum_remap = np.full(layout.n_strata, -1, dtype=np.int64)
    stratum_remap[active_strata] = np.arange(
        len(active_strata),
        dtype=np.int64,
    )
    stratum_codes = stratum_remap[raw_stratum_codes]
    outer_codes = (
        None
        if layout.outer_codes is None
        else layout.outer_codes[active_strata]
    )
    n_clusters = int(len(cluster_keys))
    sum_matrix = np.zeros(
        (n_clusters, len(active_strata)),
        dtype=np.float64,
    )
    count_matrix = np.zeros_like(sum_matrix)
    np.add.at(
        sum_matrix,
        (cluster_codes, stratum_codes),
        values[valid],
    )
    np.add.at(count_matrix, (cluster_codes, stratum_codes), 1.0)
    return (
        sum_matrix,
        count_matrix,
        outer_codes,
        n_finite_rows,
        n_clusters,
    )


def _bootstrap_and_jackknife_values(
    *,
    sum_matrix: np.ndarray,
    count_matrix: np.ndarray,
    outer_codes: np.ndarray | None,
    n_boot: int,
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = int(sum_matrix.shape[0])
    bootstrap_values = np.empty(int(n_boot), dtype=np.float64)
    for batch_start in range(0, int(n_boot), int(batch_size)):
        batch_stop = min(batch_start + int(batch_size), int(n_boot))
        current_size = batch_stop - batch_start
        sampled_clusters = rng.integers(
            0,
            n_clusters,
            size=(current_size, n_clusters),
        )
        sampled_sum = np.empty(
            (current_size, sum_matrix.shape[1]),
            dtype=np.float64,
        )
        sampled_count = np.empty_like(sampled_sum)
        for row_index, sampled_indices in enumerate(sampled_clusters):
            # Retain the legacy indexed summation order so the same seed
            # produces the same BCa interval, independent of batching.
            sampled_sum[row_index] = sum_matrix[sampled_indices].sum(axis=0)
            sampled_count[row_index] = count_matrix[sampled_indices].sum(
                axis=0
            )
        bootstrap_values[batch_start:batch_stop] = (
            _nested_statistics_from_totals(
                sampled_sum,
                sampled_count,
                outer_codes=outer_codes,
            )
        )

    total_sum = sum_matrix.sum(axis=0)
    total_count = count_matrix.sum(axis=0)
    jackknife_values = _nested_statistics_from_totals(
        total_sum[None, :] - sum_matrix,
        total_count[None, :] - count_matrix,
        outer_codes=outer_codes,
    )
    return bootstrap_values, jackknife_values


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


def _compute_group_ci_records(
    group_values: tuple[Any, ...],
    group_frame: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_items: list[tuple[str, str]],
    cluster_col: str,
    inner_cols: list[str],
    outer_cols: list[str],
    n_boot: int,
    ci_level: float,
    seed: int,
    summary_level: str,
    uncertainty_scope: str,
    bootstrap_batch_size: int,
) -> list[dict[str, Any]]:
    group_record = {
        column_name: value
        for column_name, value in zip(group_cols, group_values)
    }
    layout = _prepare_cluster_stratum_layout(
        group_frame,
        cluster_col=cluster_col,
        inner_cols=inner_cols,
        outer_cols=outer_cols,
    )
    records: list[dict[str, Any]] = []
    for metric_label, value_col in metric_items:
        (
            sum_matrix,
            count_matrix,
            outer_codes,
            n_finite_rows,
            n_clusters,
        ) = _cluster_stratum_matrices(
            group_frame,
            value_col=value_col,
            cluster_col=cluster_col,
            inner_cols=inner_cols,
            outer_cols=outer_cols,
            layout=layout,
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
        observed = _nested_statistic_from_totals(
            total_sum,
            total_count,
            outer_codes=outer_codes,
        )
        local_rng = np.random.default_rng(
            _stable_seed(
                seed,
                summary_level,
                group_values,
                metric_label,
                value_col,
            )
        )
        bootstrap_values, jackknife_values = (
            _bootstrap_and_jackknife_values(
                sum_matrix=sum_matrix,
                count_matrix=count_matrix,
                outer_codes=outer_codes,
                n_boot=int(n_boot),
                rng=local_rng,
                batch_size=int(bootstrap_batch_size),
            )
        )
        (
            ci_low,
            ci_high,
            ci_method,
            ci_status,
            n_bootstrap_valid,
        ) = _bca_interval(
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
    return records


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
    bootstrap_batch_size: int = 64,
    workers: int = 1,
    progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    group_cols = _as_list(group_cols)
    inner_cols = _as_list(inner_cols)
    outer_cols = _as_list(outer_cols)
    metric_items = _metric_items(metric_cols)
    if int(bootstrap_batch_size) < 1:
        raise ValueError("bootstrap_batch_size must be positive")
    if int(workers) < 1:
        raise ValueError("workers must be positive")

    missing = sorted(set([*group_cols, cluster_col, *inner_cols, *outer_cols, *[column for _, column in metric_items]]) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns for cluster bootstrap CI table: {missing}")

    if group_cols:
        grouped_source = frame.groupby(
            group_cols,
            dropna=False,
            sort=False,
        )
        grouped = list(grouped_source)
    else:
        grouped = [((), frame)]
    normalized_groups: list[tuple[tuple[Any, ...], pd.DataFrame]] = []
    for group_values, group_frame in grouped:
        if len(group_cols) == 1:
            group_values = (
                (group_values[0],)
                if isinstance(group_values, tuple)
                else (group_values,)
            )
        elif not isinstance(group_values, tuple):
            group_values = tuple(group_values)
        normalized_groups.append((group_values, group_frame))
    progress_bar = tqdm(
        total=len(normalized_groups) * len(metric_items),
        desc=progress_desc or summary_level,
        unit="metric",
        dynamic_ncols=True,
        disable=not progress,
    )
    common_kwargs = {
        "group_cols": group_cols,
        "metric_items": metric_items,
        "cluster_col": cluster_col,
        "inner_cols": inner_cols,
        "outer_cols": outer_cols,
        "n_boot": int(n_boot),
        "ci_level": float(ci_level),
        "seed": int(seed),
        "summary_level": summary_level,
        "uncertainty_scope": uncertainty_scope,
        "bootstrap_batch_size": int(bootstrap_batch_size),
    }
    try:
        if int(workers) == 1 or len(normalized_groups) <= 1:
            grouped_records = []
            for group_values, group_frame in normalized_groups:
                result = _compute_group_ci_records(
                    group_values,
                    group_frame,
                    **common_kwargs,
                )
                grouped_records.append(result)
                progress_bar.update(len(result))
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(int(workers), len(normalized_groups)),
                mp_context=context,
            ) as executor:
                futures = [
                    executor.submit(
                        _compute_group_ci_records,
                        group_values,
                        group_frame,
                        **common_kwargs,
                    )
                    for group_values, group_frame in normalized_groups
                ]
                grouped_records = []
                # Reading results in submission order preserves deterministic
                # table row order regardless of worker completion order.
                for future in futures:
                    result = future.result()
                    grouped_records.append(result)
                    progress_bar.update(len(result))
    finally:
        progress_bar.close()
    records = [
        record
        for group_records in grouped_records
        for record in group_records
    ]
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
