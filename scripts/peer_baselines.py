"""Per-peer (single-signature) baseline helpers shared by the reviewer-addition analyses.

The published baseline for every agreement metric is a *centroid*: the gene-wise mean
over same-context different-compound peers, scored once. Reviewer W1 notes that
averaging removes variance and preserves the common-mode response, so the centroid is
an unfairly strong competitor on rank- and distance-based metrics.

This module implements the per-peer alternative: score the metric against every peer
signature separately, then summarize the resulting distribution. It is deliberately
metric-agnostic -- callers pass an observed scalar and a vector of peer scores -- so the
same summary convention is used for retrieval similarity, DEG-restricted logFC Spearman,
logFC direction agreement, and all-gene logFC Spearman.

Field naming corresponds to ``peer_score_distribution_summary`` in
``notebooks/overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb``;
``_mean_score`` / ``_sd_score`` here are ``_mean_similarity`` / ``_sd_similarity`` there.

Run ``python scripts/peer_baselines.py`` to execute the self-tests, which check the
vectorized paths against scalar reference implementations copied from the notebooks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "DEFAULT_PEER_SAMPLING_SEED",
    "PEER_SUMMARY_FIELDS",
    "different_compound_peer_mask",
    "summarize_peer_scores",
    "empty_peer_score_summary",
    "spearman_scalar",
    "spearman_against_peers",
    "PreparedSpearmanRows",
    "prepare_spearman_rows",
    "spearman_against_prepared_peers",
    "exact_mean_for_row_mask",
    "finite_column_totals",
    "exact_mean_excluding_row_mask",
    "direction_agreement_scalar",
    "direction_agreement_against_peers",
    "select_peer_indices",
]

PEER_SUMMARY_FIELDS = (
    "n_peers",
    "mean_score",
    "sd_score",
    "n_below_observed",
    "fraction_below_observed",
    "corrected_percentile",
    "n_at_least_observed",
    "empirical_p_upper",
)

DEFAULT_PEER_SAMPLING_SEED = 20260505


@dataclass(frozen=True)
class PreparedSpearmanRows:
    """Peer rows with reusable normalized ranks for repeated all-gene scoring."""

    values: np.ndarray
    normalized_ranks: np.ndarray
    finite_rows: np.ndarray
    valid_rows: np.ndarray


def different_compound_peer_mask(
    dose_keys: np.ndarray,
    compounds: np.ndarray,
    *,
    dose_key: str,
    excluded_compound: str,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Select same-dose, different-compound peers from the supplied population.

    Retrieval analyses pass their overlap-filtered candidate arrays here, ensuring that
    individual signatures and their centroid are evaluated over the same population.
    """
    dose_keys = np.asarray(dose_keys).astype(str).reshape(-1)
    compounds = np.asarray(compounds).astype(str).reshape(-1)
    if dose_keys.shape != compounds.shape:
        raise ValueError("dose_keys and compounds must have the same shape")
    peer_mask = (
        (dose_keys == str(dose_key))
        & (compounds != str(excluded_compound))
    )
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape != peer_mask.shape:
            raise ValueError("valid_mask must match dose_keys and compounds")
        peer_mask &= valid_mask
    return peer_mask


def summarize_peer_scores(
    observed_score: float,
    peer_scores: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    """Summarize a per-peer score distribution against the observed matched score.

    ``corrected_percentile`` is ``(1 + #{peers below observed}) / (K + 1)`` and
    ``empirical_p_upper`` is ``(1 + #{peers >= observed}) / (K + 1)``; the add-one
    correction keeps both strictly inside ``(0, 1)`` for finite peer sets.
    """
    peer_scores = np.asarray(peer_scores, dtype=np.float64).reshape(-1)
    peer_scores = peer_scores[np.isfinite(peer_scores)]
    n_peers = int(peer_scores.size)
    if n_peers == 0 or not np.isfinite(observed_score):
        return empty_peer_score_summary(prefix, n_peers=n_peers)

    observed_score = float(observed_score)
    n_below = int(np.sum(peer_scores < observed_score))
    n_at_least = int(np.sum(peer_scores >= observed_score))
    return {
        f"{prefix}_n_peers": n_peers,
        f"{prefix}_mean_score": float(np.mean(peer_scores)),
        f"{prefix}_sd_score": (
            float(np.std(peer_scores, ddof=1)) if n_peers >= 2 else float("nan")
        ),
        f"{prefix}_n_below_observed": n_below,
        f"{prefix}_fraction_below_observed": float(n_below / n_peers),
        f"{prefix}_corrected_percentile": float((n_below + 1.0) / (n_peers + 1.0)),
        f"{prefix}_n_at_least_observed": n_at_least,
        f"{prefix}_empirical_p_upper": float((n_at_least + 1.0) / (n_peers + 1.0)),
    }


def empty_peer_score_summary(prefix: str, *, n_peers: int = 0) -> dict[str, float | int]:
    return {
        f"{prefix}_n_peers": int(n_peers),
        f"{prefix}_mean_score": float("nan"),
        f"{prefix}_sd_score": float("nan"),
        f"{prefix}_n_below_observed": 0,
        f"{prefix}_fraction_below_observed": float("nan"),
        f"{prefix}_corrected_percentile": float("nan"),
        f"{prefix}_n_at_least_observed": 0,
        f"{prefix}_empirical_p_upper": float("nan"),
    }


def spearman_scalar(left_values: np.ndarray, right_values: np.ndarray) -> float:
    """Signed Spearman with the notebooks' nan conventions.

    Mirrors ``signed_spearman``: drop non-finite pairs, require at least two remaining
    observations, and return nan when either rank vector is constant.
    """
    left_values = np.asarray(left_values, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[finite]
    right_values = right_values[finite]
    if left_values.size < 2:
        return float("nan")
    left_ranks = rankdata(left_values, method="average")
    right_ranks = rankdata(right_values, method="average")
    if np.allclose(left_ranks, left_ranks[0]) or np.allclose(right_ranks, right_ranks[0]):
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def spearman_against_peers(
    query_values: np.ndarray,
    peer_matrix: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Spearman of ``query_values`` against every row of ``peer_matrix``.

    ``base_mask`` restricts the evaluation to a gene subset, e.g. the query sample's own
    DEG mask under the sample-referenced convention. Rows whose values are all finite go
    through a vectorized rank-and-matmul path; rows with any non-finite value fall back
    to :func:`spearman_scalar` so per-row evaluation sets stay exact.
    """
    query_values = np.asarray(query_values, dtype=np.float64).reshape(-1)
    peer_matrix = np.atleast_2d(np.asarray(peer_matrix, dtype=np.float64))
    n_peers = int(peer_matrix.shape[0])
    scores = np.full(n_peers, np.nan, dtype=np.float64)
    if n_peers == 0:
        return scores
    if peer_matrix.shape[1] != query_values.size:
        raise ValueError(
            f"peer_matrix has {peer_matrix.shape[1]} columns but query has {query_values.size}"
        )

    if base_mask is None:
        base_mask = np.ones(query_values.size, dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    columns = np.flatnonzero(base_mask & np.isfinite(query_values))
    if columns.size < 2:
        return scores

    query_subset = query_values[columns]
    peer_subset = peer_matrix[:, columns]
    finite_rows = np.isfinite(peer_subset).all(axis=1)

    if finite_rows.any():
        query_ranks = rankdata(query_subset, method="average")
        if not np.allclose(query_ranks, query_ranks[0]):
            peer_ranks = np.atleast_2d(
                rankdata(peer_subset[finite_rows], method="average", axis=1)
            ).astype(np.float64)
            constant_rows = np.isclose(peer_ranks, peer_ranks[:, :1]).all(axis=1)
            centered_query = query_ranks - query_ranks.mean()
            query_norm = float(np.linalg.norm(centered_query))
            centered_peers = peer_ranks - peer_ranks.mean(axis=1, keepdims=True)
            peer_norms = np.linalg.norm(centered_peers, axis=1)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                row_scores = (centered_peers @ centered_query) / (peer_norms * query_norm)
            row_scores = np.clip(row_scores, -1.0, 1.0)
            row_scores[constant_rows | ~(peer_norms > 0.0)] = np.nan
            scores[finite_rows] = row_scores

    for row_index in np.flatnonzero(~finite_rows):
        scores[row_index] = spearman_scalar(query_subset, peer_subset[row_index])
    return scores


def prepare_spearman_rows(peer_matrix: np.ndarray) -> PreparedSpearmanRows:
    """Rank and normalize finite peer rows once for repeated all-gene Spearman calls.

    Rows containing non-finite values remain available in ``values`` and are evaluated
    with the scalar reference path by :func:`spearman_against_prepared_peers`.
    """
    values = np.atleast_2d(np.asarray(peer_matrix, dtype=np.float64))
    n_rows, n_columns = values.shape
    normalized_ranks = np.full((n_rows, n_columns), np.nan, dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    valid_rows = np.zeros(n_rows, dtype=bool)
    if n_columns < 2 or not finite_rows.any():
        return PreparedSpearmanRows(values, normalized_ranks, finite_rows, valid_rows)

    ranks = np.atleast_2d(
        rankdata(values[finite_rows], method="average", axis=1)
    ).astype(np.float64)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    finite_positions = np.flatnonzero(finite_rows)
    usable = norms > 0.0
    if usable.any():
        normalized_ranks[finite_positions[usable]] = (
            centered[usable] / norms[usable, None]
        )
        valid_rows[finite_positions[usable]] = True
    return PreparedSpearmanRows(values, normalized_ranks, finite_rows, valid_rows)


def spearman_against_prepared_peers(
    query_values: np.ndarray,
    prepared_peers: PreparedSpearmanRows,
    *,
    row_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score one all-gene query against reusable peer ranks.

    This is exactly equivalent to ``spearman_against_peers(query, selected_rows)`` for
    the unmasked all-gene case, including ties, constants, and per-row NaN handling.
    """
    query_values = np.asarray(query_values, dtype=np.float64).reshape(-1)
    values = prepared_peers.values
    if values.shape[1] != query_values.size:
        raise ValueError(
            f"prepared peers have {values.shape[1]} columns but query has {query_values.size}"
        )
    if row_indices is None:
        selected = np.arange(values.shape[0], dtype=np.int64)
    else:
        selected = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    scores = np.full(selected.size, np.nan, dtype=np.float64)
    if selected.size == 0 or query_values.size < 2:
        return scores

    # Query NaNs change the evaluation columns and therefore the peer ranks. Preserve
    # exact notebook semantics by falling back to the established implementation.
    if not np.isfinite(query_values).all():
        return spearman_against_peers(query_values, values[selected])

    query_ranks = rankdata(query_values, method="average").astype(np.float64)
    centered_query = query_ranks - query_ranks.mean()
    query_norm = float(np.linalg.norm(centered_query))
    if not query_norm > 0.0:
        return scores
    normalized_query = centered_query / query_norm

    selected_valid = prepared_peers.valid_rows[selected]
    if selected_valid.any():
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            scores[selected_valid] = (
                prepared_peers.normalized_ranks[selected[selected_valid]]
                @ normalized_query
            )
        scores[selected_valid] = np.clip(scores[selected_valid], -1.0, 1.0)

    fallback_positions = np.flatnonzero(~prepared_peers.finite_rows[selected])
    for output_position in fallback_positions:
        scores[output_position] = spearman_scalar(
            query_values,
            values[int(selected[output_position])],
        )
    return scores


def exact_mean_for_row_mask(
    matrix: np.ndarray,
    row_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Gene-wise finite mean over all selected rows, independent of any peer cap."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    row_mask = np.asarray(row_mask, dtype=bool).reshape(-1)
    if row_mask.size != matrix.shape[0]:
        raise ValueError("row_mask length does not match matrix rows")
    selected = matrix[row_mask]
    if selected.shape[0] == 0:
        return None
    finite = np.isfinite(selected)
    counts = finite.sum(axis=0)
    if not np.any(counts > 0):
        return None
    sums = np.where(finite, selected, 0.0).sum(axis=0)
    mean = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean


def finite_column_totals(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite sums and counts for reuse across many exclusion centroids."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    finite = np.isfinite(matrix)
    return (
        np.where(finite, matrix, 0.0).sum(axis=0),
        finite.sum(axis=0, dtype=np.int64),
    )


def exact_mean_excluding_row_mask(
    matrix: np.ndarray,
    excluded_row_mask: np.ndarray,
    *,
    total_sums: Optional[np.ndarray] = None,
    total_counts: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Exact finite mean after excluding rows, using reusable context totals."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    excluded_row_mask = np.asarray(excluded_row_mask, dtype=bool).reshape(-1)
    if excluded_row_mask.size != matrix.shape[0]:
        raise ValueError("excluded_row_mask length does not match matrix rows")
    if total_sums is None or total_counts is None:
        total_sums, total_counts = finite_column_totals(matrix)
    else:
        total_sums = np.asarray(total_sums, dtype=np.float64).reshape(-1)
        total_counts = np.asarray(total_counts, dtype=np.int64).reshape(-1)
    excluded_sums, excluded_counts = finite_column_totals(matrix[excluded_row_mask])
    remaining_sums = total_sums - excluded_sums
    remaining_counts = total_counts - excluded_counts
    if not np.any(remaining_counts > 0):
        return None
    mean = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid = remaining_counts > 0
    mean[valid] = remaining_sums[valid] / remaining_counts[valid]
    return mean


def direction_agreement_scalar(
    query_values: np.ndarray,
    peer_values: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
) -> float:
    """Sign agreement on ``base_mask`` genes where both vectors are finite.

    Mirrors ``direction_agreement_with_masks`` called with the sample's DEG mask on one
    side and the baseline's finite mask on the other, which is the sample-referenced
    convention used for the published centroid baselines.
    """
    query_values = np.asarray(query_values, dtype=np.float64).reshape(-1)
    peer_values = np.asarray(peer_values, dtype=np.float64).reshape(-1)
    if base_mask is None:
        base_mask = np.ones(query_values.size, dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    overlap = base_mask & np.isfinite(query_values) & np.isfinite(peer_values)
    if int(overlap.sum()) == 0:
        return float("nan")
    return float(
        np.mean(np.sign(query_values[overlap]) == np.sign(peer_values[overlap]))
    )


def direction_agreement_against_peers(
    query_values: np.ndarray,
    peer_matrix: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Direction agreement of ``query_values`` against every row of ``peer_matrix``."""
    query_values = np.asarray(query_values, dtype=np.float64).reshape(-1)
    peer_matrix = np.atleast_2d(np.asarray(peer_matrix, dtype=np.float64))
    n_peers = int(peer_matrix.shape[0])
    scores = np.full(n_peers, np.nan, dtype=np.float64)
    if n_peers == 0:
        return scores
    if peer_matrix.shape[1] != query_values.size:
        raise ValueError(
            f"peer_matrix has {peer_matrix.shape[1]} columns but query has {query_values.size}"
        )

    if base_mask is None:
        base_mask = np.ones(query_values.size, dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    columns = np.flatnonzero(base_mask & np.isfinite(query_values))
    if columns.size == 0:
        return scores

    query_signs = np.sign(query_values[columns])
    peer_subset = peer_matrix[:, columns]
    finite_peers = np.isfinite(peer_subset)
    agreements = (np.sign(peer_subset) == query_signs[None, :]) & finite_peers
    counts = finite_peers.sum(axis=1)
    valid = counts > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        scores[valid] = agreements.sum(axis=1)[valid] / counts[valid]
    return scores


def select_peer_indices(
    n_peers: int,
    max_peers: Optional[int],
    seed_key: str,
    sampling_seed: int = DEFAULT_PEER_SAMPLING_SEED,
) -> np.ndarray:
    """Deterministically subsample peer row indices when a cap is configured.

    The seed is derived from ``seed_key`` with blake2b rather than :func:`hash`, so the
    selection is stable across processes and machines. Returns sorted indices; returns
    all indices when no cap applies.
    """
    n_peers = int(n_peers)
    if n_peers <= 0:
        return np.empty(0, dtype=np.int64)
    if max_peers is None or int(max_peers) <= 0 or n_peers <= int(max_peers):
        return np.arange(n_peers, dtype=np.int64)
    digest = hashlib.blake2b(
        f"{int(sampling_seed)}|{seed_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    chosen = rng.choice(n_peers, size=int(max_peers), replace=False)
    return np.sort(chosen).astype(np.int64)


def _self_test() -> None:
    rng = np.random.default_rng(20260505)

    summary = summarize_peer_scores(0.60, np.asarray([0.10, 0.20, 0.55, 0.70]), "test")
    assert summary["test_n_peers"] == 4
    assert summary["test_n_below_observed"] == 3
    assert np.isclose(summary["test_fraction_below_observed"], 0.75)
    assert np.isclose(summary["test_corrected_percentile"], 0.80)
    assert np.isclose(summary["test_n_at_least_observed"], 1)
    assert np.isclose(summary["test_empirical_p_upper"], 0.40)
    assert np.isclose(summary["test_mean_score"], 0.3875)
    assert np.isclose(summary["test_sd_score"], np.std([0.10, 0.20, 0.55, 0.70], ddof=1))
    assert np.isnan(summarize_peer_scores(np.nan, np.asarray([0.1, 0.2]), "t")["t_mean_score"])
    assert summarize_peer_scores(0.5, np.asarray([]), "t")["t_n_peers"] == 0
    assert np.isnan(summarize_peer_scores(0.5, np.asarray([0.1]), "t")["t_sd_score"])

    n_genes = 60
    for trial in range(200):
        query = rng.normal(size=n_genes)
        peers = rng.normal(size=(7, n_genes))
        if trial % 3 == 0:
            peers[rng.integers(0, 7), rng.integers(0, n_genes)] = np.nan
        if trial % 5 == 0:
            query[rng.integers(0, n_genes)] = np.nan
        if trial % 7 == 0:
            peers[0, :] = 4.2
        if trial % 11 == 0:
            peers = np.round(peers)
            query = np.round(query)
        mask = rng.random(n_genes) < (0.6 if trial % 2 else 1.0)
        if not mask.any():
            continue

        fast_spearman = spearman_against_peers(query, peers, mask)
        fast_direction = direction_agreement_against_peers(query, peers, mask)
        for row_index in range(peers.shape[0]):
            eval_mask = mask & np.isfinite(query)
            reference_spearman = spearman_scalar(
                query[eval_mask], peers[row_index][eval_mask]
            )
            reference_direction = direction_agreement_scalar(
                query, peers[row_index], mask
            )
            for observed, expected in (
                (fast_spearman[row_index], reference_spearman),
                (fast_direction[row_index], reference_direction),
            ):
                if np.isnan(expected):
                    assert np.isnan(observed), (trial, row_index, observed, expected)
                else:
                    assert np.isclose(observed, expected, atol=1e-12), (
                        trial,
                        row_index,
                        observed,
                        expected,
                    )

        if np.isfinite(query).all():
            prepared = prepare_spearman_rows(peers)
            prepared_scores = spearman_against_prepared_peers(query, prepared)
            assert np.allclose(
                prepared_scores,
                spearman_against_peers(query, peers),
                atol=1e-12,
                equal_nan=True,
            )
            selected_rows = np.asarray([0, 2, 5], dtype=np.int64)
            assert np.allclose(
                spearman_against_prepared_peers(
                    query,
                    prepared,
                    row_indices=selected_rows,
                ),
                spearman_against_peers(query, peers[selected_rows]),
                atol=1e-12,
                equal_nan=True,
            )

    assert spearman_against_peers(np.arange(5.0), np.empty((0, 5))).size == 0
    assert np.all(np.isnan(spearman_against_peers(np.arange(5.0), np.ones((2, 5)))))
    assert select_peer_indices(10, None, "k").tolist() == list(range(10))
    assert select_peer_indices(10, 20, "k").tolist() == list(range(10))
    capped = select_peer_indices(1000, 25, "cigs_mce|CVCL_0062|24|10")
    assert capped.size == 25
    assert np.all(np.diff(capped) > 0)
    assert np.array_equal(
        capped, select_peer_indices(1000, 25, "cigs_mce|CVCL_0062|24|10")
    )
    assert not np.array_equal(capped, select_peer_indices(1000, 25, "other_key"))
    assert not np.array_equal(
        capped,
        select_peer_indices(
            1000,
            25,
            "cigs_mce|CVCL_0062|24|10",
            sampling_seed=20260506,
        ),
    )
    centroid_matrix = np.asarray(
        [[1.0, 2.0, np.nan], [3.0, 4.0, 9.0], [100.0, 200.0, 300.0]]
    )
    eligible_mask = np.asarray([True, True, False])
    exact_centroid = exact_mean_for_row_mask(centroid_matrix, eligible_mask)
    assert np.allclose(exact_centroid, [2.0, 3.0, 9.0], equal_nan=True)
    # Sampling changes only which individual peers are scored, never the centroid.
    for cap in (1, 2, 512):
        selected = select_peer_indices(2, cap, "centroid-test")
        assert selected.size == min(cap, 2)
        assert np.allclose(
            exact_mean_for_row_mask(centroid_matrix, eligible_mask),
            exact_centroid,
            atol=0.0,
            equal_nan=True,
        )
    context_sums, context_counts = finite_column_totals(centroid_matrix)
    exclusion_centroid = exact_mean_excluding_row_mask(
        centroid_matrix,
        ~eligible_mask,
        total_sums=context_sums,
        total_counts=context_counts,
    )
    assert np.allclose(exclusion_centroid, exact_centroid, atol=1e-12, equal_nan=True)
    print("peer_baselines self-tests passed.")


if __name__ == "__main__":
    _self_test()
