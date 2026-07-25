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
from typing import Optional

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "PEER_SUMMARY_FIELDS",
    "summarize_peer_scores",
    "empty_peer_score_summary",
    "spearman_scalar",
    "spearman_against_peers",
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
            with np.errstate(invalid="ignore", divide="ignore"):
                row_scores = (centered_peers @ centered_query) / (peer_norms * query_norm)
            row_scores = np.clip(row_scores, -1.0, 1.0)
            row_scores[constant_rows | ~(peer_norms > 0.0)] = np.nan
            scores[finite_rows] = row_scores

    for row_index in np.flatnonzero(~finite_rows):
        scores[row_index] = spearman_scalar(query_subset, peer_subset[row_index])
    return scores


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
    digest = hashlib.blake2b(str(seed_key).encode("utf-8"), digest_size=8).digest()
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
    print("peer_baselines self-tests passed.")


if __name__ == "__main__":
    _self_test()
