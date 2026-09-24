from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import scripts.replicate_retrieval as rr
from scripts.cross_source_scoring import (
    auroc_from_scores,
    normalized_best_positive_rank,
    recall_at_1,
)


def synthetic_stratum(seed: int = 3, n_genes: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    """Three doses, several compounds, two or three replicates each.

    Replicates share a condition signal plus noise, so retrieval is neither
    trivial nor hopeless, and continuous values keep scores free of ties.
    """
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for dose in ("0.1", "1", "10"):
        for compound in range(7):
            signal = rng.normal(size=n_genes)
            for replicate in range(2 + (compound % 2)):
                rows.append(
                    {
                        "dataset_name": "toy",
                        "cell_type": "CVCL_TEST",
                        "time_key": "24",
                        "pubchem_cid": f"cid{compound}",
                        "dose_key": dose,
                        "condition_key": f"CVCL_TEST|cid{compound}|24|{dose}",
                    }
                )
                vectors.append(signal + rng.normal(scale=1.5, size=n_genes))
    return pd.DataFrame(rows), np.vstack(vectors)


def naive_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    if metric == "l2":
        return -float(np.linalg.norm(a - b))
    return float(spearmanr(a, b).correlation)


def naive_records(rows: pd.DataFrame, values: np.ndarray, metric: str) -> list[dict]:
    """One query at a time, straight from the definitions."""
    condition = rows["condition_key"].to_numpy()
    compound = rows["pubchem_cid"].to_numpy()
    dose = rows["dose_key"].to_numpy()
    out = []
    for query in range(len(rows)):
        others = [j for j in range(len(rows)) if j != query]
        scores = np.array([naive_similarity(values[query], values[j], metric) for j in others])
        positive = np.array([condition[j] == condition[query] for j in others])
        _, observed = normalized_best_positive_rank(scores, positive)

        peers = [j for j in range(len(rows)) if dose[j] == dose[query] and compound[j] != compound[query]]
        centroid = values[peers].mean(axis=0)
        augmented = np.append(scores, naive_similarity(values[query], centroid, metric))
        centroid_mask = np.zeros(augmented.size, dtype=bool)
        centroid_mask[-1] = True
        _, centroid_rank = normalized_best_positive_rank(augmented, centroid_mask)

        out.append(
            {
                "row": query,
                "observed_normalized_rank": observed,
                "recall_at_1": recall_at_1(scores, positive),
                "auroc": auroc_from_scores(scores, positive),
                "centroid_normalized_rank": centroid_rank,
                "n_positives": int(positive.sum()),
                "n_candidates": len(scores),
            }
        )
    return out


class MetricDefinitionTests(unittest.TestCase):
    def test_metrics_match_cross_source_definitions_including_ties(self) -> None:
        rng = np.random.default_rng(0)
        for trial in range(200):
            n = int(rng.integers(3, 40))
            # Coarse values force exact ties, which rank "min" and AUROC
            # half-credit must both handle as cross-source does.
            scores = rng.integers(0, 6, size=n).astype(np.float64) / 5.0
            positive = rng.random(n) < 0.3
            if not positive.any() or positive.all():
                continue
            best, expected = normalized_best_positive_rank(scores, positive)
            self.assertEqual(rr.best_positive_rank(scores, positive), best)
            self.assertEqual(rr.normalized_rank(best, n), expected)
            self.assertAlmostEqual(rr.auroc(scores, positive), auroc_from_scores(scores, positive), places=12)

    def test_null_expectation_matches_simulation(self) -> None:
        rng = np.random.default_rng(1)
        for n_candidates, n_positives in ((10, 1), (50, 2), (200, 2)):
            draws = []
            for _ in range(20000):
                ranks = rng.permutation(n_candidates)[:n_positives] + 1
                draws.append(rr.normalized_rank(float(ranks.min()), n_candidates))
            self.assertAlmostEqual(
                rr.null_expected_normalized_rank(n_candidates, n_positives),
                float(np.mean(draws)),
                delta=0.005,
            )


class StratumScoringTests(unittest.TestCase):
    def test_vectorized_scoring_matches_naive_reference(self) -> None:
        # Spearman moves in steps of 6 / (n (n^2 - 1)), so with few genes two
        # candidates can tie exactly, and which one ranks first then hinges on
        # 1e-16 of rounding in any float implementation, the cross-source one
        # included. 400 genes puts the step near 1e-7, clear of ties; tie
        # handling itself is covered at the metric level above.
        for metric, n_genes in (("cosine", 40), ("spearman", 400), ("l2", 40)):
            rows, values = synthetic_stratum(n_genes=n_genes)
            # A small block forces the blocking path to be exercised.
            fast = {r["row"]: r for r in rr.score_stratum(rows, values, metric=metric, query_block=5)}
            slow = naive_records(rows, values, metric)
            self.assertEqual(len(fast), len(slow))
            for reference in slow:
                got = fast[reference["row"]]
                for key in (
                    "observed_normalized_rank", "recall_at_1", "auroc",
                    "centroid_normalized_rank",
                ):
                    self.assertAlmostEqual(got[key], reference[key], places=10,
                                           msg=f"{metric} row {reference['row']} {key}")
                self.assertEqual(got["n_positives"], reference["n_positives"])
                self.assertEqual(got["n_candidates"], reference["n_candidates"])

    def test_query_never_retrieves_itself(self) -> None:
        # Pure noise: no condition signal at all. If a query's own sample
        # were among its candidates it would sit at rank 1 with similarity 1,
        # which is exactly the bug in the scorer's original retrieval path.
        rng = np.random.default_rng(7)
        rows, _ = synthetic_stratum()
        noise = rng.normal(size=(len(rows), 40))
        # Under L2 the query's own sample would sit at distance 0, rank 1.
        for metric in ("cosine", "l2"):
            records = rr.score_stratum(rows, noise, metric=metric, query_block=64)
            observed = np.mean([r["observed_normalized_rank"] for r in records])
            null = np.mean([r["null_expected_normalized_rank"] for r in records])
            self.assertLess(observed, 0.85, metric)
            self.assertAlmostEqual(observed, null, delta=0.08, msg=metric)
            self.assertLess(np.mean([r["recall_at_1"] for r in records]), 0.3, metric)

    def test_negative_l2_matches_table9_cdist_and_its_rankings(self) -> None:
        import precompute_replicate_signature_similarity as scorer

        rng = np.random.default_rng(9)
        # Large-magnitude, near-duplicate rows stress cancellation in the
        # expansion ||q||^2 + ||c||^2 - 2 q.c.
        base = rng.normal(scale=5.0, size=(1, 300))
        candidates = np.vstack([base + rng.normal(scale=s, size=(20, 300)) for s in (0.01, 0.5, 3.0)])
        queries = candidates[:15]
        fast = rr.negative_l2_scores(
            queries, candidates, (queries**2).sum(axis=1), (candidates**2).sum(axis=1)
        )
        exact = scorer.negative_l2_similarity_matrix(queries, candidates)
        # A query's distance to itself (0) is where the expansion cancels
        # worst; it is never scored, since the query is not its own candidate.
        off_diagonal = ~np.eye(15, candidates.shape[0], dtype=bool)
        np.testing.assert_allclose(fast[off_diagonal], exact[off_diagonal], rtol=0, atol=1e-9)
        for row in range(15):
            keep = off_diagonal[row]
            np.testing.assert_array_equal(np.argsort(-fast[row][keep]), np.argsort(-exact[row][keep]))

    def test_condition_level_averages_each_conditions_queries(self) -> None:
        rows, values = synthetic_stratum()
        records = rr.score_stratum(rows, values, metric="cosine", query_block=16)
        frame = rr.condition_level(rows, records)
        self.assertEqual(len(frame), rows["condition_key"].nunique())
        per_query = pd.DataFrame(records)
        per_query["condition_key"] = rows["condition_key"].to_numpy()[per_query["row"]]
        expected = per_query.groupby("condition_key")["auroc"].mean()
        merged = frame.set_index("condition_key")["auroc"]
        pd.testing.assert_series_equal(merged.sort_index(), expected.sort_index(), check_names=False)


if __name__ == "__main__":
    unittest.main()
