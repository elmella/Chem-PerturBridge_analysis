from __future__ import annotations

import unittest

import numpy as np

from scripts.cross_source_scoring import (
    deg_mask,
    exact_cross_assay_null,
    exact_null_mid_p,
    exact_null_rank_calibration,
    expected_single_signature_metrics,
    observed_deg_metrics,
    similarity_matrix,
    summarize_retrieval_scores,
)


class CrossSourceScoringTests(unittest.TestCase):
    def test_expected_single_signature_metrics_matches_pairwise_reference(self):
        rng = np.random.default_rng(20260727)
        for n_targets, n_peers in ((1, 1), (7, 5), (31, 19)):
            target_scores = rng.normal(size=n_targets)
            peer_scores = rng.normal(size=n_peers)
            if n_targets >= 7:
                target_scores[:4] = [0.5, 0.5 + 5e-13, 0.5 + 2e-12, np.nan]
                peer_scores[:3] = [0.5, 0.5 + 1e-12, np.nan]

            finite_targets = target_scores[np.isfinite(target_scores)]
            finite_peers = peer_scores[np.isfinite(peer_scores)]
            ranks = []
            normalized_ranks = []
            recalls = []
            aurocs = []
            for peer_score in finite_peers:
                n_better = int(np.sum(finite_targets > peer_score))
                rank = 1.0 + n_better
                ranks.append(rank)
                normalized_ranks.append(
                    1.0 - n_better / len(finite_targets)
                )
                recalls.append(float(rank == 1.0))
                wins = float(np.sum(peer_score > finite_targets))
                wins += 0.5 * float(
                    np.sum(
                        np.isclose(
                            peer_score,
                            finite_targets,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                )
                aurocs.append(wins / len(finite_targets))

            observed = expected_single_signature_metrics(
                target_scores,
                peer_scores,
            )
            self.assertEqual(observed["n_peers"], len(finite_peers))
            self.assertAlmostEqual(observed["best_rank"], np.mean(ranks))
            self.assertAlmostEqual(
                observed["normalized_rank"],
                np.mean(normalized_ranks),
            )
            self.assertAlmostEqual(
                observed["recall_at_1"],
                np.mean(recalls),
            )
            self.assertAlmostEqual(observed["auroc"], np.mean(aurocs))

    def test_deg_masks_and_sample_referenced_metrics(self):
        genes = np.asarray(["a", "b", "c", "d"], dtype=object)
        left = np.asarray([2.0, -1.0, 0.1, 4.0])
        right = np.asarray([3.0, -2.0, -0.1, 1.0])
        left_p = np.asarray([0.01, 0.02, 0.01, 0.2])
        right_p = np.asarray([0.01, 0.2, 0.01, 0.01])
        self.assertEqual(
            deg_mask(left, left_p, "p05").tolist(),
            [True, True, True, False],
        )
        self.assertEqual(
            deg_mask(left, left_p, "p05_lfc02").tolist(),
            [True, True, False, False],
        )
        metrics = observed_deg_metrics(
            gene_keys=genes,
            logfc_left=left,
            logfc_right=right,
            adj_p_left=left_p,
            adj_p_right=right_p,
            definition_key="p05",
        )
        self.assertEqual(metrics["n_deg_left_p05"], 3)
        self.assertEqual(metrics["n_deg_right_p05"], 3)
        self.assertAlmostEqual(
            metrics["observed_direction_agreement_p05"],
            0.5,
        )

    def test_similarity_backends_have_expected_geometry(self):
        query = np.asarray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        target = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        cosine, valid_q, valid_t = similarity_matrix(
            query,
            target,
            "cosine",
        )
        self.assertEqual(valid_q.tolist(), [True, True])
        self.assertEqual(valid_t.tolist(), [True, True])
        self.assertAlmostEqual(cosine[0, 0], 1.0)
        self.assertAlmostEqual(cosine[0, 1], 10.0 / 14.0)
        spearman, spearman_valid_q, _ = similarity_matrix(
            query,
            target,
            "spearman",
        )
        self.assertEqual(spearman_valid_q.tolist(), [True, False])
        self.assertAlmostEqual(spearman[0, 0], 1.0)
        self.assertAlmostEqual(spearman[0, 1], -1.0)

    def test_retrieval_null_and_observed_rank(self):
        scores = np.asarray([0.1, 0.9, 0.8, 0.2])
        positives = np.asarray([False, True, True, False])
        observed = summarize_retrieval_scores(scores, positives)
        self.assertEqual(observed["best_rank"], 1.0)
        self.assertEqual(observed["recall_at_1"], 1.0)
        null = exact_cross_assay_null(n_candidates=4, n_positives=2)
        self.assertAlmostEqual(null["recall_at_1"], 0.5)
        self.assertAlmostEqual(null["auroc"], 0.5)
        mid_p = exact_null_mid_p(
            n_candidates=4,
            n_positives=2,
            best_rank=1,
        )
        self.assertAlmostEqual(mid_p, 0.25)
        calibration = exact_null_rank_calibration(
            n_candidates=4,
            n_positives=2,
            best_rank=1,
        )
        self.assertAlmostEqual(calibration["null_pit"], 0.25)
        self.assertAlmostEqual(calibration["null_p_value"], 0.5)
        peer_baseline = expected_single_signature_metrics(
            np.asarray([0.9, 0.4, 0.1]),
            np.asarray([0.8, 0.2]),
        )
        self.assertEqual(peer_baseline["n_peers"], 2)
        self.assertAlmostEqual(peer_baseline["best_rank"], 2.5)
        self.assertAlmostEqual(peer_baseline["recall_at_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
