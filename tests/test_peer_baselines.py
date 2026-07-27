import unittest

import numpy as np

from scripts.peer_baselines import (
    different_compound_peer_mask,
    exact_mean_excluding_row_mask,
    exact_mean_for_row_mask,
    finite_column_totals,
    prepare_spearman_rows,
    select_peer_indices,
    spearman_against_peers,
    spearman_against_prepared_peers,
)


class PeerBaselineTests(unittest.TestCase):
    def test_retrieval_peer_mask_stays_within_the_supplied_candidate_pool(self):
        retrieval_mask = different_compound_peer_mask(
            np.asarray(["0.05", "0.05"]),
            np.asarray(["A", "B"]),
            dose_key="0.05",
            excluded_compound="A",
        )
        full_source_mask = different_compound_peer_mask(
            np.asarray(["0.05", "0.05", "0.05", "0.05", "0.05"]),
            np.asarray(["A", "B", "C", "D", "A"]),
            dose_key="0.05",
            excluded_compound="A",
        )

        np.testing.assert_array_equal(retrieval_mask, [False, True])
        self.assertEqual(int(retrieval_mask.sum()), 1)
        self.assertEqual(int(full_source_mask.sum()), 3)

    def test_prepared_spearman_matches_reference_with_ties_constants_and_nans(self):
        rng = np.random.default_rng(20260505)
        peers = np.round(rng.normal(size=(12, 80)), decimals=1)
        peers[2] = 4.0
        peers[7, 11] = np.nan
        query = np.round(rng.normal(size=80), decimals=1)

        prepared = prepare_spearman_rows(peers)
        observed = spearman_against_prepared_peers(query, prepared)
        expected = spearman_against_peers(query, peers)
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)

    def test_prepared_spearman_selected_rows_match_reference(self):
        rng = np.random.default_rng(11)
        peers = rng.normal(size=(30, 120))
        query = rng.normal(size=120)
        selected = np.asarray([0, 3, 8, 21, 29], dtype=np.int64)

        observed = spearman_against_prepared_peers(
            query,
            prepare_spearman_rows(peers),
            row_indices=selected,
        )
        expected = spearman_against_peers(query, peers[selected])
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)

    def test_sampling_is_deterministic_and_seeded(self):
        first = select_peer_indices(5000, 512, "dataset|line|compound|24|10", 7)
        second = select_peer_indices(5000, 512, "dataset|line|compound|24|10", 7)
        changed = select_peer_indices(5000, 512, "dataset|line|compound|24|10", 8)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))

    def test_peer_cap_never_changes_exact_centroid(self):
        matrix = np.asarray(
            [
                [1.0, 2.0, np.nan],
                [3.0, 4.0, 9.0],
                [5.0, np.nan, 15.0],
                [100.0, 200.0, 300.0],
            ]
        )
        eligible = np.asarray([True, True, True, False])
        expected = np.asarray([3.0, 3.0, 12.0])
        total_sums, total_counts = finite_column_totals(matrix)

        for cap in (1, 2, 3, 512):
            select_peer_indices(int(eligible.sum()), cap, "condition", 20260505)
            observed = exact_mean_for_row_mask(matrix, eligible)
            np.testing.assert_allclose(observed, expected, atol=0.0)
            observed_from_totals = exact_mean_excluding_row_mask(
                matrix,
                ~eligible,
                total_sums=total_sums,
                total_counts=total_counts,
            )
            np.testing.assert_allclose(observed_from_totals, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
