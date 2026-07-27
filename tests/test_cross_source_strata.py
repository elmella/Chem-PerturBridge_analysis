import unittest

import numpy as np

from scripts.cross_source_strata import SignatureStratum
from scripts.peer_baselines import (
    select_peer_indices,
    spearman_against_peers,
    spearman_against_prepared_peers,
)


class SignatureStratumTests(unittest.TestCase):
    def setUp(self):
        self.compounds = np.asarray(["A", "B", "A", "C", "B"])
        self.values = np.asarray(
            [
                [1.0, np.nan, 4.0, 5.0],
                [2.0, 20.0, np.nan, 5.0],
                [3.0, 30.0, 8.0, 5.0],
                [4.0, np.nan, 10.0, 5.0],
                [np.nan, 50.0, 12.0, 5.0],
            ]
        )
        self.stratum = SignatureStratum(
            self.compounds,
            self.values,
            gene_keys=np.asarray(["g1", "g2", "g3", "g4"]),
            max_prepared_cache_bytes=1_000_000,
        )

    def test_finite_totals_and_duplicate_compound_centroid_match_numpy(self):
        np.testing.assert_allclose(
            self.stratum.finite_sums,
            np.nansum(self.values, axis=0),
            atol=0.0,
        )
        np.testing.assert_array_equal(
            self.stratum.finite_counts,
            np.isfinite(self.values).sum(axis=0),
        )

        with np.errstate(invalid="ignore"):
            expected = np.nanmean(self.values[self.compounds != "A"], axis=0)
        observed = self.stratum.different_compound_centroid("A")
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(
            self.stratum.compound_row_indices("A"),
            np.asarray([0, 2]),
        )

    def test_centroid_preserves_all_missing_gene_and_empty_population_semantics(self):
        values = np.asarray(
            [
                [1.0, np.nan, 4.0],
                [2.0, np.nan, np.nan],
                [3.0, np.nan, 8.0],
            ]
        )
        stratum = SignatureStratum(
            np.asarray(["A", "B", "B"]),
            values,
        )
        observed = stratum.different_compound_centroid("A")
        np.testing.assert_allclose(
            observed,
            np.asarray([2.5, np.nan, 8.0]),
            equal_nan=True,
        )
        only_one_compound = SignatureStratum(
            np.asarray(["B", "B", "B"]),
            values,
        )
        self.assertIsNone(only_one_compound.different_compound_centroid("B"))

    def test_centroid_can_match_legacy_nan_propagating_mean(self):
        retained = self.compounds != "A"
        expected = np.mean(self.values[retained], axis=0)
        observed = self.stratum.different_compound_centroid(
            "A",
            require_all_finite=True,
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)

    def test_peer_selection_returns_absolute_indices_and_honest_counts(self):
        seed_key = "dataset|line|time|dose|A"
        selection = self.stratum.select_different_compound_peers(
            "A",
            max_peers=2,
            seed_key=seed_key,
            sampling_seed=17,
        )
        eligible = np.flatnonzero(self.compounds != "A")
        expected_offsets = select_peer_indices(
            len(eligible),
            2,
            seed_key,
            sampling_seed=17,
        )
        self.assertEqual(selection.total_count, 3)
        self.assertEqual(selection.selected_count, 2)
        np.testing.assert_array_equal(
            selection.row_indices,
            eligible[expected_offsets],
        )
        self.assertTrue(np.all(self.compounds[selection.row_indices] != "A"))

        uncapped = self.stratum.select_different_compound_peers(
            "A",
            max_peers=None,
            seed_key=seed_key,
        )
        self.assertEqual(uncapped.total_count, 3)
        self.assertEqual(uncapped.selected_count, 3)
        np.testing.assert_array_equal(uncapped.row_indices, eligible)

    def test_prepared_spearman_matches_direct_scoring_for_selected_genes_and_rows(self):
        rng = np.random.default_rng(20260727)
        values = np.round(rng.normal(size=(18, 40)), decimals=1)
        values[4, 7] = np.nan
        values[9, :] = 3.0
        compounds = np.asarray([f"C{i // 2}" for i in range(len(values))])
        positions = np.asarray([0, 3, 4, 7, 11, 18, 29, 35], dtype=np.int64)
        stratum = SignatureStratum(compounds, values)
        peers = stratum.select_different_compound_peers(
            "C0",
            max_peers=7,
            seed_key="spearman-parity",
        )
        query = np.round(rng.normal(size=positions.size), decimals=1)

        prepared = stratum.prepared_spearman(positions)
        observed = spearman_against_prepared_peers(
            query,
            prepared,
            row_indices=peers.row_indices,
        )
        expected = spearman_against_peers(
            query,
            values[peers.row_indices][:, positions],
        )
        np.testing.assert_allclose(observed, expected, atol=1e-12, equal_nan=True)

    def test_prepared_cache_is_position_order_sensitive_and_memory_bounded(self):
        first = self.stratum.prepared_spearman(np.asarray([0, 2, 3]))
        again = self.stratum.prepared_spearman(np.asarray([0, 2, 3]))
        reordered = self.stratum.prepared_spearman(np.asarray([3, 2, 0]))
        self.assertIs(first, again)
        self.assertIsNot(first, reordered)
        self.assertGreaterEqual(self.stratum.prepared_cache_entry_count, 2)
        self.assertLessEqual(self.stratum.prepared_cache_bytes, 1_000_000)

        no_cache = SignatureStratum(
            self.compounds,
            self.values,
            max_prepared_cache_bytes=1,
        )
        no_cache.prepared_spearman(np.asarray([0, 2, 3]))
        self.assertEqual(no_cache.prepared_cache_entry_count, 0)
        self.assertEqual(no_cache.prepared_cache_bytes, 0)

    def test_constructor_and_position_validation_fail_early(self):
        with self.assertRaisesRegex(ValueError, "already be unique"):
            SignatureStratum(
                self.compounds,
                self.values,
                gene_keys=np.asarray(["g1", "g1", "g3", "g4"]),
            )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            self.stratum.prepared_spearman(np.asarray([0, 0]))
        with self.assertRaisesRegex(IndexError, "out-of-range"):
            self.stratum.prepared_spearman(np.asarray([0, 4]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.stratum.prepared_spearman(np.asarray([True, False]))


if __name__ == "__main__":
    unittest.main()
