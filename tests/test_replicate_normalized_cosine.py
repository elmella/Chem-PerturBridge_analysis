from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.population_zscore import DATASET_SCOPE, PopulationGeneStats
from scripts.precompute_replicate_signature_similarity import (
    cosine_against_peers,
    normalized_matrix_for_stats,
    overlay_condition_metric_rows,
    resolve_normalization_scopes,
    vector_cosine_similarity,
)


class ReplicateNormalizedCosineTests(unittest.TestCase):
    def test_normalization_uses_valid_overlapping_genes_in_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stats = PopulationGeneStats(
                dataset_name="dataset_a",
                cell_type="__all_cell_types__",
                gene_keys=np.asarray(["g3", "g1", "g2", "unused"]),
                finite_counts=np.asarray([4, 4, 1, 4]),
                means=np.asarray([10.0, 1.0, 2.0, 0.0]),
                population_sds=np.asarray([2.0, 2.0, np.nan, 1.0]),
                valid_mask=np.asarray([True, True, False, True]),
                population_row_count=4,
                fingerprint="test",
                cache_path=Path(temporary) / "stats.npz",
                scope=DATASET_SCOPE,
            )
            observed = normalized_matrix_for_stats(
                np.asarray([[3.0, 12.0, 99.0]]),
                gene_keys=np.asarray(["g1", "g3", "missing"]),
                stats_record=stats,
            )
        np.testing.assert_allclose(observed, np.asarray([[1.0, 1.0]]))

    def test_cosine_is_pairwise_finite_and_vectorized_peer_wrapper_matches(self) -> None:
        query = np.asarray([1.0, 0.0, np.nan])
        peers = np.asarray(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, 5.0],
            ]
        )
        np.testing.assert_allclose(
            cosine_against_peers(query, peers),
            np.asarray([1.0, 0.0]),
        )
        self.assertAlmostEqual(
            vector_cosine_similarity(np.asarray([1.0, 1.0]), np.asarray([1.0, -1.0])),
            0.0,
        )

    def test_scope_selection_is_explicit(self) -> None:
        self.assertEqual(resolve_normalization_scopes("dataset"), ("dataset",))
        self.assertEqual(
            resolve_normalization_scopes("dataset-cell-type"),
            ("dataset_cell_type",),
        )
        self.assertEqual(
            set(resolve_normalization_scopes("all")),
            {"dataset", "dataset_cell_type"},
        )

    def test_overlay_preserves_old_metrics_and_prefers_new_nonmissing_values(self) -> None:
        existing = pd.DataFrame(
            {
                "dataset_name": ["dataset_a"],
                "condition_key": ["condition_1"],
                "old_metric": [0.4],
                "recomputed_metric": [0.2],
            }
        )
        current = pd.DataFrame(
            {
                "dataset_name": ["dataset_a"],
                "condition_key": ["condition_1"],
                "recomputed_metric": [0.8],
                "normalized_cosine": [0.9],
            }
        )
        merged = overlay_condition_metric_rows(current, existing)
        self.assertEqual(merged.loc[0, "old_metric"], 0.4)
        self.assertEqual(merged.loc[0, "recomputed_metric"], 0.8)
        self.assertEqual(merged.loc[0, "normalized_cosine"], 0.9)


if __name__ == "__main__":
    unittest.main()
