from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import scripts.precompute_replicate_signature_similarity as replicate_scoring
from scripts.population_zscore import DATASET_SCOPE, PopulationGeneStats
from scripts.precompute_replicate_signature_similarity import (
    cosine_against_peers,
    complete_row_norms,
    normalized_matrix_for_stats,
    overlay_condition_metric_rows,
    resolve_processed_sep_rep_h5ad,
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

    def test_cached_peer_norms_preserve_cosine_scores(self) -> None:
        rng = np.random.default_rng(17)
        query = rng.normal(size=20)
        peers = rng.normal(size=(12, 20))
        expected = cosine_against_peers(query, peers)
        observed = cosine_against_peers(
            query,
            peers,
            peer_norms=complete_row_norms(peers),
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-14)

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

    def test_grouped_processed_metadata_is_a_candidate_inventory_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grouped = (
                root
                / "dataset_a"
                / "pseudobulk_processed"
                / "group_rep"
                / "processed.h5ad"
            )
            grouped.parent.mkdir(parents=True)
            grouped.touch()
            self.assertEqual(
                resolve_processed_sep_rep_h5ad("dataset_a", data_root=root),
                grouped,
            )

    def test_published_root_processed_file_is_a_candidate_inventory_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            published = root / "dataset_a" / "processed.h5ad"
            published.parent.mkdir(parents=True)
            published.touch()
            self.assertEqual(
                resolve_processed_sep_rep_h5ad("dataset_a", data_root=root),
                published,
            )

    def test_extracted_sep_rep_archive_layout_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = (
                root
                / "dataset_a"
                / "sep_rep_extracted"
                / "deg_data"
                / "sep_rep"
                / "full"
                / "qc_false"
                / "filter_min_cells_0"
                / "results"
            )
            expected.mkdir(parents=True)
            (expected / "line_de.h5ad").touch()
            original_root = replicate_scoring.SOURCE_DATA_ROOT
            try:
                replicate_scoring.SOURCE_DATA_ROOT = root
                self.assertEqual(
                    replicate_scoring.sep_rep_dataset_dir("dataset_a", 0),
                    expected,
                )
            finally:
                replicate_scoring.SOURCE_DATA_ROOT = original_root


if __name__ == "__main__":
    unittest.main()
