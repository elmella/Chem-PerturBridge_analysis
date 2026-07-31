from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.precompute_replicate_signature_similarity as replicate_scoring
from scripts.population_zscore import DATASET_SCOPE, PopulationGeneStats
from scripts.precompute_replicate_signature_similarity import (
    cosine_against_peers,
    complete_row_norms,
    normalized_matrix_for_stats,
    overlay_condition_metric_rows,
    resolve_deg_definitions,
    resolve_processed_sep_rep_h5ad,
    resolve_normalization_scopes,
    task_source_paths_to_open,
    vector_cosine_similarity,
)


class ReplicateNormalizedCosineTests(unittest.TestCase):
    def test_condition_scoring_uses_cached_centroid_and_selected_peers(self) -> None:
        logfc = np.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.8, 0.2, 2.2],
                [-1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
        t_stat = logfc * 3.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.h5ad"
            source.touch()
            rows = pd.DataFrame(
                {
                    "dataset_name": ["dataset_a"] * 4,
                    "cell_type": ["line_a"] * 4,
                    "pubchem_cid": ["1", "1", "2", "3"],
                    "time_key": [24.0] * 4,
                    "dose_key": [10.0] * 4,
                    "source_path": [str(source)] * 4,
                    "source_row_pos": np.arange(4),
                    "condition_key": [
                        "line_a|1|24|10",
                        "line_a|1|24|10",
                        "line_a|2|24|10",
                        "line_a|3|24|10",
                    ],
                    "perturbagen_display": ["one", "one", "two", "three"],
                }
            )
            rows = replicate_scoring.normalize_source_metadata_frame(rows)

            def load_block(block, *, gene_keys, open_adatas):
                positions = block["source_row_pos"].astype(int).to_numpy()
                return (
                    [logfc[position].copy() for position in positions],
                    [t_stat[position].copy() for position in positions],
                )

            stats_record = PopulationGeneStats(
                dataset_name="dataset_a",
                cell_type="__all_cell_types__",
                gene_keys=np.asarray(["g1", "g2", "g3"]),
                finite_counts=np.asarray([4, 4, 4]),
                means=np.asarray([0.0, 0.0, 0.0]),
                population_sds=np.asarray([1.0, 1.0, 1.0]),
                valid_mask=np.asarray([True, True, True]),
                population_row_count=4,
                fingerprint="test",
                cache_path=root / "stats.npz",
                scope=DATASET_SCOPE,
            )

            class StatsCache:
                def get(self, **kwargs):
                    return stats_record

            replicate_scoring.WORKER_CONTEXT_AGGREGATE_CACHE.clear()
            with patch.object(
                replicate_scoring,
                "load_vectors_for_rows",
                side_effect=load_block,
            ), patch.object(
                replicate_scoring,
                "shared_gene_keys_for_paths",
                return_value=np.asarray(["g1", "g2", "g3"]),
            ), patch.object(
                replicate_scoring,
                "MAX_BASELINE_PEERS",
                1,
            ):
                record = replicate_scoring.compute_condition_metric_record_from_rows(
                    rows.iloc[0],
                    rows.iloc[:2].copy(),
                    output_dir=root,
                    line_global_shared_gene_keys={
                        "line_a": np.asarray(["g1", "g2", "g3"])
                    },
                    top_k=2,
                    compute_baseline_metrics=True,
                    compute_normalized_cosine=True,
                    normalization_scopes=(DATASET_SCOPE,),
                    population_stats_cache=StatsCache(),
                    baseline_source_frame=rows.copy(),
                    baseline_context_row_indexes={
                        ("line_a", "24", "10"): np.arange(4)
                    },
                    open_adatas={},
                )

            self.assertIsNotNone(record)
            self.assertEqual(record["n_baseline_peer_rows"], 2)
            self.assertEqual(record["n_peer_rows_scored"], 1)
            self.assertTrue(
                np.isfinite(record["mean_replicate_baseline_spearman_logfc"])
            )
            self.assertTrue(
                np.isfinite(record["mean_replicate_cosine_logfc_raw"])
            )
            self.assertTrue(
                np.isfinite(
                    record["mean_replicate_cosine_logfc_normalized_dataset"]
                )
            )

    def test_context_aggregate_is_exact_and_reused_from_disk(self) -> None:
        logfc = np.asarray(
            [
                [1.0, 2.0, np.nan],
                [3.0, 4.0, 6.0],
                [5.0, np.nan, 8.0],
                [7.0, 10.0, 12.0],
            ],
            dtype=np.float64,
        )
        t_stat = logfc * 2.0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.h5ad"
            source.touch()
            rows = pd.DataFrame(
                {
                    "dataset_name": ["dataset_a"] * 4,
                    "cell_type": ["line_a"] * 4,
                    "pubchem_cid": ["1", "1", "2", "3"],
                    "time_key": [24.0] * 4,
                    "dose_key": [10.0] * 4,
                    "source_path": [str(source)] * 4,
                    "source_row_pos": np.arange(4),
                    "condition_key": [
                        "line_a|1|24|10",
                        "line_a|1|24|10",
                        "line_a|2|24|10",
                        "line_a|3|24|10",
                    ],
                }
            )

            def load_block(block, *, gene_keys, open_adatas):
                positions = block["source_row_pos"].astype(int).to_numpy()
                return (
                    [logfc[position].copy() for position in positions],
                    [t_stat[position].copy() for position in positions],
                )

            with patch.object(
                replicate_scoring,
                "load_vectors_for_rows",
                side_effect=load_block,
            ) as loader:
                aggregate = replicate_scoring.get_or_build_context_aggregate(
                    output_dir=root,
                    dataset_name="dataset_a",
                    context_rows=rows,
                    context_key=("line_a", "24", "10"),
                    gene_keys=np.asarray(["g1", "g2", "g3"]),
                    open_adatas={},
                    rows_per_batch=2,
                )
                self.assertEqual(loader.call_count, 2)

            observed = replicate_scoring.aggregate_mean_excluding_rows(
                sums=aggregate.logfc_sums,
                counts=aggregate.logfc_counts,
                excluded_rows=logfc[:2],
            )
            np.testing.assert_allclose(
                observed,
                np.nanmean(logfc[2:], axis=0),
            )

            replicate_scoring.WORKER_CONTEXT_AGGREGATE_CACHE.clear()
            with patch.object(
                replicate_scoring,
                "load_vectors_for_rows",
                side_effect=AssertionError("disk cache should be reused"),
            ):
                reused = replicate_scoring.get_or_build_context_aggregate(
                    output_dir=root,
                    dataset_name="dataset_a",
                    context_rows=rows,
                    context_key=("line_a", "24", "10"),
                    gene_keys=np.asarray(["g1", "g2", "g3"]),
                    open_adatas={},
                    rows_per_batch=2,
                )
            np.testing.assert_allclose(reused.logfc_sums, aggregate.logfc_sums)

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

    def test_context_keys_survive_mixed_numeric_tsv_inference(self) -> None:
        task_context = replicate_scoring.normalize_source_metadata_frame(
            pd.DataFrame(
                {
                    "cell_type": ["CVCL_0062"],
                    "pubchem_cid": [123],
                    "time_key": [24],
                    "dose_key": [10],
                    "condition_key": ["CVCL_0062|123|24|10"],
                }
            )
        )
        cached_rows = replicate_scoring.normalize_source_metadata_frame(
            pd.DataFrame(
                {
                    "cell_type": ["CVCL_0062", "CVCL_0062"],
                    "pubchem_cid": [123.0, 456.0],
                    "time_key": [24.0, 24.0],
                    "dose_key": [10.0, 20.0],
                    "condition_key": [
                        "CVCL_0062|123.0|24.0|10.0",
                        "CVCL_0062|456.0|24.0|20.0",
                    ],
                }
            )
        )

        self.assertEqual(task_context.loc[0, "dose_key"], "10")
        self.assertEqual(cached_rows.loc[0, "dose_key"], "10")
        self.assertEqual(
            cached_rows.loc[0, "condition_key"],
            "CVCL_0062|123|24|10",
        )
        merged = cached_rows.merge(
            task_context[["cell_type", "time_key", "dose_key"]],
            on=["cell_type", "time_key", "dose_key"],
            how="inner",
        )
        self.assertEqual(len(merged), 1)

    def test_deg_definition_selection_can_limit_reviewer_workload(self) -> None:
        self.assertEqual(resolve_deg_definitions("p05"), ("p05",))
        self.assertEqual(resolve_deg_definitions("p05_lfc02"), ("p05_lfc02",))
        self.assertEqual(
            resolve_deg_definitions("all"),
            tuple(replicate_scoring.DEG_DEFINITION_CONFIG),
        )

    def test_nonretrieval_task_opens_only_context_sources(self) -> None:
        replicates = pd.DataFrame({"source_path": ["query.h5ad"]})
        baseline_context = pd.DataFrame(
            {"source_path": ["query.h5ad", "context_peer.h5ad"]}
        )
        full_dataset = pd.DataFrame(
            {
                "source_path": [
                    "query.h5ad",
                    "context_peer.h5ad",
                    "unrelated_line.h5ad",
                ]
            }
        )

        baseline_paths = task_source_paths_to_open(
            replicates_frame=replicates,
            baseline_source_frame=baseline_context,
            full_dataset_source_frame=full_dataset,
            compute_retrieval_metrics=False,
        )
        retrieval_paths = task_source_paths_to_open(
            replicates_frame=replicates,
            baseline_source_frame=baseline_context,
            full_dataset_source_frame=full_dataset,
            compute_retrieval_metrics=True,
        )

        self.assertEqual(
            baseline_paths,
            {"query.h5ad"},
        )
        self.assertEqual(
            retrieval_paths,
            {"query.h5ad", "context_peer.h5ad", "unrelated_line.h5ad"},
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
