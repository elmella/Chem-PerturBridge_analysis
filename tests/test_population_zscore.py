import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.peer_baselines import summarize_peer_scores
from scripts.population_zscore import (
    load_or_fit_population_stats,
    standardize_matrix,
    standardize_vector,
)


def write_fixture(
    path: Path,
    matrix: np.ndarray,
    *,
    cell_type: str,
    symbols=None,
    obs_overrides=None,
) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    n_rows, n_genes = matrix.shape
    obs = pd.DataFrame(
        {
            "is_control": ["FALSE"] * n_rows,
            "pubchem_cid": [str(1000 + index) for index in range(n_rows)],
            "cell_type": [cell_type] * n_rows,
            "pert_time_h": [24.0] * n_rows,
            "pert_dose_uM": [10.0] * n_rows,
        },
        index=[f"row_{index}" for index in range(n_rows)],
    )
    for column, values in (obs_overrides or {}).items():
        obs[column] = values
    if symbols is None:
        symbols = [f"G{index}" for index in range(n_genes)]
    var = pd.DataFrame(
        {"symbol": symbols},
        index=[f"ENSG{index:05d}" for index in range(n_genes)],
    )
    adata = ad.AnnData(
        X=np.zeros_like(matrix),
        obs=obs,
        var=var,
        layers={"logFC": matrix},
    )
    adata.write_h5ad(path)


class PopulationZScoreTests(unittest.TestCase):
    def test_streaming_matches_dense_population_statistics_and_cache_reload(self):
        matrix = np.asarray(
            [
                [1.0, 100.0, 5.0, np.nan, 8.0],
                [2.0, 200.0, 5.0, 4.0, np.nan],
                [3.0, 300.0, 5.0, 6.0, np.nan],
                [1000.0, 400.0, 5.0, 8.0, 11.0],
                [2000.0, 500.0, 5.0, 10.0, 12.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            cache_root = root / "cache"
            write_fixture(
                source,
                matrix,
                cell_type="CVCL_TEST",
                symbols=["A", "A", "B", "C", "D"],
                obs_overrides={
                    "is_control": ["FALSE", "FALSE", "FALSE", "TRUE", "FALSE"],
                    "pert_dose_uM": [10.0, 10.0, 10.0, 10.0, np.nan],
                },
            )
            stats = load_or_fit_population_stats(
                source_path=source,
                dataset_name="source_a",
                cell_type="CVCL_TEST",
                cache_root=cache_root,
                row_chunk_size=2,
                verbose=False,
            )

            # Only rows 0:3 are eligible, and the duplicate A column keeps the first.
            eligible_unique = matrix[:3][:, [0, 2, 3, 4]]
            np.testing.assert_array_equal(stats.gene_keys, ["A", "B", "C", "D"])
            np.testing.assert_array_equal(
                stats.finite_counts,
                np.isfinite(eligible_unique).sum(axis=0),
            )
            np.testing.assert_allclose(
                stats.means,
                np.nanmean(eligible_unique, axis=0),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                stats.population_sds,
                np.nanstd(eligible_unique, axis=0, ddof=0),
                equal_nan=True,
            )
            self.assertEqual(stats.population_row_count, 3)
            self.assertFalse(stats.valid_mask[1])  # Constant gene B.
            self.assertFalse(stats.valid_mask[3])  # Gene D has only one finite value.
            self.assertTrue(stats.cache_path.exists())
            self.assertTrue(stats.cache_path.with_suffix(".cache.json").exists())

            cached = load_or_fit_population_stats(
                source_path=source,
                dataset_name="source_a",
                cell_type="CVCL_TEST",
                cache_root=cache_root,
                row_chunk_size=5,
                verbose=False,
            )
            self.assertEqual(stats.fingerprint, cached.fingerprint)
            np.testing.assert_allclose(stats.means, cached.means, equal_nan=True)

    def test_standardized_valid_genes_have_zero_mean_and_population_sd_one(self):
        matrix = np.asarray(
            [
                [1.0, 10.0, 2.0],
                [2.0, 20.0, 2.0],
                [4.0, 40.0, 2.0],
                [8.0, 80.0, 2.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            write_fixture(source, matrix, cell_type="CVCL_TEST")
            stats = load_or_fit_population_stats(
                source_path=source,
                dataset_name="source_a",
                cell_type="CVCL_TEST",
                cache_root=root / "cache",
                row_chunk_size=3,
                verbose=False,
            )
            standardized = standardize_matrix(
                matrix,
                gene_keys=np.asarray(["G0", "G1", "G2"]),
                stats=stats,
            )
            np.testing.assert_allclose(
                np.nanmean(standardized[:, stats.valid_mask], axis=0),
                0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                np.nanstd(standardized[:, stats.valid_mask], axis=0, ddof=0),
                1.0,
                atol=1e-12,
            )
            self.assertTrue(np.isnan(standardized[:, 2]).all())

    def test_sources_and_cell_types_are_never_mixed(self):
        base = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
        shifted = base * np.asarray([100.0, 0.25]) + np.asarray([500.0, -20.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_a = root / "a.h5ad"
            source_b = root / "b.h5ad"
            write_fixture(source_a, base, cell_type="CVCL_A")
            write_fixture(source_b, shifted, cell_type="CVCL_B")
            stats_a = load_or_fit_population_stats(
                source_path=source_a,
                dataset_name="source_a",
                cell_type="CVCL_A",
                cache_root=root / "cache",
                verbose=False,
            )
            stats_b = load_or_fit_population_stats(
                source_path=source_b,
                dataset_name="source_b",
                cell_type="CVCL_B",
                cache_root=root / "cache",
                verbose=False,
            )
            z_a = standardize_matrix(base, gene_keys=stats_a.gene_keys, stats=stats_a)
            z_b = standardize_matrix(
                shifted,
                gene_keys=stats_b.gene_keys,
                stats=stats_b,
            )
            np.testing.assert_allclose(z_a, z_b, atol=1e-12)
            self.assertNotEqual(stats_a.cache_path, stats_b.cache_path)

    def test_gene_alignment_is_strict(self):
        matrix = np.asarray([[1.0, 3.0], [2.0, 5.0], [4.0, 8.0]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            write_fixture(source, matrix, cell_type="CVCL_TEST")
            stats = load_or_fit_population_stats(
                source_path=source,
                dataset_name="source",
                cell_type="CVCL_TEST",
                cache_root=root / "cache",
                verbose=False,
            )
            with self.assertRaises(ValueError):
                standardize_vector(
                    matrix[0],
                    gene_keys=stats.gene_keys[::-1],
                    stats=stats,
                )

    def test_standardized_centroid_equals_standardized_raw_centroid(self):
        peers = np.asarray(
            [
                [1.0, 20.0, -4.0],
                [3.0, 40.0, 0.0],
                [7.0, 80.0, 8.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            write_fixture(source, peers, cell_type="CVCL_TEST")
            stats = load_or_fit_population_stats(
                source_path=source,
                dataset_name="source",
                cell_type="CVCL_TEST",
                cache_root=root / "cache",
                verbose=False,
            )
            standardized_peers = standardize_matrix(
                peers,
                gene_keys=stats.gene_keys,
                stats=stats,
            )
            standardized_centroid = standardize_vector(
                peers.mean(axis=0),
                gene_keys=stats.gene_keys,
                stats=stats,
            )
            np.testing.assert_allclose(
                standardized_peers.mean(axis=0),
                standardized_centroid,
                atol=1e-12,
            )

            summary = summarize_peer_scores(
                0.5,
                np.asarray([-0.2, 0.1, 0.7]),
                prefix="w4",
            )
            self.assertAlmostEqual(summary["w4_fraction_below_observed"], 2 / 3)
            self.assertAlmostEqual(summary["w4_corrected_percentile"], 3 / 4)


if __name__ == "__main__":
    unittest.main()
