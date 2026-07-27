import io
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd

import scripts.population_zscore as population_zscore_module
from scripts.peer_baselines import summarize_peer_scores
from scripts.precompute_population_zscore import (
    build_parser as build_precompute_parser,
    run as run_precompute,
)
from scripts.population_zscore import (
    DATASET_CELL_TYPE_SCOPE,
    DATASET_SCOPE,
    PopulationCacheReadiness,
    align_population_stats,
    check_dataset_population_cache_readiness,
    dataset_stats_cache_path,
    discover_dataset_population_sources,
    ensure_population_caches_ready,
    load_or_fit_dataset_population_stats,
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
    def test_cache_readiness_is_metadata_only_and_tracks_both_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "CVCL_A_de.h5ad"
            second_source = root / "CVCL_B_de.h5ad"
            cache_root = root / "cache"
            write_fixture(
                first_source,
                np.asarray([[1.0, 10.0], [2.0, 20.0]]),
                cell_type="CVCL_A",
            )
            write_fixture(
                second_source,
                np.asarray([[3.0, 30.0], [4.0, 40.0]]),
                cell_type="CVCL_B",
            )
            sources = {
                "CVCL_A": first_source,
                "CVCL_B": second_source,
            }

            initial = check_dataset_population_cache_readiness(
                source_paths=sources,
                dataset_name="source",
                cache_root=cache_root,
            )
            self.assertFalse(initial.ready)
            self.assertEqual(initial.ready_source_count, 0)
            self.assertEqual(initial.total_source_count, 2)
            self.assertEqual(initial.pending_sources, ("CVCL_A", "CVCL_B"))

            load_or_fit_population_stats(
                source_path=first_source,
                dataset_name="source",
                cell_type="CVCL_A",
                cache_root=cache_root,
                verbose=False,
            )
            partial = check_dataset_population_cache_readiness(
                source_paths=sources,
                dataset_name="source",
                cache_root=cache_root,
            )
            self.assertFalse(partial.ready)
            self.assertEqual(partial.ready_source_count, 1)
            self.assertEqual(partial.pending_sources, ("CVCL_B",))

            load_or_fit_dataset_population_stats(
                source_paths=sources,
                dataset_name="source",
                cache_root=cache_root,
                verbose=False,
            )
            with patch(
                "scripts.population_zscore.ad.read_h5ad",
                side_effect=AssertionError(
                    "readiness checks must not open source H5ADs"
                ),
            ):
                ready = check_dataset_population_cache_readiness(
                    source_paths=sources,
                    dataset_name="source",
                    cache_root=cache_root,
                )
                ensured = ensure_population_caches_ready(
                    dataset_sources={"source": sources},
                    cache_root=cache_root,
                    wait=False,
                    verbose=False,
                )

            self.assertTrue(ready.ready)
            self.assertTrue(ready.dataset_cache_ready)
            self.assertEqual(ready.ready_source_count, 2)
            self.assertEqual(len(ensured), 1)
            self.assertTrue(ensured[0].ready)

    def test_incomplete_cache_can_stop_or_wait_for_precompute(self):
        pending = PopulationCacheReadiness(
            dataset_name="source",
            ready_source_count=1,
            total_source_count=2,
            dataset_cache_ready=False,
            pending_sources=("CVCL_B",),
        )
        ready = PopulationCacheReadiness(
            dataset_name="source",
            ready_source_count=2,
            total_source_count=2,
            dataset_cache_ready=True,
            pending_sources=(),
        )
        with patch(
            "scripts.population_zscore.check_population_caches_ready",
            return_value=[pending],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "rerun this W4 setup cell",
            ):
                ensure_population_caches_ready(
                    dataset_sources={"source": {"CVCL_A": Path("unused")}},
                    cache_root=Path("unused"),
                    wait=False,
                    verbose=False,
                )

        with patch(
            "scripts.population_zscore.check_population_caches_ready",
            side_effect=[[pending], [ready]],
        ):
            with patch("scripts.population_zscore.time.sleep") as sleep:
                result = ensure_population_caches_ready(
                    dataset_sources={"source": {"CVCL_A": Path("unused")}},
                    cache_root=Path("unused"),
                    wait=True,
                    poll_seconds=7.0,
                    verbose=False,
                )
        sleep.assert_called_once_with(7.0)
        self.assertEqual(result, [ready])

    def test_dataset_source_discovery_matches_notebook_filename_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(
                root / "CVCL_A_de.h5ad",
                np.asarray([[1.0], [2.0]]),
                cell_type="CVCL_A",
            )
            write_fixture(
                root / "CVCL_B.h5ad",
                np.asarray([[3.0], [4.0]]),
                cell_type="CVCL_B",
            )
            (root / "ignore.txt").write_text("not an h5ad")

            sources = discover_dataset_population_sources(root)

            self.assertEqual(list(sources), ["CVCL_A", "CVCL_B"])
            self.assertEqual(sources["CVCL_A"], root / "CVCL_A_de.h5ad")
            self.assertEqual(sources["CVCL_B"], root / "CVCL_B.h5ad")

    def test_precompute_cli_discovers_lines_and_builds_both_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            write_fixture(
                source_dir / "CVCL_A_de.h5ad",
                np.asarray([[1.0, 10.0], [2.0, 20.0]]),
                cell_type="CVCL_A",
            )
            write_fixture(
                source_dir / "CVCL_B.h5ad",
                np.asarray([[3.0, 30.0], [4.0, 40.0]]),
                cell_type="CVCL_B",
            )
            cache_root = root / "cache"
            qc_path = root / "qc.tsv"
            args = build_precompute_parser().parse_args(
                [
                    "--dataset-dir",
                    f"source={source_dir}",
                    "--cache-root",
                    str(cache_root),
                    "--qc-output",
                    str(qc_path),
                    "--row-chunk-size",
                    "1",
                ]
            )
            real_read_h5ad = ad.read_h5ad
            with patch(
                "scripts.population_zscore.ad.read_h5ad",
                wraps=real_read_h5ad,
            ) as read_h5ad:
                with redirect_stdout(io.StringIO()):
                    qc = run_precompute(args)

            self.assertEqual(len(qc), 3)
            # --scope both scans each source once and pools the returned line stats
            # without reopening the H5ADs for dataset-wide aggregation.
            self.assertEqual(read_h5ad.call_count, 2)
            self.assertEqual(
                set(qc["scope"]),
                {DATASET_CELL_TYPE_SCOPE, DATASET_SCOPE},
            )
            self.assertTrue(qc_path.exists())
            self.assertTrue(
                dataset_stats_cache_path(cache_root, "source").exists()
            )

    def test_parallel_precompute_matches_serial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            write_fixture(
                source_dir / "CVCL_A_de.h5ad",
                np.asarray([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]]),
                cell_type="CVCL_A",
            )
            write_fixture(
                source_dir / "CVCL_B_de.h5ad",
                np.asarray([[3.0, 30.0], [4.0, 40.0], [8.0, 80.0]]),
                cell_type="CVCL_B",
            )

            def precompute(cache_name: str, workers: int) -> pd.DataFrame:
                args = build_precompute_parser().parse_args(
                    [
                        "--dataset-dir",
                        f"source={source_dir}",
                        "--cache-root",
                        str(root / cache_name),
                        "--row-chunk-size",
                        "1",
                        "--workers",
                        str(workers),
                    ]
                )
                with redirect_stdout(io.StringIO()):
                    return run_precompute(args)

            serial = precompute("serial_cache", 1)
            parallel = precompute("parallel_cache", 2)
            comparison_columns = [
                column for column in serial.columns if column != "cache_path"
            ]
            sort_columns = ["scope", "cell_type"]
            pd.testing.assert_frame_equal(
                serial[comparison_columns]
                .sort_values(sort_columns)
                .reset_index(drop=True),
                parallel[comparison_columns]
                .sort_values(sort_columns)
                .reset_index(drop=True),
                check_exact=True,
            )

    def test_dataset_population_pools_lines_with_different_gene_orders_and_sets(self):
        first_matrix = np.asarray(
            [
                [1.0, 10.0, 100.0],
                [3.0, 30.0, 300.0],
            ]
        )
        second_matrix = np.asarray(
            [
                [500.0, 5.0, 1000.0],
                [700.0, 7.0, 1400.0],
                [900.0, 9.0, 1800.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_source = root / "CVCL_A_de.h5ad"
            second_source = root / "CVCL_B_de.h5ad"
            cache_root = root / "cache"
            write_fixture(
                first_source,
                first_matrix,
                cell_type="CVCL_A",
                symbols=["A", "B", "C"],
            )
            write_fixture(
                second_source,
                second_matrix,
                cell_type="CVCL_B",
                symbols=["C", "A", "D"],
            )

            stats = load_or_fit_dataset_population_stats(
                # Reversed insertion order verifies canonical source ordering.
                source_paths={
                    "CVCL_B": second_source,
                    "CVCL_A": first_source,
                },
                dataset_name="source",
                cache_root=cache_root,
                row_chunk_size=2,
                verbose=False,
            )

            pooled = np.asarray(
                [
                    [1.0, 10.0, 100.0, np.nan],
                    [3.0, 30.0, 300.0, np.nan],
                    [5.0, np.nan, 500.0, 1000.0],
                    [7.0, np.nan, 700.0, 1400.0],
                    [9.0, np.nan, 900.0, 1800.0],
                ]
            )
            np.testing.assert_array_equal(stats.gene_keys, ["A", "B", "C", "D"])
            np.testing.assert_array_equal(
                stats.finite_counts,
                np.isfinite(pooled).sum(axis=0),
            )
            np.testing.assert_allclose(
                stats.means,
                np.nanmean(pooled, axis=0),
            )
            np.testing.assert_allclose(
                stats.population_sds,
                np.nanstd(pooled, axis=0, ddof=0),
            )
            self.assertEqual(stats.scope, DATASET_SCOPE)
            self.assertEqual(stats.population_row_count, 5)
            self.assertEqual(
                stats.cache_path,
                dataset_stats_cache_path(cache_root, "source"),
            )

            aligned = align_population_stats(
                stats,
                np.asarray(["C", "A", "D"]),
            )
            standardized = standardize_matrix(
                second_matrix,
                gene_keys=np.asarray(["C", "A", "D"]),
                stats=aligned,
            )
            expected = (
                second_matrix
                - np.asarray([500.0, 5.0, 1400.0])[None, :]
            ) / np.asarray(
                [
                    np.nanstd(pooled[:, 2], ddof=0),
                    np.nanstd(pooled[:, 0], ddof=0),
                    np.nanstd(pooled[:, 3], ddof=0),
                ]
            )[None, :]
            np.testing.assert_allclose(standardized, expected, atol=1e-12)

            with patch(
                "scripts.population_zscore.fit_dataset_population_stats",
                side_effect=AssertionError("dataset cache should have reloaded"),
            ):
                cached = load_or_fit_dataset_population_stats(
                    source_paths={
                        "CVCL_A": first_source,
                        "CVCL_B": second_source,
                    },
                    dataset_name="source",
                    cache_root=cache_root,
                    verbose=False,
                )
            self.assertEqual(cached.fingerprint, stats.fingerprint)
            np.testing.assert_allclose(cached.means, stats.means)

    def test_dataset_population_transform_alignment_remains_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "CVCL_A_de.h5ad"
            matrix = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
            write_fixture(
                source,
                matrix,
                cell_type="CVCL_A",
                symbols=["A", "B"],
            )
            stats = load_or_fit_dataset_population_stats(
                source_paths={"CVCL_A": source},
                dataset_name="source",
                cache_root=root / "cache",
                verbose=False,
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                align_population_stats(stats, np.asarray(["A", "MISSING"]))

            aligned = align_population_stats(stats, np.asarray(["B", "A"]))
            with self.assertRaisesRegex(ValueError, "Gene keys do not match"):
                standardize_vector(
                    matrix[0],
                    gene_keys=np.asarray(["A", "B"]),
                    stats=aligned,
                )

    def test_concurrent_first_load_fits_source_context_once(self):
        matrix = np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [4.0, 40.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            cache_root = root / "cache"
            write_fixture(source, matrix, cell_type="CVCL_TEST")

            fit_count = 0
            fit_count_lock = threading.Lock()
            start = threading.Barrier(2)
            real_fit = population_zscore_module._fit_population_stats_from_open_adata

            def counted_fit(**kwargs):
                nonlocal fit_count
                with fit_count_lock:
                    fit_count += 1
                time.sleep(0.1)
                return real_fit(**kwargs)

            def load():
                start.wait(timeout=5)
                return load_or_fit_population_stats(
                    source_path=source,
                    dataset_name="source",
                    cell_type="CVCL_TEST",
                    cache_root=cache_root,
                    row_chunk_size=2,
                    verbose=False,
                )

            with patch(
                "scripts.population_zscore._fit_population_stats_from_open_adata",
                side_effect=counted_fit,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    first, second = list(executor.map(lambda _: load(), range(2)))

            self.assertEqual(fit_count, 1)
            self.assertEqual(first.fingerprint, second.fingerprint)

    def test_cache_miss_opens_source_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.h5ad"
            write_fixture(
                source,
                np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]),
                cell_type="CVCL_TEST",
            )
            real_read_h5ad = ad.read_h5ad
            with patch(
                "scripts.population_zscore.ad.read_h5ad",
                wraps=real_read_h5ad,
            ) as read_h5ad:
                first = load_or_fit_population_stats(
                    source_path=source,
                    dataset_name="source",
                    cell_type="CVCL_TEST",
                    cache_root=root / "cache",
                    row_chunk_size=2,
                    verbose=False,
                )
            self.assertEqual(read_h5ad.call_count, 1)

            with patch(
                "scripts.population_zscore.ad.read_h5ad",
                wraps=real_read_h5ad,
            ) as read_h5ad:
                cached = load_or_fit_population_stats(
                    source_path=source,
                    dataset_name="source",
                    cell_type="CVCL_TEST",
                    cache_root=root / "cache",
                    row_chunk_size=2,
                    verbose=False,
                )
            self.assertEqual(read_h5ad.call_count, 0)
            self.assertEqual(first.fingerprint, cached.fingerprint)

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
