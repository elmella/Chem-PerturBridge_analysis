import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.cross_source_core import (
    MATCH_PAIR_COLUMNS,
    MATCH_PAIR_IDENTITY_COLUMNS,
    LineSourceCatalog,
    MatchSettings,
    analysis_output_dir,
    build_dataset_index,
    build_groups,
    build_matched_pairs,
    build_symmetric_pair_metric_matrix,
    ci_fingerprint,
    difference_if_both_defined,
    ensure_overlap_frame_schema,
    format_ci_cell,
    line_source_inventory,
    matched_dataset_lines,
    matched_dataset_names,
    mutual_nearest_dose_pairs,
    pair_match_frame,
    prepare_cross_source_scope,
    production_dataset_order,
    selected_dataset_order,
    source_dataset_dirs,
    summarize_matched_pairs,
)


def synthetic_index(dataset_name: str, doses: list[float]) -> dict[str, object]:
    frame = ensure_overlap_frame_schema(
        pd.DataFrame(
            {
                "obs_id": [f"{dataset_name}_{index}" for index in range(len(doses))],
                "plate": [f"plate_{dataset_name}"] * len(doses),
                "well": [f"A{index + 1}" for index in range(len(doses))],
                "pubchem_cid": ["1"] * len(doses),
                "cell_type": ["CVCL_TEST"] * len(doses),
                "pert_time_h": [24.0] * len(doses),
                "pert_dose_uM": doses,
            }
        ),
        dataset_name,
    )
    return {"frame": frame, "groups": build_groups(frame)}


class CrossSourceCoreTests(unittest.TestCase):
    def test_dataset_profiles_have_one_explicit_source_of_truth(self):
        signature = production_dataset_order("signature")
        deg = production_dataset_order("deg")
        retrieval = production_dataset_order("retrieval")

        self.assertEqual(deg, retrieval)
        self.assertIn("gdpx2", signature)
        self.assertIn("dilimap_train_val", signature)
        self.assertNotIn("gdpx2", deg)
        self.assertNotIn("dilimap_train_val", retrieval)
        self.assertEqual(
            selected_dataset_order("deg", ["tahoe", "sciplex", "tahoe"]),
            ["tahoe", "sciplex"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            selected_dataset_order("deg", ["tahoe", "gdpx2"])

        root = Path("/tmp/data-root")
        signature_dirs = source_dataset_dirs(root, "signature")
        self.assertEqual(
            signature_dirs["cigs_mce"],
            root
            / "cigs_mce"
            / "group_rep_extracted"
            / "deg_data"
            / "group_rep"
            / "full"
            / "qc_false"
            / "filter_min_cells_0"
            / "results",
        )

    def test_output_layout_is_shared_for_production_tags_and_subsets(self):
        production = Path("/repo/results/analysis")
        self.assertEqual(
            analysis_output_dir(
                production,
                dataset_subset=None,
                dataset_names=["A", "B"],
            ),
            production,
        )
        self.assertEqual(
            analysis_output_dir(
                production,
                dataset_subset=None,
                dataset_names=["A", "B"],
                run_tag="production_test",
            ),
            Path("/repo/results/production_runs/analysis/production_test"),
        )
        self.assertEqual(
            analysis_output_dir(
                production,
                dataset_subset=["A", "B"],
                dataset_names=["A", "B"],
            ),
            Path("/repo/results/subset_runs/analysis/A__B"),
        )

    def test_canonical_matching_returns_the_shared_superset_schema(self):
        left = synthetic_index("left", [1.0, 10.0])
        right = synthetic_index("right", [1.0, 9.0])
        settings = MatchSettings(
            max_dose_fold_difference=2.0,
            min_context_shared_drugs=1,
        )

        pairs = pair_match_frame(
            "left",
            "right",
            left,
            right,
            settings=settings,
        )

        self.assertEqual(pairs.columns.tolist(), MATCH_PAIR_COLUMNS)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            pairs[MATCH_PAIR_IDENTITY_COLUMNS]
            .astype(str)
            .to_records(index=False)
            .tolist(),
            [
                (
                    "left",
                    "right",
                    "CVCL_TEST",
                    "1",
                    "24",
                    "left_0",
                    "right_0",
                    "1",
                    "1",
                ),
                (
                    "left",
                    "right",
                    "CVCL_TEST",
                    "1",
                    "24",
                    "left_1",
                    "right_1",
                    "10",
                    "9",
                ),
            ],
        )
        self.assertAlmostEqual(
            float(pairs.loc[1, "dose_fold_difference"]),
            10.0 / 9.0,
        )

        combined = build_matched_pairs(
            {"left": left, "right": right},
            ["left", "right"],
            settings=settings,
        )
        pd.testing.assert_frame_equal(combined, pairs)
        pair_summary, line_summary = summarize_matched_pairs(combined)
        self.assertEqual(
            pair_summary.iloc[0].to_dict(),
            {
                "dataset_a": "left",
                "dataset_b": "right",
                "n_matched_sample_pairs": 2,
                "n_matching_drugs": 1,
                "n_matching_lines": 1,
                "n_matching_conditions": 2,
            },
        )
        self.assertEqual(int(line_summary.loc[0, "n_matched_sample_pairs"]), 2)

    def test_overlap_schema_uses_index_when_obs_id_is_absent(self):
        frame = ensure_overlap_frame_schema(
            pd.DataFrame(
                {
                    "pubchem_cid": ["1"],
                    "cell_type": ["CVCL_TEST"],
                    "pert_time_h": [24.0],
                    "pert_dose_uM": [1.0],
                },
                index=["source_row"],
            ),
            "source",
        )
        self.assertEqual(frame.loc[0, "obs_id"], "source_row")

    def test_fold_matching_preserves_the_legacy_log10_nearest_geometry(self):
        left = np.asarray([0.01, 0.1, 1.0, 10.0], dtype=np.float64)
        right = np.asarray([0.02, 0.2, 2.0, 20.0], dtype=np.float64)
        log_difference = np.abs(
            np.log10(left)[:, None] - np.log10(right)[None, :]
        )
        legacy_mask = (
            (log_difference <= 1.0 + 1e-12)
            & np.isclose(
                log_difference,
                log_difference.min(axis=1, keepdims=True),
                rtol=0.0,
                atol=1e-12,
            )
            & np.isclose(
                log_difference,
                log_difference.min(axis=0, keepdims=True),
                rtol=0.0,
                atol=1e-12,
            )
        )
        np.testing.assert_array_equal(
            mutual_nearest_dose_pairs(
                left,
                right,
                max_dose_fold_difference=10.0,
            ),
            np.argwhere(legacy_mask),
        )

    def test_matched_dataset_names_excludes_selected_sources_without_matches(self):
        pairs = pd.DataFrame(
            {
                "dataset_a": ["A"],
                "dataset_b": ["B"],
                "cell_type": ["line"],
            }
        )
        matched_lines = matched_dataset_lines(pairs, ["A", "B", "C"])
        self.assertEqual(
            matched_dataset_names(matched_lines, ["A", "B", "C"]),
            ["A", "B"],
        )

    def test_source_inventory_and_ci_fingerprints_track_their_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "A.h5ad"
            second = root / "B.h5ad"
            first.write_bytes(b"a")
            second.write_bytes(b"bb")
            paths = {("A", "line"): first, ("B", "line"): second}

            inventory = line_source_inventory(
                {"B": ["line"], "A": ["line"]},
                lambda dataset, line: paths[(dataset, line)],
            )
            self.assertEqual(
                [(row["dataset"], row["cell_type"]) for row in inventory],
                [("A", "line"), ("B", "line")],
            )

        baseline = ci_fingerprint(
            version="ci-v1",
            upstream_fingerprint="upstream-a",
            n_boot=2000,
            seed=123,
            metric_columns=["score"],
            group_columns=["dataset_a", "dataset_b"],
        )
        self.assertNotEqual(
            baseline,
            ci_fingerprint(
                version="ci-v1",
                upstream_fingerprint="upstream-b",
                n_boot=2000,
                seed=123,
                metric_columns=["score"],
                group_columns=["dataset_a", "dataset_b"],
            ),
        )
        self.assertNotEqual(
            baseline,
            ci_fingerprint(
                version="ci-v1",
                upstream_fingerprint="upstream-a",
                n_boot=1000,
                seed=123,
                metric_columns=["score"],
                group_columns=["dataset_a", "dataset_b"],
            ),
        )

    def test_shared_reporting_helpers_preserve_undefined_values(self):
        summary = pd.DataFrame(
            {
                "dataset_a": ["A", "A"],
                "dataset_b": ["B", "outside"],
                "score": [0.25, 0.75],
            }
        )
        matrix = build_symmetric_pair_metric_matrix(
            summary,
            "score",
            ["A", "B"],
        )
        self.assertEqual(float(matrix.loc["A", "B"]), 0.25)
        self.assertEqual(float(matrix.loc["B", "A"]), 0.25)
        self.assertTrue(np.isnan(matrix.loc["A", "A"]))
        self.assertAlmostEqual(
            difference_if_both_defined(0.5, 0.2),
            0.3,
        )
        self.assertTrue(
            np.isnan(difference_if_both_defined(np.nan, 0.2))
        )
        self.assertEqual(
            format_ci_cell(
                pd.Series({"mean": 0.25, "ci_low": 0.1, "ci_high": 0.4})
            ),
            "0.250 [0.100, 0.400]",
        )
        self.assertEqual(
            format_ci_cell(
                pd.Series(
                    {"mean": 0.25, "ci_low": np.nan, "ci_high": np.nan}
                )
            ),
            "0.250",
        )

    def test_build_dataset_index_reads_the_canonical_overlap_name(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_root = Path(directory)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "source_overlap_filtered.h5ad",
            ):
                build_dataset_index("source", missing_root)

    def test_overlap_index_prefers_harmonized_context_and_excludes_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            overlap_root = Path(directory)
            path = overlap_root / "source_overlap_filtered.h5ad"
            ad.AnnData(
                X=np.zeros((2, 1), dtype=np.float32),
                obs=pd.DataFrame(
                    {
                        "is_control": ["FALSE", "TRUE"],
                        "pubchem_cid": [1.0, 2.0],
                        "cell_type": ["raw_line", "raw_control"],
                        "harmonized_context_key": ["CVCL_TEST", "CVCL_TEST"],
                        "pert_time_h": [24.0, 24.0],
                        "pert_dose_uM": [1.0, 1.0],
                    },
                    index=["treated", "control"],
                ),
                var=pd.DataFrame(index=["gene"]),
            ).write_h5ad(path)

            index = build_dataset_index("source", overlap_root)
            self.assertEqual(index["frame"]["obs_id"].tolist(), ["treated"])
            self.assertEqual(
                index["frame"]["cell_type"].tolist(),
                ["CVCL_TEST"],
            )

    def test_prepare_scope_owns_matching_cache_and_shared_gene_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlap_root = root / "overlap"
            overlap_root.mkdir()
            output_root = root / "results" / "scope"
            source_dirs = {}
            for dataset_name in ("left", "right"):
                overlap_obs = pd.DataFrame(
                    {
                        "is_control": [False],
                        "pubchem_cid": ["1"],
                        "cell_type": ["CVCL_TEST"],
                        "pert_time_h": [24.0],
                        "pert_dose_uM": [1.0],
                    },
                    index=[f"{dataset_name}_row"],
                )
                ad.AnnData(
                    X=np.zeros((1, 2), dtype=np.float32),
                    obs=overlap_obs,
                    var=pd.DataFrame(index=["G1", "G2"]),
                ).write_h5ad(
                    overlap_root
                    / f"{dataset_name}_overlap_filtered.h5ad"
                )

                source_dir = root / dataset_name
                source_dir.mkdir()
                source_dirs[dataset_name] = source_dir
                ad.AnnData(
                    X=np.zeros((1, 2), dtype=np.float32),
                    obs=overlap_obs,
                    var=pd.DataFrame(
                        {"symbol": ["G1", "G2"]},
                        index=["ENSG1", "ENSG2"],
                    ),
                    layers={
                        "logFC": np.asarray([[1.0, 2.0]], dtype=np.float32)
                    },
                ).write_h5ad(source_dir / "CVCL_TEST_de.h5ad")

            catalog = LineSourceCatalog(source_dirs)
            try:
                scope = prepare_cross_source_scope(
                    dataset_order=["left", "right"],
                    overlap_dir=overlap_root,
                    output_dir=output_root,
                    source_catalog=catalog,
                    settings=MatchSettings(
                        max_dose_fold_difference=1.0,
                        min_context_shared_drugs=1,
                    ),
                )
                self.assertEqual(scope.active_datasets, ["left", "right"])
                self.assertEqual(len(scope.matched_pairs), 1)
                self.assertEqual(
                    scope.matched_lines,
                    {
                        "left": ["CVCL_TEST"],
                        "right": ["CVCL_TEST"],
                    },
                )
                np.testing.assert_array_equal(
                    scope.line_global_gene_keys["CVCL_TEST"],
                    ["G1", "G2"],
                )
                self.assertEqual(
                    int(
                        scope.pair_match_summary.loc[
                            0,
                            "n_matched_sample_pairs",
                        ]
                    ),
                    1,
                )
                self.assertTrue(
                    (output_root / "matched_sample_pairs.tsv").exists()
                )
                reloaded = prepare_cross_source_scope(
                    dataset_order=["left", "right"],
                    overlap_dir=overlap_root,
                    output_dir=output_root,
                    source_catalog=catalog,
                    settings=MatchSettings(
                        max_dose_fold_difference=1.0,
                        min_context_shared_drugs=1,
                    ),
                )
                pd.testing.assert_frame_equal(
                    reloaded.matched_pairs,
                    scope.matched_pairs,
                )
            finally:
                catalog.close()


if __name__ == "__main__":
    unittest.main()
