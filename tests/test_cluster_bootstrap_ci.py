from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.cluster_bootstrap_ci import (
    _bootstrap_and_jackknife_values,
    _nested_statistic_from_totals,
    cluster_bca_nested_mean_ci_table,
)


class ClusterBootstrapCITests(unittest.TestCase):
    def test_batched_resampling_preserves_legacy_summation_order(self):
        rng = np.random.default_rng(113)
        sum_matrix = rng.normal(size=(131, 17))
        count_matrix = rng.integers(
            0,
            5,
            size=sum_matrix.shape,
        ).astype(np.float64)
        sum_matrix[count_matrix == 0] = 0.0
        outer_codes = np.arange(sum_matrix.shape[1]) % 3
        n_boot = 137
        seed = 901

        legacy_rng = np.random.default_rng(seed)
        expected_bootstrap = np.empty(n_boot, dtype=np.float64)
        for index in range(n_boot):
            sampled = legacy_rng.integers(
                0,
                len(sum_matrix),
                len(sum_matrix),
            )
            expected_bootstrap[index] = _nested_statistic_from_totals(
                sum_matrix[sampled].sum(axis=0),
                count_matrix[sampled].sum(axis=0),
                outer_codes=outer_codes,
            )
        total_sum = sum_matrix.sum(axis=0)
        total_count = count_matrix.sum(axis=0)
        expected_jackknife = np.asarray(
            [
                _nested_statistic_from_totals(
                    total_sum - sum_matrix[index],
                    total_count - count_matrix[index],
                    outer_codes=outer_codes,
                )
                for index in range(len(sum_matrix))
            ]
        )

        actual_bootstrap, actual_jackknife = (
            _bootstrap_and_jackknife_values(
                sum_matrix=sum_matrix,
                count_matrix=count_matrix,
                outer_codes=outer_codes,
                n_boot=n_boot,
                rng=np.random.default_rng(seed),
                batch_size=7,
            )
        )

        np.testing.assert_array_equal(
            actual_bootstrap,
            expected_bootstrap,
        )
        np.testing.assert_array_equal(
            actual_jackknife,
            expected_jackknife,
        )

    def test_batched_kernel_preserves_seeded_scalar_results(self):
        rng = np.random.default_rng(7)
        n_rows = 48
        frame = pd.DataFrame(
            {
                "pair": np.where(
                    np.arange(n_rows) < 24,
                    "ab",
                    "cd",
                ),
                "cluster": [f"c{index % 8}" for index in range(n_rows)],
                "direction": np.where(
                    np.arange(n_rows) % 2,
                    "left",
                    "right",
                ),
                "cell": [f"l{index % 3}" for index in range(n_rows)],
                "m1": rng.normal(size=n_rows),
                "m2": rng.normal(size=n_rows),
            }
        )
        frame.loc[[2, 19, 33], "m1"] = np.nan

        result = cluster_bca_nested_mean_ci_table(
            frame,
            group_cols=["pair"],
            metric_cols=["m1", "m2"],
            cluster_col="cluster",
            inner_cols=["direction", "cell"],
            outer_cols=["direction"],
            n_boot=101,
            seed=99,
            summary_level="parity",
            bootstrap_batch_size=7,
        )
        parallel_result = cluster_bca_nested_mean_ci_table(
            frame,
            group_cols=["pair"],
            metric_cols=["m1", "m2"],
            cluster_col="cluster",
            inner_cols=["direction", "cell"],
            outer_cols=["direction"],
            n_boot=101,
            seed=99,
            summary_level="parity",
            bootstrap_batch_size=7,
            workers=3,
        )

        expected = [
            ("ab", "m1", -0.349548235142246, -0.639382564080838, 0.055702994198866, 22),
            ("ab", "m2", -0.003518824781746, -0.455407124963620, 0.350409312030263, 24),
            ("cd", "m1", -0.271419592134454, -0.474698726185760, -0.018675730748890, 23),
            ("cd", "m2", 0.143785803806419, -0.041664144563038, 0.412817932028401, 24),
        ]
        self.assertEqual(
            list(zip(result["pair"], result["metric"])),
            [(row[0], row[1]) for row in expected],
        )
        np.testing.assert_allclose(
            result[["mean", "ci_low", "ci_high"]].to_numpy(),
            np.asarray([row[2:5] for row in expected]),
            rtol=0,
            atol=1e-14,
        )
        self.assertEqual(
            result["n_finite_rows"].tolist(),
            [row[5] for row in expected],
        )
        self.assertEqual(result["n_compounds"].tolist(), [8, 8, 8, 8])
        self.assertEqual(
            result["n_bootstrap_valid"].tolist(),
            [101, 101, 101, 101],
        )
        self.assertEqual(result["ci_method"].tolist(), ["bca"] * 4)
        self.assertEqual(result["ci_status"].tolist(), ["ok"] * 4)
        pd.testing.assert_frame_equal(
            result,
            parallel_result,
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
