from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import scripts.merge_replicate_tpeers as merge
import scripts.summarize_raw_cross_source_tables as raw
from scripts.summarize_reviewer_minimal_metrics import W4_DEG_METRICS, W4_SIGNATURE_METRICS


def condition_frame(dataset: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "dataset_name": dataset,
            "condition_key": [f"{dataset}|c{i}" for i in range(n)],
            "mean_replicate_spearman_t": rng.normal(size=n),
            "mean_replicate_spearman_logfc_global": rng.normal(size=n),
            "mean_replicate_signed_overlap_t_top50": rng.normal(size=n),
            "peer_config_fingerprint": "base",
        }
    )


def with_tpeers(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for column in merge.T_PEER_COLUMNS:
        out[column] = rng.normal(size=len(out))
    out["peer_config_fingerprint"] = "tpeer"
    return out


class MergeReplicateTpeersTest(unittest.TestCase):
    def run_merge(self, base: pd.DataFrame, runs: list[pd.DataFrame]) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "base").mkdir()
            base.to_csv(root / "base/condition_metric_summary.tsv", sep="\t", index=False)
            pd.DataFrame({"x": [1]}).to_csv(root / "base/condition_deg_metric_summary.tsv", sep="\t", index=False)
            dirs = []
            for i, run in enumerate(runs):
                (root / f"run{i}").mkdir()
                run.to_csv(root / f"run{i}/condition_metric_summary.tsv", sep="\t", index=False)
                dirs.append(str(root / f"run{i}"))
            argv = ["merge", "--base-dir", str(root / "base"), "--output-dir", str(root / "out"), "--tpeer-dirs", *dirs]
            with mock.patch.object(sys, "argv", argv):
                merge.main()
            return pd.read_csv(root / "out/condition_metric_summary.tsv", sep="\t")

    def test_joins_t_peer_columns_and_keeps_base_values(self):
        a, b = condition_frame("a", 5, 1), condition_frame("b", 4, 2)
        base = pd.concat([a, b], ignore_index=True)
        out = self.run_merge(base, [with_tpeers(a, 3), with_tpeers(b, 4)])
        self.assertEqual(len(out), len(base))
        pd.testing.assert_series_equal(out["mean_replicate_spearman_t"], base["mean_replicate_spearman_t"])
        self.assertTrue(out[merge.T_PEER_COLUMNS].notna().all().all())

    def test_refuses_a_disagreeing_shared_column(self):
        a = condition_frame("a", 5, 1)
        run = with_tpeers(a, 3)
        run.loc[2, "mean_replicate_spearman_t"] += 1e-9
        with self.assertRaises(SystemExit):
            self.run_merge(a, [run])

    def test_tolerates_rounding_and_tie_sensitive_columns(self):
        a = condition_frame("a", 5, 1)
        run = with_tpeers(a, 3)
        run["mean_replicate_spearman_t"] += 1e-15
        run.loc[0, "mean_replicate_signed_overlap_t_top50"] += 0.02
        out = self.run_merge(a, [run])
        self.assertEqual(len(out), 5)

    def test_refuses_mismatched_condition_sets(self):
        a = condition_frame("a", 5, 1)
        with self.assertRaises(SystemExit):
            self.run_merge(a, [with_tpeers(a.iloc[:4], 3)])

    def test_dataset_in_two_runs_uses_the_run_matching_the_base(self):
        a = condition_frame("a", 5, 1)
        first = with_tpeers(a, 3)
        second = first.copy()
        second["mean_replicate_spearman_logfc_global"] += 0.5  # other run's gene set
        base = a.assign(n_global_shared_genes=100)
        first["n_global_shared_genes"] = 100
        second["n_global_shared_genes"] = 90
        out = self.run_merge(base, [second, first])
        np.testing.assert_allclose(out["mean_peer_baseline_spearman_t"], first["mean_peer_baseline_spearman_t"])

    def test_refuses_runs_disagreeing_on_t_peers(self):
        a = condition_frame("a", 5, 1)
        first = with_tpeers(a, 3)
        second = with_tpeers(a, 4)
        with self.assertRaises(SystemExit):
            self.run_merge(a, [first, second])


class RawScorerColumnTest(unittest.TestCase):
    def test_every_table_metric_maps_to_a_pb_column(self):
        for metric in (*W4_DEG_METRICS, *W4_SIGNATURE_METRICS):
            column = raw.scorer_column(metric)
            self.assertTrue(column.startswith("pb_"), column)
        self.assertEqual(
            raw.scorer_column("w4_observed_deg_lfc_spearman_sym_p05"),
            "pb_observed_deg_lfc_spearman_p05",
        )
        self.assertEqual(
            raw.scorer_column("w4_source_peer_spearman_logfc_pair"),
            "pb_source_peer_spearman_logfc_pair",
        )

    def test_rejects_non_w4_metric(self):
        with self.assertRaises(ValueError):
            raw.scorer_column("observed_spearman_logfc")


if __name__ == "__main__":
    unittest.main()
