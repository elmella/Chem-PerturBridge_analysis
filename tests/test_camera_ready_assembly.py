from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import scripts.merge_replicate_t_columns as merge
import scripts.summarize_raw_cross_source_tables as raw
from scripts.summarize_reviewer_minimal_metrics import W4_DEG_METRICS, W4_SIGNATURE_METRICS


NEW_METRIC = ["mean_peer_baseline_spearman_t", "mean_replicate_cosine_t"]
NEW_DEG = ["mean_replicate_deg_t_spearman_sym_p05"]


def condition_frames(dataset: str, n: int, seed: int) -> dict[str, pd.DataFrame]:
    """Base-like condition and DEG summaries for one dataset."""
    rng = np.random.default_rng(seed)
    keys = {"dataset_name": dataset, "condition_key": [f"{dataset}|c{i}" for i in range(n)]}
    return {
        "condition_metric_summary.tsv": pd.DataFrame(
            {
                **keys,
                "mean_replicate_spearman_t": rng.normal(size=n),
                "mean_replicate_spearman_logfc_global": rng.normal(size=n),
                "mean_replicate_signed_overlap_t_top50": rng.normal(size=n),
                "n_global_shared_genes": 100,
                "peer_config_fingerprint": "base",
            }
        ),
        "condition_deg_metric_summary.tsv": pd.DataFrame(
            {**keys, "mean_replicate_deg_lfc_spearman_sym_p05": rng.normal(size=n)}
        ),
    }


def with_t_columns(frames: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {name: frame.copy() for name, frame in frames.items()}
    metric = out["condition_metric_summary.tsv"]
    for column in NEW_METRIC:
        metric[column] = rng.normal(size=len(metric))
    metric["peer_config_fingerprint"] = "t-run"
    deg = out["condition_deg_metric_summary.tsv"]
    for column in NEW_DEG:
        deg[column] = rng.normal(size=len(deg))
    return out


def combine(*parts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: pd.concat([part[name] for part in parts], ignore_index=True) for name in merge.SUMMARY_FILES}


class MergeReplicateTColumnsTest(unittest.TestCase):
    def run_merge(self, base, runs, checks=()) -> dict[str, pd.DataFrame]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def write(name: str, frames: dict[str, pd.DataFrame]) -> str:
                (root / name).mkdir()
                for filename, frame in frames.items():
                    frame.to_csv(root / name / filename, sep="\t", index=False)
                return str(root / name)

            argv = ["merge", "--base-dir", write("base", base), "--output-dir", str(root / "out"),
                    "--run-dirs", *[write(f"run{i}", run) for i, run in enumerate(runs)],
                    "--check-dirs", *[write(f"check{i}", check) for i, check in enumerate(checks)]]
            with mock.patch.object(sys, "argv", argv):
                merge.main()
            return {name: pd.read_csv(root / "out" / name, sep="\t") for name in merge.SUMMARY_FILES}

    def test_joins_new_columns_into_both_summaries(self):
        a, b = condition_frames("a", 5, 1), condition_frames("b", 4, 2)
        base = combine(a, b)
        out = self.run_merge(base, [with_t_columns(a, 3), with_t_columns(b, 4)])
        metric, deg = out["condition_metric_summary.tsv"], out["condition_deg_metric_summary.tsv"]
        self.assertEqual(len(metric), 9)
        pd.testing.assert_series_equal(
            metric["mean_replicate_spearman_t"], base["condition_metric_summary.tsv"]["mean_replicate_spearman_t"]
        )
        self.assertTrue(metric[NEW_METRIC].notna().all().all())
        self.assertTrue(deg[NEW_DEG].notna().all().all())
        self.assertEqual(metric["peer_config_fingerprint"].unique().tolist(), ["base"])

    def test_refuses_a_disagreeing_shared_column(self):
        a = condition_frames("a", 5, 1)
        run = with_t_columns(a, 3)
        run["condition_deg_metric_summary.tsv"].loc[2, "mean_replicate_deg_lfc_spearman_sym_p05"] += 1e-9
        with self.assertRaises(SystemExit):
            self.run_merge(a, [run])

    def test_tolerates_rounding_and_tie_sensitive_columns(self):
        a = condition_frames("a", 5, 1)
        run = with_t_columns(a, 3)
        run["condition_metric_summary.tsv"]["mean_replicate_spearman_t"] += 1e-15
        run["condition_metric_summary.tsv"].loc[0, "mean_replicate_signed_overlap_t_top50"] += 0.02
        out = self.run_merge(a, [run])
        self.assertEqual(len(out["condition_metric_summary.tsv"]), 5)

    def test_refuses_mismatched_condition_sets(self):
        a = condition_frames("a", 5, 1)
        run = {name: frame.iloc[:4] for name, frame in with_t_columns(a, 3).items()}
        with self.assertRaises(SystemExit):
            self.run_merge(a, [run])

    def test_dataset_in_two_runs_uses_the_run_matching_the_base(self):
        a = condition_frames("a", 5, 1)
        matching = with_t_columns(a, 3)
        other = {name: frame.copy() for name, frame in matching.items()}
        other["condition_metric_summary.tsv"]["n_global_shared_genes"] = 90  # other run's gene set
        other["condition_metric_summary.tsv"]["mean_replicate_spearman_logfc_global"] += 0.5
        out = self.run_merge(a, [other, matching])
        np.testing.assert_allclose(
            out["condition_metric_summary.tsv"]["mean_replicate_spearman_logfc_global"],
            a["condition_metric_summary.tsv"]["mean_replicate_spearman_logfc_global"],
        )

    def test_refuses_runs_disagreeing_on_new_columns(self):
        a = condition_frames("a", 5, 1)
        with self.assertRaises(SystemExit):
            self.run_merge(a, [with_t_columns(a, 3), with_t_columns(a, 4)])

    def test_refuses_a_result_that_changes_an_earlier_merge(self):
        a = condition_frames("a", 5, 1)
        run = with_t_columns(a, 3)
        earlier = with_t_columns(a, 3)
        self.run_merge(a, [run], checks=[earlier])  # identical: accepted
        earlier["condition_metric_summary.tsv"].loc[1, "mean_peer_baseline_spearman_t"] += 0.1
        with self.assertRaises(SystemExit):
            self.run_merge(a, [run], checks=[earlier])


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
