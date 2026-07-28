from __future__ import annotations

import tempfile
import time
from pathlib import Path
import unittest

import pandas as pd

from scripts.replicate_reviewer_tables import (
    TABLE_METRICS,
    build_presentation_table,
    build_table_ci,
    run_table,
)


def synthetic_frame(table: int) -> pd.DataFrame:
    rows = []
    for dataset_index, dataset in enumerate(("dataset_a", "dataset_b")):
        for compound_index in range(6):
            row = {
                "dataset_name": dataset,
                "pubchem_cid": str(1000 + compound_index),
            }
            for metric_index, column in enumerate(
                TABLE_METRICS[table].values()
            ):
                row[column] = (
                    0.05 * dataset_index
                    + 0.01 * compound_index
                    + 0.001 * metric_index
                )
            rows.append(row)
    return pd.DataFrame(rows)


class ReplicateReviewerTableTests(unittest.TestCase):
    def test_each_table_builds_primary_presentation(self) -> None:
        for table in (7, 8, 10):
            with self.subTest(table=table):
                ci = build_table_ci(
                    synthetic_frame(table),
                    table=table,
                    bootstrap_iterations=40,
                    workers=1,
                )
                presentation = build_presentation_table(ci, table=table)
                self.assertEqual(len(ci), 2 * len(TABLE_METRICS[table]))
                self.assertEqual(
                    presentation["dataset_name"].tolist(),
                    ["dataset_a", "dataset_b"],
                )
                self.assertTrue(
                    presentation.iloc[0, 1].startswith("0.025 [")
                )

    def test_one_and_two_workers_are_byte_identical_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            frame = synthetic_frame(7)
            frame.to_csv(
                input_dir / "condition_deg_metric_summary.tsv",
                sep="\t",
                index=False,
            )
            output_one = root / "one"
            output_two = root / "two"
            common = {
                "table": 7,
                "input_dir": input_dir,
                "bootstrap_iterations": 60,
                "bootstrap_seed": 20260505,
                "bootstrap_batch_size": 16,
                "progress": False,
                "force": False,
            }
            paths_one = run_table(
                **common,
                output_dir=output_one,
                workers=1,
            )
            paths_two = run_table(
                **common,
                output_dir=output_two,
                workers=2,
            )
            for name in ("ci", "presentation", "ranges"):
                self.assertEqual(
                    paths_one[name].read_bytes(),
                    paths_two[name].read_bytes(),
                )

            original_mtime = paths_two["ci"].stat().st_mtime_ns
            time.sleep(0.01)
            run_table(
                **common,
                output_dir=output_two,
                workers=2,
            )
            self.assertEqual(
                original_mtime,
                paths_two["ci"].stat().st_mtime_ns,
            )

    def test_missing_peer_column_fails_with_actionable_message(self) -> None:
        frame = synthetic_frame(10).drop(
            columns="mean_peer_baseline_spearman_logfc"
        )
        with tempfile.TemporaryDirectory() as temporary:
            input_dir = Path(temporary)
            frame.to_csv(
                input_dir / "condition_metric_summary.tsv",
                sep="\t",
                index=False,
            )
            with self.assertRaisesRegex(
                KeyError,
                "Rerun precompute_replicate_signature_similarity.py",
            ):
                run_table(
                    table=10,
                    input_dir=input_dir,
                    output_dir=input_dir / "output",
                    bootstrap_iterations=20,
                    bootstrap_seed=1,
                    workers=1,
                    bootstrap_batch_size=8,
                    progress=False,
                    force=False,
                )


if __name__ == "__main__":
    unittest.main()
