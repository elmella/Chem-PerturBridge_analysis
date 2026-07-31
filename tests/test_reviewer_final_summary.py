from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.summarize_reviewer_final_tables import (
    TABLE5_METRICS,
    build_table5_ci,
    _complete_dose_ci,
    _complete_dose_pair_summary,
)
from scripts.summarize_reviewer_minimal_metrics import (
    DOSE_METRICS,
    W4_SIGNATURE_METRICS,
    _w4_long_frame,
    build_retrieval_ci,
)
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
)


class ReviewerFinalSummaryTests(unittest.TestCase):
    def test_population_summary_accepts_dataset_cell_type_scale(self):
        metric = W4_SIGNATURE_METRICS[0]
        prefix = f"{PER_GENE_DATASET_CELL_TYPE_VARIANT}__"
        frame = pd.DataFrame(
            [
                {
                    "dataset_a": "a",
                    "dataset_b": "b",
                    "cell_type": "cell",
                    "time_key": "24",
                    "pubchem_cid": "1",
                    "matched_condition_key": "condition",
                    "left_obs_id": "left",
                    "right_obs_id": "right",
                    f"{prefix}{metric}": 0.25,
                }
            ]
        )

        result, metrics = _w4_long_frame(
            frame,
            metric_candidates=(metric,),
            required_metrics=(metric,),
            label="dataset-cell-type signature",
            scale_variant=PER_GENE_DATASET_CELL_TYPE_VARIANT,
        )

        self.assertEqual(metrics, [metric])
        self.assertEqual(
            set(result["scale_variant"]),
            {PER_GENE_DATASET_CELL_TYPE_VARIANT},
        )
        self.assertEqual(float(result.loc[0, metric]), 0.25)

    def test_table5_summarizes_direction_agreement_and_peer_baselines(self):
        rows = []
        for index in range(6):
            row = {
                "dataset_a": "a",
                "dataset_b": "b",
                "cell_type": "cell",
                "time_key": "24",
                "pubchem_cid": str(100 + index),
                "matched_condition_key": f"condition-{index}",
                "left_obs_id": f"left-{index}",
            }
            row.update(
                {
                    metric: 0.5 + (0.01 * index)
                    for metric in TABLE5_METRICS
                }
            )
            rows.append(row)

        ci = build_table5_ci(
            pd.DataFrame(rows),
            n_boot=20,
            seed=20260505,
            workers=1,
            progress=False,
        )

        self.assertEqual(set(ci["metric"]), set(TABLE5_METRICS))
        self.assertEqual(set(ci["dataset_a"]), {"a"})
        self.assertEqual(set(ci["dataset_b"]), {"b"})
        self.assertEqual(set(ci["ci_status"]), {"ok"})

    def test_l2_retrieval_ci_accepts_explicit_similarity_and_scale(self):
        rows = []
        for direction in ("A_to_B", "B_to_A"):
            for index in range(6):
                value = 0.4 + 0.05 * index
                rows.append(
                    {
                        "dataset_a": "dataset_a",
                        "dataset_b": "dataset_b",
                        "direction": direction,
                        "cell_type": "cell",
                        "time_key": "24",
                        "query_pubchem_cid": str(100 + index),
                        "representation": "logFC",
                        "retrieval_variant": "strict_matched_condition",
                        "similarity_metric": "negative_l2",
                        "scale_variant": "raw",
                        "observed_normalized_best_positive_rank": value,
                        "source_individual_corrected_percentile": value - 0.1,
                        "target_individual_corrected_percentile": value + 0.1,
                    }
                )
        ci = build_retrieval_ci(
            pd.DataFrame(rows),
            n_boot=20,
            seed=20260505,
            similarity_metrics=("negative_l2",),
            scale_variants=("raw",),
            summary_level="test_l2",
            workers=1,
            progress=False,
        )
        self.assertEqual(
            set(ci["similarity_metric"].astype(str)),
            {"negative_l2"},
        )
        self.assertEqual(set(ci["scale_variant"].astype(str)), {"raw"})
        self.assertEqual(set(ci["ci_status"].astype(str)), {"ok"})

    def test_zero_match_dose_pairs_are_explicitly_nonestimable(self):
        summary = pd.DataFrame(
            [
                {
                    "dose_threshold": "exact",
                    "threshold_order": 0,
                    "max_fold_difference": 1.0,
                    "max_abs_delta_log10_dose": 0.0,
                    "is_reference": False,
                    "dataset_a": "a",
                    "dataset_b": "b",
                    "n_matched_sample_pairs": 2,
                    "n_matching_conditions": 1,
                    "n_matching_drugs": 1,
                    "n_eligible_contexts": 1,
                    "n_scored_sample_pairs": 2,
                    "n_scored_drug_line_times": 1,
                    **{f"mean_{metric}": 0.2 for metric in DOSE_METRICS},
                },
                {
                    "dose_threshold": "10x_reference",
                    "threshold_order": 3,
                    "max_fold_difference": 10.0,
                    "max_abs_delta_log10_dose": 1.0,
                    "is_reference": True,
                    "dataset_a": "a",
                    "dataset_b": "b",
                    "n_matched_sample_pairs": 4,
                    "n_matching_conditions": 2,
                    "n_matching_drugs": 2,
                    "n_eligible_contexts": 1,
                    "n_scored_sample_pairs": 4,
                    "n_scored_drug_line_times": 2,
                    **{f"mean_{metric}": 0.3 for metric in DOSE_METRICS},
                },
                {
                    "dose_threshold": "10x_reference",
                    "threshold_order": 3,
                    "max_fold_difference": 10.0,
                    "max_abs_delta_log10_dose": 1.0,
                    "is_reference": True,
                    "dataset_a": "a",
                    "dataset_b": "c",
                    "n_matched_sample_pairs": 3,
                    "n_matching_conditions": 2,
                    "n_matching_drugs": 2,
                    "n_eligible_contexts": 1,
                    "n_scored_sample_pairs": 3,
                    "n_scored_drug_line_times": 2,
                    **{f"mean_{metric}": 0.4 for metric in DOSE_METRICS},
                },
            ]
        )
        completed = _complete_dose_pair_summary(summary)
        missing = completed.loc[
            (completed["dose_threshold"] == "exact")
            & (completed["dataset_b"] == "c")
        ].iloc[0]
        self.assertEqual(int(missing["n_matched_sample_pairs"]), 0)
        self.assertTrue(
            np.isnan(
                missing[
                    "mean_observed_deg_lfc_spearman_sym_p05"
                ]
            )
        )

        ci_columns = [
            "dose_threshold",
            "threshold_order",
            "max_fold_difference",
            "max_abs_delta_log10_dose",
            "is_reference",
            "dataset_a",
            "dataset_b",
            "summary_level",
            "metric",
            "value_col",
            "mean",
            "ci_low",
            "ci_high",
            "ci_half_width",
            "ci_method",
            "ci_status",
            "ci_level",
            "n_bootstrap_iterations",
            "n_bootstrap_valid",
            "n_rows",
            "n_finite_rows",
            "n_compounds",
            "cluster_col",
            "inner_strata",
            "outer_strata",
            "uncertainty_scope",
            "cell_type",
        ]
        ci = pd.DataFrame(
            [
                {
                    column: (
                        "exact"
                        if column == "dose_threshold"
                        else "a"
                        if column == "dataset_a"
                        else "b"
                        if column == "dataset_b"
                        else "dose_threshold_dataset_pair"
                        if column == "summary_level"
                        else DOSE_METRICS[0]
                        if column == "metric"
                        else "ok"
                        if column == "ci_status"
                        else np.nan
                    )
                    for column in ci_columns
                }
            ]
        )
        completed_ci = _complete_dose_ci(ci, completed)
        added = completed_ci.loc[
            (completed_ci["dose_threshold"] == "exact")
            & (completed_ci["dataset_b"] == "c")
        ]
        self.assertEqual(len(added), len(DOSE_METRICS))
        self.assertEqual(
            set(added["ci_status"].astype(str)),
            {"no_matched_pairs"},
        )
        self.assertTrue(added["mean"].isna().all())


if __name__ == "__main__":
    unittest.main()
