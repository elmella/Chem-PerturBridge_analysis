from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.cross_source_parallel import sha256_file
from scripts.population_zscore import (
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
    load_or_fit_dataset_population_stats_from_source_stats,
    load_or_fit_population_stats,
)
from scripts.run_overlap_group_rep_deg_metrics import (
    FINAL_METRICS_NAME as DEG_FINAL,
)
from scripts.run_overlap_group_rep_deg_metrics import main as run_deg
from scripts.run_overlap_group_rep_retrieval_metrics import (
    FINAL_METRICS_NAME as RETRIEVAL_FINAL,
)
from scripts.run_overlap_group_rep_retrieval_metrics import (
    IDENTITY_COLUMNS as RETRIEVAL_IDENTITY_COLUMNS,
)
from scripts.run_overlap_group_rep_retrieval_metrics import main as run_retrieval
from scripts.run_overlap_group_rep_signature_similarity import (
    FINAL_METRICS_NAME as SIGNATURE_FINAL,
)
from scripts.run_overlap_group_rep_signature_similarity import (
    main as run_signature,
)
from scripts.summarize_reviewer_minimal_metrics import (
    DOSE_METRICS,
    _dose_coverage,
    _dose_metric_summaries,
    main as summarize_reviewer_metrics,
)


DATASETS = ("tahoe", "sciplex")
CELL_TYPES = ("CVCL_TEST_A", "CVCL_TEST_B")
N_COMPOUNDS = 10
N_GENES = 60
REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_directory(data_root: Path, dataset_name: str) -> Path:
    return data_root / dataset_name / "group_rep"


def _fixture_obs(dataset_name: str, cell_type: str) -> pd.DataFrame:
    obs_ids = [
        f"{dataset_name}_{cell_type}_{index:02d}"
        for index in range(N_COMPOUNDS)
    ]
    return pd.DataFrame(
        {
            "is_control": [False] * N_COMPOUNDS,
            "pubchem_cid": [str(10_000 + index) for index in range(N_COMPOUNDS)],
            "cell_type": [cell_type] * N_COMPOUNDS,
            "pert_time_h": [24.0] * N_COMPOUNDS,
            "pert_dose_uM": [10.0] * N_COMPOUNDS,
            "plate": [f"{dataset_name}_plate"] * N_COMPOUNDS,
            "well": [f"A{index + 1:02d}" for index in range(N_COMPOUNDS)],
        },
        index=obs_ids,
    )


def _write_source(
    path: Path,
    *,
    dataset_index: int,
    cell_index: int,
    obs: pd.DataFrame,
    logfc_only: bool = False,
) -> None:
    row = np.arange(1, N_COMPOUNDS + 1, dtype=np.float64)[:, None]
    gene = np.arange(1, N_GENES + 1, dtype=np.float64)[None, :]
    phase = 0.19 * dataset_index + 0.11 * cell_index
    logfc = np.sin(row * gene * 0.071 + phase) + 0.025 * row
    t_values = 2.5 * logfc + np.cos(gene * 0.13 + row * 0.17 + phase)
    adj_p = np.where(np.abs(logfc) > 0.42, 0.01, 0.2)
    var = pd.DataFrame(
        {"symbol": [f"G{index:03d}" for index in range(N_GENES)]},
        index=[f"ENSG{index:05d}" for index in range(N_GENES)],
    )
    layers = {"logFC": logfc.astype(np.float32)}
    if not logfc_only:
        layers.update(
            {
                "t": t_values.astype(np.float32),
                "adj.P.Value.within_one_contrast": adj_p.astype(np.float32),
            }
        )
    adata = ad.AnnData(
        X=np.zeros((N_COMPOUNDS, N_GENES), dtype=np.float32),
        obs=obs.copy(),
        var=var,
        layers=layers,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def build_fixture(
    root: Path,
    *,
    precompute_w4: bool = True,
    logfc_only: bool = False,
) -> tuple[Path, Path, Path, list[Path]]:
    data_root = root / "data"
    overlap_dir = root / "overlap"
    w4_root = root / "w4"
    overlap_dir.mkdir(parents=True)
    source_paths: list[Path] = []
    for dataset_index, dataset_name in enumerate(DATASETS):
        overlap_frames = []
        source_stats = []
        for cell_index, cell_type in enumerate(CELL_TYPES):
            obs = _fixture_obs(dataset_name, cell_type)
            overlap_frames.append(obs)
            source_path = (
                _source_directory(data_root, dataset_name)
                / f"{cell_type}_de.h5ad"
            )
            _write_source(
                source_path,
                dataset_index=dataset_index,
                cell_index=cell_index,
                obs=obs,
                logfc_only=logfc_only,
            )
            source_paths.append(source_path)
            source_stats.append(
                load_or_fit_population_stats(
                    source_path=source_path,
                    dataset_name=dataset_name,
                    cell_type=cell_type,
                    cache_root=w4_root,
                    verbose=False,
                )
            )
        if precompute_w4:
            load_or_fit_dataset_population_stats_from_source_stats(
                source_stats=source_stats,
                dataset_name=dataset_name,
                cache_root=w4_root,
                verbose=False,
            )
        overlap_obs = pd.concat(overlap_frames, axis=0)
        overlap = ad.AnnData(
            X=np.zeros((len(overlap_obs), 1), dtype=np.float32),
            obs=overlap_obs,
            var=pd.DataFrame(index=["placeholder"]),
        )
        overlap.write_h5ad(
            overlap_dir / f"{dataset_name}_overlap_filtered.h5ad"
        )
    return data_root, overlap_dir, w4_root, source_paths


def _common_arguments(
    *,
    data_root: Path,
    overlap_dir: Path,
    w4_root: Path,
    output_dir: Path,
    workers: int,
    max_baseline_peers: int = 8,
    workload: str | None = None,
) -> list[str]:
    arguments = [
        "--datasets",
        ",".join(DATASETS),
        "--data-root",
        str(data_root),
        "--overlap-dir",
        str(overlap_dir),
        "--w4-stats-root",
        str(w4_root),
        "--output-dir",
        str(output_dir),
        "--run-tag",
        "fixture",
        "--workers",
        str(workers),
        "--rows-per-shard",
        "7",
        "--max-baseline-peers",
        str(max_baseline_peers),
    ]
    if workload is not None:
        arguments.extend(["--workload", workload])
    return arguments


class ParallelScoringScriptTests(unittest.TestCase):
    def test_empty_dose_scores_keep_merge_key_schema(self):
        drug, pair, line = _dose_metric_summaries(pd.DataFrame())
        self.assertTrue(drug.empty)
        pair_coverage = pd.DataFrame(
            {
                "dataset_a": ["tahoe"],
                "dataset_b": ["sciplex"],
                "n_matched_sample_pairs": [10],
            }
        )
        merged = pair_coverage.merge(
            pair.drop(
                columns="n_scored_sample_pairs",
                errors="ignore",
            ),
            on=["dataset_a", "dataset_b"],
            how="left",
            validate="one_to_one",
        )
        self.assertEqual(len(merged), 1)
        self.assertTrue(
            {
                f"mean_{metric}" for metric in DOSE_METRICS
            }.issubset(merged.columns)
        )
        self.assertTrue(
            {"dataset_a", "dataset_b", "cell_type"}.issubset(line.columns)
        )
        empty_pair_coverage, empty_line_coverage = _dose_coverage(
            pd.DataFrame(),
            pd.DataFrame(),
        )
        for coverage in (empty_pair_coverage, empty_line_coverage):
            self.assertIn("n_matched_sample_pairs", coverage.columns)
            self.assertIn("n_eligible_contexts", coverage.columns)

    def test_summary_cli_can_be_launched_by_path(self):
        script_path = REPO_ROOT / "scripts" / "summarize_reviewer_minimal_metrics.py"
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                cwd=temporary_directory,
                capture_output=True,
                check=False,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--retrieval-metrics", completed.stdout)

    def test_reviewer_minimal_summaries_cover_w1_w3_and_w4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, _ = build_fixture(root)
            scoring_root = root / "scoring"
            common = {
                "data_root": data_root,
                "overlap_dir": overlap_dir,
                "w4_root": w4_root,
                "workers": 1,
            }
            deg_output = scoring_root / "deg"
            signature_output = scoring_root / "signature"
            retrieval_output = scoring_root / "retrieval"
            self.assertEqual(
                run_deg(
                    [
                        *_common_arguments(
                            output_dir=deg_output,
                            **common,
                        ),
                        "--w4-scales",
                        "dataset",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_signature(
                    [
                        *_common_arguments(
                            output_dir=signature_output,
                            **common,
                        ),
                        "--w4-scales",
                        "dataset",
                    ]
                ),
                0,
            )
            self.assertEqual(
                run_retrieval(
                    _common_arguments(
                        output_dir=retrieval_output,
                        workload="reviewer-minimal",
                        **common,
                    )
                ),
                0,
            )

            summary_output = root / "summary"
            summary_arguments = [
                "--deg-metrics",
                str(deg_output / "fixture" / DEG_FINAL),
                "--signature-metrics",
                str(
                    signature_output
                    / "fixture"
                    / SIGNATURE_FINAL
                ),
                "--retrieval-metrics",
                str(
                    retrieval_output
                    / "fixture"
                    / RETRIEVAL_FINAL
                ),
                "--overlap-dir",
                str(overlap_dir),
                "--datasets",
                ",".join(DATASETS),
                "--output-dir",
                str(summary_output),
                "--bootstrap-iterations",
                "50",
                "--progress",
                "off",
            ]
            self.assertEqual(
                summarize_reviewer_metrics(summary_arguments),
                0,
            )
            expected_outputs = (
                "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
                "w4_deg_cluster_bca_ci.tsv",
                "w4_signature_cluster_bca_ci.tsv",
                "dose_threshold_deg_metric_summary.tsv",
                "dose_threshold_deg_metric_line_summary.tsv",
                "dose_threshold_deg_cluster_bca_ci.tsv",
                "dose_threshold_overall_matched_pair_counts.tsv",
                "dose_mismatch_distribution.tsv",
                "run_metadata.json",
            )
            for filename in expected_outputs:
                self.assertTrue((summary_output / filename).is_file())

            retrieval_ci = pd.read_csv(
                summary_output
                / "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
                sep="\t",
            )
            self.assertEqual(
                set(retrieval_ci["similarity_metric"].astype(str)),
                {"cosine", "spearman"},
            )
            self.assertEqual(
                set(retrieval_ci["scale_variant"].astype(str)),
                {"raw", PER_GENE_DATASET_VARIANT},
            )
            self.assertTrue(
                {
                    "source_individual_n_peers",
                    "source_individual_n_below_observed",
                    "target_individual_n_peers",
                    "target_individual_n_below_observed",
                    "random_normalized_best_positive_rank",
                    "random_recall_at_1",
                    "random_auroc",
                }.issubset(set(retrieval_ci["metric"].astype(str)))
            )
            dose_counts = pd.read_csv(
                summary_output
                / "dose_threshold_overall_matched_pair_counts.tsv",
                sep="\t",
            ).sort_values("threshold_order")
            self.assertEqual(
                dose_counts["dose_threshold"].tolist(),
                ["exact", "2x", "3x", "10x_reference"],
            )
            self.assertTrue(
                (
                    np.diff(
                        dose_counts[
                            "n_matched_sample_pairs"
                        ].to_numpy(dtype=float)
                    )
                    >= 0
                ).all()
            )

            markers = sorted(
                (summary_output / "checkpoints").glob(
                    "*/*.complete.json"
                )
            )
            self.assertEqual(len(markers), 3)
            marker_mtimes = {
                path.name: path.stat().st_mtime_ns for path in markers
            }
            self.assertEqual(
                summarize_reviewer_metrics(summary_arguments),
                0,
            )
            self.assertEqual(
                marker_mtimes,
                {
                    path.name: path.stat().st_mtime_ns
                    for path in markers
                },
            )
            progress_log = (
                summary_output / "progress.log"
            ).read_text()
            self.assertIn("stage=retrieval cached", progress_log)
            self.assertIn("stage=deg cached", progress_log)
            self.assertIn("stage=signature cached", progress_log)

            retrieval_marker = next(
                path
                for path in markers
                if path.name == "retrieval.complete.json"
            )
            previous_retrieval_marker_mtime = (
                retrieval_marker.stat().st_mtime_ns
            )
            time.sleep(0.01)
            (
                summary_output
                / "reviewer_minimal_retrieval_cluster_bca_ci.tsv"
            ).write_text("corrupt\n")
            self.assertEqual(
                summarize_reviewer_metrics(summary_arguments),
                0,
            )
            self.assertGreater(
                retrieval_marker.stat().st_mtime_ns,
                previous_retrieval_marker_mtime,
            )
            repaired_retrieval = pd.read_csv(
                summary_output
                / "reviewer_minimal_retrieval_cluster_bca_ci.tsv",
                sep="\t",
            )
            self.assertIn("metric", repaired_retrieval.columns)

    def test_reviewer_minimal_requires_only_logfc_source_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, _ = build_fixture(
                root,
                logfc_only=True,
            )
            output_dir = root / "results"
            self.assertEqual(
                run_retrieval(
                    _common_arguments(
                        data_root=data_root,
                        overlap_dir=overlap_dir,
                        w4_root=w4_root,
                        output_dir=output_dir,
                        workers=1,
                        workload="reviewer-minimal",
                    )
                ),
                0,
            )
            result = pd.read_csv(
                output_dir / "fixture" / RETRIEVAL_FINAL,
                sep="\t",
            )
            self.assertFalse(result.empty)

    def test_reviewer_minimal_retrieval_runs_only_required_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, _ = build_fixture(root)
            output_dir = root / "results" / "reviewer-minimal"
            parallel_output_dir = (
                root / "results" / "reviewer-minimal-parallel"
            )
            self.assertEqual(
                run_retrieval(
                    _common_arguments(
                        data_root=data_root,
                        overlap_dir=overlap_dir,
                        w4_root=w4_root,
                        output_dir=output_dir,
                        workers=1,
                        max_baseline_peers=3,
                        workload="reviewer-minimal",
                    )
                ),
                0,
            )
            self.assertEqual(
                run_retrieval(
                    _common_arguments(
                        data_root=data_root,
                        overlap_dir=overlap_dir,
                        w4_root=w4_root,
                        output_dir=parallel_output_dir,
                        workers=2,
                        max_baseline_peers=3,
                        workload="reviewer-minimal",
                    )
                ),
                0,
            )

            serial_path = output_dir / "fixture" / RETRIEVAL_FINAL
            parallel_path = (
                parallel_output_dir / "fixture" / RETRIEVAL_FINAL
            )
            self.assertEqual(serial_path.read_bytes(), parallel_path.read_bytes())
            result = pd.read_csv(serial_path, sep="\t")
            self.assertFalse(result.empty)
            progress_log = (
                output_dir / "fixture" / "progress.log"
            ).read_text()
            self.assertIn("phase=context_loaded", progress_log)
            self.assertIn(
                "phase=similarity scale=raw metric=cosine",
                progress_log,
            )
            self.assertIn(
                f"phase=similarity scale={PER_GENE_DATASET_VARIANT} "
                "metric=spearman",
                progress_log,
            )
            self.assertIn(
                "phase=query_progress scale=raw metric=cosine "
                "direction=A_to_B processed=10/10",
                progress_log,
            )
            self.assertIn("phase=finished", progress_log)
            self.assertEqual(
                set(result["representation"].astype(str)),
                {"logFC"},
            )
            self.assertEqual(
                set(result["retrieval_variant"].astype(str)),
                {"strict_matched_condition"},
            )
            self.assertEqual(
                set(result["similarity_metric"].astype(str)),
                {"cosine", "spearman"},
            )
            self.assertEqual(
                set(result["scale_variant"].astype(str)),
                {"raw", PER_GENE_DATASET_VARIANT},
            )
            for column in (
                "observed_normalized_best_positive_rank",
                "observed_recall_at_1",
                "observed_auroc",
                "source_individual_mean_similarity",
                "source_individual_sd_similarity",
                "source_individual_n_below_observed",
                "source_individual_corrected_percentile",
                "target_individual_mean_similarity",
                "target_individual_sd_similarity",
                "target_individual_n_below_observed",
                "target_individual_corrected_percentile",
                "source_individual_total_count",
                "source_individual_scored_count",
                "target_individual_total_count",
                "target_individual_scored_count",
                "source_centroid_similarity",
                "target_centroid_similarity",
                "target_decoy_null_normalized_best_positive_rank",
            ):
                self.assertIn(column, result.columns)
            self.assertEqual(
                set(result["source_individual_total_count"].astype(int)),
                {N_COMPOUNDS - 1},
            )
            self.assertEqual(
                set(result["target_individual_total_count"].astype(int)),
                {N_COMPOUNDS - 1},
            )
            self.assertEqual(
                set(result["source_individual_scored_count"].astype(int)),
                {3},
            )
            self.assertEqual(
                set(result["target_individual_scored_count"].astype(int)),
                {3},
            )
            self.assertEqual(
                set(result["source_individual_n_peers"].astype(int)),
                {3},
            )
            self.assertEqual(
                set(result["target_individual_n_peers"].astype(int)),
                {3},
            )

    def test_reviewer_minimal_matches_the_corresponding_full_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, _ = build_fixture(root)
            full_output = root / "results" / "full"
            minimal_output = root / "results" / "minimal"
            for output_dir, workload in (
                (full_output, "full"),
                (minimal_output, "reviewer-minimal"),
            ):
                self.assertEqual(
                    run_retrieval(
                        _common_arguments(
                            data_root=data_root,
                            overlap_dir=overlap_dir,
                            w4_root=w4_root,
                            output_dir=output_dir,
                            workers=1,
                            max_baseline_peers=3,
                            workload=workload,
                        )
                    ),
                    0,
                )

            full = pd.read_csv(
                full_output / "fixture" / RETRIEVAL_FINAL,
                sep="\t",
            )
            minimal = pd.read_csv(
                minimal_output / "fixture" / RETRIEVAL_FINAL,
                sep="\t",
            )
            expected = full.loc[
                (full["representation"] == "logFC")
                & (
                    full["retrieval_variant"]
                    == "strict_matched_condition"
                )
                & full["similarity_metric"].isin(("cosine", "spearman"))
                & full["scale_variant"].isin(
                    ("raw", PER_GENE_DATASET_VARIANT)
                )
            ].copy()
            compared_columns = [
                *RETRIEVAL_IDENTITY_COLUMNS,
                "observed_normalized_best_positive_rank",
                "observed_recall_at_1",
                "observed_auroc",
                "source_individual_mean_similarity",
                "source_individual_sd_similarity",
                "source_individual_corrected_percentile",
                "target_individual_mean_similarity",
                "target_individual_sd_similarity",
                "target_individual_corrected_percentile",
                "source_centroid_similarity",
                "target_centroid_similarity",
                "target_decoy_null_normalized_best_positive_rank",
            ]
            pd.testing.assert_frame_equal(
                minimal[compared_columns].reset_index(drop=True),
                expected[compared_columns].reset_index(drop=True),
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_all_scripts_match_for_one_and_two_workers_without_mutating_inputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, source_paths = build_fixture(root)
            input_paths = source_paths + sorted(overlap_dir.glob("*.h5ad"))
            before = {
                path: (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    sha256_file(path),
                )
                for path in input_paths
            }
            analyses = (
                (run_deg, DEG_FINAL, "deg"),
                (run_signature, SIGNATURE_FINAL, "signature"),
                (run_retrieval, RETRIEVAL_FINAL, "retrieval"),
            )
            for command, final_name, analysis in analyses:
                serial_output = root / "results" / analysis / "serial"
                parallel_output = root / "results" / analysis / "parallel"
                self.assertEqual(
                    command(
                        _common_arguments(
                            data_root=data_root,
                            overlap_dir=overlap_dir,
                            w4_root=w4_root,
                            output_dir=serial_output,
                            workers=1,
                        )
                    ),
                    0,
                )
                self.assertEqual(
                    command(
                        _common_arguments(
                            data_root=data_root,
                            overlap_dir=overlap_dir,
                            w4_root=w4_root,
                            output_dir=parallel_output,
                            workers=2,
                        )
                    ),
                    0,
                )
                serial_path = serial_output / "fixture" / final_name
                parallel_path = parallel_output / "fixture" / final_name
                self.assertEqual(serial_path.read_bytes(), parallel_path.read_bytes())
                result = pd.read_csv(serial_path, sep="\t")
                self.assertFalse(result.empty)
                self.assertFalse(result.duplicated().all())
                if analysis == "deg":
                    self.assertIn(
                        "pb_source_peer_deg_lfc_spearman_pair_p05",
                        result.columns,
                    )
                    for scale_variant in POPULATION_SCALE_VARIANTS:
                        self.assertIn(
                            f"{scale_variant}__"
                            "w4_observed_deg_lfc_spearman_sym_p05",
                            result.columns,
                        )
                if analysis == "signature":
                    self.assertIn(
                        "baseline_pair_mean_spearman_logfc_global",
                        result.columns,
                    )
                    for scale_variant in POPULATION_SCALE_VARIANTS:
                        self.assertIn(
                            f"{scale_variant}__w4_observed_spearman_logfc",
                            result.columns,
                        )
                if analysis == "retrieval":
                    self.assertFalse(
                        result.duplicated(
                            subset=RETRIEVAL_IDENTITY_COLUMNS
                        ).any()
                    )
                    self.assertTrue(
                        {
                            "baseline_normalized_best_positive_rank",
                            "single_signature_baseline_normalized_best_positive_rank",
                            "source_individual_corrected_percentile",
                            "null_pit",
                        }.issubset(result.columns)
                    )
                    self.assertEqual(
                        set(result["scale_variant"].astype(str)),
                        {"raw", *POPULATION_SCALE_VARIANTS},
                    )

            self.assertEqual(
                before,
                {
                    path: (
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                        sha256_file(path),
                    )
                    for path in input_paths
                },
            )

    def test_missing_w4_caches_reports_the_exact_precompute_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root, overlap_dir, w4_root, _ = build_fixture(
                root,
                precompute_w4=False,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "precompute_population_zscore.py",
            ) as raised:
                run_signature(
                    _common_arguments(
                        data_root=data_root,
                        overlap_dir=overlap_dir,
                        w4_root=w4_root,
                        output_dir=root / "results",
                        workers=1,
                    )
                )
            message = str(raised.exception)
            for dataset_name in DATASETS:
                self.assertIn(
                    f"--dataset-dir {dataset_name}="
                    f"{_source_directory(data_root, dataset_name).resolve()}",
                    message,
                )


if __name__ == "__main__":
    unittest.main()
