import ast
import json
import unittest
from pathlib import Path

import pandas as pd

from scripts.cross_source_core import (
    global_gene_scope_lines,
    matched_dataset_lines,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PROFILES = {
    REPO_ROOT
    / "notebooks"
    / "overlap_group_rep_deg_metrics_reviewer_additions.ipynb": "deg",
    REPO_ROOT
    / "notebooks"
    / "overlap_group_rep_signature_similarity.ipynb": "signature",
    REPO_ROOT
    / "notebooks"
    / "overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb": "retrieval",
}


def notebook_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]


def notebook_code(path: Path) -> str:
    return "\n\n".join(notebook_cells(path))


def literal_assignment(code: str, variable_name: str):
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == variable_name:
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                continue
    raise AssertionError(f"Notebook code has no literal assignment {variable_name!r}")


def cached_frame_calls(code: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cached_frame"
    ]


def locally_defined_names(code: str) -> set[str]:
    return {
        node.name
        for node in ast.parse(code).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


class CrossNotebookScopeTests(unittest.TestCase):
    def test_matched_and_global_gene_line_scopes_are_distinct(self):
        available_lines = {
            "A": ["line_shared", "line_a_only", "line_unused_shared"],
            "B": ["line_shared", "line_b_only", "line_unused_shared"],
            "C": ["line_shared", "line_c_only"],
        }
        matched_pairs = pd.DataFrame(
            {
                "dataset_a": ["A"],
                "dataset_b": ["B"],
                "cell_type": ["line_shared"],
            }
        )

        matched = matched_dataset_lines(matched_pairs, ["A", "B", "C"])
        global_gene = global_gene_scope_lines(
            available_lines,
            matched_pairs,
            ["A", "B", "C"],
        )

        self.assertEqual(
            matched,
            {"A": ["line_shared"], "B": ["line_shared"], "C": []},
        )
        self.assertEqual(
            global_gene,
            {
                "A": ["line_shared"],
                "B": ["line_shared"],
                "C": ["line_shared"],
            },
        )

    def test_all_notebooks_use_the_shared_live_execution_seam(self):
        shared_main_fragments = (
            "OUTPUT_DIR = cross_source_core.analysis_output_dir(",
            "cross_source_core.prepare_cross_source_scope(",
            "matched_pairs = cross_source_scope.matched_pairs",
            "MATCHED_PAIRS_FINGERPRINT = cross_source_scope.matched_pairs_fingerprint",
            "matched_lines = cross_source_scope.matched_lines",
            "matched_active_datasets = cross_source_scope.matched_active_datasets",
            "global_gene_lines = cross_source_scope.global_gene_lines",
            "line_global_gene_keys = cross_source_scope.line_global_gene_keys",
        )
        shared_catalog_fragments = (
            "SOURCE_CATALOG = cross_source_core.LineSourceCatalog(SOURCE_DATASET_DIRS)",
            "LineSource = cross_source_core.LineSource",
            "resolve_line_path = SOURCE_CATALOG.resolve_line_path",
            "get_line_source = SOURCE_CATALOG.get_line_source",
            "shared_gene_positions = SOURCE_CATALOG.shared_gene_positions",
            "set_global_shared_gene_keys = SOURCE_CATALOG.set_global_shared_gene_keys",
            "global_gene_positions = SOURCE_CATALOG.global_gene_positions",
        )

        for path, profile in NOTEBOOK_PROFILES.items():
            with self.subTest(notebook=path.name):
                cells = notebook_cells(path)
                code = "\n\n".join(cells)
                ast.parse(code, filename=str(path))
                self.assertNotIn("scripts.cross_source_scope", code)
                self.assertEqual(
                    literal_assignment(code, "ANALYSIS_PROFILE"),
                    profile,
                )
                for fragment in (*shared_main_fragments, *shared_catalog_fragments):
                    self.assertIn(fragment, code)

                main_cell = next(
                    cell
                    for cell in cells
                    if "cross_source_core.prepare_cross_source_scope" in cell
                )
                self.assertNotIn("pair_match_frames", main_cell)
                self.assertNotIn("cached_frame(", main_cell)
                self.assertNotIn("set_global_shared_gene_keys(", main_cell)

        main_cells = [
            next(
                cell
                for cell in notebook_cells(path)
                if "cross_source_core.prepare_cross_source_scope" in cell
            )
            for path in NOTEBOOK_PROFILES
        ]
        self.assertEqual(main_cells[1:], main_cells[:-1])

    def test_notebooks_cannot_redefine_shared_core_implementations(self):
        core_owned_names = {
            "LineSource",
            "active_dataset_names",
            "build_dataset_index",
            "build_groups",
            "build_matched_pairs",
            "build_symmetric_pair_metric_matrix",
            "coerce_control_mask",
            "difference_if_both_defined",
            "empty_score_dict",
            "ensure_overlap_frame_schema",
            "filter_finite_pair",
            "format_ci_cell",
            "format_numeric",
            "format_pubchem_cid",
            "global_gene_positions",
            "load_overlap_obs",
            "mean_available",
            "normalize_pubchem_cid_values",
            "pair_match_frame",
            "pretty_label",
            "resolve_line_path",
            "sanitize_string_values",
            "score_signature_pair",
            "set_global_shared_gene_keys",
            "shared_gene_positions",
            "signed_overlap_at_k",
            "signed_spearman",
        }
        for path in NOTEBOOK_PROFILES:
            with self.subTest(notebook=path.name):
                locally_defined = locally_defined_names(notebook_code(path))
                self.assertFalse(
                    core_owned_names & locally_defined,
                    f"{path.name} shadows shared core definitions: "
                    f"{sorted(core_owned_names & locally_defined)}",
                )

    def test_every_dataframe_cache_has_an_explicit_fingerprint(self):
        for path in NOTEBOOK_PROFILES:
            with self.subTest(notebook=path.name):
                calls = cached_frame_calls(notebook_code(path))
                self.assertTrue(calls)
                for call in calls:
                    stage = (
                        call.args[0].value
                        if call.args
                        and isinstance(call.args[0], ast.Constant)
                        else "<dynamic>"
                    )
                    keyword_names = {keyword.arg for keyword in call.keywords}
                    self.assertIn(
                        "fingerprint",
                        keyword_names,
                        f"{path.name} cache stage {stage!r} has no fingerprint",
                    )

    def test_w4_only_precomputes_datasets_that_participate_in_matches(self):
        for path in NOTEBOOK_PROFILES:
            with self.subTest(notebook=path.name):
                code = notebook_code(path)
                self.assertIn(
                    "for dataset_name in matched_active_datasets",
                    code,
                )
                self.assertIn(
                    '"datasets": list(matched_active_datasets)',
                    code,
                )
                self.assertNotIn(
                    'raise ValueError(f"No matched comparison lines for {dataset_name}")',
                    code,
                )

                self.assertIn(
                    "W4_STATS_CATALOG = PopulationStatsCatalog(",
                    code,
                )
                self.assertIn(
                    "w4_dataset_population_source_paths = (\n"
                    "    W4_STATS_CATALOG.dataset_population_source_paths\n"
                    ")",
                    code,
                )
                self.assertNotIn(
                    "def w4_dataset_population_source_paths",
                    code,
                )
                self.assertNotIn(
                    "def get_w4_population_stats",
                    code,
                )

    def test_w4_signature_tables_report_standardized_peer_counts(self):
        expected_tables = {
            "deg": "w4_table4",
            "signature": "w4_table6",
        }
        for path, profile in NOTEBOOK_PROFILES.items():
            if profile not in expected_tables:
                continue
            with self.subTest(notebook=path.name):
                table_name = expected_tables[profile]
                reporting_cell = next(
                    cell
                    for cell in notebook_cells(path)
                    if f"{table_name} = pd.concat(" in cell
                )
                self.assertIn('"mean_peers_source": float(', reporting_cell)
                self.assertIn(
                    'pair_row["mean_left_peer_scored_count"]',
                    reporting_cell,
                )
                self.assertIn(
                    'pair_row["mean_right_peer_scored_count"]',
                    reporting_cell,
                )
                self.assertIn(
                    f'{table_name}["scale_variant"] != "raw_logfc"',
                    reporting_cell,
                )
                self.assertIn(
                    '"mean_peers_source"',
                    reporting_cell,
                )

    def test_source_dependent_primary_metric_caches_inventory_line_files(self):
        expected_fingerprints = {
            "deg": "DEG_METRICS_FINGERPRINT",
            "signature": "SIGNATURE_METRICS_FINGERPRINT",
            "retrieval": "PRIMARY_RETRIEVAL_FINGERPRINT",
        }
        for path, profile in NOTEBOOK_PROFILES.items():
            with self.subTest(notebook=path.name):
                code = notebook_code(path)
                fingerprint_position = code.index(expected_fingerprints[profile])
                inventory_position = code.index(
                    "cross_source_core.line_source_inventory(",
                    fingerprint_position,
                )
                self.assertGreater(inventory_position, fingerprint_position)

    def test_ci_caches_track_exact_upstream_stage_and_bootstrap_policy(self):
        expected_fragments = {
            "deg": (
                '"upstream_fingerprint": DEG_METRICS_FINGERPRINT',
                "upstream_fingerprint=W4_DEG_METRICS_FINGERPRINT",
                "n_boot=BOOTSTRAP_ITERATIONS",
                "seed=BOOTSTRAP_RANDOM_SEED",
            ),
            "signature": (
                "upstream_fingerprint=W4_SIGNATURE_METRICS_FINGERPRINT",
                "n_boot=BOOTSTRAP_ITERATIONS",
                "seed=BOOTSTRAP_RANDOM_SEED",
            ),
            "retrieval": (
                "upstream_fingerprint=W4_RETRIEVAL_METRICS_FINGERPRINT",
                "retrieval-ablation-ci-v4",
                "retrieval-null-ci-v3",
                "n_boot=BOOTSTRAP_ITERATIONS",
                "seed=BOOTSTRAP_RANDOM_SEED",
            ),
        }
        for path, profile in NOTEBOOK_PROFILES.items():
            with self.subTest(notebook=path.name):
                code = notebook_code(path)
                for fragment in expected_fragments[profile]:
                    self.assertIn(fragment, code)

    def test_peer_membership_and_retrieval_null_helpers_are_order_independent(self):
        deg_code = notebook_code(
            next(path for path, profile in NOTEBOOK_PROFILES.items() if profile == "deg")
        )
        signature_code = notebook_code(
            next(
                path
                for path, profile in NOTEBOOK_PROFILES.items()
                if profile == "signature"
            )
        )
        for code in (deg_code, signature_code):
            self.assertIn(
                "cross_source_core.select_line_source_peers(",
                code,
            )
            self.assertNotIn("def baseline_peer_stratum", code)

        retrieval_code = notebook_code(
            next(
                path
                for path, profile in NOTEBOOK_PROFILES.items()
                if profile == "retrieval"
            )
        )
        self.assertNotIn(
            "def exact_cross_assay_target_decoy_null",
            retrieval_code,
        )
        self.assertIn(
            "def ablation_exact_cross_assay_target_decoy_null",
            retrieval_code,
        )
        self.assertIn(
            "def focused_exact_cross_assay_target_decoy_null",
            retrieval_code,
        )


if __name__ == "__main__":
    unittest.main()
