import ast
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.cross_source_strata import SignatureStratum


REPO_ROOT = Path(__file__).resolve().parents[1]
LINE_SOURCE_NOTEBOOKS = (
    REPO_ROOT / "notebooks" / "overlap_group_rep_signature_similarity.ipynb",
    REPO_ROOT
    / "notebooks"
    / "overlap_group_rep_deg_metrics_reviewer_additions.ipynb",
    REPO_ROOT
    / "notebooks"
    / "overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb",
)


def coerce_control_mask(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("").astype(str).str.lower()
    return normalized.isin({"true", "1", "yes"})


def normalize_pubchem_cid_values(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").astype(str).str.strip()


def format_numeric(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def load_notebook_line_source(path: Path):
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class LineSource:" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LineSource"
        )
        namespace = {
            "SignatureStratum": SignatureStratum,
            "Path": Path,
            "ad": ad,
            "coerce_control_mask": coerce_control_mask,
            "dataclass": dataclass,
            "field": field,
            "format_numeric": format_numeric,
            "normalize_pubchem_cid_values": normalize_pubchem_cid_values,
            "np": np,
            "pd": pd,
        }
        module = ast.Module(body=[class_node], type_ignores=[])
        exec(compile(module, str(path), "exec"), namespace)
        return namespace["LineSource"]
    raise AssertionError(f"No LineSource class found in {path}")


def write_duplicate_compound_fixture(path: Path) -> None:
    obs = pd.DataFrame(
        {
            "is_control": ["FALSE"] * 4,
            "pubchem_cid": ["A", "A", "B", "C"],
            "cell_type": ["CVCL_TEST"] * 4,
            "pert_time_h": [24.0] * 4,
            "pert_dose_uM": [0.05] * 4,
        },
        index=["row_a1", "row_a2", "row_b", "row_c"],
    )
    var = pd.DataFrame(
        {"symbol": ["G1", "G2"]},
        index=["ENSG1", "ENSG2"],
    )
    values = np.asarray(
        [
            [10.0, 20.0],
            [11.0, 21.0],
            [1.0, 3.0],
            [5.0, 7.0],
        ],
        dtype=np.float32,
    )
    ad.AnnData(
        X=np.zeros_like(values),
        obs=obs,
        var=var,
        layers={"logFC": values},
    ).write_h5ad(path)


class NotebookLineSourceTests(unittest.TestCase):
    def test_centroid_cache_hit_retains_peer_count_for_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "CVCL_TEST_de.h5ad"
            write_duplicate_compound_fixture(fixture_path)

            for notebook_path in LINE_SOURCE_NOTEBOOKS:
                with self.subTest(notebook=notebook_path.name):
                    line_source_type = load_notebook_line_source(notebook_path)
                    source = line_source_type(
                        dataset_name="source",
                        cell_type="CVCL_TEST",
                        path=fixture_path,
                    )
                    try:
                        first = source.get_baseline_vector("row_a1", "logFC")
                        second = source.get_baseline_vector("row_a2", "logFC")
                        np.testing.assert_allclose(first, [3.0, 5.0])
                        np.testing.assert_allclose(second, first)
                        self.assertEqual(
                            source.baseline_peer_count("row_a1"),
                            2,
                        )
                        self.assertEqual(
                            source.baseline_peer_count("row_a2"),
                            2,
                        )
                    finally:
                        source.close()


if __name__ == "__main__":
    unittest.main()
