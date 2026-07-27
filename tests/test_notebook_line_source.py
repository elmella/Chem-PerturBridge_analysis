import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.cross_source_core import (
    LineSource,
    first_available_layer,
    line_source_stratum_arrays,
    select_line_source_peers,
)


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

            source = LineSource(
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

    def test_shared_peer_selection_reuses_the_centroid_stratum(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = Path(directory) / "CVCL_TEST_de.h5ad"
            write_duplicate_compound_fixture(fixture_path)
            source = LineSource(
                dataset_name="source",
                cell_type="CVCL_TEST",
                path=fixture_path,
            )
            try:
                selection = select_line_source_peers(
                    source,
                    "row_a1",
                    pubchem_cid="A",
                    dose_key="0.05",
                    time_key="24",
                    max_peers=None,
                    sampling_seed=2025,
                )
                self.assertEqual(selection.total_count, 2)
                self.assertEqual(selection.selected_count, 2)
                np.testing.assert_allclose(
                    selection.stratum.values[selection.row_indices],
                    [[1.0, 3.0], [5.0, 7.0]],
                )
                compounds, values = line_source_stratum_arrays(
                    source,
                    dose_key="0.05",
                    time_key="24",
                )
                np.testing.assert_array_equal(
                    compounds,
                    selection.stratum.compounds,
                )
                self.assertIs(values, selection.stratum.values)
                self.assertEqual(
                    first_available_layer(source, ["missing", "logFC"]),
                    "logFC",
                )
                with self.assertRaisesRegex(KeyError, "available layers"):
                    first_available_layer(source, ["missing"])
            finally:
                source.close()


if __name__ == "__main__":
    unittest.main()
