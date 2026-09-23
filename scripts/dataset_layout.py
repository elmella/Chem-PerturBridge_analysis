#!/usr/bin/env python3
"""One place that knows where a dataset's DGE H5ADs live.

The same datasets arrive in three different shapes:

* the HuggingFace mirror stages them flat, as ``<dataset>/<mode>/``;
* the cluster keeps the full pipeline path,
  ``<dataset>/deg_data/<mode>/full/qc_false/filter_min_cells_<n>/results/``;
* the tarball-only datasets unpack under ``<dataset>/<mode>_extracted/``, and
  their internal prefix is not consistent — the CIGS archives carry a
  ``deg_data/`` level that the GDPx2 and DILImap archives do not.

Hardcoding one shape per dataset silently loses a source when an archive is
restaged, which is exactly how GDPx2's ``group_rep`` went missing from the
population precompute while its ``sep_rep`` resolved fine. Resolution is
therefore by search, ordered from most to least specific, with a recursive
fallback that refuses to guess between two candidates.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["MODES", "dge_dataset_dir", "pipeline_relative_path"]

MODES = ("group_rep", "sep_rep")


def pipeline_relative_path(mode: str, filter_min_cells: int) -> Path:
    return (
        Path("deg_data")
        / mode
        / "full"
        / "qc_false"
        / f"filter_min_cells_{int(filter_min_cells)}"
        / "results"
    )


def _candidates(dataset_root: Path, dataset_name: str, mode: str, filter_min_cells: int):
    pipeline = pipeline_relative_path(mode, filter_min_cells)
    # The archives disagree about whether a ``deg_data`` level is present, so
    # try both with and without it.
    bare = Path(mode) / "full" / "qc_false" / f"filter_min_cells_{int(filter_min_cells)}" / "results"
    extracted = f"{mode}_extracted"
    return (
        dataset_root / mode,
        dataset_root / pipeline,
        dataset_root / extracted / pipeline,
        dataset_root / extracted / bare,
        dataset_root / extracted / dataset_name / pipeline,
        dataset_root / extracted / dataset_name / bare,
    )


def dge_dataset_dir(
    data_root: Path | str,
    dataset_name: str,
    mode: str,
    filter_min_cells: int = 0,
) -> Path:
    """Resolve one dataset's DGE directory for ``mode``.

    Returns the canonical pipeline path when nothing is present, so a missing
    input produces an error message pointing at where it belongs.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    dataset_root = Path(data_root) / dataset_name

    for candidate in _candidates(dataset_root, dataset_name, mode, filter_min_cells):
        if candidate.is_dir() and any(candidate.glob("*_de.h5ad")):
            return candidate

    # An archive can add a wrapper directory whose name is not stable. Accept
    # it only when discovery finds exactly one directory for this mode, so a
    # choice between different filtering runs is never made silently.
    other_mode = "sep_rep" if mode == "group_rep" else "group_rep"
    discovered = sorted(
        {
            path.parent
            for path in dataset_root.rglob("*_de.h5ad")
            if any(part.startswith(mode) for part in path.parts)
            and not any(part.startswith(other_mode) for part in path.parts)
        }
    )
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        locations = "\n".join(f"- {path}" for path in discovered)
        raise RuntimeError(
            f"Multiple {mode} DGE directories found for {dataset_name}; "
            f"cannot choose safely:\n{locations}"
        )

    return dataset_root / pipeline_relative_path(mode, filter_min_cells)


def _self_test() -> None:
    import tempfile

    def touch(directory: Path, name: str = "CVCL_0001_de.h5ad") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).touch()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Flat HuggingFace layout.
        touch(root / "tahoe" / "sep_rep")
        assert dge_dataset_dir(root, "tahoe", "sep_rep") == root / "tahoe" / "sep_rep"

        # CIGS archive shape, with the deg_data level.
        cigs = root / "cigs_mce" / "group_rep_extracted" / pipeline_relative_path("group_rep", 0)
        touch(cigs)
        assert dge_dataset_dir(root, "cigs_mce", "group_rep") == cigs

        # GDPx2 archive shape, without it. This is the case that regressed.
        gdpx2 = (
            root / "gdpx2" / "group_rep_extracted" / "group_rep" / "full"
            / "qc_false" / "filter_min_cells_0" / "results"
        )
        touch(gdpx2)
        assert dge_dataset_dir(root, "gdpx2", "group_rep") == gdpx2

        # Both modes present must not cross-contaminate.
        sep = (
            root / "gdpx2" / "sep_rep_extracted" / "sep_rep" / "full"
            / "qc_false" / "filter_min_cells_0" / "results"
        )
        touch(sep)
        assert dge_dataset_dir(root, "gdpx2", "sep_rep") == sep
        assert dge_dataset_dir(root, "gdpx2", "group_rep") == gdpx2

        # A non-standard wrapper resolves by discovery.
        odd = root / "weird" / "sep_rep_extracted" / "surprise" / "inner"
        touch(odd)
        assert dge_dataset_dir(root, "weird", "sep_rep") == odd

        # Ambiguity is refused rather than guessed.
        touch(root / "weird" / "sep_rep_extracted" / "surprise" / "other")
        try:
            dge_dataset_dir(root, "weird", "sep_rep")
        except RuntimeError as exc:
            assert "cannot choose safely" in str(exc)
        else:
            raise AssertionError("expected ambiguity to be refused")

        # Absent input points at the canonical location.
        missing = dge_dataset_dir(root, "absent", "sep_rep")
        assert missing == root / "absent" / pipeline_relative_path("sep_rep", 0)

        # filter_min_cells is honoured.
        touch(root / "sciplex" / pipeline_relative_path("sep_rep", 10))
        assert dge_dataset_dir(root, "sciplex", "sep_rep", 10) == (
            root / "sciplex" / pipeline_relative_path("sep_rep", 10)
        )

    print("dataset_layout self-tests passed")


if __name__ == "__main__":
    _self_test()
