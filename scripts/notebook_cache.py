"""Resumable notebook stages.

These notebooks run long scoring loops and bootstrap passes whose results are written to
TSV anyway. Wrapping each expensive stage in :func:`cached_frame` makes a notebook
restartable: a stage whose output file already exists is reloaded instead of recomputed,
so a crash late in a notebook no longer means redoing everything above it.

Usage in a notebook::

    from scripts.notebook_cache import cache_summary, cached_frame, force_recompute

    def build_matched_pair_metrics() -> pd.DataFrame:
        ...                                   # the expensive work
        return frame

    path = OUTPUT_DIR / "matched_sample_pair_metrics.tsv"
    matched_pair_metrics = cached_frame("deg_metrics", path, build_matched_pair_metrics)

To rebuild a stage, delete its TSV or name it explicitly::

    force_recompute("peer_baselines")         # in a cell, before the stage runs
    CPB_FORCE_RECOMPUTE=peer_baselines,peer_ci jupyter lab      # or from the shell
    CPB_FORCE_RECOMPUTE=all jupyter lab                        # ignore every cache

Prefer :func:`force_recompute` over rebinding ``FORCE_RECOMPUTE``: the set is shared by
reference, so reassigning the name in a notebook would silently detach it.

Run ``python scripts/notebook_cache.py`` to execute the self-tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

__all__ = [
    "CACHE_STATUS",
    "CACHE_STRING_COLUMNS",
    "FORCE_RECOMPUTE",
    "FORCE_ALL",
    "cache_summary",
    "cached_frame",
    "force_recompute",
    "is_cached",
]

# Read back with inferred dtypes, a key like "00123" becomes the integer 123 and silently
# stops matching the string keys used for row lookups -- which surfaces as mass unresolved
# rows rather than an error. These columns are therefore always reloaded as strings;
# everything else keeps its inferred dtype.
CACHE_STRING_COLUMNS: tuple[str, ...] = (
    "dataset_a",
    "dataset_b",
    "query_dataset",
    "target_dataset",
    "dataset_name",
    "cell_type",
    "time_key",
    "dose_key",
    "pubchem_cid",
    "query_pubchem_cid",
    "condition_key",
    "obs_id",
    "query_obs_id",
    "left_obs_id",
    "right_obs_id",
    "left_dose_key",
    "right_dose_key",
    "query_dose_key",
    "left_plate",
    "right_plate",
    "left_well",
    "right_well",
    "matched_condition_key",
    "left_adj_pvalue_layer",
    "right_adj_pvalue_layer",
    "perturbagen_display",
    "dose_threshold",
    "representation",
    "retrieval_variant",
    "similarity_metric",
    "baseline_type",
    "baseline_role",
    "metric",
    "value_col",
    "summary_level",
    "cluster_col",
    "inner_strata",
    "outer_strata",
    "uncertainty_scope",
    "ci_method",
    "ci_status",
    "direction",
)

_ENV_FORCE = os.environ.get("CPB_FORCE_RECOMPUTE", "")
FORCE_ALL: bool = _ENV_FORCE.strip().lower() == "all"
FORCE_RECOMPUTE: set[str] = (
    set()
    if FORCE_ALL
    else {stage.strip() for stage in _ENV_FORCE.split(",") if stage.strip()}
)
CACHE_STATUS: dict[str, str] = {}


def force_recompute(*stages: str, replace: bool = False) -> set[str]:
    """Mark stages to recompute even when their output file exists.

    Mutates the shared set rather than rebinding it, so notebook cells and this module
    stay in agreement. Pass ``replace=True`` to drop any previous selection.
    """
    if replace:
        FORCE_RECOMPUTE.clear()
    FORCE_RECOMPUTE.update(str(stage) for stage in stages)
    return set(FORCE_RECOMPUTE)


def is_cached(stage: str, path: Path) -> bool:
    """Whether :func:`cached_frame` would reload this stage rather than rebuild it.

    Useful when the expensive work is a module-level loop that would be awkward to move
    into a builder: guard the loop's input with this so it becomes a no-op on a cache hit,
    and let ``cached_frame`` assemble the result from whatever the loop produced.
    """
    return Path(path).exists() and not FORCE_ALL and stage not in FORCE_RECOMPUTE


def cached_frame(
    stage: str,
    path: Path,
    build: Callable[[], pd.DataFrame],
    *,
    string_columns: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reload ``path`` when it exists, otherwise run ``build()`` and save the result.

    ``stage`` is the name used by :func:`force_recompute` and reported in
    :data:`CACHE_STATUS`.
    """
    path = Path(path)
    columns: Iterable[str] = (
        CACHE_STRING_COLUMNS if string_columns is None else tuple(string_columns)
    )

    if path.exists() and not FORCE_ALL and stage not in FORCE_RECOMPUTE:
        frame = pd.read_csv(
            path,
            sep="\t",
            dtype={column_name: str for column_name in columns},
        )
        for column_name in columns:
            if column_name in frame.columns:
                frame[column_name] = (
                    frame[column_name].astype("string").fillna("").astype(str)
                )
        CACHE_STATUS[stage] = "reloaded"
        if verbose:
            print(f"[{stage}] reloaded {len(frame):,} rows from {path.name}")
        return frame

    frame = build()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"[{stage}] build() returned {type(frame).__name__}, not a DataFrame")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    CACHE_STATUS[stage] = "computed"
    if verbose:
        print(f"[{stage}] computed and saved {len(frame):,} rows to {path.name}")
    return frame


def cache_summary() -> pd.DataFrame:
    """One row per stage seen so far, showing whether it was reloaded or recomputed."""
    return pd.DataFrame(
        [{"stage": stage, "status": status} for stage, status in sorted(CACHE_STATUS.items())],
        columns=["stage", "status"],
    )


def _self_test() -> None:
    import tempfile

    import numpy as np

    FORCE_RECOMPUTE.clear()
    CACHE_STATUS.clear()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "stage.tsv"
        frame = pd.DataFrame(
            {
                "left_obs_id": ["00123", "P1_A01"],
                "time_key": ["24", "24"],
                "left_dose_key": ["10", "0.0003"],
                "score": [0.31, np.nan],
                "n_deg": [120, 4],
            }
        )

        calls: list[int] = []
        cached_frame("demo", path, lambda: (calls.append(1), frame)[1], verbose=False)
        assert calls == [1] and CACHE_STATUS["demo"] == "computed"
        assert path.exists(), "parent directories should be created"

        def explode() -> pd.DataFrame:
            raise AssertionError("build() must not run on a cache hit")

        again = cached_frame("demo", path, explode, verbose=False)
        assert CACHE_STATUS["demo"] == "reloaded"
        assert again["left_obs_id"].tolist() == ["00123", "P1_A01"], "leading zeros lost"
        assert again["time_key"].tolist() == ["24", "24"]
        assert again["left_dose_key"].tolist() == ["10", "0.0003"]
        assert again["n_deg"].dtype.kind == "i", again["n_deg"].dtype
        assert np.isclose(again["score"][0], 0.31) and np.isnan(again["score"][1])

        force_recompute("demo")
        calls.clear()
        cached_frame("demo", path, lambda: (calls.append(1), frame)[1], verbose=False)
        assert calls == [1] and CACHE_STATUS["demo"] == "computed"

        force_recompute("other", replace=True)
        assert FORCE_RECOMPUTE == {"other"}
        cached_frame("demo", path, explode, verbose=False)

        assert is_cached("demo", path) and not is_cached("absent", Path(directory) / "nope.tsv")
        force_recompute("demo")
        assert not is_cached("demo", path), "force_recompute must defeat is_cached"
        force_recompute("other", replace=True)

        summary = cache_summary()
        assert summary["stage"].tolist() == ["demo"]
        assert summary["status"].tolist() == ["reloaded"]

        try:
            cached_frame("bad", Path(directory) / "bad.tsv", lambda: "not a frame", verbose=False)
        except TypeError as exc:
            assert "not a DataFrame" in str(exc)
        else:
            raise AssertionError("a non-DataFrame build result should raise")

    FORCE_RECOMPUTE.clear()
    CACHE_STATUS.clear()
    print("notebook_cache self-tests passed.")


if __name__ == "__main__":
    _self_test()
