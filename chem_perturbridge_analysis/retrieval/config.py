from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

DATA_ROOT = Path(os.environ.get("PERTURB_DATA_ROOT", "data/processed"))


def _group_rep_results(dataset_name: str, min_cells: int) -> Path:
    return (
        DATA_ROOT
        / dataset_name
        / "deg_data"
        / "group_rep"
        / "full"
        / "qc_false"
        / f"filter_min_cells_{min_cells}"
        / "results"
    )


DEFAULT_DATASET_PATHS: dict[str, Path] = {
    "sciplex": _group_rep_results("sciplex", 10),
    "tahoe": _group_rep_results("tahoe", 50),
    "l1000_phase1": _group_rep_results("l1000_phase1", 0),
    "l1000_phase2": _group_rep_results("l1000_phase2", 0),
}

# Optional fallback paths for merged h5ad files.
FALLBACK_DATASET_PATHS: dict[str, Path] = {
    "l1000_phase1": _group_rep_results("l1000_phase1", 0).parent / "l1000_phase1.h5ad",
    "l1000_phase2": _group_rep_results("l1000_phase2", 0).parent / "l1000_phase2.h5ad",
}


@dataclass(frozen=True)
class RetrievalSettings:
    include_metrics: frozenset[str] = frozenset({"cosine", "pearson", "spearman", "mrrmse"})
    include_representations: frozenset[str] = frozenset()
    skip_representations: frozenset[str] = frozenset()
    excluded_layers: frozenset[str] = frozenset(
        {
            "CI.L",
            "CI.R",
            "stdev.scaled",
            "stdev.unscaled",
            "AveExpr",
            "adj.P.Value.across_all_contrasts",
            "adj.P.Value.within_one_contrast",
        }
    )
    logfc_layer: str = "logFC"
    pvalue_layer: str = "P.Value"
    p_floor: float = 1e-300
    p_clip_high: float = 1.0 - 1e-16


def resolve_dataset_paths(overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Resolve dataset paths from defaults + caller overrides + optional fallbacks."""
    merged = dict(DEFAULT_DATASET_PATHS)
    if overrides:
        merged.update({name: Path(path) for name, path in overrides.items()})

    resolved: dict[str, Path] = {}
    for name, path in merged.items():
        if path.exists():
            resolved[name] = path
            continue
        fallback = FALLBACK_DATASET_PATHS.get(name)
        if fallback is not None and fallback.exists():
            resolved[name] = fallback
            continue
        raise FileNotFoundError(
            f"Dataset path for '{name}' does not exist: {path}"
            + (f" (fallback checked: {fallback})" if fallback else "")
        )
    return resolved
