#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

DATA_ROOT = Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated")
DATASET_ORDER = [
    "l1000_phase1",
    "l1000_phase2",
    "novartis_batch_2500",
    "op3",
    "sciplex",
    "tahoe",
]
DATASET_ORDER_RANK = {
    dataset_name: index for index, dataset_name in enumerate(DATASET_ORDER)
}
DEFAULT_MAX_LOG10_DOSE_DIFF = 1.0
NUMERIC_SIG_FIGS = 12
INVALID_STRING_VALUES = {"", "nan", "none", "<na>"}
PHASE12_ONLY_DATASETS = frozenset({"l1000_phase1", "l1000_phase2"})


@dataclass(frozen=True)
class NodeKey:
    dataset_name: str
    obs_id: str


@dataclass
class DatasetMetadata:
    dataset_name: str
    h5ad_path: Path
    frame: pd.DataFrame
    groups: dict[tuple[str, str, str], np.ndarray]
    normalized_groups: dict[tuple[str, str, str], np.ndarray]
    n_input_obs: int
    n_eligible_obs: int


@dataclass
class MatchEdge:
    dataset_a: str
    dataset_b: str
    obs_id_a: str
    obs_id_b: str
    pubchem_cid: str
    harmonized_context_key: str
    time_key: str


@dataclass
class ForcedBridgeLine:
    bridge_name: str
    context_key: str
    n_matching_compounds: int
    n_matching_conditions: int
    n_matching_edges: int


def canonicalize_match_edge(edge: MatchEdge) -> MatchEdge:
    left_rank = DATASET_ORDER_RANK.get(edge.dataset_a, len(DATASET_ORDER_RANK))
    right_rank = DATASET_ORDER_RANK.get(edge.dataset_b, len(DATASET_ORDER_RANK))
    if left_rank <= right_rank:
        return edge

    return MatchEdge(
        dataset_a=edge.dataset_b,
        dataset_b=edge.dataset_a,
        obs_id_a=edge.obs_id_b,
        obs_id_b=edge.obs_id_a,
        pubchem_cid=edge.pubchem_cid,
        harmonized_context_key=edge.harmonized_context_key,
        time_key=edge.time_key,
    )


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[NodeKey, NodeKey] = {}
        self._rank: dict[NodeKey, int] = {}

    def add(self, node: NodeKey) -> None:
        if node in self._parent:
            return
        self._parent[node] = node
        self._rank[node] = 0

    def find(self, node: NodeKey) -> NodeKey:
        self.add(node)
        parent = self._parent[node]
        if parent != node:
            self._parent[node] = self.find(parent)
        return self._parent[node]

    def union(self, left: NodeKey, right: NodeKey) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        left_rank = self._rank[left_root]
        right_rank = self._rank[right_root]
        if left_rank < right_rank:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if left_rank == right_rank:
            self._rank[left_root] = left_rank + 1

    def nodes(self) -> Iterable[NodeKey]:
        return self._parent.keys()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter processed group-level h5ads down to samples that participate in "
            "cross-dataset overlap components."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root directory containing dataset folders (default: {DATA_ROOT}).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Datasets to include when building overlap components.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/overlap_filtered_h5ads"),
        help="Directory where filtered h5ads and summary tables will be written.",
    )
    parser.add_argument(
        "--max-log10-dose-diff",
        type=float,
        default=DEFAULT_MAX_LOG10_DOSE_DIFF,
        help=(
            "Maximum allowed absolute difference in log10 dose for a matched pair "
            f"(default: {DEFAULT_MAX_LOG10_DOSE_DIFF})."
        ),
    )
    parser.add_argument(
        "--keep-phase12-only-overlaps",
        action="store_true",
        help=(
            "Keep l1000_phase1/l1000_phase2 overlap edges even when their cell line "
            "or compound are only supported by that pair. By default those "
            "phase1/phase2-only context/compound overlaps are excluded."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log progress to stderr.",
    )
    return parser.parse_args()


def log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def resolve_processed_h5ad(data_root: Path, dataset_name: str) -> Path:
    group_rep_dir = data_root / dataset_name / "pseudobulk_processed" / "group_rep"
    matches = sorted(group_rep_dir.glob("*.h5ad"))
    if not matches:
        raise FileNotFoundError(f"No .h5ad file found in {group_rep_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one .h5ad file in {group_rep_dir}, found {len(matches)}")
    return matches[0]


def normalize_string_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str).str.strip()


def sanitize_string_values(series: pd.Series) -> pd.Series:
    normalized = normalize_string_values(series)
    lowered = normalized.str.lower()
    normalized.loc[lowered.isin(INVALID_STRING_VALUES)] = ""
    return normalized


def normalize_pubchem_cid_values(series: pd.Series) -> pd.Series:
    normalized = sanitize_string_values(series)
    numeric = pd.to_numeric(normalized, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite_mask.any():
        return normalized

    normalized = normalized.copy()
    normalized.loc[finite_mask] = numeric.loc[finite_mask].map(format_pubchem_cid)
    return normalized


def to_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    lowered = series.fillna(False).astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def format_numeric(value: float) -> str:
    formatted = f"{value:.{NUMERIC_SIG_FIGS}g}"
    return "0" if formatted == "-0" else formatted


def format_pubchem_cid(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format_numeric(float(value))


def derive_harmonized_context_key(obs: pd.DataFrame) -> pd.Series:
    if "cell_type" not in obs.columns:
        raise KeyError(
            "Could not derive a context key because obs['cell_type'] is missing. "
            f"Available columns: {list(obs.columns)}"
        )
    return sanitize_string_values(obs["cell_type"])


def is_dmso_control_obs(obs: pd.DataFrame) -> pd.Series:
    if "is_control" not in obs.columns:
        return pd.Series(False, index=obs.index, dtype=bool)

    is_control = to_boolean_series(obs["is_control"])
    dmso_name_mask = pd.Series(False, index=obs.index, dtype=bool)
    for column in ("perturbagen_name", "perturbagen", "perturbation_label"):
        if column not in obs.columns:
            continue
        lowered = normalize_string_values(obs[column]).str.lower()
        dmso_name_mask = dmso_name_mask | lowered.str.contains("dmso", regex=False)
        dmso_name_mask = dmso_name_mask | lowered.str.contains(
            "dimethyl sulfoxide",
            regex=False,
        )

    dmso_pubchem_mask = pd.Series(False, index=obs.index, dtype=bool)
    if "pubchem_cid" in obs.columns:
        dmso_pubchem_mask = sanitize_string_values(obs["pubchem_cid"]) == "679"

    return is_control & (dmso_name_mask | dmso_pubchem_mask)


def build_condition_groups(
    frame: pd.DataFrame,
    pubchem_column: str,
) -> dict[tuple[str, str, str], np.ndarray]:
    grouped = frame.groupby(
        [pubchem_column, "harmonized_context_key", "time_key"],
        sort=False,
    ).groups
    return {
        key: np.asarray(list(row_positions), dtype=np.int64)
        for key, row_positions in grouped.items()
    }


def load_dataset_metadata(dataset_name: str, data_root: Path, verbose: bool = False) -> DatasetMetadata:
    h5ad_path = resolve_processed_h5ad(data_root, dataset_name)
    log(f"[{dataset_name}] reading metadata from {h5ad_path}", verbose=verbose)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs.copy()
        n_input_obs = int(adata.n_obs)
    finally:
        adata.file.close()

    required_columns = {"pubchem_cid", "pert_time_h", "pert_dose_uM", "is_control"}
    missing_columns = sorted(required_columns - set(obs.columns))
    if missing_columns:
        raise KeyError(
            f"{dataset_name} is missing required obs columns {missing_columns}. "
            f"Available columns: {list(obs.columns)}"
        )

    metadata = pd.DataFrame(index=obs.index.copy())
    metadata["dataset_name"] = dataset_name
    metadata["obs_id"] = metadata.index.astype(str)
    metadata["pubchem_cid"] = sanitize_string_values(obs["pubchem_cid"])
    metadata["normalized_pubchem_cid"] = normalize_pubchem_cid_values(obs["pubchem_cid"])
    metadata["harmonized_context_key"] = derive_harmonized_context_key(obs)
    metadata["time_h"] = pd.to_numeric(obs["pert_time_h"], errors="coerce")
    metadata["dose_um"] = pd.to_numeric(obs["pert_dose_uM"], errors="coerce")
    metadata["is_control"] = to_boolean_series(obs["is_control"])

    valid_mask = (
        ~metadata["is_control"]
        & (metadata["pubchem_cid"] != "")
        & (metadata["normalized_pubchem_cid"] != "")
        & (metadata["harmonized_context_key"] != "")
        & np.isfinite(metadata["time_h"].to_numpy(dtype=float))
        & np.isfinite(metadata["dose_um"].to_numpy(dtype=float))
        & (metadata["dose_um"].to_numpy(dtype=float) > 0.0)
    )
    metadata = metadata.loc[valid_mask].copy().reset_index(drop=True)
    metadata["time_key"] = metadata["time_h"].map(lambda value: format_numeric(float(value)))
    metadata["log10_dose"] = np.log10(metadata["dose_um"].to_numpy(dtype=np.float64))
    groups = build_condition_groups(metadata, pubchem_column="pubchem_cid")
    normalized_groups = build_condition_groups(
        metadata,
        pubchem_column="normalized_pubchem_cid",
    )

    log(
        (
            f"[{dataset_name}] input_obs={n_input_obs} eligible_obs={len(metadata)} "
            f"condition_keys={len(groups)}"
        ),
        verbose=verbose,
    )
    return DatasetMetadata(
        dataset_name=dataset_name,
        h5ad_path=h5ad_path,
        frame=metadata,
        groups=groups,
        normalized_groups=normalized_groups,
        n_input_obs=n_input_obs,
        n_eligible_obs=int(len(metadata)),
    )


def mutual_nearest_logdose_pairs(
    left_log10_doses: np.ndarray,
    right_log10_doses: np.ndarray,
    max_log10_dose_diff: float,
) -> np.ndarray:
    if left_log10_doses.size == 0 or right_log10_doses.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    diff = np.abs(left_log10_doses[:, None] - right_log10_doses[None, :])
    left_min = diff.min(axis=1, keepdims=True)
    right_min = diff.min(axis=0, keepdims=True)
    # Keep all tied mutual nearest neighbors, but only within the allowed
    # log10-dose window so sparse-dose datasets do not create unreasonable pairs.
    is_mnn = (
        (diff <= max_log10_dose_diff + 1e-12)
        & np.isclose(diff, left_min, rtol=0.0, atol=1e-12)
        & np.isclose(diff, right_min, rtol=0.0, atol=1e-12)
    )
    return np.argwhere(is_mnn)


def collect_pair_match_edges(
    left: DatasetMetadata,
    right: DatasetMetadata,
    max_log10_dose_diff: float,
    *,
    groups_attr: str = "groups",
    pubchem_column: str = "pubchem_cid",
    allowed_context_keys: set[str] | None = None,
) -> tuple[list[MatchEdge], int]:
    left_groups = getattr(left, groups_attr)
    right_groups = getattr(right, groups_attr)
    left_obs_ids = left.frame["obs_id"].to_numpy(dtype=object)
    right_obs_ids = right.frame["obs_id"].to_numpy(dtype=object)
    left_log10_dose = left.frame["log10_dose"].to_numpy(dtype=np.float64)
    right_log10_dose = right.frame["log10_dose"].to_numpy(dtype=np.float64)

    shared_keys = sorted(set(left_groups) & set(right_groups))
    if allowed_context_keys is not None:
        shared_keys = [key for key in shared_keys if str(key[1]) in allowed_context_keys]

    edges: list[MatchEdge] = []
    for key in shared_keys:
        pubchem_cid, harmonized_context_key, time_key = key
        left_rows = left_groups[key]
        right_rows = right_groups[key]
        pair_positions = mutual_nearest_logdose_pairs(
            left_log10_doses=left_log10_dose[left_rows],
            right_log10_doses=right_log10_dose[right_rows],
            max_log10_dose_diff=max_log10_dose_diff,
        )
        if pair_positions.size == 0:
            continue

        for left_pos, right_pos in pair_positions:
            edges.append(
                MatchEdge(
                    dataset_a=left.dataset_name,
                    dataset_b=right.dataset_name,
                    obs_id_a=str(left_obs_ids[left_rows[left_pos]]),
                    obs_id_b=str(right_obs_ids[right_rows[right_pos]]),
                    pubchem_cid=str(pubchem_cid),
                    harmonized_context_key=str(harmonized_context_key),
                    time_key=str(time_key),
                )
            )

    return edges, int(len(shared_keys))


def process_pair_matches(
    left: DatasetMetadata,
    right: DatasetMetadata,
    max_log10_dose_diff: float,
    verbose: bool = False,
) -> tuple[dict[str, object], list[MatchEdge]]:
    edges, n_shared_keys = collect_pair_match_edges(
        left=left,
        right=right,
        max_log10_dose_diff=max_log10_dose_diff,
        groups_attr="groups",
        pubchem_column="pubchem_cid",
    )
    matched_left: set[str] = set()
    matched_right: set[str] = set()
    for edge in edges:
        matched_left.add(edge.obs_id_a)
        matched_right.add(edge.obs_id_b)
    match_edges = len(edges)

    log(
        (
            f"[pair {left.dataset_name} <-> {right.dataset_name}] "
            f"shared_keys={n_shared_keys} match_edges={match_edges} "
            f"matched_obs={len(matched_left) + len(matched_right)}"
        ),
        verbose=verbose,
    )
    return {
        "dataset_a": left.dataset_name,
        "dataset_b": right.dataset_name,
        "n_shared_condition_keys": int(n_shared_keys),
        "n_match_edges": int(match_edges),
        "n_matched_obs_dataset_a": int(len(matched_left)),
        "n_matched_obs_dataset_b": int(len(matched_right)),
    }, edges


def build_match_edge_frame(match_edges: list[MatchEdge]) -> pd.DataFrame:
    if not match_edges:
        return pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "obs_id_a",
                "obs_id_b",
                "pubchem_cid",
                "harmonized_context_key",
                "time_key",
                "phase12_pair",
            ]
        )

    frame = pd.DataFrame(
        [
            {
                "dataset_a": canonical_edge.dataset_a,
                "dataset_b": canonical_edge.dataset_b,
                "obs_id_a": canonical_edge.obs_id_a,
                "obs_id_b": canonical_edge.obs_id_b,
                "pubchem_cid": canonical_edge.pubchem_cid,
                "harmonized_context_key": canonical_edge.harmonized_context_key,
                "time_key": canonical_edge.time_key,
            }
            for edge in match_edges
            for canonical_edge in [canonicalize_match_edge(edge)]
        ]
    )
    frame["phase12_pair"] = (
        (
            (frame["dataset_a"] == "l1000_phase1") & (frame["dataset_b"] == "l1000_phase2")
        )
        | (
            (frame["dataset_a"] == "l1000_phase2") & (frame["dataset_b"] == "l1000_phase1")
        )
    )
    return frame


def filter_match_edges(
    match_edges: pd.DataFrame,
    keep_phase12_only_overlaps: bool,
    verbose: bool = False,
) -> pd.DataFrame:
    if match_edges.empty or keep_phase12_only_overlaps:
        return match_edges.copy()

    non_phase12_edges = match_edges.loc[~match_edges["phase12_pair"]].copy()
    supported_contexts = set(non_phase12_edges["harmonized_context_key"].astype(str))
    supported_pubchem_cids = set(non_phase12_edges["pubchem_cid"].astype(str))

    phase12_keep_mask = (
        match_edges["phase12_pair"]
        & (
            match_edges["harmonized_context_key"].astype(str).isin(supported_contexts)
            | match_edges["pubchem_cid"].astype(str).isin(supported_pubchem_cids)
        )
    )
    retained = match_edges.loc[~match_edges["phase12_pair"] | phase12_keep_mask].copy()

    excluded_phase12_edges = int(match_edges["phase12_pair"].sum() - phase12_keep_mask.sum())
    log(
        (
            "[phase12-filter] "
            f"supported_contexts_from_other_pairs={len(supported_contexts)} "
            f"supported_pubchem_cids_from_other_pairs={len(supported_pubchem_cids)} "
            f"excluded_phase12_edges={excluded_phase12_edges}"
        ),
        verbose=verbose,
    )
    return retained


def rank_bridge_line_candidates(
    source: DatasetMetadata,
    targets: list[DatasetMetadata],
    max_log10_dose_diff: float,
    excluded_context_keys: set[str],
) -> list[ForcedBridgeLine]:
    candidate_contexts = sorted(
        set(source.frame["harmonized_context_key"].astype(str))
        - set(excluded_context_keys)
    )
    ranked: list[ForcedBridgeLine] = []

    for context_key in candidate_contexts:
        compounds: set[str] = set()
        conditions: set[tuple[str, str, str]] = set()
        edge_count = 0
        for target in targets:
            edges, _ = collect_pair_match_edges(
                left=source,
                right=target,
                max_log10_dose_diff=max_log10_dose_diff,
                groups_attr="normalized_groups",
                pubchem_column="normalized_pubchem_cid",
                allowed_context_keys={context_key},
            )
            if not edges:
                continue
            edge_count += len(edges)
            for edge in edges:
                compounds.add(edge.pubchem_cid)
                conditions.add((edge.pubchem_cid, edge.time_key, edge.dataset_b))

        if not compounds:
            continue

        ranked.append(
            ForcedBridgeLine(
                bridge_name="",
                context_key=context_key,
                n_matching_compounds=int(len(compounds)),
                n_matching_conditions=int(len(conditions)),
                n_matching_edges=int(edge_count),
            )
        )

    return sorted(
        ranked,
        key=lambda line: (
            -line.n_matching_compounds,
            -line.n_matching_conditions,
            -line.n_matching_edges,
            line.context_key,
        ),
    )


def select_forced_bridge_lines(
    metadata_by_dataset: dict[str, DatasetMetadata],
    retained_context_keys: set[str],
    max_log10_dose_diff: float,
    verbose: bool = False,
) -> list[ForcedBridgeLine]:
    selected: list[ForcedBridgeLine] = []

    bridge_specs = [
        (
            "sciplex_l1000",
            "sciplex",
            ["l1000_phase1", "l1000_phase2"],
        ),
        (
            "tahoe_l1000",
            "tahoe",
            ["l1000_phase1", "l1000_phase2"],
        ),
    ]

    excluded_context_keys = set(retained_context_keys)
    for bridge_name, source_name, target_names in bridge_specs:
        if source_name not in metadata_by_dataset:
            continue
        if any(target_name not in metadata_by_dataset for target_name in target_names):
            continue

        ranked = rank_bridge_line_candidates(
            source=metadata_by_dataset[source_name],
            targets=[metadata_by_dataset[target_name] for target_name in target_names],
            max_log10_dose_diff=max_log10_dose_diff,
            excluded_context_keys=excluded_context_keys,
        )
        if not ranked:
            log(f"[forced-lines] no candidate line found for {bridge_name}", verbose=verbose)
            continue

        best = ranked[0]
        best.bridge_name = bridge_name
        selected.append(best)
        excluded_context_keys.add(best.context_key)
        log(
            (
                f"[forced-lines] selected {bridge_name}: context={best.context_key} "
                f"matching_compounds={best.n_matching_compounds} "
                f"matching_conditions={best.n_matching_conditions} "
                f"matching_edges={best.n_matching_edges}"
            ),
            verbose=verbose,
        )

    return selected


def build_forced_bridge_edge_frame(
    selected_lines: list[ForcedBridgeLine],
    metadata_by_dataset: dict[str, DatasetMetadata],
    max_log10_dose_diff: float,
) -> pd.DataFrame:
    if not selected_lines:
        return build_match_edge_frame([])

    bridge_targets = {
        "sciplex_l1000": ("sciplex", ["l1000_phase1", "l1000_phase2"]),
        "tahoe_l1000": ("tahoe", ["l1000_phase1", "l1000_phase2"]),
    }
    edges: list[MatchEdge] = []

    for selected_line in selected_lines:
        if selected_line.bridge_name not in bridge_targets:
            continue
        source_name, target_names = bridge_targets[selected_line.bridge_name]
        source = metadata_by_dataset[source_name]
        for target_name in target_names:
            target = metadata_by_dataset[target_name]
            pair_edges, _ = collect_pair_match_edges(
                left=source,
                right=target,
                max_log10_dose_diff=max_log10_dose_diff,
                groups_attr="normalized_groups",
                pubchem_column="normalized_pubchem_cid",
                allowed_context_keys={selected_line.context_key},
            )
            edges.extend(pair_edges)

    forced_frame = build_match_edge_frame(edges)
    if forced_frame.empty:
        return forced_frame

    forced_frame = forced_frame.drop_duplicates(
        subset=[
            "dataset_a",
            "dataset_b",
            "obs_id_a",
            "obs_id_b",
            "pubchem_cid",
            "harmonized_context_key",
            "time_key",
        ]
    ).reset_index(drop=True)
    return forced_frame


def matched_obs_counts_from_edges(match_edges: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, set[str]] = defaultdict(set)
    if match_edges.empty:
        return {}

    for row in match_edges.itertuples(index=False):
        counts[str(row.dataset_a)].add(str(row.obs_id_a))
        counts[str(row.dataset_b)].add(str(row.obs_id_b))
    return {dataset_name: len(obs_ids) for dataset_name, obs_ids in counts.items()}


def build_union_from_edges(match_edges: pd.DataFrame) -> tuple[UnionFind, dict[NodeKey, int]]:
    union_find = UnionFind()
    node_match_counts: dict[NodeKey, int] = defaultdict(int)

    for row in match_edges.itertuples(index=False):
        left_node = NodeKey(str(row.dataset_a), str(row.obs_id_a))
        right_node = NodeKey(str(row.dataset_b), str(row.obs_id_b))
        union_find.union(left_node, right_node)
        node_match_counts[left_node] += 1
        node_match_counts[right_node] += 1

    return union_find, node_match_counts


def build_component_annotations(
    union_find: UnionFind,
    node_match_counts: dict[NodeKey, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_nodes: dict[NodeKey, list[NodeKey]] = defaultdict(list)
    for node in union_find.nodes():
        component_nodes[union_find.find(node)].append(node)

    annotation_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    component_index = 1

    for nodes in sorted(
        component_nodes.values(),
        key=lambda members: (
            sorted({node.dataset_name for node in members}),
            sorted((node.dataset_name, node.obs_id) for node in members),
        ),
    ):
        datasets = sorted({node.dataset_name for node in nodes})
        dataset_set = frozenset(datasets)
        keep_component = len(dataset_set) >= 2
        component_id = f"overlap_component_{component_index:05d}"
        component_index += 1

        component_rows.append(
            {
                "overlap_component_id": component_id,
                "datasets": ",".join(datasets),
                "n_datasets": int(len(dataset_set)),
                "n_samples": int(len(nodes)),
                "sum_match_edges": int(sum(node_match_counts.get(node, 0) for node in nodes)),
                "kept": bool(keep_component),
                "phase12_only_component": bool(dataset_set == PHASE12_ONLY_DATASETS),
            }
        )

        if not keep_component:
            continue

        overlap_datasets = ",".join(datasets)
        for node in sorted(nodes, key=lambda value: (value.dataset_name, value.obs_id)):
            annotation_rows.append(
                {
                    "dataset_name": node.dataset_name,
                    "obs_id": node.obs_id,
                    "overlap_component_id": component_id,
                    "overlap_datasets": overlap_datasets,
                    "n_overlap_datasets": int(len(dataset_set)),
                    "n_overlap_matches": int(node_match_counts.get(node, 0)),
                }
            )

    annotations = pd.DataFrame(
        annotation_rows,
        columns=[
            "dataset_name",
            "obs_id",
            "overlap_component_id",
            "overlap_datasets",
            "n_overlap_datasets",
            "n_overlap_matches",
        ],
    )
    components = pd.DataFrame(
        component_rows,
        columns=[
            "overlap_component_id",
            "datasets",
            "n_datasets",
            "n_samples",
            "sum_match_edges",
            "kept",
            "phase12_only_component",
        ],
    )
    return annotations, components


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    frame.to_csv(path, sep=sep, index=False)


def export_filtered_h5ads(
    metadata_by_dataset: dict[str, DatasetMetadata],
    annotations: pd.DataFrame,
    matched_before_exclusion_counts: dict[str, int],
    output_dir: Path,
    max_log10_dose_diff: float,
    keep_phase12_only_overlaps: bool,
    verbose: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary_rows: list[dict[str, object]] = []

    for dataset_name in metadata_by_dataset:
        dataset_meta = metadata_by_dataset[dataset_name]
        dataset_annotations = annotations.loc[
            annotations["dataset_name"] == dataset_name
        ].copy()
        retained_matched_obs_ids = set(dataset_annotations["obs_id"].astype(str).tolist())
        component_count = int(dataset_annotations["overlap_component_id"].nunique())
        matched_before_exclusion = int(matched_before_exclusion_counts.get(dataset_name, 0))

        metadata_lookup = (
            dataset_meta.frame[["obs_id", "harmonized_context_key", "time_key"]]
            .drop_duplicates(subset=["obs_id"])
            .set_index("obs_id")
        )
        matched_annotations = dataset_annotations.drop(columns=["dataset_name"]).set_index("obs_id")
        matched_annotations = matched_annotations.join(metadata_lookup[["time_key"]], how="left")
        if "harmonized_context_key" not in matched_annotations.columns:
            matched_annotations = matched_annotations.join(
                metadata_lookup[["harmonized_context_key"]],
                how="left",
            )
        matched_annotations["retention_reason"] = "matched_overlap"
        matched_annotations["retained_matched_sample"] = True
        matched_annotations["retained_dmso_control"] = False

        retained_context_time_keys = set(
            matched_annotations[["harmonized_context_key", "time_key"]]
            .dropna()
            .itertuples(index=False, name=None)
        )

        log(
            f"[{dataset_name}] exporting {len(retained_matched_obs_ids)} matched samples",
            verbose=verbose,
        )
        adata = ad.read_h5ad(dataset_meta.h5ad_path, backed="r")
        try:
            control_columns = [
                column
                for column in (
                    "is_control",
                    "pubchem_cid",
                    "perturbagen",
                    "perturbagen_name",
                    "perturbation_label",
                    "cell_type",
                    "pert_time_h",
                )
                if column in adata.obs.columns
            ]
            control_obs = adata.obs[control_columns].copy()
            control_meta = pd.DataFrame(index=control_obs.index.copy())
            control_meta["obs_id"] = control_meta.index.astype(str)
            control_meta["harmonized_context_key"] = derive_harmonized_context_key(control_obs)
            control_meta["time_h"] = pd.to_numeric(control_obs["pert_time_h"], errors="coerce")
            control_meta["time_key"] = control_meta["time_h"].map(
                lambda value: format_numeric(float(value)) if np.isfinite(value) else ""
            )
            control_mask = (
                is_dmso_control_obs(control_obs)
                & (control_meta["harmonized_context_key"] != "")
                & (control_meta["time_key"] != "")
            )
            control_meta = control_meta.loc[control_mask].copy()
            control_meta["keep_for_retained_context_time"] = [
                (row.harmonized_context_key, row.time_key) in retained_context_time_keys
                for row in control_meta.itertuples(index=False)
            ]
            kept_control_meta = control_meta.loc[control_meta["keep_for_retained_context_time"]].copy()
            retained_dmso_control_obs_ids = set(kept_control_meta["obs_id"].astype(str).tolist())

            retained_obs_ids = retained_matched_obs_ids | retained_dmso_control_obs_ids
            obs_index = pd.Index(adata.obs_names.astype(str))
            keep_mask = obs_index.isin(retained_obs_ids)
            row_idx = np.flatnonzero(keep_mask).astype(np.int64, copy=False)
            filtered = adata[row_idx, :].to_memory()
        finally:
            adata.file.close()

        filtered.obs = filtered.obs.copy()
        filtered.obs_names = pd.Index(filtered.obs_names.astype(str), dtype=object)
        control_annotations = pd.DataFrame(
            {
                "harmonized_context_key": kept_control_meta["harmonized_context_key"].to_numpy(),
                "time_key": kept_control_meta["time_key"].to_numpy(),
                "overlap_component_id": "",
                "overlap_datasets": "",
                "n_overlap_datasets": 0,
                "n_overlap_matches": 0,
                "retention_reason": "dmso_control_for_retained_context_time",
                "retained_matched_sample": False,
                "retained_dmso_control": True,
            },
            index=pd.Index(kept_control_meta["obs_id"].astype(str), dtype=object),
        )
        combined_annotations = pd.concat([matched_annotations, control_annotations], axis=0)
        combined_annotations = combined_annotations.loc[
            ~combined_annotations.index.duplicated(keep="first")
        ]
        aligned_annotations = combined_annotations.reindex(filtered.obs_names.astype(str))
        filtered.obs["harmonized_context_key"] = aligned_annotations["harmonized_context_key"].to_numpy()
        filtered.obs["overlap_component_id"] = aligned_annotations["overlap_component_id"].to_numpy()
        filtered.obs["overlap_datasets"] = aligned_annotations["overlap_datasets"].to_numpy()
        filtered.obs["n_overlap_datasets"] = aligned_annotations["n_overlap_datasets"].to_numpy()
        filtered.obs["n_overlap_matches"] = aligned_annotations["n_overlap_matches"].to_numpy()
        filtered.obs["retention_reason"] = aligned_annotations["retention_reason"].to_numpy()
        filtered.obs["retained_matched_sample"] = aligned_annotations[
            "retained_matched_sample"
        ].fillna(False).to_numpy(dtype=bool)
        filtered.obs["retained_dmso_control"] = aligned_annotations[
            "retained_dmso_control"
        ].fillna(False).to_numpy(dtype=bool)
        filtered.uns["overlap_filter"] = {
            "source_h5ad": str(dataset_meta.h5ad_path),
            "max_log10_dose_diff": float(max_log10_dose_diff),
            "keep_phase12_only_overlaps": bool(keep_phase12_only_overlaps),
            "retained_control_rule": "keep DMSO controls for retained cell_type/time combinations",
            "n_retained_matched_obs": int(len(retained_matched_obs_ids)),
            "n_retained_dmso_controls": int(len(retained_dmso_control_obs_ids)),
            "n_retained_obs": int(filtered.n_obs),
        }

        output_path = output_dir / f"{dataset_name}_overlap_filtered.h5ad"
        filtered.write_h5ad(output_path)

        dataset_summary_rows.append(
            {
                "dataset_name": dataset_name,
                "source_h5ad": str(dataset_meta.h5ad_path),
                "n_input_obs": int(dataset_meta.n_input_obs),
                "n_eligible_obs": int(dataset_meta.n_eligible_obs),
                "n_matched_obs_before_exclusion": matched_before_exclusion,
                "n_retained_matched_obs": int(len(retained_matched_obs_ids)),
                "n_retained_dmso_control_obs": int(len(retained_dmso_control_obs_ids)),
                "n_retained_obs": int(filtered.n_obs),
                "n_excluded_obs": int(matched_before_exclusion - len(retained_matched_obs_ids)),
                "n_retained_components": component_count,
                "output_h5ad": str(output_path),
            }
        )

    dataset_summary = pd.DataFrame(dataset_summary_rows)
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    return dataset_summary.sort_values(
        by="dataset_name",
        key=lambda series: series.map(lambda value: order.get(value, len(order))),
    ).reset_index(drop=True)


def add_context_annotations(
    annotations: pd.DataFrame,
    metadata_by_dataset: dict[str, DatasetMetadata],
) -> pd.DataFrame:
    if annotations.empty:
        return annotations

    enriched_frames: list[pd.DataFrame] = []
    for dataset_name, dataset_meta in metadata_by_dataset.items():
        dataset_annotations = annotations.loc[annotations["dataset_name"] == dataset_name].copy()
        if dataset_annotations.empty:
            enriched_frames.append(dataset_annotations)
            continue

        context_map = (
            dataset_meta.frame[["obs_id", "harmonized_context_key"]]
            .drop_duplicates(subset=["obs_id"])
            .set_index("obs_id")
        )
        dataset_annotations = dataset_annotations.join(context_map, on="obs_id")
        enriched_frames.append(dataset_annotations)

    return pd.concat(enriched_frames, ignore_index=True)


def build_overlap_outputs(
    dataset_names: list[str],
    data_root: Path,
    output_dir: Path,
    max_log10_dose_diff: float,
    keep_phase12_only_overlaps: bool,
    verbose: bool,
) -> None:
    metadata_by_dataset = {
        dataset_name: load_dataset_metadata(dataset_name, data_root=data_root, verbose=verbose)
        for dataset_name in dataset_names
    }

    pair_rows: list[dict[str, object]] = []
    all_match_edges: list[MatchEdge] = []

    for idx, dataset_a in enumerate(dataset_names):
        left = metadata_by_dataset[dataset_a]
        for dataset_b in dataset_names[idx + 1 :]:
            right = metadata_by_dataset[dataset_b]
            pair_summary_row, pair_match_edges = process_pair_matches(
                left=left,
                right=right,
                max_log10_dose_diff=max_log10_dose_diff,
                verbose=verbose,
            )
            pair_rows.append(pair_summary_row)
            all_match_edges.extend(pair_match_edges)

    all_match_edge_frame = build_match_edge_frame(all_match_edges)
    retained_standard_match_edge_frame = filter_match_edges(
        all_match_edge_frame,
        keep_phase12_only_overlaps=keep_phase12_only_overlaps,
        verbose=verbose,
    )
    retained_context_keys = set(
        retained_standard_match_edge_frame["harmonized_context_key"].astype(str)
    )
    forced_bridge_lines = select_forced_bridge_lines(
        metadata_by_dataset=metadata_by_dataset,
        retained_context_keys=retained_context_keys,
        max_log10_dose_diff=max_log10_dose_diff,
        verbose=verbose,
    )
    forced_bridge_edge_frame = build_forced_bridge_edge_frame(
        selected_lines=forced_bridge_lines,
        metadata_by_dataset=metadata_by_dataset,
        max_log10_dose_diff=max_log10_dose_diff,
    )
    preexport_match_edge_frame = pd.concat(
        [all_match_edge_frame, forced_bridge_edge_frame],
        ignore_index=True,
    )
    if not preexport_match_edge_frame.empty:
        preexport_match_edge_frame = preexport_match_edge_frame.drop_duplicates(
            subset=[
                "dataset_a",
                "dataset_b",
                "obs_id_a",
                "obs_id_b",
                "pubchem_cid",
                "harmonized_context_key",
                "time_key",
            ]
        ).reset_index(drop=True)
    retained_match_edge_frame = pd.concat(
        [retained_standard_match_edge_frame, forced_bridge_edge_frame],
        ignore_index=True,
    )
    if not retained_match_edge_frame.empty:
        retained_match_edge_frame = retained_match_edge_frame.drop_duplicates(
            subset=[
                "dataset_a",
                "dataset_b",
                "obs_id_a",
                "obs_id_b",
                "pubchem_cid",
                "harmonized_context_key",
                "time_key",
            ]
        ).reset_index(drop=True)

    matched_before_exclusion_counts = matched_obs_counts_from_edges(preexport_match_edge_frame)
    union_find, node_match_counts = build_union_from_edges(retained_match_edge_frame)
    annotations, component_summary = build_component_annotations(
        union_find=union_find,
        node_match_counts=node_match_counts,
    )
    annotations = add_context_annotations(annotations, metadata_by_dataset)
    dataset_summary = export_filtered_h5ads(
        metadata_by_dataset=metadata_by_dataset,
        annotations=annotations,
        matched_before_exclusion_counts=matched_before_exclusion_counts,
        output_dir=output_dir,
        max_log10_dose_diff=max_log10_dose_diff,
        keep_phase12_only_overlaps=keep_phase12_only_overlaps,
        verbose=verbose,
    )

    pair_summary = pd.DataFrame(
        pair_rows,
        columns=[
            "dataset_a",
            "dataset_b",
            "n_shared_condition_keys",
            "n_match_edges",
            "n_matched_obs_dataset_a",
            "n_matched_obs_dataset_b",
        ],
    )
    retained_pair_counts = (
        retained_match_edge_frame.groupby(["dataset_a", "dataset_b"], as_index=False)
        .agg(
            n_retained_match_edges=("obs_id_a", "size"),
            n_retained_obs_dataset_a=("obs_id_a", "nunique"),
            n_retained_obs_dataset_b=("obs_id_b", "nunique"),
        )
        if not retained_match_edge_frame.empty
        else pd.DataFrame(
            columns=[
                "dataset_a",
                "dataset_b",
                "n_retained_match_edges",
                "n_retained_obs_dataset_a",
                "n_retained_obs_dataset_b",
            ]
        )
    )
    pair_summary = pair_summary.merge(
        retained_pair_counts,
        on=["dataset_a", "dataset_b"],
        how="left",
    )
    for column in [
        "n_retained_match_edges",
        "n_retained_obs_dataset_a",
        "n_retained_obs_dataset_b",
    ]:
        pair_summary[column] = pair_summary[column].fillna(0).astype(int)

    write_table(dataset_summary, output_dir / "dataset_overlap_summary.tsv")
    write_table(pair_summary, output_dir / "pair_overlap_summary.tsv")
    write_table(component_summary, output_dir / "overlap_component_summary.tsv")
    write_table(annotations, output_dir / "overlap_sample_annotations.tsv")


def main() -> None:
    args = parse_args()
    build_overlap_outputs(
        dataset_names=args.datasets,
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_log10_dose_diff=args.max_log10_dose_diff,
        keep_phase12_only_overlaps=args.keep_phase12_only_overlaps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
