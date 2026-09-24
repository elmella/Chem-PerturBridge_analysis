#!/usr/bin/env python3
"""Within-dataset replicate retrieval: can a replicate find its own condition?

Each replicate of a condition is a query. Its candidates are every *other*
replicate in the same dataset x cell line x time stratum, across all doses and
compounds; its positives are the other replicates of its own condition. This
is the replicate analogue of cross-dataset Table 9, scored with the same
definitions (``scripts/cross_source_scoring.py``):

* normalized best-positive rank, ``1 - (best_rank - 1) / (N - 1)``, 1 is best;
* Recall@1, whether a positive is the top candidate;
* AUROC of positives against negatives;
* the exact null expectation of the rank, since queries have one or two
  positives and chance is therefore above 0.5;
* similarities: cosine, Spearman, and negative Euclidean distance (``l2``),
  as in Table 9;
* a centroid baseline: the mean of same-dose, other-compound replicates,
  injected into the pool, as in the original replicate code.

There is deliberately no individual-peer baseline. In cross-dataset Table 9 a
peer is a signature from the other dataset injected into the pool, so its rank
is informative. Here every different-compound replicate is already a
candidate, so the mean rank of such peers is the middle of the pool by
construction (0.499 on OP3, with no spread), and their corrected percentile
only restates the observed rank. The exact null answers the question a peer
baseline would ask -- where does an arbitrary other signature land -- without
the tautology.

Why this is a separate module rather than the retrieval path inside
``precompute_replicate_signature_similarity.py``:

1. **That path retrieves the query itself.** It pairs replicates cyclically,
   ``(0,1), (1,2), (2,0)``, so with three replicates the query's own sample is
   also a target of the same condition, and nothing excludes it. On OP3, 402 of
   406 queries had their own sample among their positives. Here the query is
   excluded from its candidates by construction.
2. **It recomputed each stratum once per 25-condition task.** Novartis is one
   15,276-condition stratum split into 612 tasks. Here each stratum is scored
   once, as blocked matrix products.

Inputs come from a finished replicate-scoring run (``--prepared-dir``): its
replicate rows and retained conditions, so the conditions and genes here are
exactly those behind Tables 7, 8 and 10. Vectors load through the same
float32 path, including the dense layer cache.

Each stratum is checkpointed as it finishes, so an interrupted run resumes by
rerunning the same command.
"""

from __future__ import annotations

import argparse
import os
import sys


def _early_threads(argv: list[str]) -> str:
    # BLAS reads its thread count once, when numpy loads, so this has to be
    # settled before the import below. The scorer module this imports pins BLAS
    # to one thread, which suits its worker pool; here a single process does
    # large matrix products and should use several.
    for index, value in enumerate(argv):
        if value == "--threads" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--threads="):
            return value.split("=", 1)[1]
    return "8"


_THREADS = _early_threads(sys.argv)
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = _THREADS

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import time  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import precompute_replicate_signature_similarity as scorer  # noqa: E402

SIMILARITY_METRICS = ("cosine", "spearman", "l2")
SCALE_VARIANTS = ("raw", "dataset", "dataset-cell-type")
SCALE_LABELS = {
    "raw": "raw",
    "dataset": "per_gene_population_zscore_dataset",
    "dataset-cell-type": "per_gene_population_zscore_dataset_cell_type",
}
SCALE_TO_SCOPE = {
    "dataset": scorer.DATASET_SCOPE,
    "dataset-cell-type": scorer.DATASET_CELL_TYPE_SCOPE,
}
# Matches auroc_from_scores in cross_source_scoring.py.
AUROC_TIE_ATOL = 1e-12
STRATUM_DIR_NAME = "strata"
CONDITION_SUMMARY_NAME = "condition_retrieval_summary.tsv"
CONDITION_KEYS = ["dataset_name", "cell_type", "time_key", "condition_key",
                  "pubchem_cid", "dose_key"]


# ---------------------------------------------------------------------------
# Metric definitions. Each mirrors its cross_source_scoring.py counterpart;
# the self-test checks them against those functions directly.
# ---------------------------------------------------------------------------

def best_positive_rank(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    """Rank of the best positive, ``rankdata(-scores, method="min")``."""
    best = float(np.max(scores[positive_mask]))
    return 1.0 + float(np.sum(scores > best))


def normalized_rank(rank: float, n_candidates: int) -> float:
    if n_candidates < 2:
        return float("nan")
    return float(1.0 - (rank - 1.0) / (n_candidates - 1.0))


def auroc(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    """Wins plus half-ties of positives over negatives, as cross-source AUROC."""
    positives = scores[positive_mask]
    negatives = np.sort(scores[~positive_mask])
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    wins = np.searchsorted(negatives, positives, side="left").astype(np.float64)
    ties = (
        np.searchsorted(negatives, positives + AUROC_TIE_ATOL, side="right")
        - np.searchsorted(negatives, positives - AUROC_TIE_ATOL, side="left")
    ).astype(np.float64)
    return float((wins + 0.5 * ties).sum() / (positives.size * negatives.size))


def null_expected_normalized_rank(n_candidates: int, n_positives: int) -> float:
    """Expected normalized best rank when the positives are placed at random.

    The best of K uniformly placed positives among N has expected rank
    ``(N + 1) / (K + 1)``; normalization is linear, so it maps straight
    through. With one positive this is 0.5, and it rises with K.
    """
    if n_candidates < 2 or n_positives < 1:
        return float("nan")
    expected_rank = (n_candidates + 1.0) / (n_positives + 1.0)
    return normalized_rank(expected_rank, n_candidates)


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def similarity_ready_rows(matrix: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Unit rows whose dot products are the requested similarity.

    Cosine normalizes the values; Spearman normalizes centered average ranks,
    so a dot product is the Spearman correlation. Rows with no variation
    cannot be scored and are reported invalid.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if metric == "spearman":
        values = rankdata(values, method="average", axis=1).astype(np.float64)
        values -= values.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"Unsupported similarity metric: {metric!r}")
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0)
    out = np.zeros_like(values)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def negative_l2_scores(
    query: np.ndarray,
    candidates: np.ndarray,
    query_square_norms: np.ndarray,
    candidate_square_norms: np.ndarray,
) -> np.ndarray:
    """Negative Euclidean distance, ``-||q - c||``, for every query x candidate.

    Table 9 computes this with ``cdist``, which is single-threaded; replicate
    strata reach ~48,000 candidates x ~12,000 genes, so here it comes from
    ``||q||^2 + ||c||^2 - 2 q.c`` with a threaded matrix product. The two agree
    to floating-point roundoff (clipped at zero, where cancellation could dip
    below it), so rankings match; the self-test checks this against cdist.
    """
    square = query_square_norms[:, None] + candidate_square_norms[None, :] - 2.0 * (query @ candidates.T)
    return -np.sqrt(np.maximum(square, 0.0))


# ---------------------------------------------------------------------------
# One stratum
# ---------------------------------------------------------------------------

def load_stratum_matrix(
    rows: pd.DataFrame,
    open_adatas: dict,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """logFC for every replicate row, on the genes finite in all of them."""
    source_paths = sorted(rows["source_path"].astype(str).unique().tolist())
    gene_keys = np.asarray(scorer.shared_gene_keys_for_paths(source_paths), dtype=object)
    if gene_keys.size < 2:
        return None, gene_keys
    vectors, _ = scorer.load_vectors_for_rows(
        rows, gene_keys=gene_keys, open_adatas=open_adatas, load_t=False
    )
    if not all(vector is not None for vector in vectors):
        return None, gene_keys
    matrix = np.vstack(vectors).astype(np.float64)
    finite = np.isfinite(matrix).all(axis=0)
    return matrix[:, finite], gene_keys[finite]


def scaled_matrix(
    matrix: np.ndarray,
    gene_keys: np.ndarray,
    *,
    scale: str,
    dataset_name: str,
    cell_type: str,
    stats_cache: Optional["scorer.ReplicatePopulationStatsCache"],
) -> np.ndarray:
    if scale == "raw":
        return matrix
    if stats_cache is None:
        raise ValueError(f"Scale {scale!r} needs population statistics")
    stats = stats_cache.get(
        dataset_name=dataset_name, cell_type=cell_type, scope=SCALE_TO_SCOPE[scale]
    )
    normalized = scorer.normalized_matrix_for_stats(
        matrix, gene_keys=gene_keys, stats_record=stats
    )
    # Standardization can only drop genes; keep those finite in every row so
    # all queries are compared on one gene set, as for raw values.
    return normalized[:, np.isfinite(normalized).all(axis=0)]


def _group_sums(group_index: np.ndarray, n_groups: int, values: np.ndarray) -> np.ndarray:
    """Row sums per group, via a sparse indicator product.

    ``np.add.at`` is unbuffered and crawls at tens of thousands of rows by
    tens of thousands of genes; the indicator product gives the same sums.
    """
    from scipy.sparse import csr_matrix

    indicator = csr_matrix(
        (np.ones(group_index.size), (group_index, np.arange(group_index.size))),
        shape=(n_groups, group_index.size),
    )
    return np.asarray(indicator @ values, dtype=np.float64)


def score_stratum(
    rows: pd.DataFrame,
    values: np.ndarray,
    *,
    metric: str,
    query_block: int,
) -> list[dict[str, object]]:
    """Per-query records for one stratum, one representation, one metric."""
    n_rows = values.shape[0]
    # Integer codes: the per-query positive mask is compared across every
    # candidate, and on the largest strata string comparison dominates.
    _, condition = np.unique(rows["condition_key"].astype(str).to_numpy(), return_inverse=True)
    compound = rows["pubchem_cid"].astype(str).to_numpy()
    dose = rows["dose_key"].astype(str).to_numpy()
    if metric == "l2":
        # Distance needs no normalization; any row with all values finite scores.
        valid = np.isfinite(values).all(axis=1)
        finite_values = np.where(valid[:, None], values, 0.0)
        square_norms = np.einsum("ij,ij->i", finite_values, finite_values)
    else:
        unit, valid = similarity_ready_rows(values, metric)

    # Centroid baseline: mean of same-dose, other-compound replicates. Built
    # from per-dose and per-(dose, compound) sums, so each query's centroid is
    # one subtraction instead of a fresh mean over thousands of rows.
    dose_codes, dose_index = np.unique(dose, return_inverse=True)
    group_codes, group_index = np.unique(
        np.char.add(np.char.add(dose.astype(str), "\x1f"), compound.astype(str)),
        return_inverse=True,
    )
    dose_sums = _group_sums(dose_index, dose_codes.size, values)
    dose_counts = np.bincount(dose_index, minlength=dose_codes.size)
    group_sums = _group_sums(group_index, group_codes.size, values)
    group_counts = np.bincount(group_index, minlength=group_codes.size)

    records: list[dict[str, object]] = []
    for start in range(0, n_rows, query_block):
        stop = min(start + query_block, n_rows)
        block = np.arange(start, stop)
        if metric == "l2":
            similarity = negative_l2_scores(
                finite_values[block], finite_values, square_norms[block], square_norms
            )
        else:
            similarity = unit[block] @ unit.T
        similarity[:, ~valid] = np.nan

        peer_counts = dose_counts[dose_index[block]] - group_counts[group_index[block]]
        with np.errstate(invalid="ignore", divide="ignore"):
            centroids = (
                dose_sums[dose_index[block]] - group_sums[group_index[block]]
            ) / peer_counts[:, None]
        if metric == "l2":
            centroid_valid = peer_counts > 0
            safe_centroids = np.where(centroid_valid[:, None], centroids, 0.0)
            centroid_valid &= np.isfinite(safe_centroids).all(axis=1)
            centroid_scores = -np.linalg.norm(finite_values[block] - safe_centroids, axis=1)
        else:
            centroid_unit, centroid_valid = similarity_ready_rows(
                np.where(peer_counts[:, None] > 0, centroids, 0.0), metric
            )
            centroid_valid &= peer_counts > 0
            centroid_scores = np.einsum("ij,ij->i", unit[block], centroid_unit)

        for offset, query in enumerate(block):
            if not valid[query]:
                continue
            candidate = np.ones(n_rows, dtype=bool)
            candidate[query] = False
            candidate &= valid
            scores = similarity[offset][candidate]
            positive = (condition == condition[query])[candidate]
            n_candidates = int(scores.size)
            n_positives = int(positive.sum())
            if n_positives == 0 or n_candidates < 2:
                continue

            rank = best_positive_rank(scores, positive)
            observed = normalized_rank(rank, n_candidates)
            record = {
                "row": int(query),
                "n_candidates": n_candidates,
                "n_positives": n_positives,
                "observed_normalized_rank": observed,
                "recall_at_1": float(rank == 1.0),
                "auroc": auroc(scores, positive),
                "null_expected_normalized_rank": null_expected_normalized_rank(
                    n_candidates, n_positives
                ),
            }
            record["observed_minus_null"] = observed - record["null_expected_normalized_rank"]

            if centroid_valid[offset]:
                # Injected as one extra candidate: rank among N + 1, as the
                # original replicate baseline did.
                centroid_rank = 1.0 + float(np.sum(scores > centroid_scores[offset]))
                centroid_normalized = normalized_rank(centroid_rank, n_candidates + 1)
            else:
                centroid_normalized = float("nan")
            record["centroid_normalized_rank"] = centroid_normalized
            record["observed_minus_centroid"] = scorer.difference_if_both_defined(
                observed, centroid_normalized
            )
            record["n_centroid_peers"] = int(peer_counts[offset])
            records.append(record)
    return records


QUERY_METRICS = (
    "observed_normalized_rank",
    "recall_at_1",
    "auroc",
    "null_expected_normalized_rank",
    "observed_minus_null",
    "centroid_normalized_rank",
    "observed_minus_centroid",
)


def condition_level(rows: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    """Average each condition's replicate queries, so conditions weigh equally."""
    if not records:
        return pd.DataFrame()
    per_query = pd.DataFrame(records)
    meta = rows.reset_index(drop=True).iloc[per_query["row"].to_numpy()][CONDITION_KEYS]
    per_query = pd.concat([meta.reset_index(drop=True), per_query.drop(columns="row")], axis=1)
    return (
        per_query.groupby(CONDITION_KEYS, as_index=False)
        .agg(
            n_queries=("observed_normalized_rank", "size"),
            n_candidates=("n_candidates", "max"),
            n_positives=("n_positives", "mean"),
            n_centroid_peers=("n_centroid_peers", "mean"),
            **{metric: (metric, "mean") for metric in QUERY_METRICS},
        )
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def load_dataset_rows(prepared_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Replicate rows of retained conditions, in a deterministic order."""
    eligible_path = (
        prepared_dir / "task_inputs" / "dataset_metadata_cache"
        / f"{dataset_name}_eligible_rows.tsv"
    )
    rows = pd.read_csv(eligible_path, sep="\t", keep_default_na=False, low_memory=False)
    rows = scorer.normalize_source_metadata_frame(rows)
    retained = pd.read_csv(
        prepared_dir / "retained_replicate_conditions.tsv", sep="\t",
        keep_default_na=False, low_memory=False,
    )
    retained = retained[
        (retained["dataset_name"].astype(str) == dataset_name)
        & (retained["retain_for_eval"].astype(str).str.lower() == "true")
    ]
    rows = rows[rows["condition_key"].astype(str).isin(set(retained["condition_key"].astype(str)))]
    # Only conditions with a second replicate have anything to retrieve.
    replicate_counts = rows.groupby("condition_key")["condition_key"].transform("size")
    rows = rows[replicate_counts >= 2]
    return rows.sort_values(
        ["cell_type", "time_key", "condition_key", "source_path", "source_row_pos"]
    ).reset_index(drop=True)


def stratum_path(output_dir: Path, dataset_name: str, cell_type: str, time_key: str) -> Path:
    safe = "__".join(
        str(part).replace("/", "_").replace(" ", "_")
        for part in (dataset_name, cell_type, time_key)
    )
    return output_dir / STRATUM_DIR_NAME / f"{safe}.tsv"


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [value.strip() for value in args.similarity.split(",") if value.strip()]
    scales = [value.strip() for value in args.scales.split(",") if value.strip()]
    for value in metrics:
        if value not in SIMILARITY_METRICS:
            raise SystemExit(f"--similarity: unknown {value!r}; choose from {SIMILARITY_METRICS}")
    for value in scales:
        if value not in SCALE_VARIANTS:
            raise SystemExit(f"--scales: unknown {value!r}; choose from {SCALE_VARIANTS}")

    config = {
        "prepared_dir": str(prepared_dir),
        "similarity": metrics,
        "scales": scales,
        "min_compounds_per_stratum": int(args.min_compounds_per_stratum),
        "population_stats_root": str(Path(args.population_stats_root).resolve()),
    }
    config_path = output_dir / "retrieval_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config:
            raise SystemExit(
                f"{output_dir} was started with different settings; reusing its "
                f"strata would mix them.\n saved: {saved}\n asked: {config}"
            )
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    stats_cache = (
        scorer.ReplicatePopulationStatsCache(Path(args.population_stats_root))
        if any(scale != "raw" for scale in scales)
        else None
    )
    datasets = (
        [value.strip() for value in args.datasets.split(",") if value.strip()]
        if args.datasets
        else sorted(
            path.name[: -len("_eligible_rows.tsv")]
            for path in (prepared_dir / "task_inputs" / "dataset_metadata_cache").glob(
                "*_eligible_rows.tsv"
            )
        )
    )

    for dataset_name in datasets:
        rows = load_dataset_rows(prepared_dir, dataset_name)
        strata = list(rows.groupby(["cell_type", "time_key"], sort=True))
        print(f"[retrieval] {dataset_name}: {len(rows):,} replicate rows in {len(strata)} strata", flush=True)
        open_adatas: dict = {}
        for (cell_type, time_key), stratum in strata:
            target = stratum_path(output_dir, dataset_name, str(cell_type), str(time_key))
            if target.exists():
                continue
            stratum = stratum.reset_index(drop=True)
            n_compounds = int(stratum["pubchem_cid"].astype(str).nunique())
            if n_compounds < int(args.min_compounds_per_stratum):
                write_atomic(pd.DataFrame(), target)  # mark as considered
                continue
            started = time.monotonic()
            matrix, gene_keys = load_stratum_matrix(stratum, open_adatas)
            frames = []
            if matrix is not None and matrix.shape[1] >= 2:
                for scale in scales:
                    values = scaled_matrix(
                        matrix, gene_keys, scale=scale, dataset_name=dataset_name,
                        cell_type=str(cell_type), stats_cache=stats_cache,
                    )
                    if values.shape[1] < 2:
                        continue
                    for metric in metrics:
                        records = score_stratum(
                            stratum, values, metric=metric, query_block=int(args.query_block)
                        )
                        frame = condition_level(stratum, records)
                        if frame.empty:
                            continue
                        frame.insert(4, "similarity_metric", metric)
                        frame.insert(5, "scale_variant", SCALE_LABELS[scale])
                        frame["n_genes"] = int(values.shape[1])
                        frames.append(frame)
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            write_atomic(result, target)
            print(
                f"[retrieval] {dataset_name} {cell_type}/{time_key}: "
                f"{len(stratum):,} replicates, {n_compounds:,} compounds, "
                f"{0 if matrix is None else matrix.shape[1]:,} genes, "
                f"{time.monotonic() - started:.0f}s",
                flush=True,
            )
        for adata in open_adatas.values():
            adata.close()

    frames = []
    for path in sorted((output_dir / STRATUM_DIR_NAME).glob("*.tsv")):
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path, sep="\t", keep_default_na=False, dtype={"pubchem_cid": str})
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_path = output_dir / CONDITION_SUMMARY_NAME
    write_atomic(summary, summary_path)
    print(f"[retrieval] wrote {len(summary):,} condition rows to {summary_path}", flush=True)
    return summary_path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Output metric name -> condition-level column. Named to read alongside
# cross-dataset Table 9.
TABLE_METRICS = {
    "observed_normalized_rank": "observed_normalized_rank",
    "recall_at_1": "recall_at_1",
    "auroc": "auroc",
    "null_expected_normalized_rank": "null_expected_normalized_rank",
    "observed_minus_null": "observed_minus_null",
    "centroid_normalized_rank": "centroid_normalized_rank",
    "delta_vs_centroid": "observed_minus_centroid",
}
TABLE_COLUMNS = {
    "observed_normalized_rank": "Observed rank [95% CI]",
    "recall_at_1": "Recall@1 [95% CI]",
    "auroc": "AUROC [95% CI]",
    "null_expected_normalized_rank": "Null rank",
    "centroid_normalized_rank": "Centroid rank [95% CI]",
    "delta_vs_centroid": "Delta centroid [95% CI]",
    "observed_minus_null": "Observed minus null [95% CI]",
}
GROUP_COLUMNS = ["dataset_name", "similarity_metric", "scale_variant"]


def build_tables(
    summary_path: Path,
    output_dir: Path,
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 20260505,
    workers: int = 4,
) -> tuple[Path, Path]:
    """Compound-clustered BCa intervals, as for Tables 7, 8 and 10.

    Conditions are resampled by PubChem CID, so a compound tested at several
    doses counts once toward the uncertainty rather than once per dose.
    """
    from cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table
    from replicate_reviewer_tables import format_estimate

    frame = pd.read_csv(summary_path, sep="\t", dtype={"pubchem_cid": str}, low_memory=False)
    ci = cluster_bca_nested_mean_ci_table(
        frame,
        group_cols=GROUP_COLUMNS,
        metric_cols=TABLE_METRICS,
        cluster_col="pubchem_cid",
        n_boot=int(bootstrap_iterations),
        seed=int(seed),
        summary_level="replicate_retrieval_dataset",
        workers=int(workers),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ci_path = output_dir / "replicate_retrieval_cluster_bca_ci.tsv"
    write_atomic(ci, ci_path)

    counts = frame.groupby(GROUP_COLUMNS).agg(
        n_conditions=("condition_key", "size"),
        n_compounds=("pubchem_cid", "nunique"),
        mean_candidates=("n_candidates", "mean"),
    ).reset_index()
    ci = ci.copy()
    ci["estimate"] = [
        format_estimate(mean, low, high) if metric != "null_expected_normalized_rank" else f"{mean:.3f}"
        for metric, mean, low, high in zip(ci["metric"], ci["mean"], ci["ci_low"], ci["ci_high"])
    ]
    wide = (
        ci[ci["metric"].isin(TABLE_COLUMNS)]
        .pivot_table(index=GROUP_COLUMNS, columns="metric", values="estimate", aggfunc="first")
        .reindex(columns=[name for name in TABLE_COLUMNS])
        .rename(columns=TABLE_COLUMNS)
        .reset_index()
    )
    table = counts.merge(wide, on=GROUP_COLUMNS, how="right")
    table_path = output_dir / "replicate_retrieval_table.tsv"
    write_atomic(table, table_path)
    return ci_path, table_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared-dir", type=Path,
                        help="A finished replicate-scoring run whose conditions to reuse. "
                             "Required unless --tables-only.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="",
                        help="Comma-separated subset; default is every dataset in --prepared-dir.")
    parser.add_argument("--similarity", default=",".join(SIMILARITY_METRICS))
    parser.add_argument("--scales", default=",".join(SCALE_VARIANTS))
    parser.add_argument("--min-compounds-per-stratum", type=int,
                        default=scorer.DEFAULT_MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME)
    parser.add_argument("--population-stats-root", type=Path,
                        default=scorer.DEFAULT_POPULATION_STATS_ROOT)
    parser.add_argument("--query-block", type=int, default=1024,
                        help="Queries per similarity block; bounds memory.")
    parser.add_argument("--threads", type=int, default=8,
                        help="BLAS threads for the similarity products.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--tables-only", action="store_true",
                        help="Rebuild the tables from an existing condition summary.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary_path = args.output_dir.resolve() / CONDITION_SUMMARY_NAME
    if not args.tables_only:
        if args.prepared_dir is None:
            parser.error("--prepared-dir is required unless --tables-only")
        summary_path = run(args)
    elif not summary_path.is_file():
        parser.error(f"--tables-only needs an existing {summary_path}")
    ci_path, table_path = build_tables(
        summary_path,
        args.output_dir.resolve() / "tables",
        bootstrap_iterations=args.bootstrap_iterations,
        workers=max(1, min(int(args.threads), 8)),
    )
    print(f"[retrieval] tables: {table_path}", flush=True)


if __name__ == "__main__":
    main()
