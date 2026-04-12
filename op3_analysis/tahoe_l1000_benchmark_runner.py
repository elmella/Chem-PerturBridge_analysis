from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd

from .tahoe_l1000_deg_benchmark import (
    DEFAULT_LABEL_CONFIGS,
    LabelConfig,
    build_prior_score_vector,
    build_same_compound_candidate_table,
    concordant_gene_table,
    discover_filtered_l1000_paths,
    discover_tahoe_overlap_cell_types,
    donor_compound_set,
    estimate_cellline_transfer_scores,
    estimate_cellline_transfer_scores_continuous,
    grouped_compound_folds,
    normalize_pubchem_cids,
    predict_donor_scores,
    predict_from_scores,
    prepare_filtered_l1000_adata,
    prepare_tahoe_overlap_adata,
    responsive_gene_table,
    summarize_continuous_metrics,
    summarize_prediction_metrics,
    tile_gene_labels,
    train_gene_majority_labels,
    tune_threshold,
    common_gene_set,
)


DEFAULT_THRESHOLD_GRID = (0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
DEFAULT_DEG_METHODS = (
    {"name": "majority_prior", "predict_method": "majority_prior", "top_k": 0},
    {"name": "top1", "predict_method": "top1", "top_k": 1},
    {"name": "top3", "predict_method": "topk", "top_k": 3},
    {"name": "top5", "predict_method": "topk", "top_k": 5},
    {"name": "top10", "predict_method": "topk", "top_k": 10},
    {"name": "best_cellline_mean", "predict_method": "best_cellline_mean", "top_k": 0},
    {"name": "all_donor_mean", "predict_method": "all_donor_mean", "top_k": 0},
    {"name": "top5_then_prior", "predict_method": "topk_then_prior", "top_k": 5},
)
DEFAULT_LFC_METHODS = (
    {"name": "zero", "predict_method": None, "top_k": 0},
    {"name": "train_mean", "predict_method": None, "top_k": 0},
    {"name": "top1", "predict_method": "top1", "top_k": 1},
    {"name": "top3", "predict_method": "topk", "top_k": 3},
    {"name": "top5", "predict_method": "topk", "top_k": 5},
    {"name": "top10", "predict_method": "topk", "top_k": 10},
    {"name": "best_cellline_mean", "predict_method": "best_cellline_mean", "top_k": 0},
    {"name": "all_donor_mean", "predict_method": "all_donor_mean", "top_k": 0},
)


def _log(verbose: bool, *parts: object) -> None:
    if verbose:
        print(*parts, flush=True)


def _label_config_frame(label_configs: Iterable[LabelConfig]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": config.name,
                "fdr_threshold": config.fdr_threshold,
                "min_abs_logfc": config.min_abs_logfc,
            }
            for config in label_configs
        ]
    )


def _safe_read_h5ad(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path)


def _prepared_paths(prep_dir: Path, l1000_paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "tahoe": prep_dir / "tahoe_shared_compounds_shared_genes.h5ad",
        **{name: prep_dir / f"{name}_shared_genes.h5ad" for name in l1000_paths},
    }


def _write_gene_subsets(
    prep_dir: Path,
    tahoe_adata: ad.AnnData,
    donor_adatas: dict[str, ad.AnnData],
    dataset_adatas: dict[str, ad.AnnData],
    strict_layer: str,
    verbose: bool,
) -> pd.DataFrame:
    gene_subset_dir = prep_dir / "gene_subsets"
    gene_subset_dir.mkdir(parents=True, exist_ok=True)

    responsive = responsive_gene_table(dataset_adatas, label_layer=strict_layer)
    responsive.to_csv(gene_subset_dir / "responsive_gene_stats.csv", index=False)

    shared_all = pd.DataFrame({"gene_id": tahoe_adata.var_names.to_numpy(dtype=str)})
    shared_all.to_csv(gene_subset_dir / "shared_all_genes.csv", index=False)

    responsive_wide = (
        responsive.pivot_table(index="gene_id", columns="dataset_name", values="n_nonzero", fill_value=0)
        .reset_index()
    )
    donor_columns = [col for col in responsive_wide.columns if col not in {"gene_id", "tahoe"}]
    responsive_wide["l1000_total_nonzero"] = responsive_wide[donor_columns].sum(axis=1)
    frequent_responsive = responsive_wide.loc[
        (responsive_wide.get("tahoe", 0) >= 5) & (responsive_wide["l1000_total_nonzero"] >= 10),
        ["gene_id", "tahoe", "l1000_total_nonzero"],
    ].sort_values(["tahoe", "l1000_total_nonzero", "gene_id"], ascending=[False, False, True])
    frequent_responsive.to_csv(gene_subset_dir / "frequent_responsive_genes.csv", index=False)

    concordant = concordant_gene_table(
        tahoe_adata=tahoe_adata,
        donor_adatas=donor_adatas,
        label_layer=strict_layer,
        min_support=10,
    )
    concordant.to_csv(gene_subset_dir / "concordant_gene_scores.csv", index=False)
    concordant_subset = concordant.loc[
        (concordant["sign_agree_fraction"] >= 0.60) & (concordant["n_nonzero_both"] >= 5)
    ].copy()
    concordant_subset.to_csv(gene_subset_dir / "concordant_genes.csv", index=False)

    subset_manifest = pd.DataFrame(
        [
            {
                "subset_name": "shared_all",
                "n_genes": len(shared_all),
                "path": str(gene_subset_dir / "shared_all_genes.csv"),
            },
            {
                "subset_name": "frequent_responsive",
                "n_genes": len(frequent_responsive),
                "path": str(gene_subset_dir / "frequent_responsive_genes.csv"),
            },
            {
                "subset_name": "concordant",
                "n_genes": len(concordant_subset),
                "path": str(gene_subset_dir / "concordant_genes.csv"),
            },
        ]
    )
    subset_manifest.to_csv(gene_subset_dir / "subset_manifest.csv", index=False)
    _log(verbose, "wrote gene subsets", subset_manifest.to_dict("records"))
    return subset_manifest


def prepare_benchmark_inputs(
    filtered_dir: Path | str,
    prep_dir: Path | str,
    label_configs: Iterable[LabelConfig] = DEFAULT_LABEL_CONFIGS,
    strict_layer: str = "deg_fdr005_lfc025",
    verbose: bool = False,
) -> dict[str, Path]:
    start = time.time()
    filtered_root = Path(filtered_dir)
    prep_root = Path(prep_dir)
    prep_root.mkdir(parents=True, exist_ok=True)

    _log(verbose, "discovering filtered donors")
    l1000_paths = discover_filtered_l1000_paths(filtered_root)
    overlap_cell_types = discover_tahoe_overlap_cell_types(l1000_paths)
    shared_genes = common_gene_set(l1000_paths, overlap_cell_types)
    shared_compounds = donor_compound_set(l1000_paths)
    _log(verbose, "donor datasets", sorted(l1000_paths))
    _log(verbose, "overlap cell types", overlap_cell_types)
    _log(verbose, "shared compounds", len(shared_compounds))
    _log(verbose, "shared genes", len(shared_genes))

    prepared_paths = _prepared_paths(prep_root, l1000_paths)
    donor_adatas: dict[str, ad.AnnData] = {}

    for name, path in l1000_paths.items():
        prepared_path = prepared_paths[name]
        if prepared_path.exists():
            _log(verbose, "loading cached donor", name, prepared_path)
            donor_adatas[name] = _safe_read_h5ad(prepared_path)
            continue
        _log(verbose, "preparing donor", name, path)
        donor_adata = prepare_filtered_l1000_adata(
            path=path,
            dataset_name=name,
            common_genes=shared_genes,
            label_configs=label_configs,
        )
        donor_adata.write_h5ad(prepared_path)
        donor_adatas[name] = donor_adata
        _log(verbose, "prepared donor", name, donor_adata.shape)

    tahoe_path = prepared_paths["tahoe"]
    if tahoe_path.exists():
        _log(verbose, "loading cached tahoe", tahoe_path)
        tahoe_adata = _safe_read_h5ad(tahoe_path)
    else:
        _log(verbose, "preparing tahoe shared-compound overlap")
        tahoe_adata = prepare_tahoe_overlap_adata(
            l1000_paths=l1000_paths,
            output_cell_types=overlap_cell_types,
            label_configs=label_configs,
        )
        tahoe_adata.write_h5ad(tahoe_path)
        _log(verbose, "prepared tahoe", tahoe_adata.shape)

    dataset_adatas = {"tahoe": tahoe_adata, **donor_adatas}
    label_summary_rows = []
    label_config_df = _label_config_frame(label_configs)
    for dataset_name, adata in dataset_adatas.items():
        _log(
            verbose,
            "dataset",
            dataset_name,
            "shape",
            adata.shape,
            "cell_types",
            adata.obs["cell_type"].astype(str).nunique(),
            "compounds",
            normalize_pubchem_cids(adata.obs["pubchem_cid"]).dropna().astype(str).nunique(),
        )
        for config in label_config_df.to_dict("records"):
            labels = np.asarray(adata.layers[config["name"]], dtype=np.int8)
            label_summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "label_layer": config["name"],
                    "fdr_threshold": config["fdr_threshold"],
                    "min_abs_logfc": config["min_abs_logfc"],
                    "n_up": int((labels > 0).sum()),
                    "n_down": int((labels < 0).sum()),
                    "n_nonzero": int((labels != 0).sum()),
                    "nonzero_fraction": float((labels != 0).mean()),
                }
            )
    label_summary = pd.DataFrame(label_summary_rows)
    label_summary.to_csv(prep_root / "label_summary.csv", index=False)
    subset_manifest = _write_gene_subsets(
        prep_dir=prep_root,
        tahoe_adata=tahoe_adata,
        donor_adatas=donor_adatas,
        dataset_adatas=dataset_adatas,
        strict_layer=strict_layer,
        verbose=verbose,
    )
    metadata = {
        "shared_genes": int(len(shared_genes)),
        "shared_compounds": int(len(shared_compounds)),
        "overlap_cell_types": overlap_cell_types,
        "label_layers": label_config_df.to_dict("records"),
        "gene_subsets": subset_manifest.to_dict("records"),
        "runtime_seconds": round(time.time() - start, 3),
    }
    (prep_root / "benchmark_prep_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(verbose, "finished prep in seconds", metadata["runtime_seconds"])
    return prepared_paths


def load_prepared_inputs(prep_dir: Path | str) -> tuple[ad.AnnData, dict[str, ad.AnnData], pd.DataFrame]:
    prep_root = Path(prep_dir)
    tahoe_path = prep_root / "tahoe_shared_compounds_shared_genes.h5ad"
    donor_paths = {
        path.name.replace("_shared_genes.h5ad", ""): path
        for path in sorted(prep_root.glob("l1000*_shared_genes.h5ad"))
    }
    if not tahoe_path.exists():
        raise FileNotFoundError(f"Missing prepared Tahoe file: {tahoe_path}")
    if not donor_paths:
        raise FileNotFoundError(f"No prepared donor files found under {prep_root}")
    tahoe_adata = ad.read_h5ad(tahoe_path)
    donor_adatas = {name: ad.read_h5ad(path) for name, path in donor_paths.items()}
    tahoe_obs = tahoe_adata.obs.copy().reset_index().rename(columns={"index": "tahoe_obs_name"})
    tahoe_obs["query_row"] = np.arange(tahoe_adata.n_obs, dtype=np.int64)
    tahoe_obs["pubchem_cid"] = normalize_pubchem_cids(tahoe_obs["pubchem_cid"]).astype("string")
    return tahoe_adata, donor_adatas, tahoe_obs


def load_gene_subsets(
    prep_dir: Path | str,
    var_names: pd.Index,
    subset_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    prep_root = Path(prep_dir)
    gene_subset_dir = prep_root / "gene_subsets"
    manifest_path = gene_subset_dir / "subset_manifest.csv"
    var_index = pd.Index(var_names.astype(str))

    subsets: dict[str, np.ndarray] = {
        "shared_all": np.arange(var_index.size, dtype=np.int64),
    }
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            subset_name = str(row.subset_name)
            path = Path(row.path)
            if not path.exists():
                continue
            table = pd.read_csv(path)
            if "gene_id" not in table.columns or table.empty:
                subsets[subset_name] = np.array([], dtype=np.int64)
                continue
            gene_ids = pd.Index(table["gene_id"].astype(str))
            idx = var_index.get_indexer(gene_ids)
            subsets[subset_name] = idx[idx >= 0].astype(np.int64, copy=False)
    if subset_names is not None:
        requested = [str(name) for name in subset_names]
        return {name: subsets[name] for name in requested if name in subsets}
    return subsets


def _candidate_table_for_eval(
    tahoe_adata: ad.AnnData,
    donor_adatas: dict[str, ad.AnnData],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], np.ndarray]:
    candidate_table = build_same_compound_candidate_table(tahoe_adata, donor_adatas)
    if candidate_table.empty:
        raise RuntimeError("No Tahoe rows could be matched to donor rows.")
    candidate_lookup = {
        int(query_row): frame.reset_index(drop=True)
        for query_row, frame in candidate_table.groupby("query_row", sort=False)
    }
    valid_rows = np.sort(candidate_table["query_row"].unique().astype(np.int64))
    return candidate_table, candidate_lookup, valid_rows


def _outer_folds_for_rows(
    compounds: pd.Series,
    valid_rows: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    compound_values = normalize_pubchem_cids(compounds).fillna("").astype(str).to_numpy()
    unique_compounds = np.unique(compound_values)
    if unique_compounds.size < 2:
        raise RuntimeError("Need at least two compounds to create evaluation folds.")
    local_folds = grouped_compound_folds(
        compound_values,
        n_splits=max(2, min(int(n_splits), int(unique_compounds.size))),
        seed=seed,
    )
    return [(valid_rows[train_local], valid_rows[test_local]) for train_local, test_local in local_folds]


def _inner_folds_for_rows(
    train_rows: np.ndarray,
    train_compounds: pd.Series,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    compound_values = normalize_pubchem_cids(train_compounds).fillna("").astype(str).to_numpy()
    unique_compounds = np.unique(compound_values)
    if unique_compounds.size < 2:
        return []
    local_folds = grouped_compound_folds(
        compound_values,
        n_splits=max(2, min(int(n_splits), int(unique_compounds.size))),
        seed=seed,
    )
    return [(train_rows[inner_train], train_rows[inner_val]) for inner_train, inner_val in local_folds]


def _aggregate_deg_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby(["label_layer", "subset_name", "method_name"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            total_test_samples=("n_samples", "sum"),
            n_genes=("n_genes", "first"),
            macro_f1_global_mean=("macro_f1_global", "mean"),
            macro_f1_global_std=("macro_f1_global", "std"),
            macro_f1_per_sample_mean=("macro_f1_mean_per_sample", "mean"),
            signed_jaccard_mean=("signed_jaccard_mean_per_sample", "mean"),
            tuned_threshold_mean=("tuned_threshold", "mean"),
            tuned_threshold_median=("tuned_threshold", "median"),
        )
        .sort_values(
            ["label_layer", "subset_name", "macro_f1_global_mean", "signed_jaccard_mean", "method_name"],
            ascending=[True, True, False, False, True],
            ignore_index=True,
        )
    )


def _aggregate_lfc_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_metrics.groupby(["subset_name", "method_name"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            total_test_samples=("n_samples", "sum"),
            n_genes=("n_genes", "first"),
            pearson_mean=("pearson_mean", "mean"),
            pearson_std=("pearson_mean", "std"),
            spearman_mean=("spearman_mean", "mean"),
            cosine_mean=("cosine_mean", "mean"),
            mrrmse_mean=("mrrmse_mean", "mean"),
            signed_overlap_mean=("signed_overlap_mean", "mean"),
        )
        .sort_values(
            ["subset_name", "pearson_mean", "signed_overlap_mean", "method_name"],
            ascending=[True, False, False, True],
            ignore_index=True,
        )
    )


def run_deg_benchmarks(
    prep_dir: Path | str,
    results_dir: Path | str,
    label_layers: Iterable[str] | None = None,
    subset_names: Iterable[str] | None = None,
    threshold_grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
    fixed_threshold: float | None = None,
    n_outer_splits: int = 5,
    n_inner_splits: int = 4,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    start = time.time()
    prep_root = Path(prep_dir)
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    tahoe_adata, donor_adatas, tahoe_obs = load_prepared_inputs(prep_root)
    candidate_table, candidate_lookup, valid_rows = _candidate_table_for_eval(tahoe_adata, donor_adatas)
    outer_folds = _outer_folds_for_rows(
        compounds=tahoe_obs.iloc[valid_rows]["pubchem_cid"],
        valid_rows=valid_rows,
        n_splits=n_outer_splits,
        seed=seed,
    )
    subsets = load_gene_subsets(prep_root, tahoe_adata.var_names, subset_names=subset_names)
    label_config_df = pd.read_csv(prep_root / "label_summary.csv")[["label_layer", "fdr_threshold", "min_abs_logfc"]]
    available_label_layers = label_config_df["label_layer"].drop_duplicates().tolist()
    selected_label_layers = list(label_layers) if label_layers is not None else available_label_layers

    fold_rows: list[dict[str, object]] = []
    transfer_rows: list[pd.DataFrame] = []
    _log(verbose, "deg outer folds", len(outer_folds), "valid queries", len(valid_rows))

    for label_layer in selected_label_layers:
        tahoe_labels_all = np.asarray(tahoe_adata.layers[label_layer], dtype=np.int8)
        donor_labels_all = {name: np.asarray(adata.layers[label_layer], dtype=np.int8) for name, adata in donor_adatas.items()}
        tahoe_scores_all = np.asarray(tahoe_adata.layers["signed_score.shared"], dtype=np.float32)
        donor_scores_all = {
            name: np.asarray(adata.layers["signed_score.shared"], dtype=np.float32)
            for name, adata in donor_adatas.items()
        }

        for subset_name, gene_idx in subsets.items():
            gene_idx = np.asarray(gene_idx, dtype=np.int64)
            if gene_idx.size == 0:
                continue
            tahoe_labels = tahoe_labels_all[:, gene_idx]
            donor_labels = {name: matrix[:, gene_idx] for name, matrix in donor_labels_all.items()}
            donor_scores = {name: matrix[:, gene_idx] for name, matrix in donor_scores_all.items()}
            _log(verbose, "deg label", label_layer, "subset", subset_name, "genes", gene_idx.size)

            for fold_idx, (train_rows, test_rows) in enumerate(outer_folds):
                y_train = tahoe_labels[train_rows]
                y_test = tahoe_labels[test_rows]
                outer_transfer = estimate_cellline_transfer_scores(
                    candidate_table=candidate_table,
                    tahoe_labels=tahoe_labels,
                    donor_labels_by_dataset=donor_labels,
                    train_query_rows=train_rows,
                )
                if not outer_transfer.empty:
                    outer_transfer = outer_transfer.copy()
                    outer_transfer["fold"] = fold_idx
                    outer_transfer["label_layer"] = label_layer
                    outer_transfer["subset_name"] = subset_name
                    transfer_rows.append(outer_transfer)

                if fixed_threshold is None:
                    inner_folds = _inner_folds_for_rows(
                        train_rows=train_rows,
                        train_compounds=tahoe_obs.iloc[train_rows]["pubchem_cid"],
                        n_splits=n_inner_splits,
                        seed=seed + fold_idx + 1,
                    )
                    inner_transfer_scores = [
                        estimate_cellline_transfer_scores(
                            candidate_table=candidate_table,
                            tahoe_labels=tahoe_labels,
                            donor_labels_by_dataset=donor_labels,
                            train_query_rows=inner_train_rows,
                        )
                        for inner_train_rows, _ in inner_folds
                    ]
                else:
                    inner_folds = []
                    inner_transfer_scores = []
                majority_labels = train_gene_majority_labels(y_train)
                prior_score_vector = build_prior_score_vector(y_train)

                for method_spec in DEFAULT_DEG_METHODS:
                    method_name = str(method_spec["name"])
                    predict_method = str(method_spec["predict_method"])
                    top_k = int(method_spec["top_k"])
                    if method_name == "majority_prior":
                        y_pred = tile_gene_labels(majority_labels, len(test_rows))
                        tuned_threshold = np.nan
                    else:
                        tuned_threshold = (
                            float(fixed_threshold)
                            if fixed_threshold is not None
                            else tune_threshold(
                                score_method=predict_method,
                                threshold_grid=threshold_grid,
                                inner_splits=inner_folds,
                                tahoe_obs=tahoe_obs,
                                y_true=tahoe_labels,
                                candidate_table=candidate_lookup,
                                donor_score_by_dataset=donor_scores,
                                transfer_scores_by_inner_fold=inner_transfer_scores,
                                history_column="median_macro_f1",
                                top_k=top_k,
                            )
                        )
                        pred_scores = predict_donor_scores(
                            query_rows=test_rows,
                            tahoe_obs=tahoe_obs,
                            candidate_table=candidate_lookup,
                            donor_score_by_dataset=donor_scores,
                            method=predict_method,
                            transfer_scores=outer_transfer,
                            history_column="median_macro_f1",
                            top_k=top_k,
                            prior_score_vector=prior_score_vector,
                        )
                        y_pred = predict_from_scores(pred_scores, threshold=tuned_threshold)
                        if predict_method == "topk_then_prior":
                            fallback = tile_gene_labels(majority_labels, len(test_rows))
                            undecided = y_pred == 0
                            y_pred = np.where(undecided, fallback, y_pred)

                    metrics = summarize_prediction_metrics(y_test, y_pred)
                    metrics.update(
                        {
                            "fold": fold_idx,
                            "label_layer": label_layer,
                            "subset_name": subset_name,
                            "method_name": method_name,
                            "tuned_threshold": float(tuned_threshold) if np.isfinite(tuned_threshold) else np.nan,
                            "train_samples": int(len(train_rows)),
                            "test_samples": int(len(test_rows)),
                            "train_compounds": int(
                                normalize_pubchem_cids(tahoe_obs.iloc[train_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                            ),
                            "test_compounds": int(
                                normalize_pubchem_cids(tahoe_obs.iloc[test_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                            ),
                        }
                    )
                    fold_rows.append(metrics)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _aggregate_deg_summary(fold_metrics)
    transfer_scores = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame()

    fold_metrics.to_csv(results_root / "deg_fold_metrics.csv", index=False)
    summary.to_csv(results_root / "deg_summary.csv", index=False)
    if not transfer_scores.empty:
        transfer_scores.to_csv(results_root / "deg_transfer_scores.csv", index=False)

    metadata = {
        "runtime_seconds": round(time.time() - start, 3),
        "n_outer_folds": len(outer_folds),
        "threshold_grid": list(threshold_grid),
        "fixed_threshold": None if fixed_threshold is None else float(fixed_threshold),
        "label_layers": selected_label_layers,
        "subset_names": list(subsets),
        "methods": [row["name"] for row in DEFAULT_DEG_METHODS],
    }
    (results_root / "deg_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(verbose, "finished DEG benchmark in seconds", metadata["runtime_seconds"])
    return {"fold_metrics": fold_metrics, "summary": summary, "transfer_scores": transfer_scores}


def run_lfc_benchmarks(
    prep_dir: Path | str,
    results_dir: Path | str,
    subset_names: Iterable[str] | None = None,
    n_outer_splits: int = 5,
    seed: int = 0,
    history_metric: str = "pearson",
    top_k_overlap: int = 50,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    start = time.time()
    prep_root = Path(prep_dir)
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    tahoe_adata, donor_adatas, tahoe_obs = load_prepared_inputs(prep_root)
    candidate_table, candidate_lookup, valid_rows = _candidate_table_for_eval(tahoe_adata, donor_adatas)
    outer_folds = _outer_folds_for_rows(
        compounds=tahoe_obs.iloc[valid_rows]["pubchem_cid"],
        valid_rows=valid_rows,
        n_splits=n_outer_splits,
        seed=seed,
    )
    subsets = load_gene_subsets(prep_root, tahoe_adata.var_names, subset_names=subset_names)
    tahoe_logfc_all = np.asarray(tahoe_adata.layers["logFC"], dtype=np.float32)
    donor_logfc_all = {name: np.asarray(adata.layers["logFC"], dtype=np.float32) for name, adata in donor_adatas.items()}

    fold_rows: list[dict[str, object]] = []
    transfer_rows: list[pd.DataFrame] = []
    _log(verbose, "lfc outer folds", len(outer_folds), "valid queries", len(valid_rows))

    for subset_name, gene_idx in subsets.items():
        gene_idx = np.asarray(gene_idx, dtype=np.int64)
        if gene_idx.size == 0:
            continue
        tahoe_logfc = tahoe_logfc_all[:, gene_idx]
        donor_logfc = {name: matrix[:, gene_idx] for name, matrix in donor_logfc_all.items()}
        _log(verbose, "lfc subset", subset_name, "genes", gene_idx.size)

        for fold_idx, (train_rows, test_rows) in enumerate(outer_folds):
            x_train = tahoe_logfc[train_rows]
            x_test = tahoe_logfc[test_rows]
            outer_transfer = estimate_cellline_transfer_scores_continuous(
                candidate_table=candidate_table,
                tahoe_matrix=tahoe_logfc,
                donor_matrix_by_dataset=donor_logfc,
                train_query_rows=train_rows,
                metric=history_metric,
            )
            if not outer_transfer.empty:
                outer_transfer = outer_transfer.copy()
                outer_transfer["fold"] = fold_idx
                outer_transfer["subset_name"] = subset_name
                transfer_rows.append(outer_transfer)

            train_mean_vector = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
            for method_spec in DEFAULT_LFC_METHODS:
                method_name = str(method_spec["name"])
                predict_method = method_spec["predict_method"]
                top_k = int(method_spec["top_k"])

                if method_name == "zero":
                    x_pred = np.zeros_like(x_test, dtype=np.float32)
                elif method_name == "train_mean":
                    x_pred = np.broadcast_to(train_mean_vector, x_test.shape).copy()
                else:
                    x_pred = predict_donor_scores(
                        query_rows=test_rows,
                        tahoe_obs=tahoe_obs,
                        candidate_table=candidate_lookup,
                        donor_score_by_dataset=donor_logfc,
                        method=str(predict_method),
                        transfer_scores=outer_transfer,
                        history_column=f"median_{history_metric}",
                        top_k=top_k,
                        prior_score_vector=train_mean_vector,
                    )

                metrics = summarize_continuous_metrics(x_test, x_pred, top_k=top_k_overlap)
                metrics.update(
                    {
                        "fold": fold_idx,
                        "subset_name": subset_name,
                        "method_name": method_name,
                        "train_samples": int(len(train_rows)),
                        "test_samples": int(len(test_rows)),
                        "train_compounds": int(
                            normalize_pubchem_cids(tahoe_obs.iloc[train_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                        ),
                        "test_compounds": int(
                            normalize_pubchem_cids(tahoe_obs.iloc[test_rows]["pubchem_cid"]).dropna().astype(str).nunique()
                        ),
                    }
                )
                fold_rows.append(metrics)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _aggregate_lfc_summary(fold_metrics)
    transfer_scores = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame()

    fold_metrics.to_csv(results_root / "lfc_fold_metrics.csv", index=False)
    summary.to_csv(results_root / "lfc_summary.csv", index=False)
    if not transfer_scores.empty:
        transfer_scores.to_csv(results_root / "lfc_transfer_scores.csv", index=False)

    metadata = {
        "runtime_seconds": round(time.time() - start, 3),
        "n_outer_folds": len(outer_folds),
        "history_metric": history_metric,
        "top_k_overlap": int(top_k_overlap),
        "subset_names": list(subsets),
        "methods": [row["name"] for row in DEFAULT_LFC_METHODS],
    }
    (results_root / "lfc_metadata.json").write_text(json.dumps(metadata, indent=2))
    _log(verbose, "finished LFC benchmark in seconds", metadata["runtime_seconds"])
    return {"fold_metrics": fold_metrics, "summary": summary, "transfer_scores": transfer_scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Tahoe/L1000 benchmark prep and transfer evaluations.")
    parser.add_argument(
        "--mode",
        choices=("prepare", "deg", "lfc", "all"),
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--filtered-dir",
        type=Path,
        default=Path("data/l1000_narrowed_to_tahoe"),
        help="Filtered L1000 donor directory.",
    )
    parser.add_argument(
        "--prep-dir",
        type=Path,
        default=Path("data/tahoe_l1000_benchmark_prep"),
        help="Prepared shared-gene benchmark directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/tahoe_l1000_benchmarks"),
        help="Benchmark results directory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for grouped folds.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    args = parser.parse_args(argv)

    if args.mode in {"prepare", "all"}:
        prepare_benchmark_inputs(
            filtered_dir=args.filtered_dir,
            prep_dir=args.prep_dir,
            label_configs=DEFAULT_LABEL_CONFIGS,
            verbose=args.verbose,
        )
    if args.mode in {"deg", "all"}:
        run_deg_benchmarks(
            prep_dir=args.prep_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            verbose=args.verbose,
        )
    if args.mode in {"lfc", "all"}:
        run_lfc_benchmarks(
            prep_dir=args.prep_dir,
            results_dir=args.results_dir,
            seed=args.seed,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
