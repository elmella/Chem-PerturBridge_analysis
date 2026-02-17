# op3_analysis

Cross-dataset retrieval benchmark for perturbation-response signatures.

## What Is Implemented

- Dataset loading for `sciplex`, `tahoe`, `l1000_phase1`, `l1000_phase2` using paths from `notebooks/load_data.ipynb`.
- Per-cell-type retrieval with gene-overlap alignment for each query/db cell-type pair.
- Global ranking against all db cell types (including non-matching cell types) for each query sample.
- Ground-truth matching based on exact `cell_type` + `perturbagen`, with nearest `pert_time_h` and `pert_dose_uM`.
- Layer-wise scoring over all shared layers except:
  - `CI.L`, `CI.R`, `stdev.scaled`, `stdev.unscaled`, `AveExpr`
  - `adj.P.Value.across_all_contrasts`, `adj.P.Value.within_one_contrast`
- Additional derived representations:
  - `-log10(p) * sign(logFC)`
  - `sign(logFC) * Phi^-1(1 - p/2)`
- Similarity/distance metrics:
  - `cosine`, `pearson`, `spearman`, `mrrmse`
- Rank normalization to `[0, 1]`:
  - `rank_normalized` (0 is best)
  - `retrieval_score` (1 is best)

## Run

```bash
python -m op3_analysis.retrieval.cli --verbose
```

Outputs are written to:

- `results/cross_dataset_retrieval_detail.csv`
- `results/cross_dataset_retrieval_summary_by_cell_type.csv`
- `results/cross_dataset_retrieval_summary_overall.csv`

## Useful Options

```bash
python -m op3_analysis.retrieval.cli \
  --query-datasets sciplex,tahoe \
  --db-datasets l1000_phase1,l1000_phase2 \
  --cell-types CVCL_0002,CVCL_0063 \
  --representations logFC,P.Value,signed,z \
  --metrics cosine,mrrmse \
  --skip-representations P.Value \
  --output-prefix my_run \
  --verbose
```

Limit compute for faster runs:

```bash
python -m op3_analysis.retrieval.cli \
  --representations logFC,signed \
  --metrics cosine
```

Override data paths:

```bash
python -m op3_analysis.retrieval.cli \
  --dataset-path sciplex=/my/sciplex/results \
  --dataset-path l1000_phase1=/my/l1000_phase1/results
```

## Visualization

Use `notebooks/visualize_retrieval_results.ipynb` to visualize retrieval outputs (`*_detail.csv`, `*_summary_by_cell_type.csv`, `*_summary_overall.csv`) for any run prefix (default in notebook: `my_run_fast`).
