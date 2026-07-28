# Chem-PerturBridge_analysis

Cross-dataset retrieval benchmark for perturbation-response signatures.

Analyses added in response to the NeurIPS review — per-peer and target-side baselines, and
dose-threshold sensitivity — are documented separately in
[REVIEWER_ADDITIONS.md](REVIEWER_ADDITIONS.md), which maps each reviewer point to the file
that addresses it.

## Reproducibility
### Environment

The reproducible Python environment is defined by `pyproject.toml`,
`uv.lock`, and `.python-version`. The checked-in lockfile recreates the
environment currently activated with:

```bash
source .venv/bin/activate
```

from the repository root. The `.venv/` directory itself is intentionally not
tracked.

To create or refresh the environment:

```bash
uv sync --locked
source .venv/bin/activate
python --version  # Python 3.9.18
```

If Python 3.9.18 is not available locally, install or point `uv` at a 3.9.18
interpreter first, then rerun `uv sync --locked`:

```bash
uv python install 3.9.18
uv sync --locked
```

Most batch scripts in this repository call `uv run`, so they can also be
launched without manually activating `.venv` as long as `uv sync --locked` has
been run from the repository root.

### Execution
1. `scripts/build_dataset_summary.py` for table 1 content
2. `scripts/overlap_filtered_h5ads.py` for getting pseudobulks of overlapping samples (excluding those overlapping only across l1000 phases as there are so many of these that they would dominate and differ from the other considered significantly)
3. `notebooks/plot_overlap_heatmaps.ipynb` for cross-dataset overlap plots in fig1. In each dataset pair, we skip contexts that have less than 10 compounds shared between the two.
4. `notebooks/overlap_group_rep_deg_metrics.ipynb`, `notebooks/overlap_group_rep_signature_similarity.ipynb` and `notebooks/overlap_group_rep_retrieval_metrics.ipynb` to plot cross-dataset agreement.
5. `SKIP_EXISTING_DATASETS=1 EXISTING_RESULTS_DIR=./results/replicate_signature_similarity_sep_rep OUTPUT_DIR=./results/replicate_signature_similarity_sep_rep_combined COMPUTE_BASELINE_METRICS=1 COMPUTE_DEG_METRICS=1 scripts/slurm/run_replicate_signature_similarity.sh`, `notebooks/replicate_signature_similarity.ipynb` and `notebooks/replicate_deg_metrics.ipynb` to compute the cross-replicate agreement.

### Parallel reviewer scoring

The expensive matched-pair scoring from the three reviewer notebooks can be run
without modifying the notebooks:

```bash
uv run python scripts/run_overlap_group_rep_deg_metrics.py
uv run python scripts/run_overlap_group_rep_signature_similarity.py
uv run python scripts/run_overlap_group_rep_retrieval_metrics.py
```

Each command defaults to two spawned workers. Use `--workers 1` for a serial
debug run, `--datasets tahoe,sciplex` for a subset, and `--run-tag NAME` for an
isolated named run. The primary outputs are:

- `deg_scored_metrics.tsv`
- `signature_scored_metrics.tsv`
- `retrieval_scored_metrics.tsv`

They are written below `results/parallel_cross_source/<analysis>/<run>/`.
Task inputs, completion markers, checkpoints, diagnostics, and run metadata stay
beside the primary TSV. Compatible completed shards are reused automatically;
`--force` recomputes them.

All three scorers support additive computation selection. Omit `--compute` to
preserve the command's complete default behavior, list the available components
with `--list-computations`, and repeat or comma-separate `--compute` to run only
the requested kernels:

```bash
uv run python scripts/run_overlap_group_rep_deg_metrics.py \
  --compute raw \
  --compute w4-dataset
uv run python scripts/run_overlap_group_rep_signature_similarity.py \
  --compute raw,w4-dataset
uv run python scripts/run_overlap_group_rep_retrieval_metrics.py \
  --compute raw-l2 \
  --run-tag raw_l2
```

The last command is the lightweight strict-logFC L2 run: it calculates observed
normalized rank, Recall@1, AUROC, and source/target baselines without running
W4, cosine, Spearman, legacy representations, or dose-aware variants. Selected
components are included in the run fingerprint. Use a distinct `--run-tag`
when experimenting so an existing production final TSV is not replaced.
The specialized `--workload peer-only` mode remains separate because it reads
an existing retrieval TSV instead of rescoring observations.

Every scorer also appends timestamped events to `progress.log` in its run
directory. In an interactive terminal, one coordinator-owned tqdm bar reports
completed and cached checkpoint tasks. Retrieval workers additionally report
the active context, scale, similarity metric, direction, and periodic query
counts, so a large context does not remain silent until completion. Use
`--progress always` to force the bar in a non-interactive job or
`--progress off` to suppress console progress; the log is written in every
mode.

W4 population statistics must exist before scoring. For the repository-default
data layout, prepare both scopes with:

```bash
uv run python scripts/precompute_population_zscore.py \
  --all-configured \
  --scope both \
  --workers 2
```

For custom `--data-root` inputs, a missing-cache error prints the exact
`--dataset-dir` command required for that run.

For the smallest reviewer-defensible run, score only the dataset-wide W4 scale
for Tables 4 and 6 and use the strict-logFC reviewer workload for Table 9:

```bash
uv run python scripts/precompute_population_zscore.py \
  --all-configured \
  --scope dataset \
  --workers 2
uv run python scripts/run_overlap_group_rep_deg_metrics.py \
  --w4-scales dataset
uv run python scripts/run_overlap_group_rep_signature_similarity.py \
  --w4-scales dataset
uv run python scripts/run_overlap_group_rep_retrieval_metrics.py \
  --workload reviewer-minimal \
  --max-baseline-peers 0
```

The retrieval workload above computes only cosine and Spearman for raw logFC
and dataset-wide per-gene W4 logFC. It reports unadjusted observed retrieval,
source/target individual baselines, source/target centroids, and the exact
random-rank expectation. It does not compute the legacy representations,
dose-aware variants, negative-L2 parity, or within-dataset controls.

After all three scorers finish, build the compound-clustered BCa tables and
independently rematched exact/2x/3x/10x dose analysis:

```bash
uv run python scripts/summarize_reviewer_minimal_metrics.py \
  --deg-metrics results/parallel_cross_source/deg/production/deg_scored_metrics.tsv \
  --signature-metrics results/parallel_cross_source/signature/production/signature_scored_metrics.tsv \
  --retrieval-metrics results/parallel_cross_source/retrieval/production/retrieval_scored_metrics.tsv \
  --overlap-dir results/overlap_filtered_h5ads \
  --output-dir results/parallel_cross_source/reviewer_minimal_summary
```

After the optional L2 and all-peer sensitivity runs have been copied locally,
finalize the combined Table 9 and make zero-match exact-dose comparisons
explicit without reopening any H5AD:

```bash
uv run python scripts/summarize_reviewer_final_tables.py \
  --bootstrap-iterations 2000 \
  --workers 4
```

This writes raw/W4 L2 BCa intervals, one combined numeric-long and wide Table 9,
seven presentation panels, the validated all-peer sensitivity summary, and
complete nine-pair dose summary/CI tables under
`results/parallel_cross_source/reviewer_final_summary/`.

### Reviewer Tables 7, 8, and 10 without notebooks

Compute the within-dataset replicate metrics once. The same condition shards
feed all three tables, so running separate expensive jobs is unnecessary.
The normalized cosine sensitivity reuses the existing per-gene population
statistics; it does not refit them:

```bash
uv run python scripts/precompute_replicate_signature_similarity.py run-all \
  --datasets all \
  --output-dir results/replicate_signature_similarity_sep_rep_peer_full \
  --compute-baseline-metrics \
  --compute-deg-metrics \
  --compute-normalized-cosine \
  --normalization-scales all \
  --population-stats-root results/w4_population_zscore_stats \
  --max-baseline-peers 0 \
  --workers 8 \
  --progress always
```

If the raw replicate run is already complete, calculate only the added cosine
fields in a new output directory and reuse every other condition-level field:

```bash
uv run python scripts/precompute_replicate_signature_similarity.py run-all \
  --datasets all \
  --output-dir results/replicate_signature_similarity_sep_rep_normalized_cosine \
  --existing-results-dir results/replicate_signature_similarity_sep_rep_peer_full \
  --compute-normalized-cosine \
  --normalization-scales all \
  --max-baseline-peers 0 \
  --workers 8 \
  --progress always
```

Here, `--max-baseline-peers 0` scores every eligible same-context
different-compound signature. Each task writes its TSV atomically, and the
final merge occurs only after every task succeeds.

Build all three compound-clustered BCa tables without executing the replicate
notebooks:

```bash
uv run python scripts/replicate_reviewer_tables.py \
  --input-dir results/replicate_signature_similarity_sep_rep_peer_full \
  --output-dir results/parallel_cross_source/replicate_reviewer_tables \
  --bootstrap-iterations 2000 \
  --workers 8 \
  --progress always
```

The table-specific entry points
`summarize_replicate_table_7.py`, `summarize_replicate_table_8.py`, and
`summarize_replicate_table_10.py` accept the same summary options. Outputs
retain observed and original centroid results and add individual-peer means,
standard deviations, corrected percentiles, observed-minus-peer deltas, and
PubChem-clustered confidence intervals. Table 10 additionally reports cosine
agreement after dataset-wide and dataset-by-cell-type per-gene population
normalization. Compatible completed summaries are validated and reused
automatically.

The retrieval command above scores every eligible individual peer. The default
cap remains 512 for bounded exploratory runs; eligible and scored peer counts
are recorded separately. The second dataset-by-cell-type W4 scale remains
available through `--w4-scales all`.

## What Is Implemented

- Dataset loading for `sciplex`, `tahoe`, `l1000_phase1`, `l1000_phase2` using paths from `notebooks/load_data.ipynb`.
- Per-cell-type retrieval with gene-overlap alignment for each query/db cell-type pair.
- Global ranking against all db cell types (including non-matching cell types) for each query sample.
- Ground-truth matching based on exact `cell_type` + `pubchem_cid`, with nearest `pert_time_h` and nearest `log(pert_dose_uM)` (falling back to linear dose distance for non-positive doses).
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
python -m chem_perturbridge_analysis.retrieval.cli --verbose
```

Outputs are written to:

- `results/cross_dataset_retrieval_detail.csv`
- `results/cross_dataset_retrieval_summary_by_cell_type.csv`
- `results/cross_dataset_retrieval_summary_overall.csv`

Verbose mode now logs progress at pair, cell-type, and representation levels (including elapsed time and rows produced).

## Useful Options

```bash
python -m chem_perturbridge_analysis.retrieval.cli \
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
python -m chem_perturbridge_analysis.retrieval.cli \
  --representations logFC,signed \
  --metrics cosine
```

Override data paths:

```bash
python -m chem_perturbridge_analysis.retrieval.cli \
  --dataset-path sciplex=/my/sciplex/results \
  --dataset-path l1000_phase1=/my/l1000_phase1/results
```

## Visualization

Use `notebooks/visualize_retrieval_results.ipynb` to visualize retrieval outputs (`*_detail.csv`, `*_summary_by_cell_type.csv`, `*_summary_overall.csv`) for any run prefix (default in notebook: `my_run_fast`).

## Parallel Slurm Workflow

For large runs, use the parallel pipeline:

1. Precompute true cell line-drug matches and generate task matrix (`pair x representation`).
2. Run one task per row of the matrix (ideal for Slurm array jobs).
3. Merge all task outputs into the final CSVs.

Precompute now also writes one AnnData per dataset pair with matches:

- `results/<run_dir>/<prefix>_pair_matches/<dataset_a>__<dataset_b>_matches.h5ad`
- each file contains matched samples from both datasets on shared genes and includes
  directional obs columns `<dataset_a>_<dataset_b>_match_id` and `<dataset_b>_<dataset_a>_match_id`.

### Python CLI Stages

```bash
# Stage 1: precompute truth matches + task matrix
python -m chem_perturbridge_analysis.retrieval.parallel_cli precompute \
  --output-dir results/full \
  --output-prefix my_run_fast \
  --query-datasets all \
  --db-datasets all \
  --representations all \
  --verbose

# Stage 2: run one task (task_id usually from Slurm array index)
python -m chem_perturbridge_analysis.retrieval.parallel_cli run-task \
  --task-file results/full/my_run_fast_tasks.csv \
  --task-id 1 \
  --output-dir results/full/tasks \
  --output-prefix my_run_fast \
  --metrics all \
  --verbose

# Stage 3: merge all task outputs
python -m chem_perturbridge_analysis.retrieval.parallel_cli merge \
  --task-file results/full/my_run_fast_tasks.csv \
  --task-output-dir results/full/tasks \
  --output-dir results/full \
  --output-prefix my_run_fast \
  --verbose
```

### Slurm Launcher

`scripts/slurm/run_retrieval_parallel.sh` orchestrates all three stages with `sbatch -W` and an array for stage 2.

Example:

```bash
QOS=cpu_normal \
PARTITION=cpu_p \
UV_BIN=uv \
OUTPUT_PREFIX=my_run_fast \
bash scripts/slurm/run_retrieval_parallel.sh
```

Useful environment variables:

- `QUERY_DATASETS`, `DB_DATASETS` (comma-separated names or `all`)
- `REPRESENTATIONS`, `SKIP_REPRESENTATIONS`
- `METRICS`
- `CELL_TYPES`
- `INCLUDE_SELF_DATASET` (`0`/`1`)
- `UV_BIN` (default: `uv`)
- `DATASET_PATH_ARGS` (e.g. `--dataset-path sciplex=/path --dataset-path tahoe=/path`)
- `TASK_MEM`, `PREP_MEM`, `MERGE_MEM` (defaults set to `200G`)
