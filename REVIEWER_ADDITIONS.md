# Reviewer additions

Analyses added in response to the NeurIPS review, covering reviewer MDUy's **W1**
(baseline geometry), **W3** (permissive dose matching), and **W4** (uniform per-gene
population standardization). Table numbers refer to the submitted paper.

The reviewer notebooks preserve the published metric definitions and add explicit parity
checks before reporting new baseline deltas. Shared cross-source mechanics are now routed
through one module so the DEG, signature-similarity, and retrieval notebooks cannot drift
in dataset policy, dose matching, source lookup, shared-gene handling, or cache
invalidation while retaining separate metric-specific scoring.

## W1: baseline geometry

The published baseline is a **centroid**: the gene-wise mean over same-context
different-compound peers, scored once. The reviewer notes three ways this makes it an
unfairly strong competitor — averaging removes variance (smoothing), the baseline term is
computed entirely within the sample's own dataset while the observed term crosses assays
(home-field advantage), and the common-mode response shared by most compounds survives
averaging.

Four baselines are now reported side by side, which separates those mechanisms:

| Baseline | Construction | Mechanisms removed |
|---|---|---|
| `source_centroid` | The published baseline. Mean of the sample's own-dataset peers. | none (reference) |
| `target_centroid` | Same mean, built from the **other** dataset's peers, so the baseline pays the same cross-assay penalty as the observed score. | home-field |
| `source_peer` | Every own-dataset peer scored **separately**, then summarized. | smoothing, common-mode |
| `target_peer` | Every other-dataset peer scored separately. | all three |

For the per-peer baselines we report the mean and standard deviation of the peer score
distribution, the fraction of peer scores `m(i, x)` below the observed matched score
`m(i, j)`, and the add-one corrected percentile
`(1 + #{m(i, x) < m(i, j)}) / (K + 1)`.

DEG metrics keep the published **sample-referenced convention**: DEG-restricted logFC
Spearman is evaluated on the sample's own DEG mask, and direction agreement on that mask
intersected with the finite entries of the comparison signature. This keeps the new
baselines comparable to the published ones.

### Table 9 (retrieval) additionally gets

- **Cosine and Spearman** alongside negative Euclidean distance, per the reviewer's request
  to score retrieval by cosine or rank rather than Euclidean distance.
- A **cross-assay target-decoy null**: the exact expectation of the best-of-K rank under
  uniform relabeling, which is the correct random floor when a query has more than one
  positive candidate.
- A **single-signature retrieval baseline** (each peer injected into the candidate pool
  individually rather than as a centroid).
- A **within-dataset same-compound positive control**, quantifying the cross-assay penalty.
- **Exact null calibration.** The submission describes `0.5` as the random floor for
  normalized best-positive rank. That holds only when a query has exactly one positive.
  Under the strict matched-condition variant most queries have several, and the expected
  best-of-K rank is `(N + 1) / (K + 1)`, so the floor is higher and rises with K. Each
  observed rank is now calibrated against the exact null *distribution* via a mid-P
  transform that is uniform under the null and averages `0.5`.

## W3: dose matching

Dose-threshold sensitivity at exact dose equality, 2×, and 3×, with the original
inclusive 10× window retained as a parity reference, plus an ECDF of the dose mismatch
and an explicit matched-pair count at every threshold. The
`cell_type + time_key` eligibility rule is recomputed at every threshold and still
requires at least `MIN_CONTEXT_SHARED_DRUGS` shared compounds.

## W4: per-gene population standardization

W4 is an additive sensitivity analysis; it never replaces raw logFC. Two population
scopes are now reported for every gene:

```text
per_gene_population_zscore_dataset_cell_type =
    (logFC - population mean[source, cell, gene])
    / population SD[source, cell, gene]

per_gene_population_zscore_dataset =
    (logFC - population mean[source, gene])
    / population SD[source, gene]
```

Population SD uses `ddof=0`. Eligibility requires a non-control row, a valid compound,
finite time, and a finite positive dose. A gene is valid when it has at least two finite
population values and a finite, nonzero SD. The first duplicate gene symbol is retained,
matching the notebooks' `LineSource` behavior.

This applies the same gene-wise population-axis procedure to every source. It is not a
signature-global z-score (which would leave Spearman unchanged), never pools different
datasets, and does not claim to reconstruct the Broad plate-level Level 4 pipeline. The
dataset × cell-type result preserves cell-context-specific populations; the dataset-wide
result pools that dataset's eligible cell types and tests whether the conclusion depends
on the narrower population definition.

Dataset-wide fitting discovers every line-level H5AD in the configured dataset source
directory, not only the cell types retained by cross-source matching. The precompute
command and all three notebooks use the same discovery helper, so a subset analysis
still applies the same full-dataset population statistics as production.

W4 reports raw logFC beside both
`per_gene_population_zscore_dataset_cell_type` and
`per_gene_population_zscore_dataset` for:

- Table 4: DEG-restricted logFC Spearman under the fixed raw
  `adj.P.Value < 0.05` masks. Adjusted p-values, DEG membership, overlap, and biological
  direction remain raw.
- Table 6: all-gene matched-signature Spearman.
- Table 9: strict-condition logFC retrieval using negative-L2, cosine, and Spearman,
  with the exact target-decoy null, source/target centroids, source/target individual
  peers, and within-source positive controls. For each similarity metric, a retrieval
  centroid is the mean of the exact same overlap-filtered, metric-valid peer matrix used
  for the corresponding individual-peer distribution.

Within-source Tables 7, 8, and 10 and Table 5 direction agreement are intentionally
outside W4 scope.

## File map

| Reviewer point | Tables | File | Relationship to the published code |
|---|---|---|---|
| Cross-source shared core | 4, 5, 6, 9 | `scripts/cross_source_core.py` | Canonical dataset profiles, matching, source catalog, shared-gene scope, and cache fingerprints |
| W1 shared machinery | all | `scripts/peer_baselines.py` | New module |
| W4 shared machinery | 4, 6, 9 | `scripts/population_zscore.py` | Streaming, locked statistics and cache module for both population scopes |
| W4 precompute | 4, 6, 9 | `scripts/precompute_population_zscore.py` | Prepares both shared statistic scopes once before the notebooks |
| W1, W3, W4 | 4, 5 | `notebooks/overlap_group_rep_deg_metrics_reviewer_additions.ipynb` | New notebook derived from `overlap_group_rep_deg_metrics.ipynb`; W4 section appended |
| W1, W4 | 6 | `notebooks/overlap_group_rep_signature_similarity.ipynb` | Sections appended; all original cells unchanged |
| W1 | 7, 8, 10 | `scripts/precompute_replicate_signature_similarity.py` | Per-peer baselines added; source-centroid path unchanged |
| W1, W4 | 9 | `notebooks/overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb` | New notebook derived from `overlap_group_rep_retrieval_metrics.ipynb`; scores negative-L2, cosine, and Spearman together and includes the exact retrieval null |

Unchanged foundations these build on: `scripts/build_overlap_filtered_h5ads.py` produces
the overlap-filtered `.h5ad` inputs every notebook reads, and
`scripts/cluster_bootstrap_ci.py` provides the compound-clustered BCa intervals used
throughout.

`scripts/peer_baselines.py` sits alongside `scripts/cluster_bootstrap_ci.py` because that
is how the notebooks import shared statistical helpers. The `chem_perturbridge_analysis`
package is reserved for the parallel retrieval CLI.

## Running

Outputs land in `results/<analysis>/` and are **not** tracked by git.

### Resumability

Every notebook below is restartable. Each expensive stage goes through `cached_frame` in
`scripts/notebook_cache.py`: if the stage's output TSV already exists it is reloaded
instead of recomputed, so a crash late in a notebook no longer means redoing the scoring
above it. Cheap aggregations are left alone, since they derive from the cached frames in
seconds. Each notebook ends with `display(cache_summary())`, showing which stages were
reloaded and which ran.

| Notebook | Stages |
|---|---|
| `overlap_group_rep_deg_metrics_reviewer_additions` | `matched_pairs`, `deg_metrics`, `deg_ci`, `dose_ci`, `peer_baselines`, `peer_ci`, `w4_deg_metrics`, `w4_deg_ci` |
| `overlap_group_rep_signature_similarity` | `matched_pairs`, `signature_metrics`, `signature_ci`, `peer_baselines`, `peer_ci`, `w4_signature_metrics`, `w4_signature_ci` |
| `overlap_group_rep_retrieval_metrics_reviewer_additions` | `matched_pairs`, `primary_retrieval`, `retrieval_ablation`, `focused_retrieval`, `ablation_ci`, `null_calibration`, `null_ci`, `w4_retrieval_metrics`, `w4_retrieval_ci` |
| `replicate_deg_metrics` | `replicate_deg_ci` |
| `replicate_signature_similarity` | `replicate_signature_ci` |

To rebuild a stage, delete its TSV or name it:

```python
force_recompute("peer_baselines")     # in a cell, before that stage runs
```

```bash
CPB_FORCE_RECOMPUTE=peer_baselines,peer_ci jupyter lab    # or from the shell
CPB_FORCE_RECOMPUTE=all jupyter lab                       # ignore every cache
```

Dataset selection, matched-row identities, source-file size/mtime, metric settings, peer
cap, sampling seed, and engine versions are fingerprinted. Changing any of these
automatically invalidates incompatible stage outputs. Peer-baseline work is additionally
checkpointed by dataset-pair/cell/time/dose context, so an interrupted run resumes from
completed context shards. All cache and task-result writes are published atomically.

The three cross-source notebooks import `scripts/cross_source_core.py`; notebook cells no
longer retain shadow copies of the shared implementations. A single
`prepare_cross_source_scope(...)` call owns canonical indexing, mutual-nearest dose
matching, the matched-pair cache and fingerprint, matched-line scope, shared-gene scope,
and `LineSourceCatalog` lifecycle. It defers line-level H5AD loading until matches are
known. For each cell type that contributes a match, the shared-gene intersection still
includes every active dataset that contains that cell type, preserving the original
analytical gene universe. Metric caches inventory every line file that can affect that
metric: the full shared-gene scope for DEG/signature scoring and the matched
query/candidate scope for retrieval.

W4 setup likewise uses one `PopulationStatsCatalog` from
`scripts/population_zscore.py`. It owns production-versus-smoke population membership,
precompute readiness/wait policy, dataset/cell-type and pooled-dataset statistic caches,
gene alignment, and standardization. The notebooks retain only their metric-specific W4
record construction and reporting.

Dataset membership is also explicit rather than being controlled by commented list
entries. The `signature` profile includes GDPx2 and DILImap for Table 6. The `deg` and
`retrieval` profiles omit those two sources until their required layer capability is
verified; all remaining production datasets have the same canonical order. A selected
dataset that produces no retained comparison remains visible in the initial inventory
but is excluded from W4 precompute readiness, fitting, fingerprints, and QC.

Because the shared match schema and source-aware fingerprints replace the earlier
notebook-local versions, the first run after this change intentionally invalidates the
old matched-pair and downstream metric/CI caches. Subsequent runs reload the new caches
normally. Restart the notebook kernel before running so no pre-refactor function
bindings remain in memory.

Dataset selection and output isolation can be controlled without editing notebook cells:

```bash
CPB_DATASET_SUBSET=tahoe,sciplex CPB_RUN_TAG=quick_test jupyter lab
CPB_RUN_TAG=cigs_production jupyter lab  # full configured dataset list
```

W4 population-statistic caches are independent of notebook output caches and live at:

```text
results/w4_population_zscore_stats/<dataset>/<cell_type>.npz
results/w4_population_zscore_stats/<dataset>/<cell_type>.cache.json
results/w4_population_zscore_stats/dataset_wide/<dataset>.npz
results/w4_population_zscore_stats/dataset_wide/<dataset>.cache.json
```

The source path, size, modification time, layer shape, gene order, eligibility mask,
`ddof`, population scope, and W4 engine version are fingerprinted. Per-line sufficient
statistics are scanned once; dataset-wide statistics are obtained by finite-aware Chan
merges after aligning gene keys, without rereading or concatenating the H5AD matrices.
Per-cache file locks prevent concurrently running notebooks from fitting the same source
twice. On a cache miss, metadata inspection and layer fitting share one backed-H5AD
handle. In `--scope both`, the precompute command passes newly fitted or reloaded
line-stat objects directly into dataset-wide pooling rather than reopening all sources.
The line-cache metadata records the source inventory, allowing later notebooks to
fast-reload compatible statistics without opening the H5AD.

The Table 6 W4 scorer additionally keeps a bounded in-memory cache of each active
dataset/cell/time/dose/shared-gene stratum after standardization and Spearman-rank
preparation. Each stratum is prepared once, and each compound selects its deterministic
capped peer rows from that prepared object; centroid calculations still use every peer.

```bash
uv sync --locked
source .venv/bin/activate
python scripts/peer_baselines.py   # self-tests for the shared module
python scripts/precompute_population_zscore.py \
  --all-configured \
  --scope both \
  --row-chunk-size 1024 \
  --workers 2 \
  --qc-output results/w4_population_zscore_stats/precompute_qc.tsv
```

The precompute and the three notebooks may be started together. By default, each
notebook runs its raw sections immediately and, when it reaches W4, polls the shared
cache inventory every 30 seconds. The poll reads only small cache metadata files and
source file size/modification times; it neither opens the H5ADs nor duplicates fitting.
When all required line-level and dataset-wide caches are atomically published, the
notebook continues automatically.

The W4 setup behavior can be changed in a temporary first notebook cell:

```python
import os
os.environ["CPB_W4_PRECOMPUTE_MODE"] = "wait"  # wait (default), stop, or lazy
os.environ["CPB_W4_PRECOMPUTE_POLL_SECONDS"] = "30"
os.environ["CPB_W4_PRECOMPUTE_TIMEOUT_MINUTES"] = "720"
```

`stop` raises a clear error at the W4 boundary and preserves every raw output already
computed; after precompute finishes, rerun the W4 setup cell and the cells below it.
`lazy` restores the earlier behavior in which a notebook may itself fit missing W4
caches. Do not use `lazy` while the dedicated precompute is running unless that fallback
is intentional. The non-production comparison-population smoke mode described below
continues to fit its isolated, tagged caches lazily and does not wait for production
caches.

`--workers` defaults to `1`; use `2` first and try `4` only if storage throughput remains
healthy. Workers process independent line files, while dataset-wide merging remains
canonically ordered and exactly matches the serial result. Progress output distinguishes
source inspection, cache-lock waiting, matrix fitting, line completion, and dataset
pooling, with elapsed time and scan ETA.

For a fast end-to-end subset smoke test, the notebooks support an explicitly
non-production population:

```python
import os
os.environ["CPB_DATASET_SUBSET"] = "tahoe,sciplex"
os.environ["CPB_RUN_TAG"] = "tahoe_sciplex_smoke"
os.environ["CPB_W4_SMOKE_COMPARISON_POPULATION"] = "1"
```

Run this temporary cell before the configuration cell in each fresh kernel. In this
mode, dataset-wide fitting uses only cell lines retained by that notebook's selected
comparisons and writes statistics below that tagged notebook output directory. It is
therefore quick and cannot overwrite the production W4 caches. These smoke results test
execution only and must not be reported as dataset-wide reviewer results. Production
runs must omit `CPB_W4_SMOKE_COMPARISON_POPULATION`; they continue to use every line-level
H5AD in each configured dataset.

The three cross-source notebooks below construct different metric families and may run
concurrently with each other and with W4 precompute:

1. `notebooks/overlap_group_rep_deg_metrics_reviewer_additions.ipynb` — Tables 4 and 5,
   plus the W3 dose sensitivity. Writes to `results/overlap_group_rep_deg_metrics/`.
2. `notebooks/overlap_group_rep_signature_similarity.ipynb` — Table 6. Writes to
   `results/overlap_signature_similarity_group_rep/`.
3. `notebooks/overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb` — Table 9,
   including negative-L2, cosine, Spearman, individual-signature baselines, and the exact
   retrieval null in one run.

The separate within-source workflow may also run concurrently: run
`scripts/precompute_replicate_signature_similarity.py`, then re-run
   `notebooks/replicate_deg_metrics.ipynb` (Tables 7, 8) and
   `notebooks/replicate_signature_similarity.ipynb` (Table 10). Reads and writes
   `results/replicate_signature_similarity_sep_rep_combined/`. The notebooks write
   `tables_7_8_peer_baseline_reviewer_table.tsv` and
   `table_10_peer_baseline_reviewer_table.tsv`, retaining observed and centroid results
   alongside peer means, SDs, fractions, corrected percentiles, deltas, and
   compound-clustered BCa intervals.
The three cross-source notebooks each end with a rebuttal-ready per-dataset-pair table
(`peer_baseline_rebuttal_table.tsv`) carrying every baseline with its bootstrap interval.
Their appended W4 sections additionally write:

| Table | Primary W4 outputs |
|---|---|
| 4 | `w4_matched_sample_pair_deg_metrics.tsv`, `w4_deg_dataset_pair_summary.tsv`, `w4_deg_cluster_bca_ci.tsv`, `w4_table_4_standardization.tsv` |
| 6 | `w4_matched_sample_pair_metrics.tsv`, `w4_dataset_pair_summary.tsv`, `w4_cluster_bca_ci.tsv`, `w4_table_6_standardization.tsv` |
| 9 | `w4_retrieval_query_metrics.tsv`, `w4_retrieval_peer_scores.tsv`, `w4_retrieval_dataset_pair_summary.tsv`, `w4_retrieval_cluster_bca_ci.tsv`, `w4_table_9_standardization.tsv` |

Each W4 output directory also contains `w4_population_stats_qc.tsv`, including population
row counts, valid-gene counts, finite-count ranges, fingerprints, and shared cache paths.

### Compute notes

Replicate peer sets are much larger than cross-dataset ones — CIGS-MCE has 21,426
replicate-supported conditions over 5,007 compounds — so per-peer scoring costs K× the
centroid. `--max-baseline-peers` defaults to 512 and caps only the individual-peer score
distribution. The published centroid is always computed from every eligible peer, so
changing the cap cannot change the legacy result. Sampling is deterministic under
`--peer-sampling-seed` (default `20260505`); total metadata rows, available vector rows,
and scored rows are all recorded.

The expensive path loads each line/time/dose context once per task, reuses it across
conditions, and pre-ranks peer rows once for repeated all-gene Spearman scoring. Task
shards are context-local with a default size of 250 conditions. The cross-dataset
notebooks use the same 512-peer default and cover
the `p05` DEG definition; add `"p05_lfc02"` to `PEER_BASELINE_DEFINITION_KEYS` for the
`|logFC| > 0.2` variant at roughly double the peer-scoring cost.

Before accepting a capped production run, compare 128/256/512/1024 against an uncapped
reference. The validator enforces 0.01 tolerances for peer means/deltas, 0.02 for
fractions/percentiles, and unchanged signs for compound-clustered delta intervals:

```bash
python scripts/validate_peer_baseline_convergence.py \
  --run 128=results/peers_128 --run 256=results/peers_256 \
  --run 512=results/peers_512 --run 1024=results/peers_1024 \
  --run full=results/peers_full --reference-label full
```

## Status

The W1/W3 and W4 code is implemented and unit-tested against synthetic fixtures. W4 has
also completed a backed, chunked scan of a real grouped-condition `.h5ad` source. The
appended W4 notebook sections have not yet been run at production scale.

## Verification

- `python scripts/peer_baselines.py` — 200 randomized trials checking the vectorized
  Spearman and direction-agreement paths against scalar reference implementations, covering
  NaNs, tied ranks, constant rows, and gene masks; plus the add-one correction and the
  determinism of peer subsampling. The prepared-rank engine is checked to `1e-12` parity,
  and regression tests assert that peer caps never change the exact centroid.
- Parity cells in each notebook section assert the recomputed source-centroid and observed
  columns reproduce the published ones to within 1e-6, and that the per-peer matrix
  averages exactly to the existing centroid.
- The retrieval ablation asserts that its negative-Euclidean numbers reproduce the
  published Table 9 values exactly, making it a strict superset of the original analysis.
- The Spearman helper was checked against `scipy.stats.spearmanr` to 1e-17 including tied
  ranks, and the exact-null calibration against simulation (mean mid-P transform 0.4952 on
  null data, where 0.5 is expected).
- `python -m unittest discover -s tests -p 'test_population_zscore.py'` checks streaming
  statistics against dense NumPy `axis=0, ddof=0`, NaNs, duplicate genes, zero variance,
  insufficient observations, strict gene order, source-context isolation, z-score
  mean/SD, centroid equivalence, cache reloads, and the add-one peer correction.
- W4 notebook checks require identical raw and standardized matched-pair/query IDs,
  unchanged Table 4 DEG counts, identical Table 9 target-pool and positive counts, and
  affine equivalence between each checked standardized retrieval-pool centroid and the
  standardized raw centroid of that same peer matrix.
