# Reviewer additions

Analyses added in response to the NeurIPS review, covering reviewer MDUy's **W1**
(baseline geometry) and **W3** (permissive dose matching). Table numbers refer to the
submitted paper.

Nothing in the published analysis was modified in place. Every addition either lives in a
new notebook derived from the original, or is appended as a self-contained section after
all existing cells. Each addition opens with a parity check that asserts the recomputed
observed and published-baseline values reproduce the original columns, so any change in a
reported delta comes from the new baseline alone.

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

Dose-threshold sensitivity at 2×, 3×, and 5×, with the original inclusive 10× window
retained as reference, plus an ECDF of the dose mismatch across matched pairs. The
`cell_type + time_key` eligibility rule is recomputed at every threshold and still
requires at least `MIN_CONTEXT_SHARED_DRUGS` shared compounds.

## File map

| Reviewer point | Tables | File | Relationship to the published code |
|---|---|---|---|
| W1 shared machinery | all | `scripts/peer_baselines.py` | New module |
| W1, W3 | 4, 5 | `notebooks/overlap_group_rep_deg_metrics_reviewer_additions.ipynb` | New notebook derived from `overlap_group_rep_deg_metrics.ipynb`; reuses 19 of its 24 code cells verbatim |
| W1 | 6 | `notebooks/overlap_group_rep_signature_similarity.ipynb` | Section appended; all original cells unchanged |
| W1 | 7, 8, 10 | `scripts/precompute_replicate_signature_similarity.py` | Per-peer baselines added; source-centroid path unchanged |
| W1 | 9 | `notebooks/overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb` | New notebook derived from `overlap_group_rep_retrieval_metrics.ipynb`; reuses 11 of its 16 code cells verbatim |
| W1 | 9 | `notebooks/overlap_group_rep_retrieval_metrics_spearman_addendum.ipynb` | New notebook; adds Spearman and the exact null without repeating scored work |

Unchanged foundations these build on: `scripts/build_overlap_filtered_h5ads.py` produces
the overlap-filtered `.h5ad` inputs every notebook reads, and
`scripts/cluster_bootstrap_ci.py` provides the compound-clustered BCa intervals used
throughout.

`scripts/peer_baselines.py` sits alongside `scripts/cluster_bootstrap_ci.py` because that
is how the notebooks import shared statistical helpers. The `chem_perturbridge_analysis`
package is reserved for the parallel retrieval CLI.

## Running

Outputs land in `results/<analysis>/` and are **not** tracked by git.

```bash
uv sync --locked
source .venv/bin/activate
python scripts/peer_baselines.py   # self-tests for the shared module
```

Then, in order:

1. `notebooks/overlap_group_rep_deg_metrics_reviewer_additions.ipynb` — Tables 4 and 5,
   plus the W3 dose sensitivity. Writes to `results/overlap_group_rep_deg_metrics/`.
2. `notebooks/overlap_group_rep_signature_similarity.ipynb` — Table 6. Writes to
   `results/overlap_signature_similarity_group_rep/`.
3. `scripts/precompute_replicate_signature_similarity.py`, then re-run
   `notebooks/replicate_deg_metrics.ipynb` (Tables 7, 8) and
   `notebooks/replicate_signature_similarity.ipynb` (Table 10). Reads and writes
   `results/replicate_signature_similarity_sep_rep_combined/`.
4. `notebooks/overlap_group_rep_retrieval_metrics_reviewer_additions.ipynb` — Table 9.
5. `notebooks/overlap_group_rep_retrieval_metrics_spearman_addendum.ipynb` — adds Spearman
   and the exact null on top of step 4's saved output.

Steps 1, 2, and 4 each end with a rebuttal-ready per-dataset-pair table
(`peer_baseline_rebuttal_table.tsv`) carrying every baseline with its bootstrap interval.

### Compute notes

Replicate peer sets are much larger than cross-dataset ones — CIGS-MCE has 21,426
replicate-supported conditions over 5,007 compounds — so per-peer scoring costs K× the
centroid. `--max-baseline-peers` caps how many peers are scored, subsampling
deterministically per condition; both the total and scored peer counts are recorded so a
capped run is never mistaken for a full one. Start uncapped on one small dataset to see
real peer counts in `n_peer_rows_total`.

The cross-dataset notebooks set `MAX_BASELINE_PEERS = None` (score every peer) and cover
the `p05` DEG definition; add `"p05_lfc02"` to `PEER_BASELINE_DEFINITION_KEYS` for the
`|logFC| > 0.2` variant at roughly double the peer-scoring cost.

The Spearman addendum deliberately avoids re-running scored work: it reloads
`matched_sample_pairs.tsv` and `query_retrieval_metrics.tsv` rather than recomputing the
dose matching and the main retrieval loop, and scores only the Spearman similarity before
merging with the saved Euclidean and cosine rows. It backs up the existing ablation TSV
first and refuses to continue unless that backup contains both metrics.

## Status

Table 9 results are computed. The Tables 4, 5, 6, 7, 8, and 10 code is written and
unit-tested against synthetic fixtures but has not yet been run at scale.

## Verification

- `python scripts/peer_baselines.py` — 200 randomized trials checking the vectorized
  Spearman and direction-agreement paths against scalar reference implementations, covering
  NaNs, tied ranks, constant rows, and gene masks; plus the add-one correction and the
  determinism of peer subsampling.
- Parity cells in each notebook section assert the recomputed source-centroid and observed
  columns reproduce the published ones to within 1e-6, and that the per-peer matrix
  averages exactly to the existing centroid.
- The retrieval ablation asserts that its negative-Euclidean numbers reproduce the
  published Table 9 values exactly, making it a strict superset of the original analysis.
- The Spearman helper was checked against `scipy.stats.spearmanr` to 1e-17 including tied
  ranks, and the exact-null calibration against simulation (mean mid-P transform 0.4952 on
  null data, where 0.5 is expected).
