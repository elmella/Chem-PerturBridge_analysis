# Chem-PerturBridge_analysis

Cross-dataset retrieval benchmark for perturbation-response signatures.

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
Dataset scripts default to `data/processed`. Set `PERTURB_DATA_ROOT` to point at a different processed-data root:

```bash
export PERTURB_DATA_ROOT=/path/to/processed_data
```

Typical analysis order:

1. `scripts/build_dataset_summary.py` builds the dataset summary table.
2. `scripts/build_overlap_filtered_h5ads.py` creates overlap-filtered grouped-replicate `.h5ad` files.
3. `notebooks/plot_overlap_heatmaps.ipynb` plots cross-dataset overlap summaries.
4. `notebooks/overlap_group_rep_signature_similarity.ipynb`, `notebooks/overlap_group_rep_deg_metrics.ipynb`, and `notebooks/overlap_group_rep_retrieval_metrics.ipynb` analyze cross-dataset agreement.
5. `scripts/slurm/run_replicate_signature_similarity.sh`, `notebooks/replicate_signature_similarity.ipynb`, and `notebooks/replicate_deg_metrics.ipynb` analyze cross-replicate agreement.
