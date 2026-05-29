# Repository for the thesis "Integrating <i>in vivo</i> characterization and machine learning for prediction of novel outer membrane-permeabilizing peptides"

This repository contains the pipelines used in the thesis. The workflow starts from sequence data, computes descriptors, reduces correlated features, runs feature selection, evaluates models with nested CV, and trains deployable models for prediction. Additional scripts cover baselines, PCA and PLS exploration, and PDB rendering.

## Environment setup

Recommended (Conda):

```bash
# Create environment from environment file
conda env create -f environment.yml

# Activate environment
conda activate masters

# Install this repository as a local package (in editable mode)
pip install -e .
```

Notes:

- `environment.yml` holds runtime dependencies.
- `pyproject.toml` defines package metadata and install behavior.

## Pipelines (Snakemake)

There are two Snakemake pipelines: an evaluation pipeline for unbiased model assessment and a deployment pipeline for training final models. 

More in depth documentation for wrapper script used by the pipelines can be found in `scripts/README.md`.

### 1) Evaluation pipeline (nested CV + model comparison)

This pipeline builds full feature tables, creates outer CV splits, runs single/double pyramid feature selection inside each fold, tunes final models with inner CV, and aggregates results.

- Snakefile: `workflow_evaluation/Snakefile`
- Config: `workflow_evaluation/evaluation_config.yaml`

Typical commands:

```bash
# Dry run
snakemake -s workflow_evaluation/Snakefile -n

# Full run
snakemake -s workflow_evaluation/Snakefile --cores 8 --use-conda

# Force run everything
snakemake -s workflow_evaluation/Snakefile --forceall --cores 8

# Render DAG
snakemake -s workflow_evaluation/Snakefile --dag | dot -Tsvg > workflow_evaluation/snake-flow.svg
```

Key outputs:

- Run folders under `outputs/evaluation-runs/<timestamp>_.../`
- Aggregated metrics at `outputs/evaluation-runs/<run>/02_analysis/model_comparison.csv`

### 2) Deployment pipeline (feature selection + final models)

This pipeline trains deployable models on the full dataset. It prepares features and clustering, runs pyramid feature selection, trains final models, and generates training-fit diagnostics.

- Snakefile: `workflow_deployment/Snakefile`
- Config: `workflow_deployment/deployment_config.yaml`

Typical commands:

```bash
# Dry run
snakemake -s workflow_deployment/Snakefile -n

# Full run
snakemake -s workflow_deployment/Snakefile --cores 8 --use-conda

# Force run everything
snakemake -s workflow_deployment/Snakefile --forceall --cores 8

# Render DAG
snakemake -s workflow_deployment/Snakefile --dag | dot -Tsvg > workflow_deployment/snake-flow.svg
```

Key outputs:

- Run folders under `outputs/deployment-runs/<timestamp>_.../`
- Stable latest artifacts under `outputs/deployment/<target>/` (see `workflow_deployment/README.md`)

## Prediction on new sequences

Use the deployment models to predict on new sequences:

```bash
python scripts/prediction/run_deployment_predictions.py --input <input_csv with a 'sequence' column>
```

This writes predictions to `outputs/predictions/predictions_<input>.csv`. For diagnostics and plots, use:

```bash
python scripts/prediction/analyze_deployment_predictions.py --predictions outputs/predictions/predictions_<input>.csv
```

## Exploratory analysis (PCA, PLS, PDB)

- PCA: `scripts/additional/make-pca.py` runs PCA on feature tables and saves pairwise PC plots under `outputs/pca/`.
- PLS: `scripts/additional/run-PLS.py` runs PLS regression with nested CV and Optuna tuning.
- PDB rendering: `scripts/additional/view-pdbfile.py` uses PyMOL to render `.pdb` files from `data/ompp-pdb/` into `outputs/pdb-view/front-page/`.

## Project structure

```bash
Repo-base/
├── data/                      # Raw and formatted data tables
├── outputs/                   # Pipeline outputs (evaluation, deployment, predictions, plots)
├── scripts/                   # Scripts 
│   ├── additional/            # PCA, PLS, PDB rendering
│   ├── baseline/              # Baseline model scripts
│   ├── data/                  # Data formatting
│   ├── deployment/            # Deployment pipeline scripts
│   ├── evaluation/            # Evaluation pipeline scripts
│   └── prediction/            # Prediction scripts using deployment models
├── src/                       # Core library (features, selection, models, visualization)
├── workflow_deployment/       # Snakemake deployment pipeline
├── workflow_evaluation/       # Snakemake evaluation pipeline
├── environment.yml
├── pyproject.toml
└── Dockerfile
```
