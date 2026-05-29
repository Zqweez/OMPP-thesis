# Scripts overview

Scripts are grouped by purpose. Most are CLI tools used by the Snakemake pipelines under `workflow_evaluation/` and `workflow_deployment/`. Run them from the repo root and use `--help` for detailed flags. Shared logic lives in the `src/` package.

## Folder layout

- `data/` - format xlxs-files from masterframe data into csv tables used by the pipelines
- `evaluation/` - nested CV evaluation pipeline steps and analysis
- `deployment/` - deployment pipeline steps (feature selection and model creation)
- `prediction/` - run and analyze predictions using deployment models
- `baseline/` - baseline model scripts and comparisons
- `additional/` - exploratory analyses (PCA, PLS, PDB, feature importance)

---

## `data/`

`format-data.py`

- Reads the Excel file and outputs formatted CSVs with mean target values.

---

## `evaluation/`

`prepare_nested_cv_splits.py`

- Computes features for all sequences, runs global clustering diagnostics, and creates outer CV splits.
- Writes a split manifest and a copied `run_config.yaml` inside the run folder.

`run_outer_fold_pipeline.py`

- Executes all outer folds from the split manifest, running the full nested CV pipeline for each fold. This is the main entry point for evaluation runs.
- Executes one outer fold at a time: clusters features on the train fold, runs single or double pyramid selection, tunes final models with inner CV, and stores per-fold predictions and summaries, stored in `01_evaluation/`.

`aggregate_evaluation_results.py`

- Aggregates fold artifacts into `02_analysis/` and generates `model_comparison.csv` and plots evaluation metrics for each model and the entire nested CV.

`combined_evaluation.py`

- Combines evaluation outputs with baseline metrics for comparison plots, the plots included in the thesis.

---

## `deployment/`

`prepare_features_and_clustering.py`

- Computes features on the full dataset and runs global clustering for deployment runs.

`run_feature_selection.py`

- Runs pyramid selection on the full dataset and writes `feature_frequency.csv` and `selected_features.*`.

`build_final_models.py`

- Trains final models on the top features, writes `.joblib` files, and stores model summaries and feature importance tables.

`analyze_training_fit.py`

- Generates training-fit plots and summary metrics, and syncs results to `outputs/deployment/<target>/analysis/`.

---

## `prediction/`

`run_deployment_predictions.py`

- Computes features for a new input CSV and runs all deployment models under `outputs/deployment/<target>/models/`.
- Writes predictions under `outputs/predictions/`.

`analyze_deployment_predictions.py`

- Generates prediction diagnostics and evaluation metrics for existing prediction tables.

---

## `baseline/`

`baseline_functions.py`

- Shared helpers for baseline runs.

`make_baselines.py`

- Notebook-style script that runs multiple baselines (all-features, clustered, simple feature baselines).

`make_baseline_metrics.py`

- Summarizes baseline results into evaluation metrics CSVs for comparison plots.

`use_selected_features.py`

- Runs baseline evaluation using a manual feature list CSV.

---

## `additional/`

`make-pca.py`

- Runs PCA on feature tables and writes pairwise PC scatter plots plus `scaled_features.csv`.

`run-PLS.py`

- Runs PLS regression with nested CV and Optuna tuning for `n_components`.

`view-pdbfile.py`

- Uses PyMOL to render `.pdb` files from `data/ompp-pdb/` into `outputs/pdb-view/front-page/`.

`feature-importance-heatmap.py`

- Builds heatmaps of feature ranks across targets using deployment outputs.

`target-correlations.py`

- Creates target correlation plots and dendrograms from computed feature tables.

`visualize-trees.py`

- Inspects tree-based models (feature importance, partial dependence, tree previews).