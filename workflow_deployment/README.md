# Deployment Snakemake Pipeline

This workflow runs the deployment pipeline using the configuration in `deployment_config.yaml`.
It stores stable latest-run outputs under `outputs/deployment/<target>/` for predictable paths.

## Run

From the repository root:

```bash
snakemake -s workflow_deployment/Snakefile --cores 8
```

## Outputs

The final tracked output is:

- `outputs/deployment/<target>/analysis/training_fit_summary.csv`

Other stable outputs are copied alongside it:

- `outputs/deployment/<target>/latest_run.txt`
- `outputs/deployment/<target>/selection/feature_frequency.csv`
- `outputs/deployment/<target>/selection/selected_features.csv`
- `outputs/deployment/<target>/models/models_summary.csv`
- `outputs/deployment/<target>/models/<model_name>.joblib`
