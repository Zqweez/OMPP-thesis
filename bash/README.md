# Bash helpers

This folder contains small Bash helpers that run repeated evaluations and baselines across multiple targets. Most scripts write a temporary config to `bash/config.yaml` and passes it to the snakemake pipeline.

## Scripts

- `make_baselines.bash` - Runs evaluation baselines for each target using `scripts/baseline/make_baseline_metrics.py`.
- `make_deployment_baselines.bash` - Runs deployment baselines for each target using `scripts/deployment/make_deployment_baselines.py`.
- `make_deployments.bash` - Runs the deployment Snakemake pipeline for each target with `--forceall`.
- `make_evaluation_metrics.bash` - Runs the evaluation Snakemake pipeline for each target with `--forceall`.
- `make_manualfeature_eval.bash` - Evaluates manual feature sets from `data/nested_feature_sets.csv` via `scripts/baseline/use_selected_features.py`.
- `rerun_eval_analysis.bash` - Re-aggregates evaluation results for all folders under `outputs/evaluation-runs/`.
- `run_evaluation_docker.bash` - Runs baselines inside a Docker image (`masters:2`) using a clean repo snapshot and mounted outputs.
- `make_teams_backup.bash` - Tags and pushes a git backup tag after committing staged changes.

## Usage

Run from the repo root:

```bash
bash bash/make_evaluation_metrics.bash
```

## Notes

- These scripts assume the Python environment is already set up and `snakemake` is available on PATH.
- `bash/config.yaml` is a scratch config used by the scripts and is overwritten each run.
