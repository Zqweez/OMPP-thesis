#!/bin/bash
set -euo pipefail

all_run_folders=$(ls -d outputs/evaluation-runs/*/ | sort -V)

for run_folder in $all_run_folders; do
    echo "Rerunning evaluation metrics for run: $run_folder"

    # Then run the evaluation pipeline in snakemake
    python scripts/evaluation/aggregate_evaluation_results.py --run_folder "$run_folder"
done
