#!/bin/bash
set -euo pipefail

# Get all run_folders that end in *ds
#run_folders=$(find outputs/evaluation-runs -maxdepth 1 -type d -name "*ds")
#echo "Found run folders: $run_folders"

#for run_folder in $run_folders; do
#    echo "Processing run folder: $run_folder"
#    python scripts/evaluation/aggregate_evaluation_results.py --run_folder "$run_folder"
#done
