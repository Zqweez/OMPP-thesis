#!/bin/bash
set -euo pipefail

# Run jupyter notebook to make the baseline plots for each target
config_file="bash/config.yaml"

targets=("npn" "mic" "pot") # "disc" "hemo" "cyto")
# jupyter nbconvert --to python scripts/baseline/make_baselines.ipynb --output "scripts/baseline/make_baselines.py"
for target in "${targets[@]}"; do
    python -c "import yaml; c=yaml.safe_load(open('workflow_evaluation/evaluation_config.yaml')); c['target'] = '$target'; yaml.dump(c, open('$config_file', 'w'))"

    echo "Generating evaluation metrics for target: $target"
    # Then run the evaluation pipeline in snakemake
    snakemake --snakefile workflow_evaluation/Snakefile --configfile "$config_file" --cores 8
done