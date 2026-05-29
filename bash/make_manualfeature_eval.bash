#!/bin/bash
set -euo pipefail

# Run jupyter notebook to make the baseline plots for each target
config_file="bash/config.yaml"

targets=("npn" "mic" "pot" "disc" "hemo" "cyto")
feature_set_path="data/nested_feature_sets.csv"
# jupyter nbconvert --to python scripts/baseline/make_baselines.ipynb --output "scripts/baseline/make_baselines.py"
for target in "${targets[@]}"; do
    python -c "import yaml; c=yaml.safe_load(open('workflow_evaluation/evaluation_config.yaml')); c['target'] = '$target'; yaml.dump(c, open('$config_file', 'w'))"

    echo "Generating manual feature plots for target: $target"
    python scripts/baseline/use_selected_features.py --config "$config_file" --set-path "$feature_set_path"
done