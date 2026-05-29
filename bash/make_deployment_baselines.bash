#!/bin/bash
set -euo pipefail

# Run jupyter notebook to make the baseline plots for each target
config_file="bash/config.yaml"

targets=("npn" "mic" "pot" "disc" "hemo" "cyto")
# jupyter nbconvert --to python scripts/baseline/make_baselines.ipynb --output "scripts/baseline/make_baselines.py"
for target in "${targets[@]}"; do
    python -c "import yaml; c=yaml.safe_load(open('workflow_deployment/deployment_config.yaml')); c['target'] = '$target'; yaml.dump(c, open('$config_file', 'w'))"

    echo "Generating baseline plots for target: $target"
    python scripts/deployment/make_deployment_baselines.py --config "$config_file"
done