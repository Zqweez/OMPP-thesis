#!/usr/bin/env bash 
set -euo pipefail # Set modes for better error handling

# (Optional) Update env files when deps change.
# conda env export --from-history > environment.yml

# Create a unique run ID
PATH_TS=$(date +"%Y-%m-%d_%H-%M")
GIT_SHORT=$(git rev-parse --short HEAD)

RUN_ID="${PATH_TS}_${GIT_SHORT}"

RUN=../runs/run_$RUN_ID           # folder for code snapshot
OUT="$(pwd)/outputs/runs/$RUN_ID" # folder for outputs

export OUT="$OUT"

mkdir -p "$RUN" "$OUT"

# clean, tracked-only snapshot of the commit
git archive --format=tar HEAD | tar -x -C "$RUN"

# Copy over the data and generate the data sets
cp data/OMPP_master_dataframe.xlsx "$RUN/data/OMPP_master_dataframe.xlsx"
python scripts/data/format-data.py

# We build a new image do this if deps have changed, otherwise we can just reuse the old one.
# docker build -t masters:2 .

# We mount the snapshot, the data and the outputs into the container, and run snakemake inside it.
docker run --rm \
  -v "$RUN:/app" \
  # -v "$(pwd)/data:/app/data" \
  -v "$OUT:/app/outputs" \
  -w /app \
  masters:2 \
  conda run -n masters --no-capture-output bash bash/make_baselines.bash