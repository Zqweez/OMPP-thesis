#!/usr/bin/env python
# coding: utf-8

# # Make baselines
# This script is used to generate baseline performances used in comparison with the new methods. It runs nested CV with the same settings. 
# 
# <ul>
# <li>Baseline 1: No feature selection, all features used in final model</li>
# <li>Baseline 2: No feature selection, but run clustering all kept features used in final model</li>
# <li>Baseline 3: Only discrimination factor used as feature in final model, no feature selection performed.</li>
# <li>Baseline 4: Take the average of the training data and predict the average for all test samples (i.e. a dummy regressor that always predicts the mean of the training data).</li>
# </ul>

n_jobs = 8


# ## Helper Functions and imports

# In[ ]:


# General helpers used for all baselines
from src.util.config_loader import load_evaluation_config
from src.selection.shared import TARGET_MAPPING, generate_seed_list
from src.features.compute import compute_features_for_sequences
from src.features.load_features import load_and_prepare_data
from src.visualization.plot_evaluation_pipeline import (
    plot_seed_metric_violins,
    plot_train_test_diagnostics,
)
from scripts.baseline.baseline_functions import run_baseline_1_parallel, run_baseline_2_parallel

from pathlib import Path
import pandas as pd
from colorama import Fore, Style

start = Path.cwd()
for p in (start, *start.parents):
    if (p / "data").exists() or (p / "scripts").exists():
        repo_root = p
config_path = repo_root / "bash/config.yaml" #"workflow_evaluation/evaluation_config.yaml"
config = load_evaluation_config(config_path)
print(f"Repo root: {repo_root}")
input_file = Path() / config["input_file"]
amp_feature_path = Path(repo_root) / config["amp_feature_path"]
amp_sequence_path = Path(repo_root) / config["amp_sequence_path"]

target_arg = config["target"]
start_seed = config["start_seed"]
num_outer_seeds = config["num_outer_seeds"]
# Temporary override for testing
num_outer_seeds = 16
final_model_names = [str(model_name) for model_name in config["final_model_names"]]
scoring_scheme = config["scoring_scheme"]
n_trials = 100
scale_target = config["scale_target"]


cluster_threshold = config["cluster_threshold"]
cluster_method = config["cluster_method"]
max_cluster_size = config["max_cluster_size"]
variance_threshold = config["variance_threshold"]
perfect_corr_threshold = config["perfect_corr_threshold"]

baseline_output_dir = repo_root / "outputs/baseline_results" / target_arg
baseline_output_dir.mkdir(parents=True, exist_ok=True)


# ## Baseline 1

# In[ ]:


bs1_output_dir = baseline_output_dir / "baseline_1" / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
bs1_output_dir.mkdir(parents=True, exist_ok=True)
source_df = pd.read_csv(repo_root / input_file)
target = TARGET_MAPPING[target_arg]
# Only keep rows where the target value is not nan such that if the input file contains rows without target values, they are automatically dropped
source_df = source_df.dropna(subset=[target]).reset_index(drop=True)

full_feature_path = bs1_output_dir / "all_sequences_features.csv"
full_df = compute_features_for_sequences(df=source_df)
full_df.to_csv(full_feature_path, index=False)

X, y, _, _ =load_and_prepare_data(full_feature_path, target=target)

outer_seed_list = generate_seed_list(start_seed, num_outer_seeds)

baseline_1_results = run_baseline_1_parallel(
    X=X,
    y=y,
    full_df=full_df,
    outer_seed_list=outer_seed_list,
    final_model_names=final_model_names,
    target_arg=target_arg,
    scoring_scheme=scoring_scheme,
    n_trials=n_trials,
    scale_target=scale_target,
    n_jobs=n_jobs,
 )

all_test_df = baseline_1_results["all_test_df"]
all_test_df.to_csv(bs1_output_dir / "baseline_1_all_test_df.csv", index=False)
all_train_df = baseline_1_results["all_train_df"]
all_train_df.to_csv(bs1_output_dir / "baseline_1_all_train_df.csv", index=False)
seed_metrics_df = baseline_1_results["seed_metrics_df"]
seed_metrics_df.to_csv(bs1_output_dir / "baseline_1_seed_metrics_df.csv", index=False)

# Final plotting step for baseline 1 (one set of plots per model)
plot_settings = {
    "target": target_arg,
    "num_outer_seeds": num_outer_seeds,
    "n_trials": n_trials,
    "scale_target": scale_target,
    "use_standard_hyperparams": False,
}


# In[ ]:


for model_name in final_model_names:
    model_seed_metrics_df = seed_metrics_df[seed_metrics_df["model_name"] == model_name].copy()
    model_train_df = all_train_df[all_train_df["model_name"] == model_name].copy()
    model_test_df = all_test_df[all_test_df["model_name"] == model_name].copy()

    if model_seed_metrics_df.empty or model_train_df.empty or model_test_df.empty:
        print(Fore.YELLOW + f"Skipping plots for model '{model_name}' because no matching rows were found." + Style.RESET_ALL)
        continue

    model_output_dir = bs1_output_dir / f"model_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_plot_settings = {**plot_settings, "final_model_name": model_name}

    violin_plot_path = plot_seed_metric_violins(
        seed_metrics_df=model_seed_metrics_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )
    train_test_plot_path = plot_train_test_diagnostics(
        train_df=model_train_df,
        test_df=model_test_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )

    if violin_plot_path is not None:
        print(Fore.GREEN + f"[{model_name}] Saved metric violin plot: {violin_plot_path}" + Style.RESET_ALL)
    if train_test_plot_path is not None:
        print(Fore.GREEN + f"[{model_name}] Saved train/test diagnostics plot: {train_test_plot_path}" + Style.RESET_ALL)


# ## Baseline 2

# In[ ]:


bs2_output_dir = baseline_output_dir / "baseline_2" / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
bs2_output_dir.mkdir(parents=True, exist_ok=True)

source_df = pd.read_csv(repo_root / input_file)
target = TARGET_MAPPING[target_arg]
source_df = source_df.dropna(subset=[target]).reset_index(drop=True)

full_feature_path_bs2 = bs2_output_dir / "all_sequences_features.csv"
full_df_bs2 = compute_features_for_sequences(df=source_df)
full_df_bs2.to_csv(full_feature_path_bs2, index=False)

X_bs2, y_bs2, _, _ = load_and_prepare_data(full_feature_path_bs2, target=target)
outer_seed_list_bs2 = generate_seed_list(start_seed, num_outer_seeds)

baseline_2_results = run_baseline_2_parallel(
    X=X_bs2,
    y=y_bs2,
    full_df=full_df_bs2,
    full_feature_path=full_feature_path_bs2,
    amp_feature_path=amp_feature_path,
    outer_seed_list=outer_seed_list_bs2,
    final_model_names=final_model_names,
    target_arg=target_arg,
    scoring_scheme=scoring_scheme,
    n_trials=n_trials,
    scale_target=scale_target,
    n_jobs=n_jobs,
    cluster_threshold=cluster_threshold,
    cluster_method=cluster_method,
    max_cluster_size=max_cluster_size,
    variance_threshold=variance_threshold,
    perfect_corr_threshold=perfect_corr_threshold,
)

all_test_df_bs2 = baseline_2_results["all_test_df"]
all_test_df_bs2.to_csv(bs2_output_dir / "baseline_2_all_test_df.csv", index=False)
all_train_df_bs2 = baseline_2_results["all_train_df"]
all_train_df_bs2.to_csv(bs2_output_dir / "baseline_2_all_train_df.csv", index=False)
seed_metrics_df_bs2 = baseline_2_results["seed_metrics_df"]
seed_metrics_df_bs2.to_csv(bs2_output_dir / "baseline_2_seed_metrics_df.csv", index=False)
keep_summary_df_bs2 = baseline_2_results["keep_summary_df"]
keep_summary_df_bs2.to_csv(bs2_output_dir / "baseline_2_keep_summary_df.csv", index=False)

print(
    Fore.GREEN
    + f"Baseline 2 features: {baseline_2_results['n_features_before']} -> {baseline_2_results['n_features_after']}"
    + Style.RESET_ALL
)

plot_settings_bs2 = {
    "target": target_arg,
    "num_outer_seeds": num_outer_seeds,
    "n_trials": n_trials,
    "scale_target": scale_target,
    "use_standard_hyperparams": False,
}


# In[ ]:


for model_name in final_model_names:
    model_seed_metrics_df = seed_metrics_df_bs2[seed_metrics_df_bs2["model_name"] == model_name].copy()
    model_train_df = all_train_df_bs2[all_train_df_bs2["model_name"] == model_name].copy()
    model_test_df = all_test_df_bs2[all_test_df_bs2["model_name"] == model_name].copy()

    if model_seed_metrics_df.empty or model_train_df.empty or model_test_df.empty:
        continue

    model_output_dir = bs2_output_dir / f"model_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_plot_settings = {**plot_settings_bs2, "final_model_name": model_name}

    violin_plot_path = plot_seed_metric_violins(
        seed_metrics_df=model_seed_metrics_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )
    train_test_plot_path = plot_train_test_diagnostics(
        train_df=model_train_df,
        test_df=model_test_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )

    if violin_plot_path is not None:
        print(Fore.GREEN + f"[Baseline 2][{model_name}] Saved metric violin plot: {violin_plot_path}" + Style.RESET_ALL)
    if train_test_plot_path is not None:
        print(Fore.GREEN + f"[Baseline 2][{model_name}] Saved train/test diagnostics plot: {train_test_plot_path}" + Style.RESET_ALL)


# ## Baseline 3
# Discrimination factor and Hydrophobic moment only, no feature selection

# In[ ]:


bs3_output_dir = baseline_output_dir / "baseline_3" / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
bs3_output_dir.mkdir(parents=True, exist_ok=True)
source_df = pd.read_csv(repo_root / input_file)
target = TARGET_MAPPING[target_arg]
# Only keep rows where the target value is not nan such that if the input file contains rows without target values, they are automatically dropped
source_df = source_df.dropna(subset=[target]).reset_index(drop=True)

full_feature_path = bs3_output_dir / "all_sequences_features.csv"
full_df = compute_features_for_sequences(df=source_df)
full_df.to_csv(full_feature_path, index=False)

X, y, _, _ =load_and_prepare_data(full_feature_path, target=target)

# Select only the features we want to look into
selected_features = ["Discrimination_Factor", "Hydrophobic_Moment"]
X = X[selected_features]

outer_seed_list = generate_seed_list(start_seed, num_outer_seeds)

baseline_3_results = run_baseline_1_parallel(
    X=X,
    y=y,
    full_df=full_df,
    outer_seed_list=outer_seed_list,
    final_model_names=final_model_names,
    target_arg=target_arg,
    scoring_scheme=scoring_scheme,
    n_trials=n_trials,
    scale_target=scale_target,
    n_jobs=n_jobs,
 )

all_test_df_bs3 = baseline_3_results["all_test_df"]
all_test_df_bs3.to_csv(bs3_output_dir / "baseline_3_all_test_df.csv", index=False)
all_train_df_bs3 = baseline_3_results["all_train_df"]
all_train_df_bs3.to_csv(bs3_output_dir / "baseline_3_all_train_df.csv", index=False)
seed_metrics_df_bs3 = baseline_3_results["seed_metrics_df"]
seed_metrics_df_bs3.to_csv(bs3_output_dir / "baseline_3_seed_metrics_df.csv", index=False)

# Final plotting step for baseline 3 (one set of plots per model)
plot_settings = {
    "target": target_arg,
    "num_outer_seeds": num_outer_seeds,
    "n_trials": n_trials,
    "scale_target": scale_target,
    "use_standard_hyperparams": False,
}


# In[ ]:


for model_name in final_model_names:
    model_seed_metrics_df = seed_metrics_df_bs3[seed_metrics_df_bs3["model_name"] == model_name].copy()
    model_train_df = all_train_df_bs3[all_train_df_bs3["model_name"] == model_name].copy()
    model_test_df = all_test_df_bs3[all_test_df_bs3["model_name"] == model_name].copy()

    if model_seed_metrics_df.empty or model_train_df.empty or model_test_df.empty:
        print(Fore.YELLOW + f"Skipping plots for model '{model_name}' because no matching rows were found." + Style.RESET_ALL)
        continue

    model_output_dir = bs3_output_dir / f"model_{model_name}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_plot_settings = {**plot_settings, "final_model_name": model_name}

    violin_plot_path = plot_seed_metric_violins(
        seed_metrics_df=model_seed_metrics_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )
    train_test_plot_path = plot_train_test_diagnostics(
        train_df=model_train_df,
        test_df=model_test_df,
        output_dir=model_output_dir,
        settings=model_plot_settings,
    )

    if violin_plot_path is not None:
        print(Fore.GREEN + f"[{model_name}] Saved metric violin plot: {violin_plot_path}" + Style.RESET_ALL)
    if train_test_plot_path is not None:
        print(Fore.GREEN + f"[{model_name}] Saved train/test diagnostics plot: {train_test_plot_path}" + Style.RESET_ALL)

