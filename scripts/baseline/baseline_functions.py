from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from src.features.clustering import get_clustered_groups, get_features_to_keep, summarize_correlations
from src.models.train import train_final_model_inner_cv
from src.visualization.plot_evaluation_pipeline import (
    plot_seed_metric_violins,
    plot_train_test_diagnostics,
)


def build_prediction_frame(
    *,
    ompp_ids: list,
    y_true: pd.Series,
    y_pred: np.ndarray,
    split: str,
    model_name: str,
    outer_seed: int,
    outer_fold: int,
    target_arg: str,
) -> pd.DataFrame:
    """Build long-format per-sample prediction table for one fold/model/split."""
    return pd.DataFrame(
        {
            "OMPP nr": [str(ompp_id) for ompp_id in ompp_ids],
            "y_true": y_true.to_numpy(dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
            "split": str(split),
            "model_name": str(model_name),
            "outer_seed": int(outer_seed),
            "outer_fold": int(outer_fold),
            "target_arg": str(target_arg),
        }
    )


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for one prediction subset."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    spearman = float("nan")
    if y_true.size >= 2 and y_pred.size >= 2:
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(valid_mask) >= 2:
            spearman_value, _ = spearmanr(y_true[valid_mask], y_pred[valid_mask])
            if np.isfinite(spearman_value):
                spearman = float(spearman_value)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}

def plot_baselines(
        *,
        final_model_names, 
        seed_metrics_df, 
        all_test_df, 
        all_train_df,
        output_dir,
        plot_settings):
    for model_name in final_model_names:
        model_seed_metrics_df = seed_metrics_df[seed_metrics_df["model_name"] == model_name].copy()
        model_train_df = all_train_df[all_train_df["model_name"] == model_name].copy()
        model_test_df = all_test_df[all_test_df["model_name"] == model_name].copy()

        if model_seed_metrics_df.empty or model_train_df.empty or model_test_df.empty:
            print(Fore.YELLOW + f"Skipping plots for model '{model_name}' because no matching rows were found." + Style.RESET_ALL)
            continue

        model_output_dir = output_dir / f"model_{model_name}"
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


def run_one_seed(task):
    X = task["X"]
    y = task["y"]
    full_df = task["full_df"]
    outer_seed = task["outer_seed"]
    final_model_names = task["final_model_names"]
    target_arg = task["target_arg"]
    scoring_scheme = task["scoring_scheme"]
    n_trials = task["n_trials"]
    n_trials_by_model = task.get("n_trials_by_model")
    scale_target = task["scale_target"]

    kf = KFold(n_splits=5, shuffle=True, random_state=outer_seed)
    seed_train_dfs: list[pd.DataFrame] = []
    seed_test_dfs: list[pd.DataFrame] = []
    for outer_fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
        X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        train_ompp_ids = full_df.iloc[train_idx]["OMPP nr"].tolist()
        test_ompp_ids = full_df.iloc[test_idx]["OMPP nr"].tolist()

        for model_name in final_model_names:
            model_trials = n_trials
            if n_trials_by_model:
                model_trials = int(n_trials_by_model.get(str(model_name).lower(), n_trials))
            final_model, _, y_pred_train = train_final_model_inner_cv(
                X=X_train,
                y=y_train,
                model_name=model_name,
                scoring_scheme=scoring_scheme,
                inner_cv=5,
                random_state=42,
                n_trials=model_trials,
                use_default_params=False,
                n_jobs=1,
                scale_target=scale_target,
                show_progress_bar=False,
            )
            y_pred_test = np.asarray(final_model.predict(X_test), dtype=float)

            train_prediction_df = build_prediction_frame(
                ompp_ids=train_ompp_ids,
                y_true=y_train,
                y_pred=np.asarray(y_pred_train, dtype=float),
                split="train",
                model_name=model_name,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                target_arg=target_arg,
            )
            test_prediction_df = build_prediction_frame(
                ompp_ids=test_ompp_ids,
                y_true=y_test,
                y_pred=y_pred_test,
                split="test",
                model_name=model_name,
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                target_arg=target_arg,
            )

            seed_train_dfs.append(train_prediction_df)
            seed_test_dfs.append(test_prediction_df)

    seed_train_df = pd.concat(seed_train_dfs, ignore_index=True)
    seed_test_df = pd.concat(seed_test_dfs, ignore_index=True)
    seed_metrics: list[dict[str, float]] = []
    for model_name in final_model_names:
        y_pred_test = seed_test_df[seed_test_df["model_name"] == model_name]["y_pred"].to_numpy(dtype=float)
        y_true_test = seed_test_df[seed_test_df["model_name"] == model_name]["y_true"].to_numpy(dtype=float)
        test_metrics = calc_metrics(y_true_test, y_pred_test)
        seed_metrics.append(
            {
                "model_name": str(model_name),
                "outer_seed": int(outer_seed),
                **test_metrics,
            }
        )

    return {
        "train_predictions": seed_train_df,
        "test_predictions": seed_test_df,
        "seed_metrics": seed_metrics,
    }

def run_baseline_4(
        *,
        X,
        y,
        full_df,
        outer_seed_list,
        target_arg
):
    seed_train_dfs: list[pd.DataFrame] = []
    seed_test_dfs: list[pd.DataFrame] = []
    for outer_seed in outer_seed_list:
        kf = KFold(5, shuffle=True, random_state=outer_seed)
        for outer_fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
            # Mean value for y training data
            X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
            X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
            train_ompp_ids = full_df.iloc[train_idx]["OMPP nr"].tolist()
            test_ompp_ids = full_df.iloc[test_idx]["OMPP nr"].tolist()

            train_mean = np.mean(y_train)

            y_pred_train = np.full_like(y_train, fill_value=train_mean, dtype=float)
            y_pred_test = np.full_like(y_test, fill_value=train_mean, dtype=float)

            train_prediction_df = build_prediction_frame(
                ompp_ids=train_ompp_ids,
                y_true=y_train,
                y_pred=np.asarray(y_pred_train, dtype=float),
                split="train",
                model_name="mean_baseline",
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                target_arg=target_arg,
            )
            test_prediction_df = build_prediction_frame(
                ompp_ids=test_ompp_ids,
                y_true=y_test,
                y_pred=y_pred_test,
                split="test",
                model_name="mean_baseline",
                outer_seed=int(outer_seed),
                outer_fold=int(outer_fold),
                target_arg=target_arg,
            )

            seed_train_dfs.append(train_prediction_df)
            seed_test_dfs.append(test_prediction_df)

    seed_train_df = pd.concat(seed_train_dfs, ignore_index=True)
    seed_test_df = pd.concat(seed_test_dfs, ignore_index=True)
    seed_metrics: list[dict[str, float]] = []
    y_pred_test = seed_test_df["y_pred"].to_numpy(dtype=float)
    y_true_test = seed_test_df["y_true"].to_numpy(dtype=float)
    test_metrics = calc_metrics(y_true_test, y_pred_test)
    seed_metrics.append(
        {
            "model_name": "mean_baseline",
            "outer_seed": int(outer_seed),
            **test_metrics,
        }
    )
    seed_metrics_df = pd.DataFrame(seed_metrics)

    return {
        "all_test_df": seed_test_df,
        "all_train_df": seed_train_df,
        "seed_metrics_df": seed_metrics_df,
    }




def run_baseline_1_parallel(
    *,
    X,
    y,
    full_df,
    outer_seed_list,
    final_model_names,
    target_arg,
    scoring_scheme,
    n_trials,
    n_trials_by_model=None,
    scale_target,
    n_jobs,
):
    tasks = [
        {
            "X": X,
            "y": y,
            "full_df": full_df,
            "outer_seed": outer_seed,
            "final_model_names": final_model_names,
            "target_arg": target_arg,
            "scoring_scheme": scoring_scheme,
            "n_trials": n_trials,
            "n_trials_by_model": n_trials_by_model,
            "scale_target": scale_target,
        }
        for outer_seed in outer_seed_list
    ]

    all_train_prediction_frames: list[pd.DataFrame] = []
    all_test_prediction_frames: list[pd.DataFrame] = []
    seed_metric_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_task = [executor.submit(run_one_seed, task) for task in tasks]

        with tqdm(total=len(future_to_task), desc="Outer folds", unit="fold") as progress:
            for future in as_completed(future_to_task):
                result = future.result()
                all_train_prediction_frames.append(result["train_predictions"])
                all_test_prediction_frames.append(result["test_predictions"])
                seed_metric_rows.extend(result["seed_metrics"])
                progress.update(1)

    all_test_df = pd.concat(all_test_prediction_frames, ignore_index=True)
    all_train_df = pd.concat(all_train_prediction_frames, ignore_index=True)
    seed_metrics_df = pd.DataFrame(seed_metric_rows)
    if not seed_metrics_df.empty:
        seed_metrics_df = seed_metrics_df.sort_values(["model_name", "outer_seed"]).reset_index(drop=True)

    return {
        "all_test_df": all_test_df,
        "all_train_df": all_train_df,
        "seed_metrics_df": seed_metrics_df,
    }


def build_clustered_feature_matrix(
    *,
    X: pd.DataFrame,
    full_df: pd.DataFrame,
    amp_feature_path,
    full_feature_path,
    cluster_threshold: float,
    cluster_method: str,
    max_cluster_size: int,
    variance_threshold: float,
    perfect_corr_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_df, corr_abs, _, _ = get_clustered_groups(
        data_path_1=str(amp_feature_path),
        data_path_2=str(full_feature_path),
        threshold=(1 - float(cluster_threshold)),
        method=str(cluster_method),
        clustered_path=None,
    )

    summary_df = summarize_correlations(
        ompp_features=full_df,
        cluster_df=cluster_df,
        corr_abs=corr_abs,
        threshold=float(cluster_threshold),
        summary_save_path=None,
    )
    keep_summary_df, _ = get_features_to_keep(
        summary_df=summary_df,
        corr_abs=corr_abs,
        max_cluster_size=int(max_cluster_size),
        variance_threshold=float(variance_threshold),
        perfect_corr_threshold=float(perfect_corr_threshold),
        keep_save_path=None,
    )

    features_to_keep = keep_summary_df.loc[keep_summary_df["keep"], "feature_name"].astype(str).tolist()
    features_to_keep = [feature for feature in features_to_keep if feature in X.columns]
    X_clustered = X.loc[:, features_to_keep].copy()

    return X_clustered, keep_summary_df


def run_baseline_2_parallel(
    *,
    X,
    y,
    full_df,
    full_feature_path,
    amp_feature_path,
    outer_seed_list,
    final_model_names,
    target_arg,
    scoring_scheme,
    n_trials,
    scale_target,
    n_jobs,
    cluster_threshold,
    cluster_method,
    max_cluster_size,
    variance_threshold,
    perfect_corr_threshold,
):
    X_clustered, keep_summary_df = build_clustered_feature_matrix(
        X=X,
        full_df=full_df,
        amp_feature_path=amp_feature_path,
        full_feature_path=full_feature_path,
        cluster_threshold=cluster_threshold,
        cluster_method=cluster_method,
        max_cluster_size=max_cluster_size,
        variance_threshold=variance_threshold,
        perfect_corr_threshold=perfect_corr_threshold,
    )

    baseline_results = run_baseline_1_parallel(
        X=X_clustered,
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

    return {
        **baseline_results,
        "keep_summary_df": keep_summary_df,
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(X_clustered.shape[1]),
    }