from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.registry import get_strategy
from src.models.train import train_final_model_inner_cv
from src.selection.shared import generate_even_feature_sets, save_json


@dataclass
class PyramidConfig:
    model_name: str = "ridge"
    scoring_scheme: str = "neg_mean_squared_error"
    top_percentage: float = 0.5
    set_size: int = 20
    scale_target: bool = False
    random_state: int = 42
    num_pyramid_seeds: int = 6
    start_seed: int = 42


def _extract_top_features(
    all_importances: Sequence[pd.DataFrame],
    top_percentage: float,
    iteration_folder: Path | None,
) -> List[str]:
    """Extract top-percentage features from each set and keep deterministic unique order."""
    all_top_features: List[str] = []
    for importance_df in all_importances:
        ranked = importance_df.sort_values(by="abs_coefficient", ascending=False)
        keep_n = max(1, int(len(ranked) * top_percentage))
        all_top_features.extend(ranked.head(keep_n)["feature"].tolist())

    if iteration_folder is not None:
        combined_df = pd.concat(all_importances, ignore_index=True)
        combined_df["is_top_feature"] = combined_df["feature"].isin(all_top_features)
        iteration_folder.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(iteration_folder / "feature_selection_combined.csv", index=False)

    return list(dict.fromkeys(all_top_features))


def _run_single_pyramid_pass(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    features_in_set: Sequence[str],
    out_folder: Path | None,
    model_name: str,
    scoring_scheme: str,
    random_state: int,
    scale_target: bool,
) -> pd.DataFrame:
    """Fit one model on one feature set and return absolute coefficient importance."""
    strategy = get_strategy(model_name)
    X_subset = X[list(features_in_set)].reset_index(drop=True)
    y_subset = y.reset_index(drop=True)

    params = strategy.default_params()

    model = strategy.build_estimator(
        params=params,
        scale_target=scale_target,
        random_state=random_state,
    )
    model.fit(X_subset, y_subset)

    coefficients = strategy.get_coefficients(model)
    feature_importance = pd.DataFrame(
        {
            "feature": list(X_subset.columns),
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values(by="abs_coefficient", ascending=False)

    if out_folder is not None:
        out_folder.mkdir(parents=True, exist_ok=True)
        save_json(
            out_folder / "results.json",
            {
                "model_name": strategy.name,
                "scoring_scheme": scoring_scheme,
                "scale_target": scale_target,
                "params": params,
                "n_features": int(len(features_in_set)),
                "features": list(features_in_set),
                "coefficients": [float(value) for value in coefficients],
            },
        )
        pd.DataFrame([coefficients], columns=list(X_subset.columns)).to_csv(
            out_folder / "coefficients.csv", index=False
        )
        feature_importance.to_csv(out_folder / "feature_importance.csv", index=False)

    return feature_importance


def run_pyramid_until_target(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    available_features: Sequence[str],
    out_folder: Path | None,
    config: PyramidConfig,
    seed: int | None = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Run one simplified pyramid elimination and return final selected features + ranking."""
    current_features = list(available_features)
    run_seed = config.random_state if seed is None else int(seed)

    for step in range(25):
        num_sets = max(1, int(np.round(len(current_features) / config.set_size)))
        if num_sets <= 1:
            break

        iteration_folder = out_folder / f"step_{step}" if out_folder is not None else None
        if iteration_folder is not None:
            iteration_folder.mkdir(parents=True, exist_ok=True)

        feature_sets = generate_even_feature_sets(
            num_sets=num_sets,
            available_features=current_features,
            seed_value=run_seed,
        )

        all_importances: List[pd.DataFrame] = []
        for set_index, feature_set in enumerate(feature_sets, start=1):
            set_folder = iteration_folder / f"set_{set_index}" if iteration_folder is not None else None
            feature_importance = _run_single_pyramid_pass(
                X=X,
                y=y,
                features_in_set=feature_set,
                out_folder=set_folder,
                model_name=config.model_name,
                scoring_scheme=config.scoring_scheme,
                random_state=run_seed,
                scale_target=config.scale_target,
            )
            all_importances.append(feature_importance)

        current_features = _extract_top_features(
            all_importances=all_importances,
            top_percentage=config.top_percentage,
            iteration_folder=iteration_folder,
        )

    final_folder = out_folder / "final" if out_folder is not None else None
    if final_folder is not None:
        final_folder.mkdir(parents=True, exist_ok=True)

    final_importance = _run_single_pyramid_pass(
        X=X,
        y=y,
        features_in_set=current_features,
        out_folder=final_folder,
        model_name=config.model_name,
        scoring_scheme=config.scoring_scheme,
        random_state=run_seed,
        scale_target=config.scale_target,
    )

    final_ranked = final_importance.sort_values(by="abs_coefficient", ascending=False).head(config.set_size)
    final_features = final_ranked["feature"].tolist()

    if out_folder is not None:
        final_ranked.to_csv(out_folder / "top_features.csv", index=False)
        save_json(out_folder / "top_features.json", final_ranked.to_dict("records"))

    return final_features, final_ranked


def aggregate_pyramid_seed_features(
    *,
    top_feature_tables: Sequence[pd.DataFrame],
    set_size: int,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """
    Aggregate top features across pyramid seeds for one final selected subset.

    Returns
    -------
    selected_features : List[str]
        Final selected features based on selection frequency and mean absolute coefficient.
    summary_df : pd.DataFrame
        Summary of all features that were in the top sets across seeds, with their mean absolute coefficient and selection count.
    selected_summary : pd.DataFrame
        Subset of summary_df corresponding to the final selected features.
    """
    if not top_feature_tables:
        empty = pd.DataFrame(columns=["feature", "mean_abs_coefficient", "selection_count"])
        return [], empty, empty.copy()
    max_selection_count = len(top_feature_tables)
    stacked = pd.concat(top_feature_tables, ignore_index=True)
    summary = (
        stacked.groupby("feature", as_index=False)
        .agg(
            mean_abs_coefficient=("abs_coefficient", "mean"),
            selection_count=("feature", "count"),
            selection_fraction=("feature", lambda x: len(x) / max_selection_count),
        )
        .sort_values(by=["selection_count", "mean_abs_coefficient", "feature"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    selected_summary = summary.head(set_size).reset_index(drop=True)
    selected_features = selected_summary["feature"].tolist()
    return selected_features, summary, selected_summary


def run_pyramid_selection(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    available_features: Sequence[str],
    save_folder: Path | None,
    config: PyramidConfig,
    seed_list: List[int],
    fixed_params: dict | None = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Run simplified pyramid selection across seeds and aggregate to one final subset."""
    _ = fixed_params
    per_seed_tables: List[pd.DataFrame] = []
    for seed in seed_list:
        # seed_folder = save_folder / f"seed_{seed}" if save_folder is not None else None
        # if save_folder is not None: seed_folder.mkdir(parents=True, exist_ok=True) 

        _, top_features = run_pyramid_until_target(
            X=X,
            y=y,
            available_features=available_features,
            out_folder=None,
            config=config,
            seed=seed,
        )
        per_seed_tables.append(top_features[["feature", "abs_coefficient"]].copy())

    selected_features, summary_df, selected_summary = aggregate_pyramid_seed_features(
        top_feature_tables=per_seed_tables,
        set_size=config.set_size,
    )
    if save_folder is not None:
        save_folder.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_folder / "aggregated_features.csv", index=False)
        save_json(save_folder / "selected_features.json", selected_features)
    return selected_features, summary_df


def run_single_pyramid_selection(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    available_features: Sequence[str],
    config: PyramidConfig,
    seed_list: List[int],
    save_root: Path | None = None,
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """Run single pyramid selection and return selected features + summary + metadata."""
    save_folder = save_root / "single" if save_root is not None else None
    selected_features, summary_df = run_pyramid_selection(
        X=X,
        y=y,
        available_features=available_features,
        save_folder=save_folder,
        config=config,
        seed_list=seed_list,
    )

    selection_meta = {
        "feature_option": "single",
        "num_initial_features": int(len(available_features)),
        "num_selected_features": int(len(selected_features)),
    }
    return selected_features, summary_df, selection_meta


def run_double_pyramid_selection(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    available_features: Sequence[str],
    config: PyramidConfig,
    seed_list: List[int],
    double_top_percent: float,
    save_root: Path | None = None,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run double pyramid selection and return final features, summaries, and metadata."""
    stage1_folder = save_root / "double" / "stage1" if save_root is not None else None
    stage2_folder = save_root / "double" / "stage2" if save_root is not None else None

    stage1_features, stage1_summary_df = run_pyramid_selection(
        X=X,
        y=y,
        available_features=available_features,
        save_folder=stage1_folder,
        config=config,
        seed_list=seed_list,
    )

    num_features_to_select = max(1, int(len(available_features) * double_top_percent))
    stage1_selected_features = stage1_summary_df.head(num_features_to_select)["feature"].tolist()

    final_features, final_summary_df = run_pyramid_selection(
        X=X,
        y=y,
        available_features=stage1_selected_features,
        save_folder=stage2_folder,
        config=config,
        seed_list=seed_list,
    )

    selection_meta = {
        "feature_option": "double",
        "num_initial_features": int(len(available_features)),
        "num_stage1_features": int(len(stage1_features)),
        "num_selected_features": int(len(final_features)),
        "double_top_percent": float(double_top_percent),
    }

    return final_features, final_summary_df, stage1_summary_df, selection_meta


