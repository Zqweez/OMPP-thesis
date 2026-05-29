from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.tree import plot_tree

from colorama import Fore, Style


def _discover_tree_models(models_dir: Path, model_kind: str) -> list[Path]:
	"""Find tree ensemble model artifacts in a folder tree by filename pattern."""
	if not models_dir.exists():
		return []

	all_joblib = sorted(models_dir.rglob("*.joblib"))
	if model_kind == "both":
		return [path for path in all_joblib if "random_forest" in path.stem.lower() or "gradient_boosted" in path.stem.lower()]
	elif model_kind == "random_forest":
		return [path for path in all_joblib if "random_forest" in path.stem.lower()]
	elif model_kind == "gradient_boosted":
		return [path for path in all_joblib if "gradient_boosted" in path.stem.lower()]

def _unwrap_model(model: Any) -> Any:
	"""Extract the inner estimator from wrappers/pipelines when present."""
	if hasattr(model, "regressor_") and hasattr(model.regressor_, "named_steps"):
		return model.regressor_.named_steps.get("model", model.regressor_)
	if hasattr(model, "named_steps"):
		return model.named_steps.get("model", model)
	return model


def _resolve_feature_names(model: Any, model_path: Path) -> list[str]:
	"""Resolve feature names from model metadata or local sidecar files."""
	if hasattr(model, "feature_names_in_"):
		return [str(item) for item in model.feature_names_in_]

	if hasattr(model, "regressor_") and hasattr(model.regressor_, "feature_names_in_"):
		return [str(item) for item in model.regressor_.feature_names_in_]

	candidates = [
		model_path.with_name("selected_features.csv"),
		model_path.with_name("feature_importance.csv"),
		model_path.with_name(f"{model_path.stem}_selected_features.csv"),
	]

	for candidate in candidates:
		if not candidate.exists():
			continue

		df = pd.read_csv(candidate)
		if "feature" in df.columns:
			features = df["feature"].dropna().astype(str).tolist()
			if features:
				return features
		if df.shape[1] == 1:
			features = df.iloc[:, 0].dropna().astype(str).tolist()
			if features:
				return features

	raise ValueError(f"Could not resolve feature names for model: {model_path}")


def _resolve_xy(
	model_path: Path,
	feature_names: list[str],
	data_csv: Path | None,
	target_column: str | None,
) -> tuple[pd.DataFrame, pd.Series] | None:
	"""Resolve X/y either from explicit CSV input or model-sidecar files."""
	if data_csv is not None:
		if not data_csv.exists():
			raise FileNotFoundError(f"Provided data_csv does not exist: {data_csv}")
		if not target_column:
			raise ValueError("target_column must be provided when data_csv is used.")

		df = pd.read_csv(data_csv)
		missing = [feature for feature in feature_names if feature not in df.columns]
		if missing:
			raise ValueError(
				f"Missing {len(missing)} feature columns in data_csv for {model_path.name}: {missing[:10]}"
			)
		if target_column not in df.columns:
			raise ValueError(f"target_column '{target_column}' not found in {data_csv}")

		X = df[feature_names].copy()
		y = df[target_column].copy()
		return X, y

	x_candidates = [
		model_path.with_name("selected_features_matrix.csv"),
		model_path.with_name(f"{model_path.stem}_feature_matrix.csv"),
	]
	y_candidates = [
		model_path.with_name("final_model_in_sample_predictions.csv"),
	]

	x_df: pd.DataFrame | None = None
	y_series: pd.Series | None = None

	for candidate in x_candidates:
		if candidate.exists():
			x_df = pd.read_csv(candidate)
			break

	for candidate in y_candidates:
		if candidate.exists():
			pred_df = pd.read_csv(candidate)
			if "y_true" in pred_df.columns:
				y_series = pred_df["y_true"].copy()
				break

	if x_df is None or y_series is None:
		return None

	missing = [feature for feature in feature_names if feature not in x_df.columns]
	if missing:
		return None

	X = x_df[feature_names].copy()
	y = y_series.reset_index(drop=True)

	if len(X) != len(y):
		n = min(len(X), len(y))
		X = X.iloc[:n].copy()
		y = y.iloc[:n].copy()

	return X, y


def _plot_feature_importance(
	inner_model: Any,
	feature_names: list[str],
	output_path: Path,
	top_n: int,
	model_label: str,
) -> pd.DataFrame:
	"""Plot built-in tree ensemble feature importances and return ranked table."""
	importances = np.asarray(inner_model.feature_importances_, dtype=float)
	order = np.argsort(importances)[::-1]

	if top_n > 0:
		order = order[:top_n]

	ordered_importances = importances[order]
	ordered_features = np.array(feature_names, dtype=object)[order]

	fig_width = max(10, int(len(order) * 0.45))
	plt.figure(figsize=(fig_width, 6.5))
	plt.bar(range(len(order)), ordered_importances)
	plt.xticks(range(len(order)), ordered_features, rotation=60, ha="right")
	plt.ylabel("Feature importance")
	plt.title(f"{model_label} Feature Importance")
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()

	return pd.DataFrame({"feature": ordered_features, "importance": ordered_importances})


def _plot_tree_preview(
	inner_model: Any,
	feature_names: list[str],
	output_path: Path,
	max_depth: int,
	model_label: str,
) -> None:
	"""Plot first four trees as a 2x2 overview for quick structural inspection."""
	estimators = inner_model.estimators_
	if isinstance(estimators, np.ndarray):
		estimator_list = list(estimators.ravel())
	else:
		estimator_list = list(estimators)

	fig, axes = plt.subplots(2, 2, figsize=(20, 14))
	flat_axes = axes.ravel()
	n_trees = min(4, len(estimator_list))

	for tree_idx in range(n_trees):
		ax = flat_axes[tree_idx]
		plot_tree(
			estimator_list[tree_idx],
			feature_names=feature_names,
			filled=True,
			rounded=True,
			impurity=True,
			proportion=False,
			precision=3,
			fontsize=7,
			max_depth=max_depth,
			ax=ax,
		)
		ax.set_title(f"Tree #{tree_idx}")

	for tree_idx in range(n_trees, 4):
		flat_axes[tree_idx].axis("off")

	fig.suptitle(f"{model_label} Tree Preview (first {n_trees} trees)", y=1.02)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def _plot_permutation_importance(
	model: Any,
	X: pd.DataFrame,
	y: pd.Series,
	output_path: Path,
	top_n: int,
	n_repeats: int,
	random_state: int,
) -> pd.DataFrame:
    """Plot permutation importance as bar, violin, and box plots from one computation."""
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    order = np.argsort(result.importances_mean)[::-1]
    if top_n > 0:
        order = order[:top_n]

    labels = X.columns.to_numpy()[order]
    means = result.importances_mean[order]
    stds = result.importances_std[order]
    importances_all = result.importances[order]
    violin_data = [importances_all[idx] for idx in range(importances_all.shape[0])]
    x_positions = np.arange(len(order), dtype=float)
    rng = np.random.default_rng(random_state)

    # --- Bar plot (mean ± std + individual repeat dots) ---
    fig_width = max(10, int(len(order) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))
    ax.bar(x_positions, means, yerr=stds, capsize=3, zorder=2)

    for idx, x_pos in enumerate(x_positions):
        repeats = importances_all[idx]
        jitter = rng.normal(loc=0.0, scale=0.06, size=len(repeats))
        ax.scatter(
            np.full(len(repeats), x_pos) + jitter,
            repeats,
            s=10,
            c="gray",
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Permutation importance (mean ± std)")
    ax.set_title("Permutation Importance")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path.with_name("permutation_importance_bar.pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- Violin plot (distribution + mean/std + individual repeat dots) ---
    fig_violin, ax_violin = plt.subplots(figsize=(fig_width, 7.2))
    parts = ax_violin.violinplot(
        violin_data,
        positions=x_positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_alpha(0.35)
        body.set_edgecolor("black")

    ax_violin.errorbar(
        x_positions,
        means,
        yerr=stds,
        fmt="o",
        markersize=4,
        color="black",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )

    for idx, x_pos in enumerate(x_positions):
        repeats = importances_all[idx]
        jitter = rng.normal(loc=0.0, scale=0.05, size=len(repeats))
        ax_violin.scatter(
            np.full(len(repeats), x_pos) + jitter,
            repeats,
            s=10,
            c="gray",
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )

    ax_violin.set_xticks(x_positions)
    ax_violin.set_xticklabels(labels, rotation=60, ha="right")
    ax_violin.set_ylabel("Permutation importance distribution")
    ax_violin.set_title("Permutation Importance (Violin + Mean ± STD)")
    fig_violin.tight_layout(rect=(0, 0.05, 1, 1))
    fig_violin.savefig(output_path.with_name("permutation_importance_violin.pdf"), bbox_inches="tight")
    plt.close(fig_violin)

    # --- Boxplot (distribution + mean/std + individual repeat dots) ---
    fig_box, ax_box = plt.subplots(figsize=(fig_width, 7.2))
    box_parts = ax_box.boxplot(
        violin_data,
        positions=x_positions,
        widths=0.6,
        showmeans=True,
        showfliers=False,
        patch_artist=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
    )
    for box in box_parts["boxes"]:
        box.set_facecolor("lightgray")
        box.set_alpha(0.35)
        box.set_edgecolor("black")
    for median in box_parts["medians"]:
        median.set_color("black")
        median.set_alpha(0)
    for whisker in box_parts["whiskers"]:
        whisker.set_color("black")
        whisker.set_alpha(0)
    for cap in box_parts["caps"]:
        cap.set_alpha(0)
    for mean in box_parts["means"]:
        mean.set_color("black")

    ebox = ax_box.errorbar(
        x_positions,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#ce3333",
        elinewidth=1.0,
        capsize=3,
        zorder=3,
    )
    ebox[-1][0].set_linestyle("-.")

    for idx, x_pos in enumerate(x_positions):
        repeats = importances_all[idx]
        jitter = rng.normal(loc=0.0, scale=0.05, size=len(repeats))
        ax_box.scatter(
            np.full(len(repeats), x_pos) + jitter,
            repeats,
            s=10,
            c="gray",
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )

    ax_box.set_xticks(x_positions)
    ax_box.set_xticklabels(labels, rotation=60, ha="right")
    ax_box.set_ylabel("Permutation importance distribution")
    ax_box.set_title("Permutation Importance (Mean ± STD)")
    fig_box.text(
        0.01,
        0.01,
        "Box shows quartiles and median of repeat values; black error bars show mean ± std; gray dots are individual repeats.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    fig_box.tight_layout(rect=(0, 0.05, 1, 1))
    fig_box.savefig(output_path.with_name("permutation_importance_box.pdf"), bbox_inches="tight")
    plt.close(fig_box)

    return pd.DataFrame(
        {
            "feature": labels,
            "importance_mean": means,
            "importance_std": stds,
        }
    )

def _plot_partial_dependence(
	model: Any,
	X: pd.DataFrame,
	ranked_features: list[str],
	output_path: Path,
	n_features: int,
) -> list[str]:
	"""Plot PDP curves for top-ranked features to show marginal feature effects."""
	selected = ranked_features[: max(1, n_features)]

	fig, ax = plt.subplots(figsize=(12, 8))
	PartialDependenceDisplay.from_estimator(
		model,
		X,
		features=selected,
		feature_names=list(X.columns),
		ax=ax,
        kind="both"
	)
	fig.suptitle("Partial Dependence Plots", y=1.03)
	fig.tight_layout()
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return selected


def visualize_tree_ensemble_models(
	models_dir: str | Path = "data/final-models",
	output_dir: str | Path | None = None,
	data_csv: str | Path | None = None,
	target_column: str | None = None,
	model_kind: str = "random_forest",
	top_n_importance: int = 20,
	permutation_repeats: int = 50,
	pdp_features: int = 3,
	tree_max_depth: int = 3,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Run tree-ensemble visual diagnostics for matching models in a directory."""
	models_path = Path(models_dir)
	if output_dir:
		output_root = Path(output_dir)
	else:
		output_root = models_path / "tree-visualizations"
	output_root.mkdir(parents=True, exist_ok=True)

	discovered = _discover_tree_models(models_path, model_kind=model_kind)
	if not discovered:
		print(f"No matching '{model_kind}' model artifacts found in: {models_path}")
		return {"models_found": 0, "models_processed": 0, "results": []}

	external_data = Path(data_csv) if data_csv else None

	results: list[dict[str, Any]] = []
	for model_path in discovered:
		model_name = model_path.stem
		model_out = output_root / model_name
		model_out.mkdir(parents=True, exist_ok=True)

		print(f"\nProcessing: {model_path}")
		status: dict[str, Any] = {
			"model": str(model_path),
			"output_dir": str(model_out),
			"plots": {},
			"warnings": [],
		}

		try:
			model = joblib.load(model_path)
			inner_model = _unwrap_model(model)

			if not hasattr(inner_model, "feature_importances_"):
				status["warnings"].append("Model does not expose feature_importances_; skipping.")
				results.append(status)
				continue

			feature_names = _resolve_feature_names(model, model_path)
			print(f"{Fore.GREEN}Resolved {len(feature_names)} feature names.{Style.RESET_ALL}")
			model_label = model_kind.replace("_", " ").title()

			print(f"{Fore.GREEN}Plotting feature importance for {len(feature_names)} features...{Style.RESET_ALL}")
			fi_path = model_out / "feature_importance_bar.pdf"
			importance_df = _plot_feature_importance(
				inner_model=inner_model,
				feature_names=feature_names,
				output_path=fi_path,
				top_n=top_n_importance,
				model_label=model_label,
			)
			importance_df.to_csv(model_out / "feature_importance_values.csv", index=False)
			status["plots"]["feature_importance"] = str(fi_path)

			tree_path = model_out / f"tree_preview_2x2_depth{tree_max_depth}.pdf"
			_plot_tree_preview(
				inner_model=inner_model,
				feature_names=feature_names,
				output_path=tree_path,
				max_depth=tree_max_depth,
				model_label=model_label,
			)
			status["plots"]["tree_preview"] = str(tree_path)

			xy = _resolve_xy(
				model_path=model_path,
				feature_names=feature_names,
				data_csv=external_data,
				target_column=target_column,
			)
			if xy is None:
				status["warnings"].append(
					"Could not resolve X/y automatically; permutation importance and PDP skipped. "
					"Provide --data-csv and --target-column to enable these plots."
				)
				results.append(status)
				continue

			X, y = xy

			perm_path = model_out / "permutation_importance.pdf"
			perm_df = _plot_permutation_importance(
				model=model,
				X=X,
				y=y,
				output_path=perm_path,
				top_n=top_n_importance,
				n_repeats=permutation_repeats,
				random_state=random_state,
			)
			perm_df.to_csv(model_out / "permutation_importance_values.csv", index=False)
			status["plots"]["permutation_importance_bar"] = str(perm_path.with_name("permutation_importance_bar.pdf"))
			status["plots"]["permutation_importance_violin"] = str(perm_path.with_name("permutation_importance_violin.pdf"))
			status["plots"]["permutation_importance_box"] = str(perm_path.with_name("permutation_importance_box.pdf"))

			ranked_for_pdp = perm_df["feature"].astype(str).tolist()
			if not ranked_for_pdp:
				ranked_for_pdp = importance_df["feature"].astype(str).tolist()

			pdp_path = model_out / "partial_dependence.pdf"
			used_features = _plot_partial_dependence(
				model=model,
				X=X,
				ranked_features=ranked_for_pdp,
				output_path=pdp_path,
				n_features=pdp_features,
			)
			status["plots"]["partial_dependence"] = str(pdp_path)
			status["pdp_features"] = used_features

			results.append(status)

		except Exception as error:
			status["warnings"].append(f"Failed to process model: {error}")
			results.append(status)

	summary = {
		"models_found": len(discovered),
		"models_processed": sum(1 for item in results if item.get("plots")),
		"output_dir": str(output_root),
		"results": results,
	}

	with (output_root / "summary.json").open("w", encoding="utf-8") as file_handle:
		json.dump(summary, file_handle, indent=2)

	print(f"\nDone. Visualizations saved in: {output_root}")
	return summary


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate tree ensemble visualizations for .joblib models containing "
			"'random_forest' or 'gradient_boosted' in filename."
		)
	)
	parser.add_argument("--models-dir", default="data/final-models", help="Folder to scan for model files.")
	parser.add_argument("--output-dir", default=None, help="Folder for generated figures.")
	parser.add_argument("--data-csv", default=None, help="Optional dataset CSV containing feature columns and target column for permutation/PDP.")
	parser.add_argument("--target-column", default=None, help="Target column in --data-csv (required when --data-csv is provided).")
	parser.add_argument("--model-kind", choices=["random_forest", "gradient_boosted", "both"], default="random_forest", help="Model family to visualize by filename pattern.")
	parser.add_argument("--top-n-importance", type=int, default=20, help="Number of top features to plot.")
	parser.add_argument("--permutation-repeats", type=int, default=50, help="n_repeats for permutation importance.")
	parser.add_argument("--pdp-features", type=int, default=6, help="Number of top features for PDP plots.")
	parser.add_argument("--tree-max-depth", type=int, default=3, help="max_depth in tree preview plot.")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed for permutation importance.")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	visualize_tree_ensemble_models(
		models_dir=args.models_dir,
		output_dir=args.output_dir,
		data_csv=args.data_csv,
		target_column=args.target_column,
		model_kind=args.model_kind,
		top_n_importance=args.top_n_importance,
		permutation_repeats=args.permutation_repeats,
		pdp_features=args.pdp_features,
		tree_max_depth=args.tree_max_depth,
		random_state=args.random_state,
	)
