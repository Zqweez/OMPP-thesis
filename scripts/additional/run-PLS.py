from __future__ import annotations

import argparse
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from colorama import Fore, Style
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_MAPPING = {
	"npn": "NPN_Mean",
	"mic": "MIC_Mean",
	"pot": "Potentiation_Mean",
	"disc": "DISC_Mean",
	"cyto": "Cytotoxicity_Mean",
	"hemo": "Hemolysis_Mean",
}

EXCLUDE_COLS = {
	"Sequence",
	"OMPP nr",
	"Peptide",
	"Potentiation_Mean",
	"MIC_Mean",
	"NPN_Mean",
	"DISC_Mean",
	"Cytotoxicity_Mean",
	"Hemolysis_Mean",
}


def _load_json(path: str | Path) -> dict:
	json_path = Path(path)
	if not json_path.exists():
		raise FileNotFoundError(f"Could not find JSON file: {json_path}")
	with json_path.open("r") as file_handle:
		return json.load(file_handle)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	with path.open("r") as file_handle:
		return json.load(file_handle)


def _build_pls_pipeline(n_components: int) -> Pipeline:
	return Pipeline(
		[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
			("pls", PLSRegression(n_components=n_components)),
		]
	)


def _compute_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	mse = float(mean_squared_error(y_true, y_pred))
	return {
		"mse": mse,
		"rmse": float(np.sqrt(mse)),
		"r2": float(r2_score(y_true, y_pred)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
	}


def _sanitize_prediction_array(pred: np.ndarray) -> np.ndarray:
	arr = np.asarray(pred)
	if arr.ndim == 2 and arr.shape[1] == 1:
		return arr.ravel()
	return arr

def _resolve_run_folder(
	*,
	run_folder_arg: str | None,
	settings_file_arg: str | None,
) -> tuple[Path, dict]:
	if run_folder_arg:
		run_folder = Path(run_folder_arg)
		settings = _load_json_if_exists(run_folder / "settings.json") if (run_folder / "settings.json").exists() else {}
		return run_folder, settings

	if settings_file_arg:
		settings = _load_json(settings_file_arg)
		run_folder_value = settings.get("run_folder_relative") or settings.get("run_folder")
		if not run_folder_value:
			raise ValueError("Provided settings file does not contain run_folder info")
		return Path(run_folder_value), settings

	latest_settings_path = Path("outputs/feature-selection/latest/double_pyramid_latest_settings.json")
	if not latest_settings_path.exists():
		raise FileNotFoundError(
			"Could not resolve double pyramid run folder. Provide --run_folder or --settings_file."
		)

	settings = _load_json(latest_settings_path)
	run_folder_value = settings.get("run_folder_relative") or settings.get("run_folder")
	if not run_folder_value:
		raise ValueError("Latest double pyramid settings does not contain run_folder info")

	return Path(run_folder_value), settings


def _resolve_data_source(
	*,
	feature_source: str,
	input_file: str | None,
	run_folder: Path | None,
	settings: Dict[str, Any],
) -> Path:
	if feature_source == "reduced":
		return Path(input_file) if input_file else Path("data/OMPP_master_long_features_reduced.csv")

	if run_folder is None:
		raise ValueError("run_folder must be set for pyramid feature_source")

	if input_file:
		return Path(input_file)
	if settings.get("input_file"):
		return Path(str(settings["input_file"]))
	return Path("data/OMPP_master_long_features_reduced.csv")


def _resolve_target_column(
	*,
	target_arg: str | None,
	settings: Dict[str, Any],
	df: pd.DataFrame,
) -> str:
	if target_arg:
		target_col = TARGET_MAPPING[target_arg]
	elif settings.get("target_variable"):
		target_col = str(settings["target_variable"])
	elif settings.get("target_arg") in TARGET_MAPPING:
		target_col = TARGET_MAPPING[str(settings["target_arg"])]
	else:
		target_col = "NPN_Mean"

	if target_col not in df.columns:
		raise ValueError(
			f"Target column '{target_col}' not found in data. Available columns include: {list(df.columns[:10])}..."
		)
	return target_col


def _load_feature_frequency(run_folder: Path) -> pd.DataFrame:
	frequency_path = run_folder / "feature_frequency.csv"
	if not frequency_path.exists():
		raise FileNotFoundError(f"Could not find feature frequency file: {frequency_path}")

	frequency_df = pd.read_csv(frequency_path)
	required_cols = {"feature", "selection_count"}
	if not required_cols.issubset(frequency_df.columns):
		raise ValueError("feature_frequency.csv must contain 'feature' and 'selection_count' columns")

	frequency_df = frequency_df.sort_values(["selection_count", "feature"], ascending=[False, True]).reset_index(drop=True)
	return frequency_df


def _select_features(
	*,
	feature_source: str,
	run_folder: Path | None,
	top_n: int,
	full_feature_cols: List[str],
) -> Tuple[List[str], pd.DataFrame | None, List[str]]:
	if feature_source == "reduced":
		return full_feature_cols, None, []

	if run_folder is None:
		raise ValueError("run_folder must be set for pyramid feature_source")

	frequency_df = _load_feature_frequency(run_folder)
	top_df = frequency_df.head(top_n).copy()
	ranked_features = top_df["feature"].astype(str).tolist()

	selected_features = [feature for feature in ranked_features if feature in full_feature_cols]
	missing_features = [feature for feature in ranked_features if feature not in full_feature_cols]

	if not selected_features:
		raise ValueError(
			"No top_n features from feature_frequency.csv were found in the input dataset columns."
		)

	return selected_features, top_df, missing_features


def _get_max_components_for_cv(X: pd.DataFrame, y: pd.Series, cv: KFold) -> int:
	"""Compute a safe global n_components upper bound for all folds in a CV splitter."""
	# PLSRegression requires n_components <= min(n_train_samples, n_features)
	train_sizes = [len(train_idx) for train_idx, _ in cv.split(X, y)]

	min_train_size = min(train_sizes)
	max_components = min(8, int(min_train_size)-1)

	return max_components


def _tune_n_components_optuna(
	*,
	X: pd.DataFrame,
	y: pd.Series,
	cv: KFold,
	n_trials: int,
	random_state: int,
	scoring: str,
	n_jobs: int,
) -> Tuple[int, float, List[Dict[str, Any]]]:
	max_components = _get_max_components_for_cv(X, y, cv)

	sampler = optuna.samplers.TPESampler(seed=random_state)
	study = optuna.create_study(direction="maximize", sampler=sampler)
	optuna.logging.set_verbosity(optuna.logging.WARNING)

	def objective(trial: optuna.Trial) -> float:
		n_components = trial.suggest_int("n_components", 1, max_components)
		model = _build_pls_pipeline(n_components=n_components)
		try:
			scores = cross_val_score(
				model,
				X,
				y,
				cv=cv,
				scoring=scoring,
				n_jobs=n_jobs,
				error_score="raise",
			)
		except ValueError as exc:
			raise optuna.TrialPruned(f"Invalid trial for n_components={n_components}: {exc}") from exc

		mean_score = float(np.mean(scores))
		if not np.isfinite(mean_score):
			raise optuna.TrialPruned(f"Non-finite CV score for n_components={n_components}")

		return mean_score

	optuna.logging.set_verbosity(optuna.logging.WARNING)
	study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

	completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
	if not completed_trials:
		raise ValueError(
			"Optuna could not find a valid n_components value. "
			"Try fewer CV folds or check data after filtering features."
		)

	trial_rows = []
	for trial in study.trials:
		trial_rows.append(
			{
				"trial_number": int(trial.number),
				"n_components": trial.params.get("n_components"),
				"objective_value": float(trial.value) if trial.value is not None else np.nan,
				"state": str(trial.state),
			}
		)

	return int(study.best_params["n_components"]), float(study.best_value), trial_rows


def _run_nested_cv(
	*,
	X: pd.DataFrame,
	y: pd.Series,
	outer_cv: int,
	inner_cv: int,
	n_trials: int,
	random_state: int,
	scoring: str,
	n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)

	fold_rows: List[Dict[str, Any]] = []
	pred_rows: List[Dict[str, Any]] = []
	trial_rows_all: List[Dict[str, Any]] = []

	for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, y), start=1):
		X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
		y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

		inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state + fold)
		best_n, best_inner_score, trial_rows = _tune_n_components_optuna(
			X=X_train,
			y=y_train,
			cv=inner_kf,
			n_trials=n_trials,
			random_state=random_state + fold,
			scoring=scoring,
			n_jobs=n_jobs,
		)

		for trial_row in trial_rows:
			trial_row["outer_fold"] = fold
			trial_rows_all.append(trial_row)

		model = _build_pls_pipeline(n_components=best_n)
		model.fit(X_train, y_train)

		y_test_pred = _sanitize_prediction_array(model.predict(X_test))
		metrics = _compute_metrics(y_test, y_test_pred)

		fold_rows.append(
			{
				"outer_fold": fold,
				"best_n_components": best_n,
				"inner_best_score": best_inner_score,
				"outer_test_mse": metrics["mse"],
				"outer_test_rmse": metrics["rmse"],
				"outer_test_r2": metrics["r2"],
				"outer_test_mae": metrics["mae"],
				"outer_train_size": int(len(train_idx)),
				"outer_test_size": int(len(test_idx)),
			}
		)

		for sample_idx, y_true_val, y_pred_val in zip(test_idx, y_test.to_numpy(), y_test_pred):
			pred_rows.append(
				{
					"sample_index": int(sample_idx),
					"outer_fold": fold,
					"y_true": float(y_true_val),
					"y_pred": float(y_pred_val),
					"best_n_components": best_n,
				}
			)

	fold_df = pd.DataFrame(fold_rows)
	pred_df = pd.DataFrame(pred_rows).sort_values("sample_index").reset_index(drop=True)
	trials_df = pd.DataFrame(trial_rows_all)

	summary_df = pd.DataFrame(
		[
			{
				"metric": "outer_test_mse",
				"mean": float(fold_df["outer_test_mse"].mean()),
				"std": float(fold_df["outer_test_mse"].std(ddof=0)),
			},
			{
				"metric": "outer_test_rmse",
				"mean": float(fold_df["outer_test_rmse"].mean()),
				"std": float(fold_df["outer_test_rmse"].std(ddof=0)),
			},
			{
				"metric": "outer_test_r2",
				"mean": float(fold_df["outer_test_r2"].mean()),
				"std": float(fold_df["outer_test_r2"].std(ddof=0)),
			},
			{
				"metric": "outer_test_mae",
				"mean": float(fold_df["outer_test_mae"].mean()),
				"std": float(fold_df["outer_test_mae"].std(ddof=0)),
			},
			{
				"metric": "best_n_components",
				"mean": float(fold_df["best_n_components"].mean()),
				"std": float(fold_df["best_n_components"].std(ddof=0)),
			},
		]
	)

	return fold_df, pred_df, trials_df, summary_df


def _run_final_cv_and_fit(
	*,
	X: pd.DataFrame,
	y: pd.Series,
	cv_folds: int,
	n_trials: int,
	random_state: int,
	scoring: str,
	n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	final_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
	best_n, best_cv_score, final_trial_rows = _tune_n_components_optuna(
		X=X,
		y=y,
		cv=final_cv,
		n_trials=n_trials,
		random_state=random_state,
		scoring=scoring,
		n_jobs=n_jobs,
	)

	best_model = _build_pls_pipeline(best_n)

	cv_scores_r2 = cross_val_score(best_model, X, y, cv=final_cv, scoring="r2", n_jobs=n_jobs)
	cv_scores_neg_mse = cross_val_score(best_model, X, y, cv=final_cv, scoring="neg_mean_squared_error", n_jobs=n_jobs)
	cv_scores_neg_mae = cross_val_score(best_model, X, y, cv=final_cv, scoring="neg_mean_absolute_error", n_jobs=n_jobs)

	cv_pred = cross_val_predict(best_model, X, y, cv=final_cv, n_jobs=n_jobs)
	cv_pred = _sanitize_prediction_array(cv_pred)
	cv_pred_metrics = _compute_metrics(y, cv_pred)

	best_model.fit(X, y)
	in_sample_pred = _sanitize_prediction_array(best_model.predict(X))
	in_sample_metrics = _compute_metrics(y, in_sample_pred)

	final_tuning_summary = pd.DataFrame(
		[
			{
				"best_n_components": best_n,
				"best_cv_objective_score": best_cv_score,
				"cv_folds": cv_folds,
				"n_trials": n_trials,
				"scoring": scoring,
			}
		]
	)

	final_cv_metrics = pd.DataFrame(
		[
			{
				"metric": "cv_r2",
				"mean": float(np.mean(cv_scores_r2)),
				"std": float(np.std(cv_scores_r2, ddof=0)),
			},
			{
				"metric": "cv_neg_mean_squared_error",
				"mean": float(np.mean(cv_scores_neg_mse)),
				"std": float(np.std(cv_scores_neg_mse, ddof=0)),
			},
			{
				"metric": "cv_neg_mean_absolute_error",
				"mean": float(np.mean(cv_scores_neg_mae)),
				"std": float(np.std(cv_scores_neg_mae, ddof=0)),
			},
			{
				"metric": "cv_predictions_mse",
				"mean": cv_pred_metrics["mse"],
				"std": np.nan,
			},
			{
				"metric": "cv_predictions_rmse",
				"mean": cv_pred_metrics["rmse"],
				"std": np.nan,
			},
			{
				"metric": "cv_predictions_r2",
				"mean": cv_pred_metrics["r2"],
				"std": np.nan,
			},
			{
				"metric": "cv_predictions_mae",
				"mean": cv_pred_metrics["mae"],
				"std": np.nan,
			},
		]
	)

	in_sample_metrics_df = pd.DataFrame(
		[
			{
				"mse": in_sample_metrics["mse"],
				"rmse": in_sample_metrics["rmse"],
				"r2": in_sample_metrics["r2"],
				"mae": in_sample_metrics["mae"],
				"best_n_components": best_n,
			}
		]
	)

	in_sample_pred_df = pd.DataFrame(
		{
			"sample_index": np.arange(len(y)),
			"y_true": y.to_numpy(dtype=float),
			"y_pred_in_sample": in_sample_pred,
		}
	)

	cv_pred_df = pd.DataFrame(
		{
			"sample_index": np.arange(len(y)),
			"y_true": y.to_numpy(dtype=float),
			"y_pred_cv": cv_pred,
		}
	)

	final_trials_df = pd.DataFrame(final_trial_rows)

	return final_tuning_summary, final_cv_metrics, in_sample_metrics_df, in_sample_pred_df.merge(cv_pred_df, on=["sample_index", "y_true"]), final_trials_df


def _build_output_folder(*, output_root: str, feature_source: str, top_n: int, target_col: str, n_features: int) -> Path:
	timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	source_text = "reduced-all" if feature_source == "reduced" else f"pyramid-top{top_n}"
	folder_name = f"{timestamp}_pls_{target_col}_{source_text}_{n_features}feat"
	output_dir = Path(output_root) / folder_name
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir



def main(
	*,
	input_file: str | None,
	target_arg: str | None,
	feature_source: str,
	top_n: int,
	outer_cv: int,
	inner_cv: int,
	final_cv: int,
	n_trials: int,
	scoring: str,
	random_state: int,
	n_jobs: int,
	output_root: str,
	double_pyramid_run_folder: str | None,
	double_pyramid_settings_file: str | None,
) -> dict:
	if top_n < 1:
		raise ValueError("top_n must be >= 1")

	run_folder: Path | None = None
	settings: Dict[str, Any] = {}
	if feature_source == "pyramid":
		run_folder, settings = _resolve_run_folder(
			run_folder_arg=double_pyramid_run_folder,
			settings_file_arg=double_pyramid_settings_file,
		)

	data_path = _resolve_data_source(
		feature_source=feature_source,
		input_file=input_file,
		run_folder=run_folder,
		settings=settings,
	)
	if not data_path.exists():
		raise FileNotFoundError(f"Input CSV does not exist: {data_path}")

	print(f"{Fore.CYAN}Loading data from: {data_path}{Style.RESET_ALL}")
	df = pd.read_csv(data_path)

	resolved_target_arg = target_arg or settings.get("target_arg") or "npn"
	if resolved_target_arg not in TARGET_MAPPING:
		raise ValueError(f"Unknown target '{resolved_target_arg}'. Choose one of: {list(TARGET_MAPPING.keys())}")

	target_col = _resolve_target_column(
		target_arg=resolved_target_arg,
		settings=settings,
		df=df,
	)
	feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]
	feature_cols = [col for col in feature_cols if col != target_col]

	selected_features, selected_from_frequency_df, missing_features = _select_features(
		feature_source=feature_source,
		run_folder=run_folder,
		top_n=top_n,
		full_feature_cols=feature_cols,
	)

	X = df[selected_features].copy().reset_index(drop=True)
	y = df[target_col].copy().reset_index(drop=True)

	if len(y) < max(outer_cv, inner_cv, final_cv):
		raise ValueError(
			"Number of samples is smaller than requested CV folds. "
			f"Samples={len(y)}, outer_cv={outer_cv}, inner_cv={inner_cv}, final_cv={final_cv}"
		)

	output_dir = _build_output_folder(
		output_root=output_root,
		feature_source=feature_source,
		top_n=top_n,
		target_col=target_col,
		n_features=len(selected_features),
	)
	print(f"{Fore.GREEN}Saving outputs to: {output_dir}{Style.RESET_ALL}")

	print(f"{Fore.CYAN}Running nested CV with Optuna tuning for n_components...{Style.RESET_ALL}")
	nested_fold_df, nested_pred_df, nested_trials_df, nested_summary_df = _run_nested_cv(
		X=X,
		y=y,
		outer_cv=outer_cv,
		inner_cv=inner_cv,
		n_trials=n_trials,
		random_state=random_state,
		scoring=scoring,
		n_jobs=n_jobs,
	)

	print(f"{Fore.CYAN}Running full-data CV tuning + final in-sample fit...{Style.RESET_ALL}")
	(
		final_tuning_summary_df,
		final_cv_metrics_df,
		final_in_sample_metrics_df,
		final_predictions_df,
		final_trials_df,
	) = _run_final_cv_and_fit(
		X=X,
		y=y,
		cv_folds=final_cv,
		n_trials=n_trials,
		random_state=random_state,
		scoring=scoring,
		n_jobs=n_jobs,
	)

	selected_features_df = pd.DataFrame({"feature": selected_features})
	selected_features_df.to_csv(output_dir / "selected_features.csv", index=False)

	if selected_from_frequency_df is not None:
		selected_from_frequency_df.to_csv(output_dir / "selected_features_from_frequency_top_n.csv", index=False)

	if missing_features:
		pd.DataFrame({"missing_feature": missing_features}).to_csv(output_dir / "missing_top_n_features_in_data.csv", index=False)

	nested_fold_df.to_csv(output_dir / "nested_outer_fold_metrics.csv", index=False)
	nested_pred_df.to_csv(output_dir / "nested_outer_predictions.csv", index=False)
	nested_trials_df.to_csv(output_dir / "nested_inner_optuna_trials.csv", index=False)
	nested_summary_df.to_csv(output_dir / "nested_summary.csv", index=False)

	final_tuning_summary_df.to_csv(output_dir / "final_cv_tuning_summary.csv", index=False)
	final_cv_metrics_df.to_csv(output_dir / "final_cv_metrics.csv", index=False)
	final_in_sample_metrics_df.to_csv(output_dir / "final_in_sample_metrics.csv", index=False)
	final_predictions_df.to_csv(output_dir / "final_predictions_in_sample_and_cv.csv", index=False)
	final_trials_df.to_csv(output_dir / "final_cv_optuna_trials.csv", index=False)

	metadata = {
		"run_timestamp": datetime.now().isoformat(),
		"feature_source": feature_source,
		"input_csv": str(data_path),
		"run_folder": str(run_folder) if run_folder is not None else None,
		"target_column": target_col,
		"target_arg": resolved_target_arg,
		"n_samples": int(X.shape[0]),
		"n_selected_features": int(X.shape[1]),
		"outer_cv": outer_cv,
		"inner_cv": inner_cv,
		"final_cv": final_cv,
		"n_trials": n_trials,
		"scoring": scoring,
		"seed": random_state,
		"n_jobs": n_jobs,
		"output_dir": str(output_dir),
	}
	with (output_dir / "run_metadata.json").open("w") as file_handle:
		json.dump(metadata, file_handle, indent=2)

	nested_r2_mean = float(nested_fold_df["outer_test_r2"].mean())
	final_in_sample_r2 = float(final_in_sample_metrics_df.iloc[0]["r2"])
	print(f"{Fore.GREEN}Nested CV mean R2: {nested_r2_mean:.4f}{Style.RESET_ALL}")
	print(f"{Fore.GREEN}Final in-sample R2 (fit on all data): {final_in_sample_r2:.4f}{Style.RESET_ALL}")
	print(f"{Fore.GREEN}Done. Results saved to: {output_dir}{Style.RESET_ALL}")

	return {
		"run_folder": str(output_dir),
		"resolved_input_file": str(data_path),
		"resolved_target_arg": resolved_target_arg,
		"target_column": target_col,
		"selected_feature_count": int(X.shape[1]),
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run PLS with nested CV and Optuna-tuned n_components")
	parser.add_argument("--input", "-i", type=str, default=None, help="Path to input CSV file with features and target")
	parser.add_argument("--target", type=str, choices=list(TARGET_MAPPING.keys()), default=None)
	parser.add_argument("--feature_source", type=str, choices=["reduced", "pyramid"], default="reduced")
	parser.add_argument("--top_n", type=int, default=20, help="Top N features from feature_frequency.csv when using pyramid")

	parser.add_argument("--outer_cv", type=int, default=5)
	parser.add_argument("--inner_cv", type=int, default=5)
	parser.add_argument("--final_cv", type=int, default=5)
	parser.add_argument("--n_trials", type=int, default=40)
	parser.add_argument("--scoring", type=str, default="neg_mean_squared_error", choices=["neg_mean_squared_error", "r2", "neg_mean_absolute_error"])
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--n_jobs", type=int, default=-1)

	parser.add_argument("--run_folder", type=str, default=None, help="Path to completed double-pyramid run folder")
	parser.add_argument("--settings_file", type=str, default=None, help="Path to double-pyramid settings file")

	parser.add_argument("--output_root", type=str, default="outputs/pls")
	args = parser.parse_args()

	print(f"{Fore.LIGHTGREEN_EX}\nStarting PLS run at {datetime.now()}{Style.RESET_ALL}")

	run_result = main(
		input_file=args.input,
		target_arg=args.target,
		feature_source=args.feature_source,
		top_n=args.top_n,
		outer_cv=args.outer_cv,
		inner_cv=args.inner_cv,
		final_cv=args.final_cv,
		n_trials=args.n_trials,
		scoring=args.scoring,
		random_state=args.seed,
		n_jobs=args.n_jobs,
		output_root=args.output_root,
		double_pyramid_run_folder=args.run_folder,
		double_pyramid_settings_file=args.settings_file,
	)

	print(f"{Fore.LIGHTGREEN_EX}PLS run completed at {datetime.now()}{Style.RESET_ALL}")
