from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from .registry import get_strategy
from .strategies import get_tuning_config

import warnings
from sklearn.exceptions import ConvergenceWarning

# Supress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def _get_model_coefficients_or_none(strategy, fitted_model) -> np.ndarray | None:
	"""Return model coefficients when available, otherwise None."""
	try:
		coefficients = strategy.get_coefficients(fitted_model)
		return np.asarray(coefficients, dtype=float)
	except (NotImplementedError, AttributeError):
		return None


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
	"""Compute standard regression metrics for one evaluation split."""
	mse = float(mean_squared_error(y_true, y_pred))
	rmse = float(np.sqrt(mse))
	return {
		"mse": mse,
		"rmse": rmse,
		"r2": float(r2_score(y_true, y_pred)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
	}


def _score_fixed_params(
	strategy,
	params: Dict[str, Any],
	X: pd.DataFrame,
	y: pd.Series,
	cv,
	scoring_scheme: str,
	scale_target: bool,
	random_state: int,
	n_jobs: int,
) -> float:
	"""Evaluate fixed params using inner CV."""
	model = strategy.build_estimator(
		params=params,
		scale_target=scale_target,
		random_state=random_state,
	)
	scores = cross_val_score(
		model,
		X,
		y,
		cv=cv,
		scoring=scoring_scheme,
		n_jobs=n_jobs,
	)
	return float(np.mean(scores))


def _tune_params_optuna(
	strategy,
	tune: Any,
	X: pd.DataFrame,
	y: pd.Series,
	cv,
	scoring_scheme: str,
	scale_target: bool,
	random_state: int,
	n_trials: int,
	n_jobs: int,
	fold_idx: int | None = None,
	show_progress_bar: bool = False,
) -> Tuple[Dict[str, Any], float]:
	"""Tune model params with Optuna and return best params and best CV score."""

	def objective(trial) -> float:
		params = strategy.suggest_params(trial, tune)
		model = strategy.build_estimator(
			params=params,
			scale_target=scale_target,
			random_state=random_state,
		)
		if fold_idx is not None:
			trial.set_user_attr("outer_fold", fold_idx)

		scores = cross_val_score(
			model,
			X,
			y,
			cv=cv,
			scoring=scoring_scheme,
			n_jobs=n_jobs,
		)
		return float(np.mean(scores))

	optuna.logging.set_verbosity(optuna.logging.WARNING)
	sampler = optuna.samplers.TPESampler(seed=random_state)
	study = optuna.create_study(direction="maximize", sampler=sampler)
	study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=show_progress_bar)

	return study.best_params, float(study.best_value)

def train_final_model_inner_cv(
	X: pd.DataFrame,
	y: pd.Series,
	model_name: str,
	scoring_scheme: str = "neg_mean_squared_error",
	inner_cv: int = 5,
	random_state: int = 42,
	n_trials: int = 200,
	use_default_params: bool = False,
	n_jobs: int = 1,
	scale_target: bool = True,
	show_progress_bar: bool = False,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
	"""
	Train final model on full dataset using inner CV (or default params).

	Parameters
	----------
	X : pd.DataFrame
		Feature matrix.
	y : pd.Series
		Target vector.
	model_name : str
		Model key from registry, e.g. "ridge", "lasso", "elasticnet", "random_forest".
	scoring_scheme : str, default='neg_mean_squared_error'
		Scikit-learn scoring name.
	inner_cv : int, default=5
		Number of folds used to select hyperparameters.
	random_state : int, default=42
		Random seed for CV and estimators.
	n_trials : int, default=200
		Number of Optuna trials when tuning is enabled.
	use_default_params: bool, default=False
		If True, skip Optuna tuning and use strategy default params.
	n_jobs : int, default=1
		Parallel jobs used by cross-validation scoring.
	scale_target : bool, default=True
		Whether to scale target using TransformedTargetRegressor.
	Returns
	-------
	final_model : Any
		Trained estimator from selected strategy.
	final_details : Dict[str, Any]
		Selected params, CV score, feature importance, and fit diagnostics on full data.
	y_pred : np.ndarray
		Predictions on full X for downstream analysis.
	"""
	strategy = get_strategy(model_name)
	tune = get_tuning_config(model_name)

	inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
	
	if use_default_params:
		best_params = strategy.default_params()
		best_inner_score = _score_fixed_params(
			strategy=strategy,
			params=best_params,
			X=X,
			y=y,
			cv=inner_kf,
			scoring_scheme=scoring_scheme,
			scale_target=scale_target,
			random_state=random_state,
			n_jobs=n_jobs,
		)
	else:
		best_params, best_inner_score = _tune_params_optuna(
			strategy=strategy,
			tune=tune,
			X=X,
			y=y,
			cv=inner_kf,
			scoring_scheme=scoring_scheme,
			scale_target=scale_target,
			random_state=random_state,
			n_trials=n_trials,
			n_jobs=n_jobs,
			show_progress_bar=show_progress_bar,
		)
	
	final_model = strategy.build_estimator(
		params=best_params,
		scale_target=scale_target,
		random_state=random_state,
	)
	final_model.fit(X, y)

	y_pred = final_model.predict(X)
	metrics = _evaluate_predictions(y, y_pred)
	feature_importance = strategy.feature_importance(final_model, X.columns)
	coefficients = _get_model_coefficients_or_none(strategy, final_model)

	best_params_clean = {
		key: float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value
		for key, value in best_params.items()
	}

	final_details = {
		"model_name": strategy.name,
		"coefficients": coefficients,
		"feature_importance": feature_importance.to_numpy(dtype=float),
		"best_params": best_params_clean,
		"inner_cv_score": float(best_inner_score),
		"mse": metrics["mse"],
		"rmse": metrics["rmse"],
		"r2": metrics["r2"],
		"mae": metrics["mae"],
	}

	for key, value in best_params_clean.items():
		final_details[f"best_{key}"] = value

	return final_model, final_details, y_pred

