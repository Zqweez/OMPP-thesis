from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class LinearTuningConfig:
    """Shared tuning config for linear models."""
    alpha_min: float = 1e-3
    alpha_max: float = 1e5
    l1_ratio_min: float = 1e-2
    l1_ratio_max: float = 0.99

@dataclass(frozen=True)
class RandomForestTuningConfig:
    n_estimators_min: int = 50
    n_estimators_max: int = 1000

    max_depth_min: int = 2
    max_depth_max: int = 20

    min_samples_split_min: int = 2
    min_samples_split_max: int = 20

    min_samples_leaf_min: int = 1
    min_samples_leaf_max: int = 10

    max_features_options: tuple = ("sqrt", "log2", None)

class GradientBoostingTuningConfig:
    n_estimators_min: int = 50
    n_estimators_max: int = 1000

    learning_rate_min: float = 0.01
    learning_rate_max: float = 0.2

    max_depth_min: int = 2
    max_depth_max: int = 20

    min_samples_split_min: int = 2
    min_samples_split_max: int = 20

    min_samples_leaf_min: int = 1
    min_samples_leaf_max: int = 10

    subsample_min: float = 0.5
    subsample_max: float = 1.0

    max_features_options: tuple = ("sqrt", "log2", None)

@dataclass(frozen=True)
class RidgeLassoTuningConfig(LinearTuningConfig):
    """Default tuning space for Ridge/Lasso."""
    alpha_min: float = 1e-3
    alpha_max: float = 1e4


@dataclass(frozen=True)
class ElasticNetTuningConfig(LinearTuningConfig):
    """Narrower ElasticNet alpha search space."""
    alpha_min: float = 1e-2
    alpha_max: float = 1e4


def get_tuning_config(model_name: str) -> Any:
    key = model_name.lower().strip()
    if key in {"ridge", "lasso"}:
        return RidgeLassoTuningConfig()
    if key == "elasticnet":
        return ElasticNetTuningConfig()
    if key == "random_forest":
        return RandomForestTuningConfig()
    if key == "gradient_boosted":
        return GradientBoostingTuningConfig()
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: ridge, lasso, elasticnet, random_forest, gradient_boosted")


def _make_x_pipeline(model) -> Pipeline:
    """X-side pipeline shared by linear models."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _wrap_target_if_needed(x_pipeline: Pipeline, scale_target: bool):
    """Optionally scale y inside fits."""
    if not scale_target:
        return x_pipeline
    return TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler()
    )


def _get_inner_model(fitted_estimator):
    """Returns the fitted sklearn linear model object, can handle TransformedTargetRegressor."""
    if hasattr(fitted_estimator, "regressor_"):  # TransformedTargetRegressor
        return fitted_estimator.regressor_.named_steps["model"]
    return fitted_estimator.named_steps["model"]

# --- STRATEGIES ---
class RidgeStrategy:
    name: str = "ridge"

    def suggest_params(self, trial: Any, tune: LinearTuningConfig) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", tune.alpha_min, tune.alpha_max, log=True),
        }

    def default_params(self) -> Dict[str, Any]:
        return {"alpha": 1.0}

    def build_estimator(self, params: Dict[str, Any], *, scale_target: bool, random_state: int):
        model = Ridge(alpha=float(params["alpha"]), max_iter=20000, random_state=random_state)
        return _wrap_target_if_needed(_make_x_pipeline(model), scale_target)

    def feature_importance(self, fitted_estimator, feature_names: Sequence[str]) -> pd.Series:
        coefs = self.get_coefficients(fitted_estimator)
        return pd.Series(np.abs(coefs), index=list(feature_names))

    def get_coefficients(self, fitted_estimator) -> np.ndarray:
        return _get_inner_model(fitted_estimator).coef_
    
class LassoStrategy:
    name: str = "lasso"

    def suggest_params(self, trial: Any, tune: LinearTuningConfig) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", tune.alpha_min, tune.alpha_max, log=True),
        }

    def default_params(self) -> Dict[str, Any]:
        return {"alpha": 1.0}

    def build_estimator(self, params: Dict[str, Any], *, scale_target: bool, random_state: int):
        model = Lasso(alpha=float(params["alpha"]), max_iter=20000, random_state=random_state)
        return _wrap_target_if_needed(_make_x_pipeline(model), scale_target)

    def feature_importance(self, fitted_estimator, feature_names: Sequence[str]) -> pd.Series:
        coefs = self.get_coefficients(fitted_estimator)
        return pd.Series(np.abs(coefs), index=list(feature_names))

    def get_coefficients(self, fitted_estimator) -> np.ndarray:
        return _get_inner_model(fitted_estimator).coef_

class ElasticNetStrategy:
    name: str = "elasticnet"

    def suggest_params(self, trial: Any, tune: LinearTuningConfig) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", tune.alpha_min, tune.alpha_max, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", tune.l1_ratio_min, tune.l1_ratio_max),
        }

    def default_params(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "l1_ratio": 0.5}

    def build_estimator(self, params: Dict[str, Any], *, scale_target: bool, random_state: int):
        model = ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=50000,
            random_state=random_state,
        )
        return _wrap_target_if_needed(_make_x_pipeline(model), scale_target)

    def feature_importance(self, fitted_estimator, feature_names: Sequence[str]) -> pd.Series:
        coefs = self.get_coefficients(fitted_estimator)
        return pd.Series(np.abs(coefs), index=list(feature_names))

    def get_coefficients(self, fitted_estimator) -> np.ndarray:
        return _get_inner_model(fitted_estimator).coef_

class RandomForestStrategy:
    name: str = "random_forest"

    def suggest_params(self, trial: Any, tune: RandomForestTuningConfig) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", tune.n_estimators_min, tune.n_estimators_max),
            "max_depth": trial.suggest_int("max_depth", tune.max_depth_min, tune.max_depth_max),
            "min_samples_split": trial.suggest_int("min_samples_split", tune.min_samples_split_min, tune.min_samples_split_max),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", tune.min_samples_leaf_min, tune.min_samples_leaf_max),
            "max_features": trial.suggest_categorical("max_features", tune.max_features_options),
        }

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "max_features": "log2",
        }

    def build_estimator(self, params: Dict[str, Any], *, scale_target: bool, random_state: int):
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=1,
        )
        # RF is scale invariant but we might as well keep the same structure for all models, and scaling y can still be helpful.
        return _wrap_target_if_needed(_make_x_pipeline(model), scale_target)

    def feature_importance(self, fitted_estimator, feature_names: Sequence[str]) -> pd.Series:
        model = _get_inner_model(fitted_estimator)
        importance = model.feature_importances_
        return pd.Series(importance, index=list(feature_names))

    def get_coefficients(self, fitted_estimator) -> np.ndarray:
        raise NotImplementedError("Random Forest does not produce coefficients")
    
class GradientBoostingStrategy:
    name: str = "gradient_boosting"

    def suggest_params(self, trial: Any, tune: GradientBoostingTuningConfig) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", tune.n_estimators_min, tune.n_estimators_max),
            "learning_rate": trial.suggest_float("learning_rate", tune.learning_rate_min, tune.learning_rate_max, log=True),
            "max_depth": trial.suggest_int("max_depth", tune.max_depth_min, tune.max_depth_max),
            "min_samples_split": trial.suggest_int("min_samples_split", tune.min_samples_split_min, tune.min_samples_split_max),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", tune.min_samples_leaf_min, tune.min_samples_leaf_max),
            "subsample": trial.suggest_float("subsample", tune.subsample_min, tune.subsample_max),
            "max_features": trial.suggest_categorical("max_features", tune.max_features_options),
        }

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_split": 5,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "max_features": None,
        }

    def build_estimator(self, params: Dict[str, Any], *, scale_target: bool, random_state: int):
        model = GradientBoostingRegressor(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            subsample=float(params["subsample"]),
            max_features=params["max_features"],
            random_state=random_state,
        )

        return _wrap_target_if_needed(_make_x_pipeline(model), scale_target)

    def feature_importance(self, fitted_estimator, feature_names: Sequence[str]) -> pd.Series:
        model = _get_inner_model(fitted_estimator)
        return pd.Series(model.feature_importances_, index=list(feature_names))

    def get_coefficients(self, fitted_estimator) -> np.ndarray:
        raise NotImplementedError("Gradient Boosting does not produce coefficients")