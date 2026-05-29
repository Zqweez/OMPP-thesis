from __future__ import annotations
from .strategies import RidgeStrategy, LassoStrategy, ElasticNetStrategy, RandomForestStrategy, GradientBoostingStrategy

STRATEGIES = {
    "ridge": RidgeStrategy(),
    "lasso": LassoStrategy(),
    "elasticnet": ElasticNetStrategy(),
    "random_forest": RandomForestStrategy(),
    "gradient_boosted": GradientBoostingStrategy(),
}
ALL_MODEL_NAMES = set(STRATEGIES.keys())

def get_strategy(model_name: str):
    key = model_name.lower().strip()
    if key not in STRATEGIES:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {list(STRATEGIES.keys())}")
    return STRATEGIES[key]