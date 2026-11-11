"""Deterministic machine-learning estimator construction and importance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.forecasting.ml.config import DEFAULT_MODEL_PARAMETERS


def make_model(
    name: str,
    *,
    random_seed: int,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Create one configured estimator; scaling is fit only on training rows."""
    params = dict(DEFAULT_MODEL_PARAMETERS.get(name, {}))
    if name not in DEFAULT_MODEL_PARAMETERS:
        raise ValueError(f"Unsupported ML model {name!r}.")
    if parameters:
        params.update(parameters)
    if name == "random_forest":
        return RandomForestRegressor(
            **params,
            random_state=random_seed,
            n_jobs=1,
        )
    if name == "xgboost":
        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_seed,
            n_jobs=1,
            verbosity=0,
        )
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(**params)),
        ]
    )


def feature_importance(model: Any, model_name: str, feature_names: tuple[str, ...]) -> np.ndarray:
    """Return comparable non-negative importance values in canonical feature order."""
    if model_name in {"random_forest", "xgboost"}:
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    else:
        coefficients = np.asarray(
            model.named_steps["ridge"].coef_, dtype=np.float64
        )
        values = np.abs(coefficients)
    if values.shape != (len(feature_names),):
        raise ValueError(
            f"{model_name} returned {values.shape} importance values for "
            f"{len(feature_names)} features."
        )
    return values

