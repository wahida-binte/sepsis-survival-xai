"""Imbalance-aware XGBoost and post-hoc sigmoid (Platt) calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    seed: int = 42,
) -> XGBClassifier:
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=neg / max(1, pos),
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def calibrate(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid"):
    """Fit a calibrator on the held-out calibration split (works on sklearn <1.6 and >=1.6)."""
    try:
        from sklearn.frozen import FrozenEstimator

        clf = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
    except ImportError:
        clf = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    clf.fit(X_val, y_val)
    return clf
