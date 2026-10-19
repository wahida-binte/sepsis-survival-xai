"""Smoke test on a synthetic cohort with the same schema as the real dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sepsis_xai.data import load_cohort, split_cohort  # noqa: E402
from sepsis_xai.evaluate import evaluate, f1_optimal_threshold  # noqa: E402
from sepsis_xai.model import calibrate, train_xgboost  # noqa: E402


def _synthetic_csv(tmp_path: Path, n: int = 4000) -> Path:
    rng = np.random.default_rng(0)
    age = rng.integers(0, 100, n)
    sex = rng.integers(0, 2, n)
    ep = rng.integers(1, 6, n)
    logit = -5 + 0.045 * age - 0.2 * sex + 0.15 * ep
    dead = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        "age_years": age,
        "sex_0male_1female": sex,
        "episode_number": ep,
        "hospital_outcome_1alive_0dead": (~dead).astype(int),
    })
    p = tmp_path / "sepsis_primary_cohort.csv"
    df.to_csv(p, index=False)
    return p


def test_target_is_mortality(tmp_path):
    X, y = load_cohort(_synthetic_csv(tmp_path))
    assert "hospital_outcome_1alive_0dead" not in X.columns
    assert y.mean() < 0.5  # deaths are the minority class


def test_end_to_end(tmp_path):
    X, y = load_cohort(_synthetic_csv(tmp_path))
    s = split_cohort(X, y)
    model = train_xgboost(s.X_train, s.y_train, n_estimators=30)
    cal = calibrate(model, s.X_val, s.y_val)
    thr = f1_optimal_threshold(s.y_val, cal.predict_proba(s.X_val)[:, 1])
    m = evaluate(s.y_test, model.predict_proba(s.X_test)[:, 1],
                 cal.predict_proba(s.X_test)[:, 1], thr, n_boot=50)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert m["brier_calibrated"] < m["brier_uncalibrated"]
