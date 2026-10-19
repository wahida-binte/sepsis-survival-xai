"""Data loading, target construction and stratified splitting."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_KEYWORDS = ("outcome", "survival", "dead", "mortality")
ID_COLUMNS = ("Unnamed: 0", "patient_id", "id", "case_no")
DEFAULT_GLOBS = ("data/raw/*primary_cohort*.csv", "data/raw/*sepsis*.csv")


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def find_csv(path: str | None = None) -> Path:
    """Return an explicit CSV path, or search the default data locations."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p
    for pattern in DEFAULT_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(
        "No sepsis CSV found in data/raw/. See data/README.md for download steps."
    )


def load_cohort(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cohort and build a binary target: 1 = in-hospital death."""
    df = pd.read_csv(csv_path)
    candidates = [c for c in df.columns if any(k in c.lower() for k in TARGET_KEYWORDS)]
    if not candidates:
        raise KeyError("Outcome column not found.")
    target_col = candidates[0]

    # Source encodes 1 = alive, 0 = dead; flip so the positive class is mortality.
    if "alive" in target_col.lower() and df[target_col].max() == 1:
        y = (df[target_col] == 0).astype(int)
    else:
        y = df[target_col].astype(int)
    y.name = "mortality"

    drop = [target_col, *[c for c in ID_COLUMNS if c in df.columns]]
    X = df.drop(columns=drop).copy()
    return X, y


def split_cohort(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.30, seed: int = 42
) -> Splits:
    """70/15/15 stratified split: train / calibration+threshold / held-out test."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)
