"""Threshold selection, bootstrap CIs and the metrics report."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)

CLASS_NAMES = ["Survived", "Mortality"]


def f1_optimal_threshold(y_true, probs) -> float:
    """Threshold maximising F1 on the PR curve.

    Call this on the validation split, never on the test split, or the
    test-set metrics at that threshold are optimistically biased.
    """
    p, r, t = precision_recall_curve(y_true, probs)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-10)
    return float(t[np.nanargmax(f1)])


def bootstrap_ci(y_true, probs, metric=roc_auc_score, n: int = 500, seed: int = 42):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric(y_true[idx], probs[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(y_test, uncal_probs, cal_probs, threshold: float, n_boot: int = 500, seed: int = 42) -> dict:
    auc_lo, auc_hi = bootstrap_ci(y_test, cal_probs, roc_auc_score, n_boot, seed)
    ap_lo, ap_hi = bootstrap_ci(y_test, cal_probs, average_precision_score, n_boot, seed)
    preds_default = (cal_probs >= 0.5).astype(int)
    preds_opt = (cal_probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, cal_probs)),
        "roc_auc_95ci": [auc_lo, auc_hi],
        "pr_auc": float(average_precision_score(y_test, cal_probs)),
        "pr_auc_95ci": [ap_lo, ap_hi],
        "prevalence": float(np.mean(y_test)),
        "brier_uncalibrated": float(brier_score_loss(y_test, uncal_probs)),
        "brier_calibrated": float(brier_score_loss(y_test, cal_probs)),
        "threshold_f1_val": threshold,
        "report_default_0.5": classification_report(
            y_test, preds_default, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
        "report_tuned": classification_report(
            y_test, preds_opt, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }


def print_report(m: dict) -> None:
    print("=" * 60)
    print("CLINICAL EVALUATION REPORT (held-out test set)")
    print("=" * 60)
    print(f"Prevalence:            {m['prevalence']:.2%}")
    print(f"ROC-AUC:               {m['roc_auc']:.4f}  95% CI [{m['roc_auc_95ci'][0]:.4f}, {m['roc_auc_95ci'][1]:.4f}]")
    print(f"PR-AUC:                {m['pr_auc']:.4f}  95% CI [{m['pr_auc_95ci'][0]:.4f}, {m['pr_auc_95ci'][1]:.4f}]")
    print(f"Brier (uncalibrated):  {m['brier_uncalibrated']:.4f}")
    print(f"Brier (calibrated):    {m['brier_calibrated']:.4f}")
    print(f"F1 threshold (val):    {m['threshold_f1_val']:.4f}")
    mort = m["report_tuned"]["Mortality"]
    print(f"Mortality @ tuned thr: precision {mort['precision']:.2f} | recall {mort['recall']:.2f} | F1 {mort['f1-score']:.2f}")
