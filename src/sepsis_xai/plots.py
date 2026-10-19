"""Publication figures: calibration curve, SHAP summary, SHAP waterfall."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_calibration(y_test, uncal_probs, cal_probs, out: Path, n_bins: int = 10) -> Path:
    tu, pu = calibration_curve(y_test, uncal_probs, n_bins=n_bins)
    tc, pc = calibration_curve(y_test, cal_probs, n_bins=n_bins)
    plt.figure(figsize=(7, 6))
    plt.plot(pu, tu, "s--", color="#e53935",
             label=f"Uncalibrated XGBoost (Brier: {brier_score_loss(y_test, uncal_probs):.3f})")
    plt.plot(pc, tc, "o-", color="#1e88e5", linewidth=2,
             label=f"Sigmoid calibrated (Brier: {brier_score_loss(y_test, cal_probs):.3f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    plt.title("Probability Calibration: In-Hospital Sepsis Mortality Risk", fontsize=12, pad=12)
    plt.xlabel("Predicted mortality probability")
    plt.ylabel("Observed event frequency")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def plot_shap_summary(explainer, X_test, out: Path) -> Path:
    sv = explainer.shap_values(X_test)
    plt.figure(figsize=(9, 5))
    shap.summary_plot(sv, X_test, feature_names=list(X_test.columns), show=False)
    plt.title("Clinical Predictor Attribution: Sepsis Mortality Risk (log-odds)", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


def plot_shap_waterfall(explainer, X_test, preds, out: Path) -> Path | None:
    flagged = np.where(preds == 1)[0]
    if len(flagged) == 0:
        return None
    i = int(flagged[0])
    explanation = explainer(X_test)
    plt.figure(figsize=(8, 4))
    shap.plots.waterfall(explanation[i], show=False)
    plt.title(f"Patient-Level Audit Trail (test sample #{i}, log-odds scale)", fontsize=11, pad=15)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out
