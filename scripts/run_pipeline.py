"""End-to-end pipeline: load -> split -> train -> calibrate -> evaluate -> explain.

Usage:
    python scripts/run_pipeline.py --csv data/raw/s41598-020-73558-3_sepsis_survival_primary_cohort.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shap  # noqa: E402

from sepsis_xai.data import find_csv, load_cohort, split_cohort  # noqa: E402
from sepsis_xai.evaluate import evaluate, f1_optimal_threshold, print_report  # noqa: E402
from sepsis_xai.model import calibrate, train_xgboost  # noqa: E402
from sepsis_xai.plots import plot_calibration, plot_shap_summary, plot_shap_waterfall  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="Path to the cohort CSV (default: search data/raw/)")
    ap.add_argument("--out", default=str(ROOT / "results"), help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv = find_csv(args.csv)
    print(f"Loading {csv}")
    X, y = load_cohort(csv)
    print(f"Cohort: {len(X)} episodes | features: {list(X.columns)} | mortality: {y.mean():.2%}")

    s = split_cohort(X, y, seed=args.seed)
    print(f"Split: train {len(s.X_train)} | val {len(s.X_val)} | test {len(s.X_test)}")

    model = train_xgboost(s.X_train, s.y_train, seed=args.seed)
    cal = calibrate(model, s.X_val, s.y_val)

    # Threshold is tuned on the validation split to keep the test set untouched.
    threshold = f1_optimal_threshold(s.y_val, cal.predict_proba(s.X_val)[:, 1])

    uncal_p = model.predict_proba(s.X_test)[:, 1]
    cal_p = cal.predict_proba(s.X_test)[:, 1]
    metrics = evaluate(s.y_test, uncal_p, cal_p, threshold, n_boot=args.n_bootstrap, seed=args.seed)
    print_report(metrics)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    explainer = shap.TreeExplainer(model)
    plot_calibration(s.y_test, uncal_p, cal_p, fig_dir / "calibration_curve.png")
    plot_shap_summary(explainer, s.X_test, fig_dir / "shap_summary.png")
    plot_shap_waterfall(explainer, s.X_test, (cal_p >= threshold).astype(int), fig_dir / "shap_patient_waterfall.png")
    print(f"Saved metrics and figures to {out}")


if __name__ == "__main__":
    main()
