"""Probability calibration (invariant 4: outputs are calibrated probabilities).

A class-weighted / ``scale_pos_weight`` XGBoost model ranks well but its raw
outputs are not probabilities — they're inflated toward the minority class. The
cost-based threshold needs *true* probabilities (a 0.7 must mean ~70% of such
transactions are fraud), so we fit an isotonic calibrator on a held-out
calibration slice that is later in time than training and earlier than
validation. Isotonic is non-parametric and handles the sigmoid-unfriendly score
distribution of a reweighted booster better than Platt scaling, and the
calibration slice is large enough to support it.

We wrap the already-fitted booster in ``FrozenEstimator`` so
``CalibratedClassifierCV`` fits only the calibrator and never refits (or
re-splits) the booster — the modern replacement for the deprecated
``cv="prefit"``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def calibrate_classifier(fitted_clf, X_calib, y_calib, *, method: str = "isotonic"):
    """Fit a calibrator on top of an already-fitted classifier.

    ``fitted_clf`` is frozen, so only the calibration map is learned (from the
    calibration slice). Returns a ``CalibratedClassifierCV`` whose
    ``predict_proba`` yields calibrated probabilities.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_clf), method=method)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def reliability_table(
    y_true: np.ndarray, scores: np.ndarray, *, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions and return (mean_predicted, observed_fraction, weight).

    Bins are equal-width in probability. ``weight`` is the share of rows in each
    bin (so sparse high-probability bins can be shown but not over-read).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_frac = np.full(n_bins, np.nan)
    weight = np.zeros(n_bins)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred[b] = scores[mask].mean()
            obs_frac[b] = y_true[mask].mean()
            weight[b] = mask.mean()
    return mean_pred, obs_frac, weight


def plot_calibration_curve(
    y_true: np.ndarray,
    scores_uncal: np.ndarray,
    scores_cal: np.ndarray,
    out_path: Path,
    *,
    n_bins: int = 10,
) -> Path:
    """Plot reliability diagrams for the uncalibrated vs calibrated scores."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    for scores, label in ((scores_uncal, "uncalibrated"), (scores_cal, "calibrated (isotonic)")):
        mean_pred, obs_frac, _ = reliability_table(y_true, scores, n_bins=n_bins)
        ok = ~np.isnan(mean_pred)
        ax.plot(mean_pred[ok], obs_frac[ok], marker="o", lw=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration (time-based validation)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved calibration curve to %s", out_path)
    return out_path


def brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities); lower is better."""
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)))
