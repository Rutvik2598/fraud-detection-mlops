"""Honest evaluation for imbalanced fraud detection.

Accuracy is banned as a headline (invariant 3): predicting "never fraud" scores
~96.5% accuracy here and catches zero fraud. The headline is PR-AUC (average
precision). We also report precision@k under fixed alert budgets — the metric an
analyst team actually lives with — and ROC-AUC only as a secondary reference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / notebook-safe
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

logger = logging.getLogger(__name__)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    """Precision and recall within the top-``k`` highest-scoring transactions.

    This models a fixed alert budget: if analysts can only review k transactions,
    what fraction of those are truly fraud (precision) and what share of all
    fraud do we catch (recall)?
    """
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1]
    top_idx = order[:k]
    n_fraud_in_top = int(y_true[top_idx].sum())
    total_fraud = int(y_true.sum())
    return {
        "k": k,
        "precision": n_fraud_in_top / k if k else 0.0,
        "recall": n_fraud_in_top / total_fraud if total_fraud else 0.0,
        "n_fraud_caught": n_fraud_in_top,
    }


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k_values: tuple[int, ...] = (100, 500, 1000, 2000, 5000),
) -> dict[str, float]:
    """Compute the headline + supporting metrics for a set of probability scores."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, scores))
    metrics: dict[str, float] = {
        "pr_auc": pr_auc,  # HEADLINE
        "roc_auc": float(roc_auc_score(y_true, scores)),  # secondary reference
        "base_rate": base_rate,
        "lift_over_base": pr_auc / base_rate if base_rate else float("nan"),
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
        # Calibration smell test: a well-calibrated model should have almost no
        # mass pinned at p~=1.0. A class-weighted, uncalibrated LogReg saturates
        # many rows at 1.0, which makes precision@k degenerate (tie-breaking) for
        # k inside that mass. This number motivates calibration in M1 (invariant 4).
        "n_scores_saturated": int((scores >= 0.999).sum()),
        "frac_scores_saturated": float((scores >= 0.999).mean()),
    }
    for k in k_values:
        res = precision_at_k(y_true, scores, k)
        metrics[f"precision_at_{k}"] = res["precision"]
        metrics[f"recall_at_{k}"] = res["recall"]

    logger.info(
        "PR-AUC=%.4f (base=%.4f, lift=%.1fx) | ROC-AUC=%.4f",
        metrics["pr_auc"],
        base_rate,
        metrics["lift_over_base"],
        metrics["roc_auc"],
    )
    return metrics


def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    out_path: Path,
    *,
    title: str = "Baseline precision-recall (time-based validation)",
) -> Path:
    """Render and save the precision-recall curve; return the saved path."""
    y_true = np.asarray(y_true).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    base_rate = float(y_true.mean())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(
        base_rate,
        ls="--",
        color="grey",
        label=f"random baseline (base rate = {base_rate:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved PR curve to %s", out_path)
    return out_path
