"""Cost-based decision threshold (invariant 4: thresholds set by cost, not 0.5).

The model outputs a calibrated probability; the business decides where to block.
We pick the threshold that minimizes expected cost under an explicit cost model:

    cost(tau) =  (amount of every fraud we ALLOW, i.e. score < tau)        [missed fraud]
               + COST_PER_FALSE_BLOCK * (count of legit txns we BLOCK)      [false blocks]

A blocked fraud costs 0 (loss prevented); an allowed legit txn costs 0. The
threshold is selected on the **calibration** slice (never on validation) and then
applied to validation for honest reporting — selecting on the same data we report
would understate cost. We also report the two trivial policies (block none /
block all) as sanity rails and plot cost-vs-threshold with the chosen point.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    cost_per_false_block: float = config.COST_PER_FALSE_BLOCK,
) -> float:
    """Total expected cost of blocking every txn scoring >= ``threshold``."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    blocked = scores >= threshold
    missed_fraud_cost = amounts[(y_true == 1) & (~blocked)].sum()
    false_block_cost = cost_per_false_block * int(((y_true == 0) & blocked).sum())
    return float(missed_fraud_cost + false_block_cost)


def cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    *,
    cost_per_false_block: float = config.COST_PER_FALSE_BLOCK,
    n_grid: int = 501,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds over [0, 1] and return (thresholds, total_cost)."""
    thresholds = np.linspace(0.0, 1.0, n_grid)
    costs = np.array(
        [
            expected_cost(y_true, scores, amounts, t, cost_per_false_block=cost_per_false_block)
            for t in thresholds
        ]
    )
    return thresholds, costs


def select_cost_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    *,
    cost_per_false_block: float = config.COST_PER_FALSE_BLOCK,
    n_grid: int = 501,
) -> dict[str, float]:
    """Pick the cost-minimizing threshold on the given (calibration) data.

    Returns the chosen threshold plus the cost there and for the block-none /
    block-all rails, so the saving from the model is explicit.
    """
    thresholds, costs = cost_curve(
        y_true, scores, amounts, cost_per_false_block=cost_per_false_block, n_grid=n_grid
    )
    best = int(np.argmin(costs))
    # Rails: threshold > 1 blocks nothing; threshold 0 blocks everything.
    cost_block_none = expected_cost(
        y_true, scores, amounts, 1.01, cost_per_false_block=cost_per_false_block
    )
    cost_block_all = expected_cost(
        y_true, scores, amounts, 0.0, cost_per_false_block=cost_per_false_block
    )
    result = {
        "threshold": float(thresholds[best]),
        "cost_at_threshold": float(costs[best]),
        "cost_block_none": float(cost_block_none),
        "cost_block_all": float(cost_block_all),
        "cost_per_false_block": float(cost_per_false_block),
    }
    logger.info(
        "Cost-optimal threshold=%.4f -> cost=%.0f (block-none=%.0f, block-all=%.0f)",
        result["threshold"],
        result["cost_at_threshold"],
        cost_block_none,
        cost_block_all,
    )
    return result


def operating_point(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    cost_per_false_block: float = config.COST_PER_FALSE_BLOCK,
) -> dict[str, float]:
    """Confusion-style summary of applying ``threshold`` (e.g. to validation)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    blocked = scores >= threshold

    tp = int(((y_true == 1) & blocked).sum())
    fp = int(((y_true == 0) & blocked).sum())
    fn = int(((y_true == 1) & ~blocked).sum())
    total_fraud = int((y_true == 1).sum())
    fraud_amt_total = float(amounts[y_true == 1].sum())
    fraud_amt_caught = float(amounts[(y_true == 1) & blocked].sum())

    return {
        "threshold": float(threshold),
        "n_blocked": int(blocked.sum()),
        "block_rate": float(blocked.mean()),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / total_fraud if total_fraud else 0.0,
        "fraud_dollars_recall": fraud_amt_caught / fraud_amt_total if fraud_amt_total else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cost": expected_cost(
            y_true, scores, amounts, threshold, cost_per_false_block=cost_per_false_block
        ),
    }


def plot_cost_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    chosen_threshold: float,
    out_path: Path,
    *,
    cost_per_false_block: float = config.COST_PER_FALSE_BLOCK,
    title: str = "Expected cost vs. block threshold (validation)",
) -> Path:
    """Plot cost vs threshold and mark the chosen operating point."""
    thresholds, costs = cost_curve(
        y_true, scores, amounts, cost_per_false_block=cost_per_false_block
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, costs, lw=2, label="expected cost")
    chosen_cost = expected_cost(
        y_true, scores, amounts, chosen_threshold, cost_per_false_block=cost_per_false_block
    )
    ax.axvline(
        chosen_threshold, ls="--", color="crimson", label=f"chosen tau = {chosen_threshold:.3f}"
    )
    ax.scatter([chosen_threshold], [chosen_cost], color="crimson", zorder=5)
    ax.set_xlabel("Block threshold (calibrated probability)")
    ax.set_ylabel("Total expected cost ($)")
    ax.set_title(title)
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved cost curve to %s", out_path)
    return out_path
