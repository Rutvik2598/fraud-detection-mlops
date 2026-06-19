"""Evidently drift detection: production traffic vs. the training distribution.

Drift is the early-warning signal that needs **no labels** (which arrive late,
M4): if the feature distribution the model sees in production moves away from
what it was trained on — or its own score distribution shifts — quality is
probably decaying even before the chargebacks come back to confirm it.

We report two things against a fixed training reference:
  - **feature drift**: the share of monitored features whose distribution drifted
    (Evidently auto-selects a per-feature statistical test);
  - **prediction drift**: whether the model's output score distribution drifted.

A rich HTML report is saved for humans; a small summary dict drives the trigger.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)

PREDICTION_COL = "prediction"

# Amount and amount-derived feature columns the synthetic drift perturbs.
_DRIFT_AMOUNT_COLS: tuple[str, ...] = (
    config.AMOUNT_COL,
    "card_amt_mean_prior",
    "card_amt_sum_1h",
    "card_amt_sum_24h",
    "card_amt_sum_7d",
    "amt_vs_card_mean_ratio",
)


def inject_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a synthetic covariate shift to a feature frame (population change).

    Scales the amount and amount-derived features and shifts the raw amount, so
    the population moves out of the training support — exactly what the
    ``producer --drift`` does to the live stream, but on a precomputed frame.
    """
    out = df.copy()
    for col in _DRIFT_AMOUNT_COLS:
        if col in out.columns:
            out[col] = out[col] * config.DRIFT_AMOUNT_MULTIPLIER
    if config.AMOUNT_COL in out.columns:
        out[config.AMOUNT_COL] = out[config.AMOUNT_COL] + config.DRIFT_AMOUNT_SHIFT
    return out


def build_monitoring_frame(features: pd.DataFrame, model, model_cols: list[str]) -> pd.DataFrame:
    """Monitored features + the model's score, for one window of transactions."""
    cols = [
        c
        for c in (*config.DRIFT_NUMERIC_FEATURES, *config.DRIFT_CATEGORICAL_FEATURES)
        if c in features.columns
    ]
    out = features[cols].copy()
    out[PREDICTION_COL] = model.predict_proba(features[model_cols])[:, 1]
    return out


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, *, html_path: Path | None = None
) -> dict:
    """Compare ``current`` to ``reference``; return a drift summary dict.

    Keys: n_features, n_drifted, feature_drift_share, dataset_drift (bool),
    prediction_drift (bool), html (path or None).
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import DriftedColumnsCount
    from evidently.presets import DataDriftPreset

    numeric = [c for c in config.DRIFT_NUMERIC_FEATURES if c in reference.columns]
    categorical = [c for c in config.DRIFT_CATEGORICAL_FEATURES if c in reference.columns]
    feature_cols = numeric + categorical

    data_def = DataDefinition(
        numerical_columns=[*numeric, PREDICTION_COL], categorical_columns=categorical
    )
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    # Explicit, index-stable metrics: [0] feature drift count, [1] prediction
    # drift count. The preset enriches the saved HTML.
    report = Report(
        metrics=[
            DriftedColumnsCount(columns=feature_cols),
            DriftedColumnsCount(columns=[PREDICTION_COL]),
            DataDriftPreset(),
        ]
    )
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)
    metrics = snapshot.dict()["metrics"]

    feat = metrics[0]["value"]  # {'count': ..., 'share': ...}
    pred = metrics[1]["value"]
    n_drifted = int(feat["count"])
    share = float(feat["share"])
    prediction_drift = int(pred["count"]) > 0

    html_out = None
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
        html_out = str(html_path)

    summary = {
        "n_features": len(feature_cols),
        "n_drifted": n_drifted,
        "feature_drift_share": round(share, 4),
        "dataset_drift": share >= config.DRIFT_SHARE_THRESHOLD,
        "prediction_drift": prediction_drift,
        "html": html_out,
    }
    logger.info("Drift: %s", {k: v for k, v in summary.items() if k != "html"})
    return summary


def detect_drift(summary: dict) -> bool:
    """Trigger condition: dataset feature drift OR the prediction distribution drifted."""
    return bool(summary["dataset_drift"] or summary["prediction_drift"])
