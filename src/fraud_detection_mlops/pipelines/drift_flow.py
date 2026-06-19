"""Drift-check Prefect flow that triggers retraining (M5).

The bridge between monitoring and the feedback loop: run a drift report on a
recent window vs the training reference, publish the drift gauges, and — if drift
is detected — fire the M4 retraining flow. This is the "drift -> retrain" wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from prefect import flow

from fraud_detection_mlops import config
from fraud_detection_mlops.monitoring import drift, metrics
from fraud_detection_mlops.pipelines.retrain_flow import retraining_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("drift_flow")


@flow(name="fraud-drift-check")
def drift_check_flow(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    clock: int,
    registered_model_name: str = config.FEEDBACK_MODEL_NAME,
    trigger_retrain: bool = True,
    html_path: Path | None = None,
) -> dict:
    """Report drift; if detected, trigger retraining. Returns summary + retrained flag."""
    summary = drift.drift_report(reference, current, html_path=html_path)
    metrics.record_drift(summary)

    retrained = False
    if trigger_retrain and drift.detect_drift(summary):
        logger.warning(
            "DRIFT DETECTED (feature share %.2f, prediction_drift=%s) -> triggering retraining",
            summary["feature_drift_share"], summary["prediction_drift"],
        )
        metrics.DRIFT_RETRAINS.inc()
        retraining_flow(clock, registered_model_name=registered_model_name)
        retrained = True
    else:
        logger.info("No drift trigger (healthy).")

    return {**{k: v for k, v in summary.items() if k != "html"}, "retrained": retrained}
