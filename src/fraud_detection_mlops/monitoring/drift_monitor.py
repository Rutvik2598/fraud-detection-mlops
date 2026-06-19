"""Periodic drift monitor that publishes gauges for Prometheus (M5).

Exposes the drift metrics on ``MONITORING_PORT`` and, on each tick, scores a
window of recent transactions, runs the drift-check flow (which updates the
gauges and triggers retraining when drift is detected), and sleeps. Run it
alongside the scoring service so Prometheus scrapes both; watch the gauges on
Grafana. Set ``DRIFT_INJECT=1`` (or run ``producer --drift``) to make the world
move and see detection + the drift-triggered retrain fire.

Run:  python -m fraud_detection_mlops.monitoring.drift_monitor --interval 15
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import mlflow
from prometheus_client import start_http_server

from fraud_detection_mlops import config
from fraud_detection_mlops.monitoring import drift
from fraud_detection_mlops.pipelines import labels
from fraud_detection_mlops.pipelines.drift_flow import drift_check_flow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("drift_monitor")


def run(*, interval: float, ticks: int | None, trigger_retrain: bool) -> None:
    cache = labels.load_features()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(
        f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    )
    cols = list(model.named_steps["preprocess"].feature_names_in_)

    v_start = labels.val_start_dt(cache)
    train = cache[cache[config.TIME_COL] < v_start]
    recent = cache[cache[config.TIME_COL] >= v_start]
    ref_sample = train.sample(config.DRIFT_WINDOW_SIZE, random_state=1)
    reference = drift.build_monitoring_frame(ref_sample, model, cols)

    start_http_server(config.MONITORING_PORT)
    logger.info(
        "Drift monitor exposing metrics on :%d (interval=%.0fs)", config.MONITORING_PORT, interval
    )

    tick = 0
    while ticks is None or tick < ticks:
        window = recent.sample(min(config.DRIFT_WINDOW_SIZE, len(recent)), random_state=tick + 2)
        if os.environ.get("DRIFT_INJECT") == "1":
            window = drift.inject_drift_features(window)
        current = drift.build_monitoring_frame(window, model, cols)
        result = drift_check_flow(
            reference, current, clock=int(v_start), trigger_retrain=trigger_retrain
        )
        logger.info(
            "tick %d: feature_drift_share=%.2f prediction_drift=%s retrained=%s",
            tick,
            result["feature_drift_share"],
            result["prediction_drift"],
            result["retrained"],
        )
        tick += 1
        if ticks is None or tick < ticks:
            time.sleep(interval)


def main() -> None:
    p = argparse.ArgumentParser(description="Periodic drift monitor (Prometheus gauges).")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--ticks", type=int, default=None, help="Stop after N ticks (default: forever).")
    p.add_argument("--no-retrain", action="store_true", help="Report drift but don't retrain.")
    args = p.parse_args()
    run(interval=args.interval, ticks=args.ticks, trigger_retrain=not args.no_retrain)


if __name__ == "__main__":
    main()
