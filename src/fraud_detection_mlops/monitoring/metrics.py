"""Prometheus metrics for the scoring service and the drift monitor (M5).

Four things the Grafana dashboards visualize:
  - **latency**: per-request scoring time (histogram -> p50/p99);
  - **throughput**: request rate (counter -> rate());
  - **alert rate**: share of decisions that are blocks (decision counter);
  - **score distribution**: histogram of fraud probabilities.

Plus drift gauges the monitor publishes (feature drift share, prediction drift).
The scoring service exposes these at ``/metrics``; the drift monitor serves its
gauges on its own port. Both share this registry of definitions.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- Scoring service -----------------------------------------------------------
SCORING_REQUESTS = Counter(
    "fraud_scoring_requests_total", "Total scoring requests served."
)
SCORING_DECISIONS = Counter(
    "fraud_scoring_decisions_total", "Scoring decisions by outcome.", ["decision"]
)
SCORING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "Server-side scoring latency (seconds).",
    buckets=(0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5),
)
SCORE_DISTRIBUTION = Histogram(
    "fraud_score",
    "Distribution of predicted fraud probabilities.",
    buckets=(0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
)

# --- Drift monitor -------------------------------------------------------------
FEATURE_DRIFT_SHARE = Gauge(
    "fraud_feature_drift_share", "Share of monitored features drifting vs training."
)
PREDICTION_DRIFT = Gauge(
    "fraud_prediction_drift", "1 if the prediction distribution has drifted, else 0."
)
DRIFT_RETRAINS = Counter(
    "fraud_drift_retrains_total", "Retrains triggered by drift detection."
)


def record_scoring(latency_seconds: float, probability: float, decision: str) -> None:
    """Update the scoring-service metrics for one request."""
    SCORING_REQUESTS.inc()
    SCORING_DECISIONS.labels(decision=decision).inc()
    SCORING_LATENCY.observe(latency_seconds)
    SCORE_DISTRIBUTION.observe(probability)


def record_drift(summary: dict) -> None:
    """Publish drift gauges from a drift summary."""
    FEATURE_DRIFT_SHARE.set(float(summary["feature_drift_share"]))
    PREDICTION_DRIFT.set(1.0 if summary["prediction_drift"] else 0.0)
