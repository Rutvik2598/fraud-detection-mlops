"""FastAPI online scoring service (M3).

A transaction arrives, we look up the card's velocity features from Feast by card
key, assemble the full model input (request fields + looked-up features), score
with the calibrated champion model, and apply the cost-based block threshold —
all under the latency budget. Lookup + score only; no heavy computation on the
request path (the rolling state was precomputed and materialized).

  uvicorn fraud_detection_mlops.serving.app:app --port 8001

POST /score   body = a transaction JSON object (raw columns incl. card1, amount, …)
GET  /health  model version, online store, threshold
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraud_detection_mlops import config
from fraud_detection_mlops.features.online import VELOCITY_FEATURES, card_key
from fraud_detection_mlops.monitoring import metrics
from fraud_detection_mlops.serving import store as store_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("serving")

# Process-wide state, populated on startup.
STATE: dict = {}


def _load_threshold(model_version) -> float:
    """Prefer the cost threshold logged with the champion run; fall back to config."""
    try:
        client = mlflow.MlflowClient()
        run = client.get_run(model_version.run_id)
        return float(run.data.metrics["chosen_threshold"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read chosen_threshold from MLflow (%s); using default", exc)
        return config.DECISION_THRESHOLD


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    logger.info("Loading champion model %s", uri)
    STATE["model"] = mlflow.sklearn.load_model(uri)
    STATE["feature_cols"] = list(STATE["model"].named_steps["preprocess"].feature_names_in_)
    mv = mlflow.MlflowClient().get_model_version_by_alias(
        config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS
    )
    STATE["model_version"] = mv.version
    STATE["threshold"] = _load_threshold(mv)
    STATE["store"] = store_mod.get_store()
    STATE["online_store"] = STATE["store"].config.online_store.type
    logger.info(
        "Ready: model v%s | %d features | threshold=%.4f | online_store=%s",
        STATE["model_version"],
        len(STATE["feature_cols"]),
        STATE["threshold"],
        STATE["online_store"],
    )
    yield
    STATE.clear()


app = FastAPI(title="fraud-detection online scoring", lifespan=lifespan)


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint (latency, throughput, decisions, score dist)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if STATE.get("model") is not None else "loading",
        "model_version": STATE.get("model_version"),
        "online_store": STATE.get("online_store"),
        "threshold": STATE.get("threshold"),
        "latency_budget_ms": config.LATENCY_BUDGET_MS,
    }


@app.post("/score")
def score(txn: dict) -> dict:
    """Score one transaction: Feast feature lookup by card key -> model -> decision."""
    t0 = time.perf_counter()

    # 1. Look up + compute the velocity features from the online store by card key.
    velocity = store_mod.online_velocity_features(STATE["store"], txn)
    t1 = time.perf_counter()

    # 2. Assemble the model input: request raw columns + looked-up velocity features.
    row = {col: txn.get(col) for col in STATE["feature_cols"]}
    row.update({f: velocity[f] for f in VELOCITY_FEATURES if f in row})
    X = pd.DataFrame([row], columns=STATE["feature_cols"])

    # 3. Calibrated probability + cost-based decision.
    prob = float(STATE["model"].predict_proba(X)[0, 1])
    threshold = STATE["threshold"]
    decision = "block" if prob >= threshold else "allow"
    t2 = time.perf_counter()

    metrics.record_scoring(t2 - t0, prob, decision)
    return {
        "transaction_id": txn.get(config.ID_COL),
        "card_id": card_key(txn),
        "fraud_probability": prob,
        "decision": decision,
        "threshold": threshold,
        "latency_ms": round((t2 - t0) * 1000, 2),
        "feature_lookup_ms": round((t1 - t0) * 1000, 2),
        "model_ms": round((t2 - t1) * 1000, 2),
    }
