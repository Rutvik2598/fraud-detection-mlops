# Real-time fraud detection — end-to-end MLOps

A production-shaped credit-card fraud detection system, not just a model in a notebook. It scores transactions inline with authorization (sub-50 ms), learns from chargeback labels that only arrive weeks later, watches itself for decay, and retrains automatically when the data drifts. Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (~590k transactions, 3.5% fraud).

The interesting engineering here is the **loop around the model**: streaming feature computation that exactly matches training, a feature store for low-latency lookups, a delayed-label feedback path, and drift-triggered retraining.

## The problem

Fraud detection has two properties that make it a systems problem, not a modelling exercise:

1. **You must decide now, but learn the truth later.** A card is swiped; you have milliseconds to allow or block. Whether it was actually fraud is only known weeks later, when the customer disputes the charge (a chargeback).
2. **The world drifts.** Fraud patterns, products, and spending all change, so a model that was accurate last month quietly gets worse — with no error, no crash.

The system is built as two planes connected by a feedback loop:

```
                 ONLINE PLANE  (fast, every transaction)
   transaction ─► look up features (Redis) ─► calibrated score ─► allow / block
        │                                            │  (<50 ms)
        │                                            ▼
        │                                   Prometheus / Grafana
        ▼                                   (latency, drift, alerts)
   ┌─────────────────────────── feedback loop ───────────────────────────┐
   │  chargeback labels arrive late ─► retrain ─► promote if better       │
   │            ▲                                                          │
   │            └──── triggered when drift is detected ◄──────────────────┘
                 OFFLINE PLANE  (slow, occasional)
```

## Results

All metrics use a **time-based split** (train on the past, validate on the most recent ~20% by transaction time) — the realistic setup. Random cross-validation leaks future information here and inflates scores, so it is never used. The Kaggle holdout is never touched for evaluation. Headline metric is **PR-AUC** (average precision), because at a 3.5% fraud rate accuracy is meaningless.

| Metric | Baseline (logistic regression) | Gradient-boosted model |
| --- | --- | --- |
| PR-AUC | 0.184 | **0.475** |
| ROC-AUC | 0.832 | 0.891 |
| Precision @ top 500 alerts | 0.034 | **0.916** |
| Brier score (calibration) | — | 0.083 → **0.023** after calibration |

The model outputs **calibrated probabilities**, and the allow/block threshold is chosen by **expected cost** (a missed fraud costs the transaction amount; a wrongly blocked customer costs a fixed amount), not a default 0.5. At the cost-optimal threshold, expected loss on the validation window is **~34% lower than blocking nothing** and a fraction of blocking everything.

Operationally the model is strong where it matters — **92% of the top-500 flagged transactions are truly fraud** — even though PR-AUC (which also grades the hard high-recall tail) leaves clear headroom. See [Limitations](#limitations--next-steps).

**Serving latency** (warm, server-side): **p50 ≈ 14 ms, p99 ≈ 28 ms**, within the 50 ms budget — a Redis feature lookup (~1.4 ms) plus model inference (~13 ms).

## What makes it production-grade

- **No data leakage.** Every behavioural feature ("card's spend in the last 24h", "time since last transaction", "size of the card's fraud-ring") is computed from *strictly earlier* transactions only. This is enforced mechanically and covered by tests that assert a future transaction can never change a past one's features.
- **Train/serve parity.** The rolling features are defined **once** and reused by offline training and online serving. A test replays 30k real transactions through both paths and asserts the features — and the resulting model scores — are identical. Two code paths that "should match" are a bug waiting to happen.
- **Honest evaluation.** Time-based split, PR-AUC over accuracy, calibrated probabilities, cost-based thresholds, and the unlabelled holdout left untouched.
- **A real feedback loop.** Labels are treated as arriving late; retraining only ever uses labels that have matured, and a new model is promoted **only if it beats the current one** on a fixed held-out window.
- **Self-monitoring.** Feature and prediction drift are tracked against the training distribution (no labels required), surfaced on Grafana, and wired to trigger retraining automatically.
- **Reproducible.** Pinned dependencies, deterministic seeds, and a single `docker compose up` for the infrastructure.

## Tech stack

| Concern | Choice |
| --- | --- |
| Model | XGBoost (gradient-boosted trees) + isotonic calibration |
| Streaming | Redpanda (Kafka API) |
| Feature store | Feast — online: Redis, offline: parquet |
| Serving | FastAPI + Uvicorn |
| Experiment tracking & registry | MLflow |
| Orchestration | Prefect |
| Monitoring | Evidently (drift) + Prometheus + Grafana |
| Packaging | Python 3.11+, `uv`, Docker Compose |

## Repository layout

```
src/fraud_detection_mlops/
├── config.py          # paths, seeds, thresholds, window sizes — one place, no magic numbers
├── data/              # loading, validation, time-based splits
├── features/          # feature definitions shared by training and serving
│   ├── velocity.py    #   per-card rolling aggregates (batch)
│   ├── online.py      #   the same definitions, computed incrementally for serving
│   ├── graph.py       #   fraud-ring graph features (connected components)
│   ├── encoders.py    #   leakage-safe frequency encoding
│   └── build.py       #   model feature-matrix assembly
├── models/            # train, evaluate, calibrate, cost-based threshold
├── streaming/         # Kafka producer (replay) + consumer (rolling features)
├── serving/           # Feast feature views + FastAPI scoring service
├── monitoring/        # Evidently drift + Prometheus metrics
└── pipelines/         # Prefect flows: delayed-label retraining, drift-check
tests/                 # leakage, parity, drift, and gating tests
infra/                 # Prometheus + Grafana provisioning
```

## Getting started

### 1. Install

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
```

### 2. Get the data

The dataset is not committed (`dataset/` is gitignored). Download from Kaggle into `dataset/`:

```bash
kaggle competitions download -c ieee-fraud-detection -p dataset/ && unzip 'dataset/*.zip' -d dataset/
```

Only `train_transaction.csv` + `train_identity.csv` (the labelled data) are used for modelling. The test files are the unlabelled holdout, used only as a source of unseen transactions to replay through the stream.

### 3. Train the model

```bash
make train-offline        # feature engineering, calibration, cost threshold; registers the model in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs and the model registry at :5000
```

### 4. Run the live system

```bash
docker compose up -d      # Redpanda + Console, Redis, Prometheus, Grafana
make feast-build          # build per-card state and load it into the online store
make topic                # create the transactions stream

# in separate terminals:
make serve                # scoring API on :8001 (exposes /metrics)
make consume              # keep rolling features fresh from the stream
make produce              # replay transactions onto the stream
make drift-monitor        # detect drift, publish gauges, trigger retraining
```

Dashboards: **Grafana** at `:3000`, **Redpanda Console** at `:8080`, **Prometheus** at `:9090`.

### 5. Self-contained demos

Each runs end-to-end and prints its result; most need only the dataset.

```bash
make score-verify     # prove online features + scores exactly match the offline definitions
make loadtest         # measure scoring latency (p50/p99) under load
make feedback-demo    # delayed labels → retrain rounds → gated promotion, with a label traced through
make drift-demo       # decay detection → drift-triggered retrain → recovery
make graph-experiment # measure the lift from fraud-ring graph features
```

`make help` lists every task.

## How the pieces fit together

- **Feature engineering** computes per-card velocity features (trailing-window counts and sums, time-since-last, amount-vs-average, new-device/new-location flags) and fraud-ring graph features (how many cards share a device, connected-component size). All point-in-time correct.
- **Streaming** replays transactions in time order onto Redpanda, keyed by card so each card's events stay ordered. A consumer maintains the rolling state incrementally — the same logic the batch path uses.
- **Feature store + serving** materialise per-card state into Redis; FastAPI looks features up by card key and scores with the calibrated model. The velocity transform is a single function exposed as a Feast on-demand view, so offline and online cannot diverge.
- **Feedback loop** simulates chargebacks arriving after a delay, joins matured labels back to their transactions, and runs a Prefect retraining flow that promotes a challenger only if it beats the champion on the held-out window.
- **Monitoring** compares recent traffic to the training distribution with Evidently, exports serving and drift metrics to Prometheus/Grafana, and trips the retraining flow when drift is detected.

## Testing

```bash
pytest -q
```

The suite focuses on the things that are easy to get silently wrong:

- **Leakage** — future transactions never change a past transaction's features (velocity and graph).
- **Train/serve parity** — the incremental online features equal the batch offline features, including a real-data slice and concurrent same-second transactions.
- **Feedback correctness** — retraining never uses an unmatured label, and the validation window is fixed across rounds.
- **Promotion gating** — a challenger is promoted only when it actually beats the champion.
- **Drift** — clean traffic raises no alarm; an injected shift is detected.

## Limitations & next steps

- **Model accuracy has headroom.** PR-AUC ~0.48 on the honest time split is solid at the top of the ranking but not state-of-the-art. The biggest untapped levers are a stable client identifier with per-client aggregations, and hyperparameter tuning — both target the bulk of the ranking. (This was a deliberate scope choice: the project's focus is the production system, not leaderboard chasing.)
- **Graph features are coverage-limited.** They add a real but small lift because device information exists for only ~24% of transactions.
- **Online serving of graph features** is not yet wired into the feature store — the same path the velocity features already took from batch to online.
- **Single-node infrastructure.** Everything runs locally via Docker Compose; a cloud deployment (k8s, managed Kafka/Redis) is out of scope here.
