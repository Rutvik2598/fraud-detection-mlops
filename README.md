# fraud-detection-mlops

Real-time fraud detection MLOps pipeline on the IEEE-CIS dataset — streaming inference, train/serve feature parity, a delayed-label feedback loop, and drift-triggered retraining.

**Status:** M0 — scaffold ✅ · M1 — offline model ✅ · M2 — streaming ✅ · M3 — feature store + serving ✅ · M4 — feedback loop ✅ · M5 — monitoring + drift-triggered retrain ✅ · _next: M6 (stretch)_

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # source .venv/bin/activate on Unix
pip install -e ".[dev]"
cp .env.example .env

python -m fraud_detection_mlops.models.train_baseline   # M0: train + log baseline
python -m fraud_detection_mlops.models.train_offline    # M1: XGBoost + calibration + cost threshold
pytest -q                                               # leakage / encoder tests
mlflow ui --backend-store-uri sqlite:///mlflow.db       # view runs at :5000
```

EDA lives in [notebooks/00_eda.ipynb](notebooks/00_eda.ipynb).

## Data

The IEEE-CIS dataset is **not committed** (`dataset/` is gitignored). Download from Kaggle into `dataset/`:

```bash
kaggle competitions download -c ieee-fraud-detection -p dataset/ && unzip dataset/'*.zip' -d dataset/
```

Only `train_transaction.csv` (+ `train_identity.csv`) is used for modeling; the test files are the unlabeled holdout, reserved for later stream replay.

## M0 result

Time-based validation (latest ~20% by `TransactionDT`): fraud rate **3.44%**, **PR-AUC 0.184** (5.3× over base), ROC-AUC 0.832. A deliberately simple, class-weighted logistic-regression floor for later milestones to beat.

## M1 result

XGBoost on point-in-time velocity features (per-card trailing-window counts/sums, time-since-last, amount-vs-card-mean, new-device/new-location flags) + frequency-encoded categoricals, with `scale_pos_weight` for imbalance, isotonic calibration, and a cost-minimizing block threshold. Evaluated on the **same** validation window as M0.

| metric | M0 (LogReg) | M1 (XGBoost) |
| --- | --- | --- |
| PR-AUC (headline) | 0.184 | **0.475** (+158%) |
| ROC-AUC | 0.832 | 0.891 |
| precision@500 | 0.034 | 0.916 |
| Brier (calibrated) | — | 0.083 → **0.023** |

The cost model (missed fraud = txn amount; false block = \$25) selects its threshold on a held-out calibration slice, then is applied to validation: expected cost **\$404k** vs **\$610k** for blocking nothing (~34% lower) and \$2.85M for blocking everything. The calibrated model is logged to MLflow and registered as `fraud-detection-offline@champion`. Point-in-time correctness is covered by `tests/`.

## M2 — streaming backbone

Redpanda (Kafka API) via `docker-compose`, a **producer** that replays transactions in `TransactionDT` order at configurable speed, and a **consumer** that keeps the per-card rolling aggregates current as transactions flow. No scoring yet (that's M3).

```bash
docker compose up -d                 # start Redpanda + Console (http://localhost:8080)
make topic                           # create the `transactions` topic (6 partitions)

# end-to-end check: replay a slice, consume it, verify streamed aggregates == offline
python -m fraud_detection_mlops.streaming.verify --limit 50000

# or run the two sides separately
make produce LIMIT=50000 SPEED=0     # SPEED = sim TransactionDT-seconds per real second (0 = flood)
make consume                         # updates rolling features, writes reports/stream_features.jsonl
```

Messages are **keyed by card** so all of a card's transactions stay ordered on one partition — the guarantee the rolling aggregates depend on. The online aggregator (`features/online.py`) is the serve-side twin of the offline `features/velocity.py`; `tests/test_online_aggregator.py` replays data through both and asserts they produce **identical** features (incl. a 30k-row real-data slice and concurrent same-second transactions) — train/serve parity (invariant 5) verified without a broker. The transactions stream deliberately carries **no fraud label** (chargebacks arrive late; that's the M4 feedback loop).

## M3 — feature store + online inference

Feast (online = Redis, offline = parquet) serves the per-card rolling state; a FastAPI service looks features up by card key and scores with the calibrated champion model under the latency budget. The 12 velocity features are defined **once** (`features/online.py::compute_velocity_features`) and exposed as a Feast on-demand feature view — the same function the streaming aggregator uses, so offline (`get_historical_features`) and online (`get_online_features`) cannot drift.

```bash
docker compose up -d redis                              # online store
make feast-build                                        # build snapshot + apply + materialize
make serve                                              # FastAPI on :8001 (4 workers)
make score-verify                                       # prove parity: online features + scores == offline
make loadtest                                           # p50/p99 latency under load
```
For local dev without Docker, set `FEAST_ONLINE_STORE=sqlite`.

**Train/serve parity:** `make score-verify` materializes state at the cutoff, then for the first "live" transaction of each card compares online vs offline — **300 transactions, all 12 velocity features and all calibrated model scores matched exactly**. `tests/test_serving_parity.py` guards the Feast on-demand transform without a broker.

**Latency (server-side processing, the scoring budget):** warm per-request **p50 ≈ 14 ms, p99 ≈ 28 ms** — under the 50 ms budget (Feast lookup ~1.4 ms, model ~13 ms). Under concurrent load the single-worker GIL serializes the CPU-bound model call, so throughput scales with workers/replicas; the sqlite dev store also adds tail latency under contention that Redis removes. Measured with `make loadtest`; re-run against Redis for production numbers.

## M4 — feedback loop + retraining

Labels (chargebacks) arrive **late**: a transaction at `TransactionDT = t` only gets a usable label once the clock passes `t + LABEL_DELAY_SECONDS`. A Prefect flow retrains on the matured labels (joined back by `TransactionID`) and promotes the challenger in the MLflow registry **only if it beats the current champion** on the fixed held-out validation window.

```bash
make feedback-demo                 # simulate the clock advancing: retrain rounds + gated promotion + label trace
make retrain CLOCK=8745782         # run one Prefect retraining round at a given clock
```

Demo output (3 rounds, label delay 7 days; validation window held out every round):

| round | clock (TransactionDT) | matured labels | challenger PR-AUC | champion PR-AUC | promoted |
|---|---|---|---|---|---|
| 1 | 5,592,303 | 214,078 | 0.4103 | — | ✅ |
| 2 | 8,745,782 | 331,489 | 0.4352 | 0.4103 | ✅ |
| 3 | 12,192,853 | 453,779 | 0.4529 | 0.4352 | ✅ |

PR-AUC climbs as delayed labels mature, each round gated-promoted because it genuinely beat the incumbent (the gate rejects regressions — covered by `tests/test_feedback.py`). The demo then traces one late fraud label from "not yet arrived → excluded" to "matured → joined back → trained on → part of a promoted model." Point-in-time features are computed once and cached; the retraining flow only uses labels matured by the clock (never the future — invariant 1/2). The demo registers under a separate model name (`fraud-detection-feedback`) so it never disturbs the M3 serving champion.

## M5 — monitoring + drift-triggered retrain

Evidently drift reports (feature + prediction drift vs. the training distribution), Prometheus metrics on the scoring service + drift monitor, Grafana dashboards, and a drift-check Prefect flow that fires the M4 retraining flow when the world moves.

```bash
docker compose up -d prometheus grafana          # Grafana :3000 (anon admin), Prometheus :9090
make drift-demo                                  # in-process: decay detection -> drift-triggered retrain -> recovery
# live version:
make serve                                       # scoring service exposes /metrics
make drift-monitor                               # publishes drift gauges; triggers retrain on drift
make produce-drift LIMIT=60000 DRIFT_AFTER=25000 # stream a covariate shift mid-run
```

`make drift-demo` output (decay → detect → trigger → recover):

| state | feature drift share | dataset drift | prediction drift | PR-AUC |
|---|---|---|---|---|
| 1. healthy | 0.10 | False | False | 0.4751 |
| 2. drift injected (champion) | 0.30 | **True** | **True** | 0.4447 |
| 4. after retrain (new model) | 0.10 | False | False | 0.4516 |

Healthy traffic raises no alert; an injected covariate shift trips feature **and** prediction drift and the champion decays; the drift-check flow detects it and fires the retraining flow; the retrained model brings drift back to healthy and recovers PR-AUC on the drifted population. Drift detection needs **no labels** (the early-warning signal, since chargebacks arrive late). Monitored features are deliberately the roughly-stationary ones — cumulative velocity counts trend over time and would always "drift." Grafana panels: latency p50/p99, throughput, alert (block) rate, score distribution, feature/prediction drift, drift-triggered retrains. Evidently HTML reports land in `reports/drift/`.
