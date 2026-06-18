# fraud-detection-mlops

Real-time fraud detection MLOps pipeline on the IEEE-CIS dataset — streaming inference, train/serve feature parity, a delayed-label feedback loop, and drift-triggered retraining.

**Status:** M0 — scaffold + honest baseline ✅ · M1 — offline model done right ✅ · _next: M2 (streaming backbone)_

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
