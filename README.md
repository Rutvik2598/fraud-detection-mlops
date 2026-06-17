# fraud-detection-mlops

Real-time fraud detection MLOps pipeline on the IEEE-CIS dataset — streaming inference, train/serve feature parity, a delayed-label feedback loop, and drift-triggered retraining.

**Status:** M0 — scaffold + honest baseline ✅ · _next: M1 (offline model done right)_

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # source .venv/bin/activate on Unix
pip install -e ".[dev]"
cp .env.example .env

python -m fraud_detection_mlops.models.train_baseline   # train + log baseline
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
