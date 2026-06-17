"""Central configuration: paths, seeds, split fractions, column blocks.

No magic numbers scattered across the codebase (CLAUDE.md convention). Anything
tunable that more than one module cares about lives here.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------------
# Repo root = three parents up from this file (src/fraud_detection_mlops/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATASET_DIR: Path = Path(os.environ.get("DATASET_DIR", PROJECT_ROOT / "dataset"))

TRAIN_TRANSACTION_CSV: Path = DATASET_DIR / "train_transaction.csv"
TRAIN_IDENTITY_CSV: Path = DATASET_DIR / "train_identity.csv"
# Test files are the UNLABELED Kaggle holdout. Referenced here for completeness
# only — they must NEVER be loaded for training, validation, or metrics
# (invariant 2). They are reserved for stream replay in later milestones.
TEST_TRANSACTION_CSV: Path = DATASET_DIR / "test_transaction.csv"
TEST_IDENTITY_CSV: Path = DATASET_DIR / "test_identity.csv"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# --- Reproducibility -----------------------------------------------------------
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# --- Time-based split (invariant 2: never random-split) ------------------------
# Earliest TRAIN_FRACTION of rows (by TransactionDT) -> train; the rest -> val.
TRAIN_FRACTION: float = 0.8

# --- Schema --------------------------------------------------------------------
TARGET: str = "isFraud"
ID_COL: str = "TransactionID"
TIME_COL: str = "TransactionDT"
AMOUNT_COL: str = "TransactionAmt"

# Columns that must never become model features:
#   - ID_COL: a primary key (pure identifier, no signal).
#   - TIME_COL: absolute time. In a time-based split the validation window has
#     strictly larger TransactionDT than train, so feeding raw time lets the
#     model key on "later = different" — it won't generalize and won't exist in
#     the same range at serving. Time-derived *relative* features come in M1.
NON_FEATURE_COLS: tuple[str, ...] = (ID_COL, TARGET, TIME_COL)

# IEEE-CIS column blocks (used for missingness reporting and feature typing).
# Curated low-cardinality categoricals for the baseline one-hot encoder. The
# high-cardinality fields (card1, addr1, DeviceInfo, emaildomains, id_30/31/33)
# are deliberately left out of the *simple* baseline; real encoding lands in M1.
BASELINE_CATEGORICAL_COLS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
)

# MLflow. The legacy file store is rejected by MLflow >=3 (maintenance mode) and
# does not support the model registry we need in M1, so we default to a local
# SQLite backend with a local artifact directory. Both are gitignored.
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_LOCATION: str = os.environ.get(
    "MLFLOW_ARTIFACT_LOCATION", (PROJECT_ROOT / "mlartifacts").as_uri()
)
MLFLOW_EXPERIMENT: str = os.environ.get("MLFLOW_EXPERIMENT", "fraud-baseline")
