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
# Earliest TRAIN_FRACTION of rows (by TransactionDT) -> train+calibration; the
# rest -> validation. The validation window is held identical between M0 and M1
# so PR-AUC is compared on the exact same rows.
TRAIN_FRACTION: float = 0.8
# M1 three-way split: of the full timeline, the first MODEL_TRAIN_FRACTION trains
# the model, MODEL_TRAIN_FRACTION..TRAIN_FRACTION is the calibration slice (fits
# the isotonic calibrator + selects the cost threshold), and the final
# 1-TRAIN_FRACTION is validation. All boundaries are by TransactionDT, so
# train < calibration < validation in time — no future leaks backward.
MODEL_TRAIN_FRACTION: float = 0.7

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

# --- M1 feature engineering ----------------------------------------------------
# Card entity for velocity/aggregate features. IEEE-CIS is anonymized and has no
# explicit card id; ``card1`` is the most granular card field and the standard
# proxy for "the card/account". Kept as a tuple so it can grow into a composite
# uid (e.g. card1+addr1) later without touching the feature code.
CARD_ID_COLS: tuple[str, ...] = ("card1",)

# Trailing windows for velocity features, in seconds (TransactionDT is seconds
# from a fixed reference). 1 hour / 24 hours / 7 days.
VELOCITY_WINDOWS_SECONDS: dict[str, int] = {"1h": 3600, "24h": 86400, "7d": 604800}

# Columns whose membership a card "has seen before" is itself a fraud signal:
# a card transacting from a brand-new shipping region or device is riskier.
NEW_LOCATION_COL: str = "addr1"
NEW_DEVICE_COL: str = "DeviceInfo"

# ID-like numeric columns: stored as numbers but semantically categorical (high
# cardinality, no ordinal meaning). They are frequency-encoded, not fed raw, so
# the tree never treats e.g. card1=15000 as "greater than" card1=1000.
ID_LIKE_NUMERIC_COLS: tuple[str, ...] = (
    "card1", "card2", "card3", "card5", "addr1", "addr2",
)

# --- M1 cost model (invariant 4: thresholds chosen by cost, not 0.5) -----------
# Expected-cost framing for the allow/block decision:
#   - Missing fraud (a fraudulent txn we allow) costs the transaction amount: the
#     issuer eats the chargeback. (Fees could be added; amount is the floor.)
#   - Blocking a legitimate txn costs a fixed amount: lost margin on the sale plus
#     customer friction / support cost. A flat per-incident cost keeps the
#     trade-off legible and gives the cost-vs-threshold curve a clear minimum.
# These drive the threshold sweep; tune them to a business's real economics.
COST_PER_FALSE_BLOCK: float = 25.0

# --- MLflow --------------------------------------------------------------------
# The legacy file store is rejected by MLflow >=3 (maintenance mode) and
# does not support the model registry we need in M1, so we default to a local
# SQLite backend with a local artifact directory. Both are gitignored.
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_LOCATION: str = os.environ.get(
    "MLFLOW_ARTIFACT_LOCATION", (PROJECT_ROOT / "mlartifacts").as_uri()
)
MLFLOW_EXPERIMENT: str = os.environ.get("MLFLOW_EXPERIMENT", "fraud-baseline")
# M1 logs to its own experiment and registers the promoted model under this name.
# The "champion" alias points at the current best model the registry serves.
MLFLOW_OFFLINE_EXPERIMENT: str = os.environ.get("MLFLOW_OFFLINE_EXPERIMENT", "fraud-offline")
REGISTERED_MODEL_NAME: str = os.environ.get("REGISTERED_MODEL_NAME", "fraud-detection-offline")
CHAMPION_ALIAS: str = "champion"
