"""Load and validate the labeled IEEE-CIS training data.

Only ``train_transaction.csv`` (+ ``train_identity.csv``) is loaded here. The
test files are the unlabeled Kaggle holdout and are never read for modeling or
metrics (invariant 2) — they are reserved for stream replay in later milestones.
"""

from __future__ import annotations

import logging

import pandas as pd

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)


def load_training_data(
    *,
    transaction_csv=config.TRAIN_TRANSACTION_CSV,
    identity_csv=config.TRAIN_IDENTITY_CSV,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load ``train_transaction`` and LEFT JOIN ``train_identity`` on TransactionID.

    Only ~24% of transactions have a matching identity row, so identity columns
    are NaN for most rows after the join — that is expected and handled
    downstream. Whether identity data exists *at all* is itself a potential
    fraud signal, so we engineer a boolean ``has_identity`` flag before the
    identity NaNs make that information unrecoverable.

    Args:
        transaction_csv: Path to ``train_transaction.csv``.
        identity_csv: Path to ``train_identity.csv`` (optional; handled if absent).
        nrows: If set, read only the first N transaction rows (for quick smoke
            tests). Note: a head-slice is NOT time-ordered cleanly, so never use
            ``nrows`` for real metrics — only for plumbing checks.

    Returns:
        The joined transaction+identity frame with a ``has_identity`` column.
    """
    logger.info("Loading transactions from %s", transaction_csv)
    txn = pd.read_csv(transaction_csv, nrows=nrows)
    logger.info("Loaded %d transaction rows, %d columns", len(txn), txn.shape[1])

    if identity_csv is not None and identity_csv.exists():
        logger.info("Loading identity from %s", identity_csv)
        identity = pd.read_csv(identity_csv)
        # Mark identity-bearing transactions BEFORE the join, so the flag is not
        # contaminated by the NaNs the LEFT JOIN introduces for non-matches.
        identity_ids = set(identity[config.ID_COL])
        merged = txn.merge(identity, on=config.ID_COL, how="left", validate="one_to_one")
        merged["has_identity"] = merged[config.ID_COL].isin(identity_ids).astype("int8")
        match_rate = merged["has_identity"].mean()
        logger.info("Identity match rate after LEFT JOIN: %.1f%%", 100 * match_rate)
    else:
        logger.warning("Identity file not found at %s — proceeding without it.", identity_csv)
        merged = txn
        merged["has_identity"] = 0

    return merged


def validate_training_data(df: pd.DataFrame) -> dict[str, object]:
    """Run basic structural checks on the loaded training frame.

    Verifies row count, presence of the key columns (isFraud / TransactionDT /
    TransactionAmt / TransactionID), that the label is binary, and that the time
    key has no nulls (it is the ordering key for the split). Raises on a broken
    invariant; returns a small summary dict otherwise.
    """
    summary: dict[str, object] = {}

    if len(df) == 0:
        raise ValueError("Training frame is empty.")
    summary["n_rows"] = len(df)
    summary["n_cols"] = df.shape[1]

    required = [config.ID_COL, config.TARGET, config.TIME_COL, config.AMOUNT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    label_values = set(df[config.TARGET].dropna().unique())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"{config.TARGET} must be binary 0/1; saw {sorted(label_values)}")
    if df[config.TARGET].isna().any():
        raise ValueError(f"{config.TARGET} has nulls — the training data must be fully labeled.")
    summary["fraud_rate"] = float(df[config.TARGET].mean())

    if df[config.TIME_COL].isna().any():
        raise ValueError(
            f"{config.TIME_COL} has nulls — it is the ordering key and must be complete."
        )
    summary["transactiondt_min"] = int(df[config.TIME_COL].min())
    summary["transactiondt_max"] = int(df[config.TIME_COL].max())

    if df[config.ID_COL].duplicated().any():
        raise ValueError(f"{config.ID_COL} contains duplicates — join may have fanned out.")

    if "has_identity" in df.columns:
        summary["identity_match_rate"] = float(df["has_identity"].mean())

    logger.info("Validation passed: %s", summary)
    return summary
