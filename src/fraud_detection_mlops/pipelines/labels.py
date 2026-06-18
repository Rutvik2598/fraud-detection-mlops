"""Delayed-label simulation, feature cache, and label join-back (M4).

Models the central fact of fraud feedback: a transaction's label (fraud or not)
is **not known when it happens** — the chargeback arrives later. We simulate that
with a fixed delay: a transaction at ``TransactionDT = t`` only has a usable label
once the clock passes ``t + LABEL_DELAY_SECONDS``. Retraining at clock ``T`` may
therefore only use labels that have *matured* by ``T`` — never the rest (that would
be peeking at the future, invariant 1/2).

The "join-back" makes this explicit: transactions carry features but no label;
labels arrive separately into a label store and are joined back by
``TransactionID``, keeping only the matured ones. The point-in-time features are
computed once and cached (they depend only on prior transactions, not on labels),
so retraining rounds are cheap.
"""

from __future__ import annotations

import logging

import pandas as pd

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features import add_velocity_features, select_model_columns

logger = logging.getLogger(__name__)


def load_features(*, rebuild: bool = False) -> pd.DataFrame:
    """Point-in-time feature frame for every transaction (cached parquet).

    Columns = the model's feature columns + label + TransactionID + TransactionDT,
    time-ordered. Features depend only on prior transactions, so this is identical
    regardless of the retraining clock — compute once, reuse every round.
    """
    path = config.FEATURE_CACHE_PARQUET
    if path.exists() and not rebuild:
        logger.info("Loading cached features from %s", path)
        return pd.read_parquet(path)

    logger.info("Building point-in-time feature cache (one-time)...")
    df = add_velocity_features(load_training_data())
    numeric, categorical = select_model_columns(df)
    keep = list(
        dict.fromkeys([*numeric, *categorical, config.TARGET, config.TIME_COL, config.ID_COL])
    )
    out = df[keep].sort_values([config.TIME_COL, config.ID_COL]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    logger.info("Cached %d rows x %d cols -> %s", len(out), out.shape[1], path)
    return out


def label_store(features: pd.DataFrame, *, delay: int = config.LABEL_DELAY_SECONDS) -> pd.DataFrame:
    """The label 'store': each label becomes available at TransactionDT + delay."""
    return pd.DataFrame(
        {
            config.ID_COL: features[config.ID_COL].to_numpy(),
            config.TARGET: features[config.TARGET].to_numpy(),
            "label_available_dt": features[config.TIME_COL].to_numpy() + delay,
        }
    )


def join_back(
    features: pd.DataFrame, labels: pd.DataFrame, available_until_dt: int
) -> pd.DataFrame:
    """Join matured labels back onto (unlabeled) feature rows by TransactionID.

    Drops the label that rides along with ``features`` and re-attaches only labels
    that have matured by ``available_until_dt`` — an inner join, so transactions
    whose chargeback hasn't come back yet are simply absent from training.
    """
    matured = labels[labels["label_available_dt"] <= available_until_dt]
    unlabeled = features.drop(columns=[config.TARGET])
    return unlabeled.merge(matured[[config.ID_COL, config.TARGET]], on=config.ID_COL, how="inner")


def val_start_dt(features: pd.DataFrame, *, val_fraction: float = config.VAL_FRACTION) -> int:
    """TransactionDT at which the held-out validation window begins (last fraction)."""
    return int(features[config.TIME_COL].quantile(1.0 - val_fraction, interpolation="lower"))


def build_training_data(
    features: pd.DataFrame,
    available_until_dt: int,
    *,
    delay: int = config.LABEL_DELAY_SECONDS,
    val_fraction: float = config.VAL_FRACTION,
    model_train_fraction: float = config.MODEL_TRAIN_FRACTION,
) -> dict:
    """Assemble (train, calib, val) for a retrain at clock ``available_until_dt``.

    - val: the fixed held-out window (latest ``val_fraction``), labels known — the
      yardstick, identical every round, never trained on.
    - training pool: matured-label transactions strictly before the val window,
      assembled via the join-back. Split by time into train (earliest
      ``model_train_fraction``) and calib (the rest) for isotonic calibration.
    """
    v_start = val_start_dt(features, val_fraction=val_fraction)
    val_df = features[features[config.TIME_COL] >= v_start].copy()

    labeled = join_back(features, label_store(features, delay=delay), available_until_dt)
    pool = labeled[labeled[config.TIME_COL] < v_start].sort_values(
        [config.TIME_COL, config.ID_COL]
    )
    if pool.empty:
        raise ValueError(f"No matured training labels at clock {available_until_dt}.")

    cut = pool[config.TIME_COL].quantile(model_train_fraction, interpolation="lower")
    train_df = pool[pool[config.TIME_COL] <= cut].copy()
    calib_df = pool[pool[config.TIME_COL] > cut].copy()
    both_classes = train_df[config.TARGET].nunique() >= 2 and calib_df[config.TARGET].nunique() >= 2
    if calib_df.empty or not both_classes:
        raise ValueError(
            f"Insufficient matured data at clock {available_until_dt} "
            f"(train={len(train_df)}, calib={len(calib_df)}) — advance the clock."
        )

    logger.info(
        "Clock %d: %d matured training rows (train=%d, calib=%d), val=%d",
        available_until_dt, len(pool), len(train_df), len(calib_df), len(val_df),
    )
    return {
        "train_df": train_df,
        "calib_df": calib_df,
        "val_df": val_df,
        "n_matured": len(pool),
        "val_start_dt": v_start,
    }
