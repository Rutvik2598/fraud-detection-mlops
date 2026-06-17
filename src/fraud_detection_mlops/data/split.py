"""Time-based train/validation split.

Invariant 2: NEVER random-split. Fraud detection is a forecasting problem — we
train on the past and validate on the future. A random split leaks future
information into training and produces dishonestly optimistic metrics. We order
by ``TransactionDT`` and cut at a fraction so the validation window is strictly
later in time than the training window.
"""

from __future__ import annotations

import logging

import pandas as pd

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    *,
    train_fraction: float = config.TRAIN_FRACTION,
    time_col: str = config.TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, val) by time.

    The earliest ``train_fraction`` of rows (ordered by ``time_col``) become the
    training set; the latest rows become validation. The cut is placed at the
    ``train_fraction`` quantile of ``time_col``; all rows at or before the cut
    go to train, the rest to val. Ties on the boundary timestamp are kept
    together on the train side so no single instant straddles the split.

    Args:
        df: Loaded (and validated) training frame.
        train_fraction: Fraction of the timeline assigned to training.
        time_col: Ordering column (the IEEE-CIS time delta in seconds).

    Returns:
        (train_df, val_df), each a copy sorted by time then TransactionID.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}")

    ordered = df.sort_values([time_col, config.ID_COL]).reset_index(drop=True)

    cut_value = ordered[time_col].quantile(train_fraction, interpolation="lower")
    train_mask = ordered[time_col] <= cut_value

    train_df = ordered[train_mask].copy()
    val_df = ordered[~train_mask].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            "Time split produced an empty side — check train_fraction and the "
            f"time distribution (cut={cut_value})."
        )

    logger.info(
        "Time split @ %s=%s: train=%d (%.1f%%, fraud %.3f%%) | val=%d (%.1f%%, fraud %.3f%%)",
        time_col,
        cut_value,
        len(train_df),
        100 * len(train_df) / len(ordered),
        100 * train_df[config.TARGET].mean(),
        len(val_df),
        100 * len(val_df) / len(ordered),
        100 * val_df[config.TARGET].mean(),
    )

    # Sanity: no time overlap across the boundary (the whole point of the split).
    assert train_df[time_col].max() <= val_df[time_col].min(), "Time overlap across split!"

    return train_df, val_df
