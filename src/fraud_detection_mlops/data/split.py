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


def time_based_split_three(
    df: pd.DataFrame,
    *,
    model_train_fraction: float = config.MODEL_TRAIN_FRACTION,
    train_fraction: float = config.TRAIN_FRACTION,
    time_col: str = config.TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into (train, calibration, validation) by time.

    Two cuts on the ``time_col`` timeline produce three strictly-ordered windows:
    the earliest ``model_train_fraction`` trains the model, the slice up to
    ``train_fraction`` calibrates it (and selects the cost threshold), and the
    final ``1 - train_fraction`` is held out for metrics. Because the validation
    boundary equals the M0 two-way cut, the validation set is identical to M0's —
    so M1's PR-AUC is comparable to the baseline on the exact same rows.

    train < calibration < validation in time, so no future leaks backward
    (invariant 2). Validation is touched only for final metrics, never for any
    fitting or threshold selection.
    """
    if not 0.0 < model_train_fraction < train_fraction < 1.0:
        raise ValueError(
            "Need 0 < model_train_fraction < train_fraction < 1; got "
            f"{model_train_fraction} and {train_fraction}."
        )

    ordered = df.sort_values([time_col, config.ID_COL]).reset_index(drop=True)
    train_cut = ordered[time_col].quantile(model_train_fraction, interpolation="lower")
    val_cut = ordered[time_col].quantile(train_fraction, interpolation="lower")

    train_df = ordered[ordered[time_col] <= train_cut].copy()
    calib_df = ordered[(ordered[time_col] > train_cut) & (ordered[time_col] <= val_cut)].copy()
    val_df = ordered[ordered[time_col] > val_cut].copy()

    if len(train_df) == 0 or len(calib_df) == 0 or len(val_df) == 0:
        raise ValueError(
            "Three-way time split produced an empty window — check the fractions "
            f"and time distribution (train_cut={train_cut}, val_cut={val_cut})."
        )

    logger.info(
        "3-way time split: train=%d (fraud %.3f%%) | calib=%d (fraud %.3f%%) | "
        "val=%d (fraud %.3f%%)",
        len(train_df),
        100 * train_df[config.TARGET].mean(),
        len(calib_df),
        100 * calib_df[config.TARGET].mean(),
        len(val_df),
        100 * val_df[config.TARGET].mean(),
    )

    assert train_df[time_col].max() <= calib_df[time_col].min(), "train/calib time overlap!"
    assert calib_df[time_col].max() <= val_df[time_col].min(), "calib/val time overlap!"

    return train_df, calib_df, val_df
