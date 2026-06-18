"""Point-in-time velocity / aggregate features per card.

INVARIANT 1 (no leakage, point-in-time correctness) lives or dies here. Every
feature below describes a card's behaviour *as of the moment of the current
transaction*, computed from **strictly earlier** transactions only. The current
transaction never contributes to its own features, and a *later* transaction can
never change an *earlier* transaction's features.

How the guarantee is enforced mechanically:
  - We sort by ``(card, TransactionDT, TransactionID)`` and run all per-card
    operations in that time order, so "prior rows" == "earlier in time".
  - Cumulative/shift ops (``cumcount``, ``shift(1)``, ``cumsum`` minus self)
    exclude the current row by construction.
  - Trailing time-window aggregates use ``rolling(window, closed="left")`` on a
    time index, which includes the left edge of the window but **excludes the
    right edge (the current instant)** — so concurrent same-second transactions
    don't see each other either (the conservative, safe choice).

Why compute on the *full* timeline before the train/val split? Because that is
both correct and realistic: a validation-time transaction genuinely has the
card's earlier history available to it at serving time (including rows that fall
in the training window). Splitting first would blind validation rows to that
history and make the features inconsistent with what online scoring will see
(invariant 5). Training rows still never see validation rows — those are strictly
in the future, hence never "prior". The leakage tests assert this directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)

_CARD_ID = "_card_id"
_NAN_TOKEN = "__nan__"

# Feature column names this module adds (used by the model column selector so the
# engineered features are picked up as numeric passthrough).
VELOCITY_FEATURES: tuple[str, ...] = (
    "card_txn_count_prior",
    "time_since_last_txn",
    "card_amt_mean_prior",
    "amt_vs_card_mean_ratio",
    "card_txn_count_1h",
    "card_amt_sum_1h",
    "card_txn_count_24h",
    "card_amt_sum_24h",
    "card_txn_count_7d",
    "card_amt_sum_7d",
    "new_location",
    "new_device",
)


def _build_card_key(df: pd.DataFrame, card_cols: tuple[str, ...]) -> pd.Series:
    """Build a single string card key from ``card_cols`` (NaN -> sentinel).

    A string key keeps grouping stable and lets the key grow into a composite
    uid later without changing call sites.
    """
    present = [c for c in card_cols if c in df.columns]
    if not present:
        raise ValueError(f"None of the card id columns {card_cols} are in the frame.")
    key = df[present[0]].astype("object").where(df[present[0]].notna(), _NAN_TOKEN).astype(str)
    for col in present[1:]:
        part = df[col].astype("object").where(df[col].notna(), _NAN_TOKEN).astype(str)
        key = key.str.cat(part, sep="|")
    return key


def _rolling_window_count_sum(
    sorted_work: pd.DataFrame, time_col: str, amount_col: str, window_seconds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window count and amount-sum of *prior* txns for each card.

    ``sorted_work`` must already be sorted by ``(_card_id, time_col, ...)`` so
    that, within each card group, the time index is monotonic. We index by a
    timedelta built from ``TransactionDT`` and roll with ``closed="left"`` so the
    window ``[t - window, t)`` covers strictly-earlier transactions and excludes
    the current instant. Output arrays are positionally aligned to
    ``sorted_work`` rows. A card's first transaction has count 0 / sum 0.
    """
    tmp = sorted_work[[_CARD_ID, amount_col]].copy()
    tmp.index = pd.to_timedelta(sorted_work[time_col].to_numpy(), unit="s")
    grouped = tmp.groupby(_CARD_ID, sort=False)[amount_col]
    win = f"{window_seconds}s"
    # closed="left" => exclude the right endpoint (the current transaction's time).
    # min_periods=0 => an empty trailing window yields 0, not NaN (a card's first
    # txn, or one with no recent activity, legitimately has zero prior txns).
    count = grouped.rolling(win, closed="left", min_periods=0).count().to_numpy()
    total = grouped.rolling(win, closed="left", min_periods=0).sum().to_numpy()
    # No prior rows in the window -> rolling returns 0 for count and 0.0 for sum.
    return count, np.nan_to_num(total, nan=0.0)


def _is_new_value_for_card(sorted_work: pd.DataFrame, value_col: str) -> np.ndarray:
    """Flag the first time a card is seen with a given value (point-in-time).

    Returns 1.0 when the (card, value) pair has not appeared in any strictly
    earlier transaction for that card, 0.0 when it has, and NaN when the value
    itself is missing (a missing device/region is "unknown", not "new"). Relies
    on ``sorted_work`` being in time order within each card so ``cumcount`` counts
    only prior occurrences.
    """
    occ = sorted_work.groupby([_CARD_ID, value_col], dropna=False).cumcount()
    flag = (occ == 0).astype("float32").to_numpy()
    flag[sorted_work[value_col].isna().to_numpy()] = np.nan
    return flag


def add_velocity_features(
    df: pd.DataFrame,
    *,
    time_col: str = config.TIME_COL,
    amount_col: str = config.AMOUNT_COL,
    card_cols: tuple[str, ...] = config.CARD_ID_COLS,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add point-in-time velocity/aggregate features per card to ``df``.

    Returns a copy of ``df`` (in its original row order) with these columns added:

    - ``card_txn_count_prior``: number of strictly-earlier transactions for this
      card over all time (lifetime prior count). 0 for a card's first txn.
    - ``time_since_last_txn``: seconds since this card's previous transaction;
      NaN for the first one.
    - ``card_amt_mean_prior``: mean amount of this card's strictly-earlier
      transactions; NaN when there is no history.
    - ``amt_vs_card_mean_ratio``: current amount / ``card_amt_mean_prior`` — how
      unusual this amount is versus the card's own past. NaN with no history.
    - ``card_txn_count_{1h,24h,7d}`` / ``card_amt_sum_{1h,24h,7d}``: count and
      amount-sum of prior transactions in the trailing window (excludes current).
    - ``new_location``: 1 if the card's ``addr1`` is unseen in its prior txns.
    - ``new_device``: 1 if the card's ``DeviceInfo`` is unseen in its prior txns.

    Every value uses only transactions strictly earlier than the current one for
    the same card (invariant 1). See the module docstring for the mechanism.
    """
    windows = windows if windows is not None else config.VELOCITY_WINDOWS_SECONDS
    original_index = df.index
    if not original_index.is_unique:
        raise ValueError("add_velocity_features requires a unique index to restore row order.")

    work = df.copy()
    work[_CARD_ID] = _build_card_key(work, card_cols)
    # Sort by card then time so per-card ops run in time order. mergesort is
    # stable, so TransactionID ties break deterministically (reproducibility).
    sort_keys = [_CARD_ID, time_col]
    if config.ID_COL in work.columns:
        sort_keys.append(config.ID_COL)
    work = work.sort_values(sort_keys, kind="mergesort")

    grp = work.groupby(_CARD_ID, sort=False)

    # Lifetime prior count: cumcount is 0 for the first row of each card and
    # counts only earlier rows thereafter (time-ordered) -> point-in-time safe.
    work["card_txn_count_prior"] = grp.cumcount().astype("int32")

    # Seconds since the card's previous transaction (shift looks one row back).
    prev_dt = grp[time_col].shift(1)
    work["time_since_last_txn"] = (work[time_col] - prev_dt).astype("float64")

    # Prior mean amount = (running sum - current) / prior count; NaN when none.
    cum_sum = grp[amount_col].cumsum()
    prior_sum = cum_sum - work[amount_col]
    prior_count = work["card_txn_count_prior"]
    prior_mean = prior_sum / prior_count.where(prior_count > 0)
    work["card_amt_mean_prior"] = prior_mean.astype("float64")
    work["amt_vs_card_mean_ratio"] = (work[amount_col] / prior_mean).astype("float64")

    # Trailing time-window count + amount sum (closed="left" excludes current).
    for name, secs in windows.items():
        count, total = _rolling_window_count_sum(work, time_col, amount_col, secs)
        work[f"card_txn_count_{name}"] = count.astype("float32")
        work[f"card_amt_sum_{name}"] = total.astype("float32")

    # New-region / new-device flags (NaN where the underlying value is missing).
    loc_col = config.NEW_LOCATION_COL
    dev_col = config.NEW_DEVICE_COL
    work["new_location"] = (
        _is_new_value_for_card(work, loc_col) if loc_col in work.columns else np.float32(np.nan)
    )
    work["new_device"] = (
        _is_new_value_for_card(work, dev_col) if dev_col in work.columns else np.float32(np.nan)
    )

    work = work.drop(columns=[_CARD_ID])
    # Restore the caller's original row order so labels/splits stay aligned.
    result = work.loc[original_index]

    logger.info(
        "Added %d velocity features (windows=%s, card=%s)",
        len(VELOCITY_FEATURES),
        list(windows),
        card_cols,
    )
    return result
