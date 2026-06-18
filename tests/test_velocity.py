"""Leakage tests for the point-in-time velocity features (invariant 1).

The central property: a transaction's features use ONLY strictly-earlier
transactions of the same card. We assert the concrete values on a hand-built
timeline and, most importantly, that adding or removing *later* transactions
never changes an *earlier* transaction's features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_detection_mlops.features import add_velocity_features

# Column names match the project schema (config.TIME_COL etc.).
DT, AMT, CARD, ADDR, DEV, TID, FRAUD = (
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "DeviceInfo",
    "TransactionID",
    "isFraud",
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def timeline() -> pd.DataFrame:
    # card 1: three txns; card 2: one txn (with a missing device).
    return _frame(
        [
            {TID: 1, DT: 1000, AMT: 100.0, CARD: 1, ADDR: 10, DEV: "X", FRAUD: 0},
            {TID: 2, DT: 1500, AMT: 200.0, CARD: 1, ADDR: 10, DEV: "Y", FRAUD: 0},
            {TID: 3, DT: 10000, AMT: 300.0, CARD: 1, ADDR: 20, DEV: "X", FRAUD: 1},
            {TID: 4, DT: 1200, AMT: 50.0, CARD: 2, ADDR: 99, DEV: np.nan, FRAUD: 0},
        ]
    )


def _by_tid(df: pd.DataFrame) -> dict[int, pd.Series]:
    out = add_velocity_features(df).set_index(TID)
    return {tid: out.loc[tid] for tid in out.index}


def test_first_txn_has_no_history(timeline):
    r = _by_tid(timeline)
    first = r[1]
    assert first["card_txn_count_prior"] == 0
    assert np.isnan(first["time_since_last_txn"])
    assert np.isnan(first["card_amt_mean_prior"])
    assert np.isnan(first["amt_vs_card_mean_ratio"])
    assert first["card_txn_count_1h"] == 0
    assert first["card_amt_sum_1h"] == 0


def test_prior_counts_and_amounts(timeline):
    r = _by_tid(timeline)
    # 2nd txn of card 1: one prior txn 500s earlier, amount 100.
    assert r[2]["card_txn_count_prior"] == 1
    assert r[2]["time_since_last_txn"] == 500
    assert r[2]["card_amt_mean_prior"] == pytest.approx(100.0)
    assert r[2]["amt_vs_card_mean_ratio"] == pytest.approx(2.0)  # 200 / 100
    assert r[2]["card_txn_count_1h"] == 1  # txn 1 is within 3600s and strictly earlier
    assert r[2]["card_amt_sum_1h"] == pytest.approx(100.0)
    # 3rd txn: two priors (mean 150), but both are >3600s before t=10000.
    assert r[3]["card_txn_count_prior"] == 2
    assert r[3]["time_since_last_txn"] == 8500
    assert r[3]["card_amt_mean_prior"] == pytest.approx(150.0)
    assert r[3]["amt_vs_card_mean_ratio"] == pytest.approx(2.0)  # 300 / 150
    assert r[3]["card_txn_count_1h"] == 0  # both priors fell out of the 1h window
    assert r[3]["card_amt_sum_1h"] == 0


def test_window_excludes_current_transaction(timeline):
    # closed="left" must exclude the current instant: a card's solo txn within a
    # window sees count 0, never 1 (it must not count itself).
    r = _by_tid(timeline)
    assert r[1]["card_txn_count_24h"] == 0
    assert r[4]["card_txn_count_24h"] == 0


def test_new_location_and_device_flags(timeline):
    r = _by_tid(timeline)
    # card 1: addr 10 new on txn1, repeated on txn2, addr 20 new on txn3.
    assert r[1]["new_location"] == 1
    assert r[2]["new_location"] == 0
    assert r[3]["new_location"] == 1
    # devices: X new on txn1, Y new on txn2, X already seen by txn3.
    assert r[1]["new_device"] == 1
    assert r[2]["new_device"] == 1
    assert r[3]["new_device"] == 0
    # card 2 device is missing -> "unknown", encoded NaN (not "new").
    assert np.isnan(r[4]["new_device"])
    assert r[4]["new_location"] == 1


def test_cards_are_independent(timeline):
    # card 2's lone txn must not be influenced by card 1's activity.
    r = _by_tid(timeline)
    assert r[4]["card_txn_count_prior"] == 0
    assert np.isnan(r[4]["time_since_last_txn"])


def test_later_transactions_do_not_change_earlier_features(timeline):
    """The core anti-leakage property: the future cannot alter the past."""
    full = _by_tid(timeline)

    # Drop the latest txn (txn 3) -> earlier rows must be byte-for-byte identical.
    without_last = _by_tid(timeline[timeline[TID] != 3].copy())
    for tid in (1, 2, 4):
        pd.testing.assert_series_equal(
            full[tid].drop(labels=[FRAUD]),
            without_last[tid].drop(labels=[FRAUD]),
            check_names=False,
        )

    # Append an even-later txn for card 1 -> existing rows must be unchanged.
    later = _frame([{TID: 5, DT: 20000, AMT: 999.0, CARD: 1, ADDR: 10, DEV: "Z", FRAUD: 0}])
    extended = pd.concat([timeline, later], ignore_index=True)
    with_future = _by_tid(extended)
    for tid in (1, 2, 3, 4):
        pd.testing.assert_series_equal(
            full[tid].drop(labels=[FRAUD]),
            with_future[tid].drop(labels=[FRAUD]),
            check_names=False,
        )


def test_row_order_is_preserved(timeline):
    # Output must align to input rows (labels/splits depend on this).
    shuffled = timeline.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = add_velocity_features(shuffled)
    assert (out[TID].to_numpy() == shuffled[TID].to_numpy()).all()
