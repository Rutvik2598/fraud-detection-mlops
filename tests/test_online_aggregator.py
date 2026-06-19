"""Train/serve parity for the streaming aggregator (invariants 1 and 5).

The online ``OnlineCardAggregator`` (serve-side, event-at-a-time) must reproduce
the offline ``add_velocity_features`` (train-side, vectorized) exactly. We feed
the same transactions through both and assert equality feature-by-feature. A
mismatch here means the model would see different features at training and at
serving time — the silent bug invariant 5 exists to prevent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection_mlops import config
from fraud_detection_mlops.features import (
    VELOCITY_FEATURES,
    OnlineCardAggregator,
    add_velocity_features,
)

DT, AMT, CARD, ADDR, DEV, TID, FRAUD = (
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "DeviceInfo",
    "TransactionID",
    "isFraud",
)


def _run_online(df: pd.DataFrame) -> pd.DataFrame:
    """Replay df through the aggregator in arrival order, indexed by TransactionID."""
    agg = OnlineCardAggregator()
    ordered = df.sort_values([DT, TID]).to_dict("records")
    rows = {}
    for rec in ordered:
        rows[rec[TID]] = agg.update(rec)
    return pd.DataFrame.from_dict(rows, orient="index")[list(VELOCITY_FEATURES)]


def _assert_parity(df: pd.DataFrame) -> None:
    offline = add_velocity_features(df).set_index(TID)[list(VELOCITY_FEATURES)]
    online = _run_online(df)
    for tid in offline.index:
        for feat in VELOCITY_FEATURES:
            off = float(offline.loc[tid, feat])
            on = float(online.loc[tid, feat])
            if math.isnan(off) or math.isnan(on):
                assert math.isnan(off) and math.isnan(on), f"NaN mismatch {feat} @ tid={tid}"
            else:
                assert on == pytest.approx(off, rel=1e-6, abs=1e-6), (
                    f"{feat} @ tid={tid}: online={on} offline={off}"
                )


@pytest.fixture
def timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {TID: 1, DT: 1000, AMT: 100.0, CARD: 1, ADDR: 10, DEV: "X", FRAUD: 0},
            {TID: 2, DT: 1500, AMT: 200.0, CARD: 1, ADDR: 10, DEV: "Y", FRAUD: 0},
            {TID: 3, DT: 10000, AMT: 300.0, CARD: 1, ADDR: 20, DEV: "X", FRAUD: 1},
            {TID: 4, DT: 1200, AMT: 50.0, CARD: 2, ADDR: 99, DEV: np.nan, FRAUD: 0},
        ]
    )


def test_online_matches_offline_basic(timeline):
    _assert_parity(timeline)


def test_online_known_values(timeline):
    # A couple of concrete checkpoints (the rest is covered by parity).
    online = _run_online(timeline)
    assert online.loc[1, "card_txn_count_prior"] == 0
    assert math.isnan(online.loc[1, "time_since_last_txn"])
    assert online.loc[2, "time_since_last_txn"] == 500
    assert online.loc[2, "amt_vs_card_mean_ratio"] == pytest.approx(2.0)
    assert online.loc[3, "card_txn_count_1h"] == 0  # priors fell out of the 1h window
    assert online.loc[2, "new_device"] == 1
    assert online.loc[3, "new_device"] == 0
    assert math.isnan(online.loc[4, "new_device"])  # device missing -> unknown


def test_concurrent_same_second_transactions():
    # Two txns of the same card at the SAME TransactionDT: the windows must
    # exclude the concurrent sibling (closed="left"), but the later-id one still
    # counts the earlier as a prior txn (lifetime/prev/seen). This is the most
    # error-prone parity case, so assert offline==online handles it.
    df = pd.DataFrame(
        [
            {TID: 10, DT: 5000, AMT: 100.0, CARD: 7, ADDR: 1, DEV: "A", FRAUD: 0},
            {TID: 11, DT: 5000, AMT: 200.0, CARD: 7, ADDR: 1, DEV: "A", FRAUD: 0},
            {TID: 12, DT: 5100, AMT: 300.0, CARD: 7, ADDR: 2, DEV: "B", FRAUD: 0},
        ]
    )
    _assert_parity(df)
    online = _run_online(df)
    # txn 11 (same second as 10): window count excludes 10, but prior count is 1.
    assert online.loc[11, "card_txn_count_24h"] == 0
    assert online.loc[11, "card_txn_count_prior"] == 1
    assert online.loc[11, "time_since_last_txn"] == 0
    # txn 12 (100s later): both prior txns are within 24h and strictly earlier.
    assert online.loc[12, "card_txn_count_24h"] == 2


def test_bounded_state_evicts_old_events():
    # Events older than the largest window must be evicted (memory stays bounded).
    big = config.VELOCITY_WINDOWS_SECONDS["7d"]
    df = pd.DataFrame(
        [
            {TID: 1, DT: 0, AMT: 10.0, CARD: 1, ADDR: 1, DEV: "A", FRAUD: 0},
            {TID: 2, DT: big + 1000, AMT: 20.0, CARD: 1, ADDR: 1, DEV: "A", FRAUD: 0},
        ]
    )
    agg = OnlineCardAggregator()
    for rec in df.sort_values([DT, TID]).to_dict("records"):
        agg.update(rec)
    # After the second (far-future) txn, the first must have been evicted.
    state = agg._state["1"]
    assert len(state.events) == 1
    assert state.lifetime_count == 2  # lifetime counters are NOT affected by eviction


@pytest.mark.skipif(
    not config.TRAIN_TRANSACTION_CSV.exists(), reason="IEEE-CIS dataset not present"
)
def test_parity_on_real_data_slice():
    # The strongest check: real transactions (with real ties, gaps, and missing
    # devices) must produce identical online and offline features.
    from fraud_detection_mlops.data import load_training_data

    df = load_training_data()
    df = df.sort_values([DT, TID]).head(30000).reset_index(drop=True)
    _assert_parity(df)
