"""Parity test for the Feast on-demand transform (invariant 5).

The ODFV that serves velocity features online must reproduce the offline batch
definition. We build a card's stored state with the same snapshot the feature
store materializes, feed a "joined" row (state + live request) through the ODFV's
transform, and assert it equals ``add_velocity_features`` for that transaction —
including the array round-trip shape, missing-value handling, and cold start.
This runs without standing up Feast or Redis.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection_mlops.features import VELOCITY_FEATURES, add_velocity_features
from fraud_detection_mlops.features.online import OnlineCardAggregator
from fraud_detection_mlops.serving.feature_defs import velocity_udf

DT, AMT, CARD, ADDR, DEV, TID = (
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "DeviceInfo",
    "TransactionID",
)


def _state_row_for_card(prior: pd.DataFrame) -> dict:
    """Build the materialized state row (Feast schema) from a card's prior txns."""
    agg = OnlineCardAggregator()
    for rec in prior.sort_values([DT, TID]).to_dict("records"):
        agg.ingest(rec)
    snap = agg.snapshot()
    if not snap:  # cold start: no prior state
        return {
            "last_dt": None, "lifetime_count": None, "lifetime_sum": np.nan,
            "event_dts": None, "event_amts": None, "seen_loc": None, "seen_dev": None,
        }
    s = snap[0]
    # Arrays arrive from the online store as numpy arrays — emulate that.
    return {
        "last_dt": s["last_dt"], "lifetime_count": s["lifetime_count"],
        "lifetime_sum": s["lifetime_sum"],
        "event_dts": np.array(s["event_dts"]), "event_amts": np.array(s["event_amts"]),
        "seen_loc": np.array(s["seen_loc"]), "seen_dev": np.array(s["seen_dev"], dtype=object),
    }


def _odfv_features(prior: pd.DataFrame, current: dict) -> dict:
    row = _state_row_for_card(prior)
    row.update({DT: current[DT], AMT: current[AMT], ADDR: current[ADDR], DEV: current[DEV]})
    out = velocity_udf(pd.DataFrame([row]))
    return {f: float(out[f].iloc[0]) for f in VELOCITY_FEATURES}


def _assert_matches_offline(all_txns: pd.DataFrame, current_tid: int) -> None:
    offline = add_velocity_features(all_txns).set_index(TID)
    current = all_txns[all_txns[TID] == current_tid].iloc[0].to_dict()
    ck = current[CARD]
    prior = all_txns[(all_txns[CARD] == ck) & (all_txns[DT] < current[DT])]
    online = _odfv_features(prior, current)
    for f in VELOCITY_FEATURES:
        o, v = float(offline.loc[current_tid, f]), online[f]
        if math.isnan(o) or math.isnan(v):
            assert math.isnan(o) and math.isnan(v), f"NaN mismatch {f}: online={v} offline={o}"
        else:
            assert v == pytest.approx(o, rel=1e-6, abs=1e-6), f"{f}: online={v} offline={o}"


@pytest.fixture
def txns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {TID: 1, DT: 1000, AMT: 100.0, CARD: 1, ADDR: 10.0, DEV: "X"},
            {TID: 2, DT: 1500, AMT: 200.0, CARD: 1, ADDR: 10.0, DEV: "Y"},
            {TID: 3, DT: 90000, AMT: 300.0, CARD: 1, ADDR: 20.0, DEV: "X"},
            {TID: 4, DT: 1200, AMT: 50.0, CARD: 2, ADDR: 99.0, DEV: np.nan},
        ]
    )


def test_odfv_matches_offline_with_history(txns):
    # txn 3 has two priors; windows must drop the old ones, new_location must fire.
    _assert_matches_offline(txns, current_tid=3)


def test_odfv_matches_offline_second_txn(txns):
    _assert_matches_offline(txns, current_tid=2)


def test_odfv_cold_start_matches_offline(txns):
    # A card's very first transaction: no stored state at all.
    _assert_matches_offline(txns, current_tid=1)


def test_odfv_missing_device_is_unknown(txns):
    # txn 4's device is missing -> new_device must be NaN (not 1.0).
    online = _odfv_features(txns[txns[CARD] == 2].iloc[:0], txns[txns[TID] == 4].iloc[0].to_dict())
    assert math.isnan(online["new_device"])
    assert online["new_location"] == 1.0
