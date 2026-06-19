"""Leakage and correctness tests for the fraud-ring graph features.

A ring is many cards sharing one device. We assert the structural features grow
as the ring forms, that they are strictly point-in-time (a later transaction
never changes an earlier one's features), and that missing entities are NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_detection_mlops.features import GRAPH_FEATURES, add_graph_features

DT, CARD, DEV, ADDR, TID = "TransactionDT", "card1", "DeviceInfo", "addr1", "TransactionID"


def _by_tid(df: pd.DataFrame) -> dict:
    out = add_graph_features(df).set_index(TID)
    return {tid: out.loc[tid] for tid in out.index}


@pytest.fixture
def ring() -> pd.DataFrame:
    # Cards 1, 2, 3 all transact on the same device X over time -> a ring forms.
    # Card 9 is a loner on device Z.
    return pd.DataFrame(
        [
            {TID: 1, DT: 100, CARD: 1, DEV: "X", ADDR: 10},
            {TID: 2, DT: 200, CARD: 2, DEV: "X", ADDR: 10},
            {TID: 3, DT: 300, CARD: 3, DEV: "X", ADDR: 20},
            {TID: 4, DT: 400, CARD: 1, DEV: "X", ADDR: 10},
            {TID: 5, DT: 150, CARD: 9, DEV: "Z", ADDR: 99},
        ]
    )


def test_device_sharing_grows(ring):
    r = _by_tid(ring)
    assert r[1]["device_n_cards"] == 0
    assert r[2]["device_n_cards"] == 1
    assert r[3]["device_n_cards"] == 2
    assert r[1]["device_n_txn"] == 0
    assert r[4]["device_n_txn"] == 3
    assert r[5]["device_n_cards"] == 0  # loner device Z


def test_ring_component_grows(ring):
    r = _by_tid(ring)
    assert r[1]["ring_card_count"] == 1  # card 1 alone
    assert r[2]["ring_card_count"] == 2  # card 2 joins card 1 via device X
    assert r[3]["ring_card_count"] == 3  # card 3 joins the ring
    assert r[5]["ring_card_count"] == 1  # loner card 9


def test_addr_sharing(ring):
    r = _by_tid(ring)
    assert r[1]["addr_n_cards"] == 0
    assert r[2]["addr_n_cards"] == 1
    assert r[4]["addr_n_cards"] == 2  # cards 1 and 2 seen at addr 10 before


def test_missing_device_is_nan():
    df = pd.DataFrame(
        [
            {TID: 1, DT: 100, CARD: 1, DEV: np.nan, ADDR: 10},
            {TID: 2, DT: 200, CARD: 2, DEV: "X", ADDR: np.nan},
        ]
    )
    r = _by_tid(df)
    assert math.isnan(r[1]["device_n_cards"])
    assert math.isnan(r[1]["device_n_txn"])
    assert math.isnan(r[2]["addr_n_cards"])
    assert r[1]["card_n_devices"] == 0  # never NaN: a card with no device yet has 0


def test_future_does_not_change_past(ring):
    full = _by_tid(ring)

    without_last = _by_tid(ring[ring[TID] != 4].copy())
    for tid in (1, 2, 3, 5):
        for f in GRAPH_FEATURES:
            a, b = full[tid][f], without_last[tid][f]
            assert (math.isnan(a) and math.isnan(b)) or a == b, f"{f} changed at tid={tid}"

    later = pd.DataFrame([{TID: 6, DT: 999, CARD: 4, DEV: "X", ADDR: 10}])
    extended = _by_tid(pd.concat([ring, later], ignore_index=True))
    for tid in (1, 2, 3, 4, 5):
        for f in GRAPH_FEATURES:
            a, b = full[tid][f], extended[tid][f]
            assert (math.isnan(a) and math.isnan(b)) or a == b, f"{f} changed at tid={tid}"


def test_row_order_preserved(ring):
    shuffled = ring.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = add_graph_features(shuffled)
    assert (out[TID].to_numpy() == shuffled[TID].to_numpy()).all()
