"""Leakage + gating tests for the feedback loop (M4).

The two things that must not break:
  1. At retrain clock T, training may use ONLY labels that have matured by T
     (TransactionDT + delay <= T) and must never touch the held-out validation
     window — using an unmatured label is peeking at the future (invariant 1/2).
  2. The gate promotes a challenger only when it genuinely beats the champion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection_mlops import config
from fraud_detection_mlops.pipelines import labels
from fraud_detection_mlops.pipelines.promote import should_promote

DT, TID, TARGET = config.TIME_COL, config.ID_COL, config.TARGET


def _features(n: int = 100) -> pd.DataFrame:
    # Evenly spaced transactions over a timeline; a dummy feature column.
    rng = np.random.default_rng(0)
    dt = np.arange(n) * 100  # 0,100,...,(n-1)*100
    return pd.DataFrame(
        {
            TID: np.arange(1, n + 1),
            DT: dt,
            TARGET: rng.integers(0, 2, n),
            "feat": rng.normal(size=n),
        }
    )


def test_join_back_includes_only_matured_labels():
    feats = _features()
    delay = 1000
    store = labels.label_store(feats, delay=delay)
    clock = 5000
    joined = labels.join_back(feats, store, clock)
    # Every joined row's label must have matured by the clock.
    assert (joined[DT] + delay <= clock).all()
    # No transaction whose label is still pending should appear.
    pending = feats[feats[DT] + delay > clock]
    assert not joined[TID].isin(pending[TID]).any()


def test_label_store_available_dt():
    feats = _features()
    store = labels.label_store(feats, delay=1000)
    # label_available_dt == TransactionDT + delay, aligned by TransactionID.
    merged = feats.merge(store, on=TID)
    assert (merged["label_available_dt"] == merged[DT] + 1000).all()


def test_build_training_data_no_future_label_leak():
    feats = _features(n=200)
    delay, val_fraction, mtf = 1000, 0.2, 0.7
    clock = 12000
    out = labels.build_training_data(
        feats, clock, delay=delay, val_fraction=val_fraction, model_train_fraction=mtf
    )
    v_start = out["val_start_dt"]
    train, calib, val = out["train_df"], out["calib_df"], out["val_df"]

    # 1. No training/calibration row uses an unmatured label.
    for part in (train, calib):
        assert (part[DT] + delay <= clock).all(), "unmatured label leaked into training"
    # 2. Training is strictly before the held-out validation window.
    assert train[DT].max() < v_start
    assert calib[DT].max() < v_start
    # 3. Validation is exactly the last val_fraction of the timeline.
    assert (val[DT] >= v_start).all()
    # 4. Time order within the split: train < calib.
    assert train[DT].max() <= calib[DT].min()


def test_validation_window_is_fixed_across_clocks():
    feats = _features(n=300)
    a = labels.build_training_data(feats, 8000, delay=1000)
    b = labels.build_training_data(feats, 16000, delay=1000)
    # The yardstick (validation window) must be identical regardless of the clock.
    assert a["val_start_dt"] == b["val_start_dt"]
    pd.testing.assert_frame_equal(a["val_df"], b["val_df"])
    # A later clock matures at least as many training labels.
    assert b["n_matured"] >= a["n_matured"]


def test_gate_promotes_only_on_improvement():
    # First champion: always promote.
    assert should_promote(0.30, None) is True
    # Strictly better -> promote.
    assert should_promote(0.41, 0.40) is True
    # Worse or equal -> reject (no noisy regressions).
    assert should_promote(0.39, 0.40) is False
    assert should_promote(0.40, 0.40) is False
    # Margin must be cleared.
    assert should_promote(0.405, 0.40, margin=0.01) is False
    assert should_promote(0.42, 0.40, margin=0.01) is True
