"""Drift-detection tests (M5).

No drift -> no alert; an injected covariate shift -> detected. Plus the pure
trigger logic and the injection helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection_mlops import config
from fraud_detection_mlops.monitoring import drift


def _frame(n: int, *, amt_loc: float, ratio_loc: float, pred_a: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            config.AMOUNT_COL: rng.normal(amt_loc, 20, n),
            "amt_vs_card_mean_ratio": rng.normal(ratio_loc, 0.5, n),
            "time_since_last_txn": rng.exponential(1000, n),
            "card_txn_count_24h": rng.poisson(2, n).astype(float),
            "new_location": rng.integers(0, 2, n).astype(float),
            "new_device": rng.integers(0, 2, n).astype(float),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "DeviceType": rng.choice(["desktop", "mobile"], n),
            drift.PREDICTION_COL: rng.beta(pred_a, 20, n),
        }
    )


def test_no_drift_no_alert():
    ref = _frame(1500, amt_loc=100, ratio_loc=1.0, pred_a=2, seed=1)
    cur = _frame(1500, amt_loc=100, ratio_loc=1.0, pred_a=2, seed=2)
    summary = drift.drift_report(ref, cur)
    assert not summary["dataset_drift"]
    assert not summary["prediction_drift"]
    assert not drift.detect_drift(summary)


def test_injected_drift_is_detected():
    ref = _frame(1500, amt_loc=100, ratio_loc=1.0, pred_a=2, seed=1)
    # Strong covariate shift in amount + ratio, and a shifted score distribution.
    cur = _frame(1500, amt_loc=400, ratio_loc=4.0, pred_a=8, seed=2)
    summary = drift.drift_report(ref, cur)
    assert summary["prediction_drift"]
    assert drift.detect_drift(summary)


def test_detect_drift_logic():
    base = {"dataset_drift": False, "prediction_drift": False, "feature_drift_share": 0.0}
    assert not drift.detect_drift(base)
    assert drift.detect_drift({**base, "dataset_drift": True})
    assert drift.detect_drift({**base, "prediction_drift": True})


def test_inject_drift_scales_amount():
    df = pd.DataFrame({config.AMOUNT_COL: [100.0, 200.0], "card_amt_sum_24h": [50.0, 80.0]})
    out = drift.inject_drift_features(df)
    expected = 100.0 * config.DRIFT_AMOUNT_MULTIPLIER + config.DRIFT_AMOUNT_SHIFT
    assert out[config.AMOUNT_COL].iloc[0] == expected
    assert out["card_amt_sum_24h"].iloc[0] == 50.0 * config.DRIFT_AMOUNT_MULTIPLIER
