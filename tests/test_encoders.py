"""Leakage tests for FrequencyEncoder (invariant 1).

Frequencies must come only from the rows seen in ``fit`` (the training split).
Categories appearing only later must map to 0.0, and NaN must be its own level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection_mlops.features import FrequencyEncoder


def test_frequencies_are_proportions_from_fit_only():
    train = pd.DataFrame({"c": ["a", "a", "b", "b"]})  # a:0.5, b:0.5
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out.shape == (2, 1)
    np.testing.assert_allclose(out[:, 0], [0.5, 0.5])


def test_unseen_category_maps_to_zero():
    train = pd.DataFrame({"c": ["a", "a", "a", "b"]})  # a:0.75, b:0.25
    enc = FrequencyEncoder().fit(train)
    # "z" never appears in training (e.g. a category only in the future window).
    out = enc.transform(pd.DataFrame({"c": ["a", "z"]}))
    np.testing.assert_allclose(out[:, 0], [0.75, 0.0])


def test_validation_categories_do_not_leak_into_fit():
    train = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    enc = FrequencyEncoder().fit(train)
    # Even if "b" is abundant at transform time, its frequency stays the TRAIN value.
    val = pd.DataFrame({"c": ["b"] * 100})
    out = enc.transform(val)
    assert np.allclose(out[:, 0], 0.5)


def test_nan_is_its_own_category():
    train = pd.DataFrame({"c": ["a", "a", np.nan, np.nan]})  # a:0.5, nan:0.5
    enc = FrequencyEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": [np.nan, "a"]}))
    np.testing.assert_allclose(out[:, 0], [0.5, 0.5])


def test_feature_names_out():
    enc = FrequencyEncoder().fit(pd.DataFrame({"card1": [1, 2], "addr1": [3, 4]}))
    assert list(enc.get_feature_names_out()) == ["card1_freq", "addr1_freq"]
