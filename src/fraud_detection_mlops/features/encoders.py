"""Leakage-safe categorical encoding for tree models.

High-cardinality categoricals (``card1``, ``addr1``, email domains, device
strings, …) carry strong fraud signal but blow up one-hot encoding. We use
**frequency encoding**: replace each category with how often it occurred *in the
training split*. It is:

  - leakage-safe — frequencies are learned in ``fit`` from training rows only,
    never from validation (invariant 1); applying them to later windows is a
    pure lookup;
  - robust to unseen categories — a level that only appears in the future maps to
    0.0 ("never seen in training"), which is itself meaningful;
  - NaN-aware — missing is treated as its own level, since *whether* a field is
    populated is often predictive here.

Trees split happily on the resulting frequency values, so we skip one-hot
entirely. The encoder is a standard sklearn transformer so it drops into a
``ColumnTransformer`` / ``Pipeline`` and serializes with the model (parity).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

_NAN_TOKEN = "__nan__"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace every input column's categories with their training frequency.

    Frequencies are proportions (count / n_train) in ``[0, 1]``. Categories not
    seen during ``fit`` (including those appearing only in a later validation
    window) map to ``unseen_value``. NaN is encoded as its own category, so a
    consistently-missing field gets a stable, informative frequency rather than
    being silently dropped.

    Args:
        unseen_value: Value for categories absent from the training data.
    """

    def __init__(self, unseen_value: float = 0.0):
        self.unseen_value = unseen_value

    def _as_keys(self, col: pd.Series) -> pd.Series:
        # NaN -> sentinel so it forms its own, countable category.
        return col.astype("object").where(col.notna(), _NAN_TOKEN)

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoder:
        X = pd.DataFrame(X)
        n = len(X)
        if n == 0:
            raise ValueError("FrequencyEncoder.fit received an empty frame.")
        self.columns_ = list(X.columns)
        # Frequencies come ONLY from the rows passed here (the training split) —
        # this is the entire leakage guarantee.
        self.frequencies_ = {
            col: (self._as_keys(X[col]).value_counts() / n).to_dict() for col in self.columns_
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = np.empty((len(X), len(self.columns_)), dtype="float32")
        for j, col in enumerate(self.columns_):
            mapped = self._as_keys(X[col]).map(self.frequencies_[col])
            out[:, j] = mapped.fillna(self.unseen_value).to_numpy(dtype="float32")
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray([f"{col}_freq" for col in self.columns_], dtype=object)
