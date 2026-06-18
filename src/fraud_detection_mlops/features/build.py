"""Assemble the M1 model feature matrix.

Given a frame that already has the engineered velocity features
(``add_velocity_features``), this module decides which columns are model inputs
and how each is treated, then builds the preprocessing transformer:

  - **categorical** (object-dtype columns + ID-like numeric columns such as
    ``card1``/``addr1``): frequency-encoded (see ``FrequencyEncoder``);
  - **numeric** (everything else, including the velocity features, the C/D/V
    blocks, ``TransactionAmt``, ``has_identity``): passed through untouched so
    XGBoost can use its native NaN handling.

The raw id, label, and absolute-time columns are never features (``NON_FEATURE_COLS``).
The same ``ColumnTransformer`` is fit on training rows and reused for calibration
and validation, and it is serialized inside the registered model — one feature
definition for train and serve (invariant 5).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer

from fraud_detection_mlops import config
from fraud_detection_mlops.features.encoders import FrequencyEncoder

logger = logging.getLogger(__name__)


def select_model_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split ``df``'s columns into (numeric_passthrough, categorical_to_encode).

    Categorical = every object-dtype column plus the configured ID-like numeric
    columns present in the frame. Numeric = all remaining numeric columns. The
    id/label/raw-time columns (``NON_FEATURE_COLS``) are excluded from both.
    """
    non_feature = set(config.NON_FEATURE_COLS)
    id_like = {c for c in config.ID_LIKE_NUMERIC_COLS if c in df.columns}

    categorical: list[str] = []
    numeric: list[str] = []
    for col in df.columns:
        if col in non_feature:
            continue
        if col in id_like or not pd.api.types.is_numeric_dtype(df[col]):
            categorical.append(col)
        else:
            numeric.append(col)

    logger.info(
        "Model columns: %d numeric (passthrough) + %d categorical (freq-encoded)",
        len(numeric),
        len(categorical),
    )
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build the preprocessing ``ColumnTransformer`` for the XGBoost model.

    Numeric columns pass through (NaN preserved for XGBoost); categorical columns
    are frequency-encoded. ``remainder="drop"`` ensures nothing un-selected (e.g.
    a stray helper column) leaks into the model matrix.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("freq", FrequencyEncoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
