"""Deliberately simple baseline: logistic regression with basic preprocessing.

M0 is about correct methodology, not performance. This baseline sets an honest,
beatable floor that the gradient-boosted model in M1 must clearly beat.

Methodology guarantees:
  - All preprocessing (impute / scale / one-hot) lives inside a single sklearn
    Pipeline. Fitting the pipeline on the *train* split only means the imputer
    medians, scaler statistics, and encoder categories are learned from past
    data alone — no leakage from the validation window (invariant 1).
  - Imbalance is handled with class weighting, not resampling (invariant 6).
  - The model exposes ``predict_proba`` — scores, not hard 0.5 labels
    (invariant 4). Proper calibration and cost-based thresholds come in M1.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the frame's columns into (numeric_features, categorical_features).

    Excludes the id, label, and raw-time columns (see ``NON_FEATURE_COLS`` for
    why time is excluded). Categorical features are the curated low-cardinality
    set from config; everything else numeric (incl. ``has_identity`` and numeric
    id_* columns) is treated as numeric. Columns absent from ``df`` are skipped.
    """
    categorical = [c for c in config.BASELINE_CATEGORICAL_COLS if c in df.columns]
    cat_set = set(categorical)

    numeric: list[str] = []
    for col in df.columns:
        if col in config.NON_FEATURE_COLS or col in cat_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        # Non-numeric columns outside the curated categorical set (e.g.
        # high-cardinality DeviceInfo, emaildomains, id_30/31) are dropped from
        # the simple baseline by design — real encoding lands in M1.

    logger.info("Selected %d numeric + %d categorical features", len(numeric), len(categorical))
    return numeric, categorical


def build_baseline_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    seed: int = config.RANDOM_SEED,
) -> Pipeline:
    """Build the preprocessing + logistic-regression pipeline.

    - Numeric: median imputation (robust to the heavy, skewed missingness in the
      C/D/V blocks) + standardization (LogReg needs comparable scales).
    - Categorical: most-frequent imputation + one-hot encoding. ``min_frequency``
      folds rare levels into an "infrequent" bucket and ``handle_unknown="ignore"``
      tolerates categories seen only in the later validation window — both keep
      the encoder honest across the time split.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # numeric block dominates; keep output dense for LogReg
    )

    # class_weight="balanced" => imbalance handled by reweighting, not resampling.
    # Default penalty is L2; we leave it at the default (sklearn >=1.8 deprecates
    # the explicit penalty= arg). lbfgs is single-threaded, so no n_jobs.
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])
