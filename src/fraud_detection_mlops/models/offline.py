"""Reusable offline model fitting (shared by M1 training and M4 retraining).

One definition of "how we fit the calibrated fraud model": frequency-encode +
pass-through preprocessing, class-imbalance via ``scale_pos_weight`` (invariant 6),
XGBoost, then isotonic calibration on a held-out slice (invariant 4). The M1
training script and the M4 retraining flow both call ``fit_calibrated_model`` so
the model the feedback loop produces is identical in construction to the original.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_detection_mlops import config
from fraud_detection_mlops.features import build_preprocessor, select_model_columns
from fraud_detection_mlops.models.calibrate import calibrate_classifier

logger = logging.getLogger(__name__)

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
)


def fit_calibrated_model(
    train_df: pd.DataFrame, calib_df: pd.DataFrame, *, seed: int = config.RANDOM_SEED
) -> dict:
    """Fit preprocessing + XGBoost on train, isotonic-calibrate on calib.

    Returns a dict with the fitted inference ``pipeline`` (preprocess -> calibrated
    model), the raw ``booster``, the chosen ``scale_pos_weight``, and the resolved
    feature column lists.
    """
    numeric, categorical = select_model_columns(train_df)
    model_cols = numeric + categorical
    preprocess = build_preprocessor(numeric, categorical)

    y_train = train_df[config.TARGET].to_numpy()
    y_calib = calib_df[config.TARGET].to_numpy()
    Xtr = preprocess.fit_transform(train_df[model_cols], y_train)
    Xcal = preprocess.transform(calib_df[model_cols])

    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / n_pos
    booster = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight, random_state=seed)
    booster.fit(Xtr, y_train)

    calibrated = calibrate_classifier(booster, Xcal, y_calib, method="isotonic")
    pipeline = Pipeline([("preprocess", preprocess), ("calibrated", calibrated)])

    logger.info(
        "Fitted calibrated model: %d train, %d calib, %d features, scale_pos_weight=%.2f",
        len(train_df), len(calib_df), len(model_cols), scale_pos_weight,
    )
    return {
        "pipeline": pipeline,
        "booster": booster,
        "preprocess": preprocess,
        "scale_pos_weight": scale_pos_weight,
        "numeric": numeric,
        "categorical": categorical,
        "model_cols": model_cols,
    }


def score_frame(pipeline: Pipeline, df: pd.DataFrame, model_cols: list[str]) -> np.ndarray:
    """Calibrated fraud probabilities for ``df`` (positional-class column)."""
    return pipeline.predict_proba(df[model_cols])[:, 1]
