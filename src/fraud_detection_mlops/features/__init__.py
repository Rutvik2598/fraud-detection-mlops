"""Feature engineering shared by offline training and (later) online serving.

Point-in-time velocity features, leakage-safe categorical encoding, and the
model feature-matrix assembly. These definitions are the single source of truth
for train/serve parity (invariant 5).
"""

from fraud_detection_mlops.features.build import build_preprocessor, select_model_columns
from fraud_detection_mlops.features.encoders import FrequencyEncoder
from fraud_detection_mlops.features.online import OnlineCardAggregator
from fraud_detection_mlops.features.velocity import VELOCITY_FEATURES, add_velocity_features

__all__ = [
    "add_velocity_features",
    "VELOCITY_FEATURES",
    "FrequencyEncoder",
    "select_model_columns",
    "build_preprocessor",
    "OnlineCardAggregator",
]
