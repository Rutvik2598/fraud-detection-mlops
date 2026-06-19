"""Feature engineering shared by offline training and online serving.

Point-in-time velocity features, fraud-ring graph features, leakage-safe
categorical encoding, and model feature-matrix assembly. These definitions are
the single source of truth that keeps training and serving in step.
"""

from fraud_detection_mlops.features.build import build_preprocessor, select_model_columns
from fraud_detection_mlops.features.encoders import FrequencyEncoder
from fraud_detection_mlops.features.graph import GRAPH_FEATURES, add_graph_features
from fraud_detection_mlops.features.online import OnlineCardAggregator
from fraud_detection_mlops.features.velocity import VELOCITY_FEATURES, add_velocity_features

__all__ = [
    "add_velocity_features",
    "VELOCITY_FEATURES",
    "add_graph_features",
    "GRAPH_FEATURES",
    "FrequencyEncoder",
    "select_model_columns",
    "build_preprocessor",
    "OnlineCardAggregator",
]
