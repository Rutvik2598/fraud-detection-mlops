"""Feast object definitions for the `feast` CLI.

The real definitions live in the package so application code and the CLI share
one source of truth. `feast apply` discovers the objects re-exported here.
"""

from fraud_detection_mlops.serving.feature_defs import (  # noqa: F401
    card,
    card_state,
    txn_request,
    velocity_features,
)
