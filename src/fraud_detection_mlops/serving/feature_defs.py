"""Feast feature definitions — the feature layer training and serving share.

What lives behind Feast:
  - ``card`` entity (join key ``card_id``).
  - ``card_state`` feature view: the per-card rolling **state** (lifetime
    count/sum, last transaction time, the recent event window as parallel
    arrays, and the locations/devices seen). Offline source = parquet; this is
    what gets materialized into the online store (Redis in prod, sqlite in dev).
  - ``txn_request``: the live transaction fields that arrive with a scoring
    request (time, amount, region, device).
  - ``velocity_features`` on-demand feature view: combines the looked-up
    ``card_state`` with the request and emits the 12 velocity features by calling
    :func:`compute_velocity_features` — the *same* function the streaming
    aggregator uses. One definition, used offline (``get_historical_features``)
    and online (``get_online_features``): train/serve parity by construction
    (invariant 5).

The parquet path is taken from config (env-overridable) so a test harness can
point the same definitions at a throwaway store.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from feast import Entity, FeatureView, Field, FileSource, RequestSource, ValueType
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Array, Float64, Int64, String

from fraud_detection_mlops import config
from fraud_detection_mlops.features.online import VELOCITY_FEATURES, compute_velocity_features

card = Entity(name="card", join_keys=["card_id"], value_type=ValueType.STRING)

card_state_source = FileSource(
    name="card_state_source",
    path=str(config.CARD_STATE_PARQUET),
    timestamp_field="event_timestamp",
)

card_state = FeatureView(
    name="card_state",
    entities=[card],
    # Long TTL: a card's state stays valid until its next transaction updates it.
    ttl=timedelta(days=3650),
    source=card_state_source,
    online=True,
    schema=[
        Field(name="last_dt", dtype=Int64),
        Field(name="lifetime_count", dtype=Int64),
        Field(name="lifetime_sum", dtype=Float64),
        Field(name="event_dts", dtype=Array(Int64)),
        Field(name="event_amts", dtype=Array(Float64)),
        Field(name="seen_loc", dtype=Array(Float64)),
        Field(name="seen_dev", dtype=Array(String)),
    ],
)

txn_request = RequestSource(
    name="txn_request",
    schema=[
        Field(name=config.TIME_COL, dtype=Int64),
        Field(name=config.AMOUNT_COL, dtype=Float64),
        Field(name=config.NEW_LOCATION_COL, dtype=Float64),
        Field(name=config.NEW_DEVICE_COL, dtype=String),
    ],
)

_VELOCITY_SCHEMA = [Field(name=f, dtype=Float64) for f in VELOCITY_FEATURES]


def _as_list(value) -> list:
    """Array features come back as numpy arrays (or None for a cold card)."""
    if value is None:
        return []
    return list(value)


def _as_int(value, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return int(value)


def velocity_udf(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12 velocity features per row via the shared definition.

    Kept as a plain (undecorated) function so it can be unit-tested directly with
    a synthetic joined frame, without standing up a Feast store.
    """
    windows = config.VELOCITY_WINDOWS_SECONDS
    out: dict[str, list[float]] = {f: [] for f in VELOCITY_FEATURES}
    for _, r in inputs.iterrows():
        last_dt = r["last_dt"]
        last_dt = None if (last_dt is None or pd.isna(last_dt)) else int(last_dt)
        events = list(zip(_as_list(r["event_dts"]), _as_list(r["event_amts"]), strict=False))
        feats = compute_velocity_features(
            dt=float(r[config.TIME_COL]),
            amount=float(r[config.AMOUNT_COL]),
            location=r[config.NEW_LOCATION_COL],
            device=r[config.NEW_DEVICE_COL],
            last_dt=last_dt,
            lifetime_count=_as_int(r["lifetime_count"]),
            lifetime_sum=float(r["lifetime_sum"]) if not pd.isna(r["lifetime_sum"]) else 0.0,
            events=events,
            seen_loc=set(_as_list(r["seen_loc"])),
            seen_dev=set(_as_list(r["seen_dev"])),
            windows=windows,
        )
        for f in VELOCITY_FEATURES:
            out[f].append(feats[f])
    return pd.DataFrame(out)


# The Feast on-demand feature view: same transform, registered behind Feast.
# name is explicit so it is "velocity_features" regardless of the udf's name.
velocity_features = on_demand_feature_view(
    name="velocity_features", sources=[card_state, txn_request], schema=_VELOCITY_SCHEMA
)(velocity_udf)
