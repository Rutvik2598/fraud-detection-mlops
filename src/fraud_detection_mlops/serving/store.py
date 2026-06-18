"""Build, apply, materialize, and query the Feast feature store.

The card-state snapshot is built by replaying transactions up to a cutoff through
the streaming aggregator (state-only ``ingest``), exporting each card's state, and
writing it to the offline parquet source. ``feast apply`` registers the
definitions and ``materialize`` pushes the latest state per card into the online
store (Redis in prod, sqlite in dev/test). At serving time we look features up by
card key — never recomputing from history.

Online store selection is by ``FEAST_ONLINE_STORE`` (``redis`` | ``sqlite``) so the
exact same definitions and code run locally without Docker and against Redis in
the compose stack.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from feast import FeatureStore
from feast.repo_config import RepoConfig

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features.online import (
    VELOCITY_FEATURES,
    OnlineCardAggregator,
    card_key,
)

logger = logging.getLogger(__name__)

_DATA_DIR = config.FEAST_REPO_PATH / "data"


def get_store(online_store: str | None = None) -> FeatureStore:
    """Construct a FeatureStore programmatically (online store chosen by env)."""
    online_store = (online_store or config.FEAST_ONLINE_STORE).lower()
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if online_store == "redis":
        online_cfg: dict = {"type": "redis", "connection_string": config.REDIS_CONNECTION_STRING}
    elif online_store == "sqlite":
        online_cfg = {"type": "sqlite", "path": str(_DATA_DIR / "online.db")}
    else:
        raise ValueError(f"Unknown FEAST_ONLINE_STORE={online_store!r} (use redis|sqlite)")

    repo_config = RepoConfig(
        project="fraud_detection",
        provider="local",
        registry=str(_DATA_DIR / "registry.db"),
        online_store=online_cfg,
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
    )
    return FeatureStore(config=repo_config)


def apply_definitions(store: FeatureStore) -> None:
    """Register the entity / feature views / ODFV with the store's registry."""
    from fraud_detection_mlops.serving.feature_defs import (
        card,
        card_state,
        txn_request,
        velocity_features,
    )

    store.apply([card, card_state, txn_request, velocity_features])
    logger.info("Applied Feast definitions (online store=%s)", store.config.online_store.type)


def build_card_state_snapshot(
    cutoff_fraction: float = config.CARD_STATE_CUTOFF_FRACTION,
) -> tuple[pd.DataFrame, int]:
    """Replay transactions up to the cutoff and export each card's state.

    Returns (snapshot_df, cutoff_dt). The snapshot has one row per card with the
    state-after-all-pre-cutoff-transactions plus an ``event_timestamp`` (the card's
    last transaction time, mapped to a synthetic datetime).
    """
    df = load_training_data()
    df = df.sort_values([config.TIME_COL, config.ID_COL]).reset_index(drop=True)
    cutoff = int(df[config.TIME_COL].quantile(cutoff_fraction, interpolation="lower"))
    pre = df[df[config.TIME_COL] <= cutoff]
    logger.info(
        "Building card-state from %d pre-cutoff txns (cutoff TransactionDT=%d)", len(pre), cutoff
    )

    agg = OnlineCardAggregator()
    for rec in pre[list(config.STREAM_FIELDS)].to_dict("records"):
        agg.ingest(rec)

    snap = pd.DataFrame(agg.snapshot())
    snap["event_timestamp"] = snap["last_dt"].apply(
        lambda d: config.FEAST_BASE_DATETIME + timedelta(seconds=int(d))
    )
    logger.info("Snapshot: %d cards", len(snap))
    return snap, cutoff


def write_snapshot(snap: pd.DataFrame) -> None:
    config.CARD_STATE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(config.CARD_STATE_PARQUET, index=False)
    logger.info("Wrote card-state parquet -> %s", config.CARD_STATE_PARQUET)


def materialize(store: FeatureStore) -> None:
    """Push the latest state per card from parquet into the online store."""
    start = config.FEAST_BASE_DATETIME - timedelta(days=1)
    end = datetime.now(UTC)
    store.materialize(start_date=start, end_date=end)
    logger.info("Materialized card_state into the online store.")


def setup_store(
    cutoff_fraction: float = config.CARD_STATE_CUTOFF_FRACTION,
    online_store: str | None = None,
) -> tuple[FeatureStore, int]:
    """End-to-end: build snapshot -> write parquet -> apply -> materialize."""
    snap, cutoff = build_card_state_snapshot(cutoff_fraction)
    write_snapshot(snap)
    store = get_store(online_store)
    apply_definitions(store)
    materialize(store)
    return store, cutoff


def _nan_if_missing(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    return float(value)


def online_velocity_features(store: FeatureStore, txn: dict) -> dict[str, float]:
    """Look up + compute the 12 velocity features for one transaction by card key."""
    device = txn.get(config.NEW_DEVICE_COL)
    entity_row = {
        "card_id": card_key(txn),
        config.TIME_COL: int(txn[config.TIME_COL]),
        config.AMOUNT_COL: float(txn[config.AMOUNT_COL]),
        config.NEW_LOCATION_COL: _nan_if_missing(txn.get(config.NEW_LOCATION_COL)),
        config.NEW_DEVICE_COL: None if device is None or pd.isna(device) else str(device),
    }
    resp = store.get_online_features(
        features=[f"velocity_features:{f}" for f in VELOCITY_FEATURES],
        entity_rows=[entity_row],
    ).to_dict()
    # Feast returns NaN feature values as None in to_dict(); restore NaN so a
    # missing-device/new-location flag stays "unknown" rather than crashing.
    return {
        f: (float("nan") if resp[f][0] is None else float(resp[f][0])) for f in VELOCITY_FEATURES
    }


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Build, apply, materialize the Feast feature store.")
    p.add_argument("--cutoff-fraction", type=float, default=config.CARD_STATE_CUTOFF_FRACTION)
    p.add_argument("--online-store", default=None, help="redis|sqlite (default: env)")
    args = p.parse_args()
    _, cutoff = setup_store(cutoff_fraction=args.cutoff_fraction, online_store=args.online_store)
    logger.info("Feature store ready (cutoff TransactionDT=%d).", cutoff)


if __name__ == "__main__":
    main()
