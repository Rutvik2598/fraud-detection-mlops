"""End-to-end streaming check: produce a slice, consume it, verify parity.

Proves two things at once against a live Redpanda:
  1. transactions flow end-to-end (producer -> topic -> consumer), and
  2. the rolling aggregates the consumer computes match the offline batch
     definitions exactly (invariant 5) — i.e. they update correctly.

It uses a throwaway topic so it never disturbs the main stream, replays the
earliest ``--limit`` transactions, consumes them back, rebuilds the per-card
features from the consumed messages, and compares to ``add_velocity_features``
on the same rows. Exits non-zero on any mismatch.

Run (Redpanda must be up):  python -m fraud_detection_mlops.streaming.verify --limit 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features import (
    VELOCITY_FEATURES,
    OnlineCardAggregator,
    add_velocity_features,
)
from fraud_detection_mlops.features.online import card_key
from fraud_detection_mlops.streaming.producer import _clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify")


def _broker_reachable(bootstrap: str) -> bool:
    try:
        AdminClient({"bootstrap.servers": bootstrap}).list_topics(timeout=5)
        return True
    except Exception as exc:  # noqa: BLE001 — any failure means "can't reach broker"
        logger.error("Cannot reach Redpanda at %s: %s", bootstrap, exc)
        logger.error("Start it first:  docker compose up -d   (or: make redpanda-up)")
        return False


def run(*, bootstrap: str, topic: str, limit: int) -> bool:
    if not _broker_reachable(bootstrap):
        return False

    admin = AdminClient({"bootstrap.servers": bootstrap})
    verify_topic = f"{topic}-verify-{int(time.time())}"
    admin.create_topics([NewTopic(verify_topic, num_partitions=config.TRANSACTIONS_PARTITIONS)])
    time.sleep(1.0)  # let the topic propagate
    logger.info("Using throwaway topic %s", verify_topic)

    try:
        df = load_training_data()
        df = df.sort_values([config.TIME_COL, config.ID_COL]).head(limit).reset_index(drop=True)
        fields = [c for c in config.STREAM_FIELDS if c in df.columns]
        offline = add_velocity_features(df).set_index(config.ID_COL)[list(VELOCITY_FEATURES)]

        # Produce the slice.
        producer = Producer({"bootstrap.servers": bootstrap})
        for rec in df[fields].to_dict("records"):
            message = json.dumps({k: _clean(v) for k, v in rec.items()})
            producer.produce(verify_topic, key=card_key(rec), value=message)
        producer.flush()
        logger.info("Produced %d transactions; consuming them back...", len(df))

        # Consume them back and rebuild features online.
        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": f"verify-{int(time.time())}",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([verify_topic])
        agg = OnlineCardAggregator()
        online: dict[int, dict[str, float]] = {}
        idle_deadline = time.time() + 20.0
        while len(online) < len(df) and time.time() < idle_deadline:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            event = json.loads(msg.value())
            online[event[config.ID_COL]] = agg.update(event)
            idle_deadline = time.time() + 20.0
        consumer.close()

        if len(online) != len(df):
            logger.error("Consumed %d/%d messages — flow incomplete.", len(online), len(df))
            return False
        logger.info(
            "Round-trip OK: %d in, %d out. Checking aggregate parity...", len(df), len(online)
        )

        mismatches = 0
        for tid in offline.index:
            on = online[tid]
            for feat in VELOCITY_FEATURES:
                off_v, on_v = float(offline.loc[tid, feat]), float(on[feat])
                both_nan = math.isnan(off_v) and math.isnan(on_v)
                close = abs(off_v - on_v) <= 1e-6 + 1e-6 * abs(off_v)
                if not (both_nan or close):
                    mismatches += 1
                    if mismatches <= 10:
                        logger.error(
                            "MISMATCH %s @ tid=%s: online=%s offline=%s", feat, tid, on_v, off_v
                        )
        if mismatches:
            logger.error("FAILED: %d feature mismatches.", mismatches)
            return False
        logger.info("PASS: streamed aggregates match offline definitions for all %d txns.", len(df))
        return True
    finally:
        admin.delete_topics([verify_topic])
        logger.info("Cleaned up topic %s", verify_topic)


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end streaming parity check.")
    p.add_argument("--bootstrap", default=config.REDPANDA_BOOTSTRAP)
    p.add_argument("--topic", default=config.TRANSACTIONS_TOPIC)
    p.add_argument("--limit", type=int, default=50000)
    args = p.parse_args()
    ok = run(bootstrap=args.bootstrap, topic=args.topic, limit=args.limit)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
