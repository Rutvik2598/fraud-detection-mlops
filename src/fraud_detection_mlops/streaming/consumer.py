"""Consume the transactions stream and update per-card rolling features.

This is the feature-update worker of the online plane. For each transaction it
computes the point-in-time velocity features from the card's prior activity
(``OnlineCardAggregator``) and advances that card's state. In M2 it just logs /
records the features so we can verify they update correctly and match the offline
definitions; M3 writes them into the feature store and adds scoring.

There is deliberately **no model and no scoring here** (that is M3). This worker's
single job is keeping the rolling aggregates current as the stream flows.

Run:  python -m fraud_detection_mlops.streaming.consumer --output reports/stream_features.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from confluent_kafka import Consumer, KafkaError

from fraud_detection_mlops import config
from fraud_detection_mlops.features.online import OnlineCardAggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("consumer")


def consume(
    *,
    bootstrap: str,
    topic: str,
    group: str,
    output: Path | None,
    max_messages: int | None,
    idle_timeout: float,
    log_every: int,
) -> int:
    """Consume + update aggregates until idle or ``max_messages``; return count."""
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",  # replay from the start of the topic
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([topic])
    agg = OnlineCardAggregator()

    out = output.open("w") if output is not None else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing per-transaction features to %s", output)

    n = 0
    last_msg_time = time.time()
    logger.info("Consuming '%s' (group=%s); idle stop after %.1fs", topic, group, idle_timeout)
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                if time.time() - last_msg_time > idle_timeout:
                    logger.info("Idle for %.1fs — stopping.", idle_timeout)
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Consumer error: %s", msg.error())
                continue

            last_msg_time = time.time()
            event = json.loads(msg.value())
            feats = agg.update(event)
            n += 1
            if out is not None:
                out.write(json.dumps({config.ID_COL: event.get(config.ID_COL), **feats}) + "\n")
            if n % log_every == 0:
                logger.info(
                    "Consumed %d | cards tracked=%d | last %s=%s count_24h=%s",
                    n,
                    agg.n_cards,
                    config.ID_COL,
                    event.get(config.ID_COL),
                    feats["card_txn_count_24h"],
                )
            if max_messages is not None and n >= max_messages:
                logger.info("Reached max-messages=%d — stopping.", max_messages)
                break
    finally:
        consumer.close()
        if out is not None:
            out.close()

    logger.info("Done: consumed %d messages, tracked %d cards.", n, agg.n_cards)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Consume transactions and update rolling features.")
    p.add_argument("--bootstrap", default=config.REDPANDA_BOOTSTRAP)
    p.add_argument("--topic", default=config.TRANSACTIONS_TOPIC)
    p.add_argument("--group", default=config.FEATURE_CONSUMER_GROUP)
    p.add_argument("--output", type=Path, default=None, help="Optional JSONL of per-txn features.")
    p.add_argument("--max-messages", type=int, default=None)
    p.add_argument("--idle-timeout", type=float, default=10.0, help="Stop after N idle seconds.")
    p.add_argument("--log-every", type=int, default=5000)
    args = p.parse_args()
    consume(
        bootstrap=args.bootstrap,
        topic=args.topic,
        group=args.group,
        output=args.output,
        max_messages=args.max_messages,
        idle_timeout=args.idle_timeout,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
