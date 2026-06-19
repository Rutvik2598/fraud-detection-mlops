"""Replay IEEE-CIS transactions onto a Redpanda topic in TransactionDT order.

This is the head of the online plane: it turns the static training table into a
stream, as if authorizations were arriving live. Transactions are emitted in
``TransactionDT`` order (the dataset's ordering key) and **keyed by card** so all
of a card's transactions land on one partition and are consumed in order — the
guarantee the per-card rolling aggregates rely on.

Replay speed is configurable: ``--speed`` is simulated TransactionDT-seconds per
real second (e.g. 3600 = one hour of history per second); ``--speed 0`` floods as
fast as possible. No labels are sent (a real transaction has no fraud label yet;
chargebacks are M4) and no scoring happens here (M3).

Run:  python -m fraud_detection_mlops.streaming.producer --limit 50000 --speed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time

from confluent_kafka import Producer

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features.online import card_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("producer")


def _clean(value: object) -> object:
    """JSON-safe scalar: NaN/NA -> None, numpy ints/floats -> Python scalars."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    # numpy scalar -> python scalar
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _delivery_report(err, msg) -> None:
    if err is not None:
        logger.error("Delivery failed for key=%s: %s", msg.key(), err)


def replay(
    *,
    bootstrap: str,
    topic: str,
    limit: int | None,
    speed: float,
    max_sleep: float,
    log_every: int,
    drift: bool = False,
    drift_after: int = 0,
) -> int:
    """Stream transactions to ``topic``; return the number of messages produced.

    With ``drift=True``, transactions sent after ``drift_after`` messages have their
    amount perturbed (scaled + shifted) to inject a covariate shift — the M5 decay
    scenario. Downstream the velocity features recompute from the shifted amounts,
    so the drift propagates exactly as a real population change would.
    """
    df = load_training_data()
    df = df.sort_values([config.TIME_COL, config.ID_COL]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    fields = [c for c in config.STREAM_FIELDS if c in df.columns]
    logger.info(
        "Replaying %d transactions onto '%s' (speed=%s, fields=%s)", len(df), topic, speed, fields
    )

    producer = Producer(
        {"bootstrap.servers": bootstrap, "linger.ms": 5, "enable.idempotence": True}
    )

    prev_dt: float | None = None
    n = 0
    t_start = time.time()
    for rec in df[fields].to_dict("records"):
        dt = float(rec[config.TIME_COL])
        # Pace to wall-clock by the simulated-seconds-per-second factor.
        if speed > 0 and prev_dt is not None:
            sleep_s = min((dt - prev_dt) / speed, max_sleep)
            if sleep_s > 0:
                time.sleep(sleep_s)
        prev_dt = dt

        message = {k: _clean(v) for k, v in rec.items()}
        if drift and n >= drift_after and message.get(config.AMOUNT_COL) is not None:
            message[config.AMOUNT_COL] = (
                message[config.AMOUNT_COL] * config.DRIFT_AMOUNT_MULTIPLIER
                + config.DRIFT_AMOUNT_SHIFT
            )
        key = card_key(rec)
        producer.produce(topic, key=key, value=json.dumps(message), on_delivery=_delivery_report)
        producer.poll(0)  # serve delivery callbacks without blocking
        n += 1
        if n % log_every == 0:
            tag = " [DRIFT]" if drift and n >= drift_after else ""
            logger.info("Produced %d (sim TransactionDT=%d)%s", n, int(dt), tag)

    producer.flush()
    elapsed = time.time() - t_start
    logger.info(
        "Done: produced %d messages in %.1fs (%.0f msg/s)", n, elapsed, n / max(elapsed, 1e-9)
    )
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Replay transactions onto Redpanda.")
    p.add_argument("--bootstrap", default=config.REDPANDA_BOOTSTRAP)
    p.add_argument("--topic", default=config.TRANSACTIONS_TOPIC)
    p.add_argument("--limit", type=int, default=None, help="Cap rows replayed (by time).")
    p.add_argument("--speed", type=float, default=config.REPLAY_SPEED,
                   help="Simulated TransactionDT-seconds per real second; 0 = as fast as possible.")
    p.add_argument("--max-sleep", type=float, default=config.REPLAY_MAX_SLEEP_SECONDS)
    p.add_argument("--log-every", type=int, default=5000)
    p.add_argument("--drift", action="store_true", help="Inject a covariate shift (M5 decay).")
    p.add_argument("--drift-after", type=int, default=0, help="Start drifting after N messages.")
    args = p.parse_args()
    replay(
        bootstrap=args.bootstrap,
        topic=args.topic,
        limit=args.limit,
        speed=args.speed,
        max_sleep=args.max_sleep,
        log_every=args.log_every,
        drift=args.drift,
        drift_after=args.drift_after,
    )


if __name__ == "__main__":
    main()
