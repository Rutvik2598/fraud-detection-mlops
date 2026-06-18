"""Streaming backbone (M2): replay producer + feature-update consumer.

The transport layer of the online plane. The producer replays transactions onto
Redpanda in ``TransactionDT`` order; the consumer keeps the per-card rolling
aggregates current as they flow. No scoring yet — that is M3.
"""
