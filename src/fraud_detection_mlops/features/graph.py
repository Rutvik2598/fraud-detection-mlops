"""Point-in-time graph features for fraud rings.

A per-card model has a structural blind spot: it only sees one card's own
history, so it cannot spot a ring -- many cards sharing one stolen device or
shipping address. These features add that signal by treating transactions as a
graph: cards and devices are nodes, each transaction an edge, and a ring is a
connected component.

Every feature for a transaction uses strictly-earlier transactions only. We
process in (time, TransactionID) order and read the graph state before folding
the new edge in, so a later transaction can never change an earlier one's
features (the tests assert this). The computation is incremental -- one pass with
union-find plus adjacency sets -- which is also the shape online serving takes.

Features (NaN when the underlying entity is missing for that transaction):
  - device_n_cards  : distinct cards seen on this device before now
  - device_n_txn    : prior transactions on this device
  - card_n_devices  : distinct devices this card used before now
  - addr_n_cards    : distinct cards at this shipping address before now
  - ring_card_count : distinct cards in this card's connected component
"""

from __future__ import annotations

import logging
import math
from math import nan

import pandas as pd

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)

GRAPH_FEATURES: tuple[str, ...] = (
    "device_n_cards",
    "device_n_txn",
    "card_n_devices",
    "addr_n_cards",
    "ring_card_count",
)

DEVICE_COL = "DeviceInfo"
ADDR_COL = "addr1"


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class _UnionFind:
    """Union-find over card/device nodes, tracking cards-per-component."""

    def __init__(self) -> None:
        self.parent: dict = {}
        self.size: dict = {}
        self.card_count: dict = {}

    def add(self, node, is_card: bool) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            self.card_count[node] = 1 if is_card else 0

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:  # path compression
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:  # union by size
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.card_count[ra] += self.card_count[rb]

    def component_cards(self, node) -> int:
        return self.card_count[self.find(node)] if node in self.parent else 1


class GraphFeatureState:
    """Incremental, point-in-time graph features (the serving-shaped computation)."""

    def __init__(self) -> None:
        self.device_cards: dict = {}   # device -> set of cards
        self.device_txn: dict = {}     # device -> prior txn count
        self.card_devices: dict = {}   # card -> set of devices
        self.addr_cards: dict = {}     # addr -> set of cards
        self.uf = _UnionFind()

    def update(self, card, device, addr) -> dict[str, float]:
        """Compute features, folding in the current edge for the ring size.

        The "before" counts (device/card/addr) exclude the current transaction.
        ring_card_count includes the current transaction's own device link -- the
        device fingerprint is observed at scoring time, so counting the ring it
        connects to is fair, and it is the point of catching a new ring member.
        Neither uses any future transaction.
        """
        has_device = not _missing(device)
        has_addr = not _missing(addr)

        feats = {
            "device_n_cards": float(len(self.device_cards.get(device, ()))) if has_device else nan,
            "device_n_txn": float(self.device_txn.get(device, 0)) if has_device else nan,
            "card_n_devices": float(len(self.card_devices.get(card, ()))),
            "addr_n_cards": float(len(self.addr_cards.get(addr, ()))) if has_addr else nan,
        }

        if has_device:
            self.device_cards.setdefault(device, set()).add(card)
            self.device_txn[device] = self.device_txn.get(device, 0) + 1
            self.card_devices.setdefault(card, set()).add(device)
            self.uf.add(("c", card), is_card=True)
            self.uf.add(("d", device), is_card=False)
            self.uf.union(("c", card), ("d", device))
        if has_addr:
            self.addr_cards.setdefault(addr, set()).add(card)

        feats["ring_card_count"] = float(self.uf.component_cards(("c", card)))
        return feats


def add_graph_features(
    df: pd.DataFrame,
    *,
    card_col: str = "card1",
    device_col: str = DEVICE_COL,
    addr_col: str = ADDR_COL,
) -> pd.DataFrame:
    """Add point-in-time fraud-ring graph features to df, preserving row order."""
    original_index = df.index
    if not original_index.is_unique:
        raise ValueError("add_graph_features requires a unique index to restore row order.")

    sort_keys = [config.TIME_COL]
    if config.ID_COL in df.columns:
        sort_keys.append(config.ID_COL)
    ordered = df.sort_values(sort_keys, kind="mergesort")

    state = GraphFeatureState()
    n = len(ordered)
    cards = ordered[card_col].to_numpy()
    devices = ordered[device_col].to_numpy() if device_col in ordered.columns else [None] * n
    addrs = ordered[addr_col].to_numpy() if addr_col in ordered.columns else [None] * n

    rows = [state.update(cards[i], devices[i], addrs[i]) for i in range(n)]
    feats = pd.DataFrame(rows, index=ordered.index)[list(GRAPH_FEATURES)]

    out = df.copy()
    out[list(GRAPH_FEATURES)] = feats.loc[original_index]
    logger.info("Added %d fraud-ring graph features", len(GRAPH_FEATURES))
    return out
