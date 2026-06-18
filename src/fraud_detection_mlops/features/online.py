"""Online (streaming) computation of the per-card velocity features.

This is the **serve-side twin** of ``velocity.py``. The offline module computes
the same features in one vectorized pass over the whole history; this one
maintains per-card state and updates it one transaction at a time, the way a
streaming consumer must. Invariant 5 (train/serve parity) demands the two agree
*exactly*, so ``tests/test_online_aggregator.py`` replays data through this
aggregator and asserts the output equals ``add_velocity_features`` row-for-row.

Point-in-time correctness (invariant 1) falls out of the update order: when a
transaction arrives we compute its features from the state accumulated by
**strictly-earlier** transactions, and only *afterwards* fold the new
transaction into the state. The window aggregates use a half-open ``[t-W, t)``
interval — the current instant is excluded — mirroring the offline
``rolling(..., closed="left")``. Concurrent same-second transactions therefore
don't see each other in the windows (but a same-second earlier-arriving txn
still counts toward lifetime count / previous-txn / seen-sets, matching the
offline sort by ``(card, time, TransactionID)``).

State is per card and bounded: the event deque only retains transactions inside
the largest window, so memory is O(active cards x txns-per-largest-window), not
O(history). This is what makes online feature serving viable at scale (M3).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

from fraud_detection_mlops import config
from fraud_detection_mlops.features.velocity import _NAN_TOKEN, VELOCITY_FEATURES

__all__ = ["OnlineCardAggregator", "VELOCITY_FEATURES"]


def _is_missing(value: object) -> bool:
    """True for None / NaN — the same notion of "missing" the offline code uses."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def card_key(row: Mapping[str, object], card_cols: tuple[str, ...] = config.CARD_ID_COLS) -> str:
    """Build the card key for one transaction, identically to the offline join.

    Mirrors ``velocity._build_card_key``: missing parts become a sentinel, parts
    join with ``|``. Keeping this in lock-step is what makes a card group the same
    set of transactions online and offline.
    """
    parts = []
    for col in card_cols:
        value = row.get(col)
        parts.append(_NAN_TOKEN if _is_missing(value) else str(value))
    return "|".join(parts)


class _CardState:
    """Mutable rolling state for a single card."""

    __slots__ = ("events", "lifetime_count", "lifetime_sum", "last_dt", "seen_loc", "seen_dev")

    def __init__(self) -> None:
        self.events: deque[tuple[float, float]] = deque()  # (TransactionDT, amount), in time order
        self.lifetime_count: int = 0
        self.lifetime_sum: float = 0.0
        self.last_dt: float | None = None
        self.seen_loc: set[object] = set()
        self.seen_dev: set[object] = set()


class OnlineCardAggregator:
    """Incremental per-card velocity features matching the offline definitions.

    Call :meth:`update` once per transaction, in arrival (time) order. It returns
    the feature dict for that transaction (computed from prior transactions only)
    and then advances the card's state. Windows and the card entity come from the
    same ``config`` constants the offline path uses, so the two cannot drift.
    """

    def __init__(
        self,
        *,
        windows: dict[str, int] | None = None,
        card_cols: tuple[str, ...] = config.CARD_ID_COLS,
        time_col: str = config.TIME_COL,
        amount_col: str = config.AMOUNT_COL,
        location_col: str = config.NEW_LOCATION_COL,
        device_col: str = config.NEW_DEVICE_COL,
    ) -> None:
        self.windows = dict(windows if windows is not None else config.VELOCITY_WINDOWS_SECONDS)
        self.max_window = max(self.windows.values())
        self.card_cols = card_cols
        self.time_col = time_col
        self.amount_col = amount_col
        self.location_col = location_col
        self.device_col = device_col
        self._state: dict[str, _CardState] = {}

    @property
    def n_cards(self) -> int:
        return len(self._state)

    def update(self, row: Mapping[str, object]) -> dict[str, float]:
        """Compute features for ``row`` (from prior txns), then fold it into state."""
        key = card_key(row, self.card_cols)
        state = self._state.get(key)
        if state is None:
            state = _CardState()
            self._state[key] = state

        dt = float(row[self.time_col])
        amount = float(row[self.amount_col])
        location = row.get(self.location_col)
        device = row.get(self.device_col)

        feats = self._compute(state, dt, amount, location, device)
        self._advance(state, dt, amount, location, device)
        return feats

    def _compute(
        self, state: _CardState, dt: float, amount: float, location: object, device: object
    ) -> dict[str, float]:
        prior_count = state.lifetime_count
        feats: dict[str, float] = {}

        feats["card_txn_count_prior"] = float(prior_count)
        last = state.last_dt
        feats["time_since_last_txn"] = float(dt - last) if last is not None else math.nan

        if prior_count > 0:
            prior_mean = state.lifetime_sum / prior_count
            feats["card_amt_mean_prior"] = prior_mean
            feats["amt_vs_card_mean_ratio"] = amount / prior_mean
        else:
            feats["card_amt_mean_prior"] = math.nan
            feats["amt_vs_card_mean_ratio"] = math.nan

        # Windowed counts/sums over [dt - W, dt): strictly-earlier txns only.
        for name, width in self.windows.items():
            lower = dt - width
            count = 0
            total = 0.0
            # events are time-ordered; iterate from the most recent backwards and
            # stop once we fall before the window's lower bound.
            for ev_dt, ev_amt in reversed(state.events):
                if ev_dt >= dt:  # exclude the current instant (closed="left")
                    continue
                if ev_dt < lower:  # left edge is inclusive; earlier -> out of window
                    break
                count += 1
                total += ev_amt
            feats[f"card_txn_count_{name}"] = float(count)
            feats[f"card_amt_sum_{name}"] = float(total)

        feats["new_location"] = self._new_flag(location, state.seen_loc)
        feats["new_device"] = self._new_flag(device, state.seen_dev)
        return feats

    @staticmethod
    def _new_flag(value: object, seen: set[object]) -> float:
        if _is_missing(value):
            return math.nan  # missing region/device is "unknown", not "new"
        return 0.0 if value in seen else 1.0

    def _advance(
        self, state: _CardState, dt: float, amount: float, location: object, device: object
    ) -> None:
        state.events.append((dt, amount))
        # Evict events older than the largest window relative to *this* txn; they
        # can never fall inside a window for this or any later (>= dt) txn.
        cutoff = dt - self.max_window
        while state.events and state.events[0][0] < cutoff:
            state.events.popleft()
        state.lifetime_count += 1
        state.lifetime_sum += amount
        state.last_dt = dt
        if not _is_missing(location):
            state.seen_loc.add(location)
        if not _is_missing(device):
            state.seen_dev.add(device)
