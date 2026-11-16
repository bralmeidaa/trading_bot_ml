"""
Event detectors that wake the agent between scheduled LLM cycles. Pure
functions over stream state; the engine decides debounce and budgets.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class WakeEvent:
    kind: str           # "price_move" | "liquidation_burst" | "fresh_news" | "near_stop"
    symbol: Optional[str]
    detail: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.symbol or '*'}"


def price_move(symbol: str, closes: List[float], last_price: Optional[float],
               atr_pct: Optional[float], window_min: int, atr_mult: float) -> Optional[WakeEvent]:
    """Price moved ≥ atr_mult × ATR within the last `window_min` minutes."""
    if last_price is None or not atr_pct or len(closes) < window_min:
        return None
    ref = closes[-window_min]
    move = (last_price - ref) / ref
    if abs(move) < atr_mult * atr_pct:
        return None
    return WakeEvent("price_move", symbol,
                     f"{symbol} moved {move:+.2%} in {window_min} min "
                     f"({abs(move) / atr_pct:.1f}× ATR15m)")


def liquidation_burst(symbol: str, buckets: List[float], burst_mult: float, min_usd: float,
                      min_history: int = 12) -> Optional[WakeEvent]:
    """Current bucket ≥ burst_mult × median of previous buckets and ≥ min_usd.
    Needs `min_history` past buckets so a cold start can't fire on noise."""
    if len(buckets) < min_history + 1:
        return None
    current, history = buckets[-1], buckets[:-1]
    baseline = statistics.median(history)
    if current < min_usd or current < burst_mult * max(baseline, 1.0):
        return None
    return WakeEvent("liquidation_burst", symbol,
                     f"${current / 1e6:.2f}M liquidated on {symbol} in the last bucket "
                     f"(median ${baseline / 1e6:.2f}M)")


def near_stop(symbol: str, direction: int, entry: float, stop: float, price: Optional[float],
              fraction_of_r: float) -> Optional[WakeEvent]:
    """Price within `fraction_of_r` of the initial risk distance from the stop."""
    if price is None or not stop:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    distance = (price - stop) * direction
    if distance > fraction_of_r * risk:
        return None
    return WakeEvent("near_stop", symbol,
                     f"{symbol} at {price:.4f} is {max(distance, 0) / risk:.2f}R from its stop {stop:.4f}")


def fresh_news(items: List[dict], seen: set) -> List[WakeEvent]:
    """News items whose key isn't in `seen` (mutates `seen`)."""
    events = []
    for item in items:
        key = item.get("url") or item.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        syms = item.get("symbols_mentioned") or []
        events.append(WakeEvent("fresh_news", syms[0] if len(syms) == 1 else None,
                                f"new headline: {item.get('title', '')[:140]}"))
    return events
