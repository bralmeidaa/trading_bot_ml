"""ATR-based default bracket for agent orders, with a small per-symbol cache
so the LLM turn doesn't refetch candles on every tool call."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 300


def atr_pct_from_ohlcv(rows: List[list], period: int = 14) -> Optional[float]:
    """ATR as a fraction of the last close, from ccxt OHLCV rows
    [ts, open, high, low, close, volume]. Same true-range definition as
    backend/data/intraday_features.add_trend_structure."""
    if not rows or len(rows) < 2:
        return None
    trs = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        high, low, prev_close = float(cur[2]), float(cur[3]), float(prev[4])
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    window = trs[-period:]
    last_close = float(rows[-1][4])
    if not window or last_close <= 0:
        return None
    return (sum(window) / len(window)) / last_close


class AtrCache:
    def __init__(self, exchange, timeframe: str = "15m", period: int = 14):
        self.exchange = exchange
        self.timeframe = timeframe
        self.period = period
        self._cache: Dict[str, Tuple[float, Optional[float]]] = {}

    def atr_pct(self, symbol: str) -> Optional[float]:
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < _CACHE_TTL_SEC:
            return cached[1]
        try:
            rows = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.period + 20)
            value = atr_pct_from_ohlcv(rows, self.period)
        except Exception as exc:
            logger.warning(f"ATR fetch failed for {symbol}: {exc}")
            value = None
        self._cache[symbol] = (time.time(), value)
        return value


def default_bracket(atr_pct: Optional[float], stop_mult: float, target_mult: float,
                    min_stop_pct: float, max_stop_pct: float) -> Dict[str, float]:
    """Stop = stop_mult×ATR clamped to [min, max]; target keeps the same
    reward/risk ratio as target_mult/stop_mult. Without ATR, the minimum stop."""
    stop = min_stop_pct if atr_pct is None else min(max(stop_mult * atr_pct, min_stop_pct), max_stop_pct)
    return {"stop_pct": stop, "target_pct": stop * (target_mult / stop_mult)}
