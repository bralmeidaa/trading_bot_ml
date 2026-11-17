"""
Month-sharded OHLCV cache for the replay simulator.

One pickle per (symbol, timeframe, UTC month). A month that has fully ended is
fetched once and never again; the current month is topped up incrementally
from its last cached bar. Only closed bars are kept.

The directory comes from KLINES_CACHE_DIR (prod: a bind mount on the
persistent data volume, so redeploys don't re-download), defaulting to
backend/data/_cache/klines.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Iterator, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def cache_dir() -> str:
    return os.getenv("KLINES_CACHE_DIR") or os.path.join(os.path.dirname(__file__), "_cache", "klines")


def _month_start(ms: int) -> datetime:
    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return datetime(d.year, d.month, 1, tzinfo=timezone.utc)


def _next_month(d: datetime) -> datetime:
    return datetime(d.year + (d.month == 12), d.month % 12 + 1, 1, tzinfo=timezone.utc)


def months(start_ms: int, end_ms: int) -> Iterator[Tuple[int, int, str]]:
    """(month_start_ms, month_end_ms, 'YYYY-MM') covering [start_ms, end_ms)."""
    d = _month_start(start_ms)
    while int(d.timestamp() * 1000) < end_ms:
        nxt = _next_month(d)
        yield int(d.timestamp() * 1000), int(nxt.timestamp() * 1000), d.strftime("%Y-%m")
        d = nxt


def _path(root: str, symbol: str, timeframe: str, tag: str) -> str:
    return os.path.join(root, f"{symbol.replace('/', '')}_{timeframe}_{tag}.pkl")


def _fetch(exchange, symbol: str, timeframe: str, since: int, until: int,
           sleep: float = 0.2) -> pd.DataFrame:
    step = TF_MS[timeframe]
    out = []
    while since < until:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        out.extend(b for b in batch if b[0] < until)
        nxt = batch[-1][0] + step
        if nxt <= since or len(batch) < 1000:
            break
        since = nxt
        time.sleep(sleep)
    return pd.DataFrame(out, columns=COLUMNS)


def _clean(df: pd.DataFrame, now_ms: int, step: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df[df["timestamp"] + step <= now_ms]                     # closed bars only
    return (df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
            .reset_index(drop=True))


def load(exchange, symbol: str, timeframe: str, start_ms: int, end_ms: int,
         root: Optional[str] = None, now_ms: Optional[int] = None) -> pd.DataFrame:
    """Closed bars of `symbol` with open time in [start_ms, end_ms)."""
    root = root or cache_dir()
    os.makedirs(root, exist_ok=True)
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    step = TF_MS[timeframe]
    parts = []
    for m_start, m_end, tag in months(start_ms, min(end_ms, now_ms)):
        path = _path(root, symbol, timeframe, tag)
        cached = None
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
            except Exception as exc:
                logger.warning(f"klines cache unreadable ({path}): {exc}")
        complete = m_end <= now_ms
        if cached is not None and not cached.empty:
            last = int(cached["timestamp"].iloc[-1])
            if complete and last + step >= m_end:
                parts.append(cached)
                continue
            fresh = _fetch(exchange, symbol, timeframe, last, min(m_end, now_ms))
            df = _clean(pd.concat([cached, fresh]), now_ms, step)
        else:
            df = _clean(_fetch(exchange, symbol, timeframe, m_start, min(m_end, now_ms)), now_ms, step)
        if not df.empty:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(parts)
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
