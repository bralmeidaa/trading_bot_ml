"""
Multi-symbol data layer for cross-sectional strategies.

Builds an aligned panel (timestamp × symbol) of OHLCV for a basket of majors,
with per-symbol pickle cache so research iterations are instant.

Used by research_cross_sectional.py and (later) the live portfolio engine.
"""
from __future__ import annotations

import os
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

# ~15 liquid majors on Binance spot.
DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "LTC/USDT", "ATOM/USDT", "UNI/USDT", "AAVE/USDT", "NEAR/USDT",
]

# Expanded ~35-coin universe for survivorship-aware testing: includes many
# coins that crashed hard but still trade (dilutes the "only winners" bias).
# NOTE: fully delisted coins (LUNA, FTT) are gone from Binance's API — this
# reduces but does not fully eliminate survivorship bias.
EXPANDED_UNIVERSE = DEFAULT_UNIVERSE + [
    "DOGE/USDT", "TRX/USDT", "ETC/USDT", "FIL/USDT", "ALGO/USDT",
    "ICP/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT",
    "SAND/USDT", "MANA/USDT", "AXS/USDT", "GRT/USDT", "FTM/USDT",
    "XLM/USDT", "VET/USDT", "THETA/USDT", "RUNE/USDT", "SUSHI/USDT",
    "CRV/USDT", "COMP/USDT", "MKR/USDT", "SNX/USDT", "ZEC/USDT",
]

_TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
               "1h": 60, "2h": 120, "4h": 240, "1d": 1440}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")


def _cache_path(symbol: str, timeframe: str, days: int) -> str:
    safe = symbol.replace("/", "")
    return os.path.join(CACHE_DIR, f"{safe}_{timeframe}_{days}d.pkl")


def fetch_symbol_ohlcv(exchange, symbol: str, timeframe: str, days: int,
                       sleep: float = 0.25) -> pd.DataFrame:
    """Paginated OHLCV fetch for one symbol (same pattern as run_backtest.fetch_history)."""
    tf_min = _TF_MINUTES.get(timeframe, 5)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    out: list = []
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as exc:
            print(f"    {symbol}: fetch error {exc}")
            break
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][0] + tf_min * 60 * 1000
        if len(batch) < 1000:
            break
        time.sleep(sleep)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def load_symbol(exchange, symbol: str, timeframe: str, days: int,
                use_cache: bool = True) -> pd.DataFrame:
    """Load one symbol from cache or fetch+cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(symbol, timeframe, days)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    df = fetch_symbol_ohlcv(exchange, symbol, timeframe, days)
    if not df.empty:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    return df


def build_panel(symbols: Optional[List[str]] = None, timeframe: str = "1h",
                days: int = 365, min_coverage: float = 0.95,
                use_cache: bool = True, verbose: bool = True,
                point_in_time: bool = False, min_bars: int = 200
                ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (close, volume, kept_symbols), wide DataFrames indexed by timestamp.

    point_in_time=False (default): survivorship-enforcing — drops low-coverage
      symbols and keeps only the dense common timeline (all symbols present).

    point_in_time=True: survivorship-aware — keeps every symbol with >= min_bars
      of data, aligned on the UNION of timestamps. A symbol is NaN before its
      first listing and only forward-filled AFTER it starts trading (no backfill).
      The strategy must then build the eligible universe per-bar from non-NaN
      columns + liquidity, mimicking what was actually tradeable at each date.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    exchange = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})

    closes: Dict[str, pd.Series] = {}
    vols: Dict[str, pd.Series] = {}
    for sym in symbols:
        if verbose:
            print(f"  loading {sym} {timeframe} {days}d...", end="", flush=True)
        df = load_symbol(exchange, sym, timeframe, days, use_cache=use_cache)
        if df.empty or len(df) < min_bars:
            if verbose:
                print(" skip (insufficient data)")
            continue
        closes[sym] = df.set_index("timestamp")["close"]
        vols[sym] = df.set_index("timestamp")["volume"]
        if verbose:
            print(f" {len(df):,}")

    if not closes:
        return pd.DataFrame(), pd.DataFrame(), []

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(vols).sort_index()

    if not point_in_time:
        # Survivorship-enforcing: dense common timeline
        max_bars = close.notna().sum().max()
        keep = [c for c in close.columns if close[c].notna().sum() >= min_coverage * max_bars]
        close = close[keep].ffill().dropna(how="any")
        volume = volume[keep].reindex(close.index).ffill()
    else:
        # Survivorship-aware: keep NaN before listing, ffill only after first valid
        keep = list(close.columns)
        close = close.ffill()       # forward-fill after first valid; NaN stays before listing
        volume = volume.reindex(close.index)
        # drop rows where even BTC isn't present yet (need a base timeline)
        close = close.dropna(how="all")
        volume = volume.reindex(close.index)

    if verbose:
        mode = "PIT" if point_in_time else "dense"
        print(f"  panel ({mode}): {close.shape[0]:,} bars × {close.shape[1]} symbols")
    return close, volume, keep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    c, v, k = build_panel(timeframe=args.timeframe, days=args.days)
    print(f"\nPanel ready: {c.shape[0]} bars × {len(k)} symbols")
    print(f"Date range: {pd.to_datetime(c.index[0], unit='ms')} → "
          f"{pd.to_datetime(c.index[-1], unit='ms')}")
