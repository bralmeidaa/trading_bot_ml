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

# ~15 liquid majors on Binance spot. Adjust as needed.
DEFAULT_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "LTC/USDT", "ATOM/USDT", "UNI/USDT", "AAVE/USDT", "NEAR/USDT",
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
                use_cache: bool = True, verbose: bool = True
                ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (close, volume, kept_symbols) where close/volume are wide DataFrames
    indexed by timestamp (ms) with one column per symbol, aligned on the common
    timeline. Symbols with < min_coverage of the max bar count are dropped.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    exchange = ccxt.binance({"enableRateLimit": True, "rateLimit": 1200})

    closes: Dict[str, pd.Series] = {}
    vols: Dict[str, pd.Series] = {}
    for sym in symbols:
        if verbose:
            print(f"  loading {sym} {timeframe} {days}d...", end="", flush=True)
        df = load_symbol(exchange, sym, timeframe, days, use_cache=use_cache)
        if df.empty or len(df) < 100:
            if verbose:
                print(" skip (no data)")
            continue
        closes[sym] = df.set_index("timestamp")["close"]
        vols[sym] = df.set_index("timestamp")["volume"]
        if verbose:
            print(f" {len(df):,}")

    if not closes:
        return pd.DataFrame(), pd.DataFrame(), []

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(vols).sort_index()

    # Drop symbols with insufficient coverage, then forward-fill small gaps
    max_bars = close.notna().sum().max()
    keep = [c for c in close.columns if close[c].notna().sum() >= min_coverage * max_bars]
    close = close[keep].ffill().dropna(how="any")
    volume = volume[keep].reindex(close.index).ffill()

    if verbose:
        print(f"  panel: {close.shape[0]:,} bars × {close.shape[1]} symbols "
              f"(kept: {', '.join(keep)})")
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
