from typing import Optional
import pandas as pd
import pandas_ta as ta
try:
    from .cache import cache_indicators
except ImportError:
    # Fallback if cache not available
    def cache_indicators(ttl=3600):
        def decorator(func):
            return func
        return decorator


@cache_indicators(ttl=3600)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ['ts','open','high','low','close','volume']
    Returns a new DataFrame with technical indicators appended.
    """
    if df.empty:
        return df

    df = df.copy()
    # Build a separate datetime column to avoid index name collisions with 'ts'
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("dt", inplace=True)
    # Basic indicators
    df["rsi14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
    df["ema20"] = ta.ema(df["close"], length=20)
    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Bring back datetime as column while preserving existing 'ts'
    df.reset_index(drop=False, inplace=True)
    return df
