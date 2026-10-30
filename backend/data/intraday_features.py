"""
Intraday feature layer — everything derivable from OHLCV + timestamp.

Powers the ORB + VWAP + session + multi-timeframe trend strategy. No order book
needed (that's the parallel live-collection track). All features are causal
(use only data up to and including bar t — no look-ahead).

Reused by backend/strategy/intraday_setups.py and the research harness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Time / session ──────────────────────────────────────────────────────────
def add_session_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Hour-of-day (UTC), day-of-week and session flags from ms timestamps.

    Documented effects: activity/volatility peak ~16-17 UTC ("UK tea time"),
    NY session breakouts most reliable, weekend thinness.
    """
    dt = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek                 # 0=Mon
    df["is_weekend"] = (df["dow"] >= 5).astype(float)
    h = df["hour"]
    df["sess_asia"] = ((h >= 0) & (h < 8)).astype(float)
    df["sess_london"] = ((h >= 7) & (h < 16)).astype(float)
    df["sess_ny"] = ((h >= 13) & (h < 21)).astype(float)
    df["sess_teatime"] = ((h >= 16) & (h < 17)).astype(float)   # peak window
    df["_date"] = dt.dt.date
    df["_dt"] = dt
    return df


# ── VWAP (session-anchored) ─────────────────────────────────────────────────
def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Session-anchored VWAP (resets each UTC day) + deviation + bands.

    Requires add_session_features first (uses _date). Causal: cumulative within
    the day up to bar t.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["_date"]).cumsum()
    cum_v = df["volume"].groupby(df["_date"]).cumsum()
    df["vwap"] = (cum_pv / cum_v.replace(0, np.nan)).ffill()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]
    # rolling std of deviation for bands
    dev_std = df["vwap_dev"].rolling(20, min_periods=5).std()
    df["vwap_upper"] = df["vwap"] * (1 + 2 * dev_std)
    df["vwap_lower"] = df["vwap"] * (1 - 2 * dev_std)
    df["above_vwap"] = (df["close"] > df["vwap"]).astype(float)
    return df


# ── Opening range ───────────────────────────────────────────────────────────
def add_opening_range(df: pd.DataFrame, or_minutes: int = 30,
                      bar_minutes: int = 5) -> pd.DataFrame:
    """First-N-minutes high/low of each UTC session, broadcast forward.

    or_high/or_low are the opening-range bounds; only valid AFTER the range
    completes (NaN during the range → no signal then). Causal.
    """
    bars_in_or = max(1, or_minutes // bar_minutes)
    or_high = pd.Series(np.nan, index=df.index)
    or_low = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby("_date", sort=False).groups.items():
        rows = list(idx)
        if len(rows) <= bars_in_or:
            continue
        hi = df.loc[rows[:bars_in_or], "high"].max()
        lo = df.loc[rows[:bars_in_or], "low"].min()
        # valid only from the bar AFTER the opening range closes
        or_high.loc[rows[bars_in_or:]] = hi
        or_low.loc[rows[bars_in_or:]] = lo
    df["or_high"] = or_high
    df["or_low"] = or_low
    df["or_broke_up"] = (df["close"] > df["or_high"]).astype(float)
    df["or_broke_down"] = (df["close"] < df["or_low"]).astype(float)
    return df


# ── Trend / structure / volatility ──────────────────────────────────────────
def add_trend_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe-ish trend (EMAs), swing structure, ATR, volume confirm."""
    c = df["close"]
    df["ema_fast"] = c.ewm(span=20).mean()
    df["ema_slow"] = c.ewm(span=50).mean()
    df["ema_htf"] = c.ewm(span=200).mean()            # higher-timeframe proxy
    df["trend_up"] = ((df["ema_fast"] > df["ema_slow"]) & (c > df["ema_htf"])).astype(float)
    df["trend_dn"] = ((df["ema_fast"] < df["ema_slow"]) & (c < df["ema_htf"])).astype(float)

    # ATR
    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift(1)).abs()
    lc = (df["low"] - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=5).mean()
    df["atr_pct"] = df["atr"] / c

    # swing structure (causal): rolling max/min of prior highs/lows
    df["swing_high"] = df["high"].rolling(20).max().shift(1)
    df["swing_low"] = df["low"].rolling(20).min().shift(1)
    df["dist_swing_high"] = (df["swing_high"] - c) / c
    df["dist_swing_low"] = (c - df["swing_low"]) / c

    # volume confirmation
    df["vol_sma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma"].replace(0, np.nan)

    # realized vol
    df["ret1"] = c.pct_change()
    df["rvol"] = df["ret1"].rolling(20).std()
    return df


def build_intraday_features(df: pd.DataFrame, or_minutes: int = 30,
                            bar_minutes: int = 5) -> pd.DataFrame:
    """Full causal intraday feature set. Input: OHLCV with 'timestamp' (ms)."""
    df = df.copy().reset_index(drop=True)
    df = add_session_features(df)
    df = add_vwap(df)
    df = add_opening_range(df, or_minutes, bar_minutes)
    df = add_trend_structure(df)
    return df


# feature columns used by the meta-model (context at setup time)
CONTEXT_FEATURES = [
    "hour", "dow", "is_weekend", "sess_london", "sess_ny", "sess_teatime",
    "vwap_dev", "above_vwap", "trend_up", "trend_dn",
    "atr_pct", "rvol", "vol_ratio", "dist_swing_high", "dist_swing_low",
]
