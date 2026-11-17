"""Causal features on 5m bars for the hunters. Every value at row t uses only
bars that had closed by the end of bar t. Reuses the daily-anchored VWAP and
session flags from backend.data.intraday_features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.data.intraday_features import add_session_features, add_vwap

BAR_MIN = 5
SESSIONS = {"london": 7 * 60, "ny": 13 * 60 + 30}    # UTC open, minutes from midnight
OR_MINUTES = 30
SESSION_WINDOW_MIN = 180                              # breakouts only in the first 3h


def _atr15_pct(df: pd.DataFrame) -> pd.Series:
    """ATR(14) of 15m bars as a fraction of price, mapped to each 5m bar using
    only 15m bars completed by that 5m bar's close."""
    s = df.set_index("_dt")[["open", "high", "low", "close"]]
    r = s.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev_close = r["close"].shift(1)
    tr = pd.concat([r["high"] - r["low"], (r["high"] - prev_close).abs(),
                    (r["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = pd.DataFrame({"available_at": r.index + pd.Timedelta(minutes=15),
                        "atr15_pct": (tr.rolling(14, min_periods=5).mean() / r["close"]).values})
    bars = pd.DataFrame({"close_at": df["_dt"] + pd.Timedelta(minutes=BAR_MIN), "pos": np.arange(len(df))})
    merged = pd.merge_asof(bars.sort_values("close_at"), atr.sort_values("available_at"),
                           left_on="close_at", right_on="available_at", direction="backward")
    return merged.sort_values("pos")["atr15_pct"].to_numpy()


def _prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("_date").agg(day_high=("high", "max"), day_low=("low", "min"),
                                    n=("high", "size"))
    full_day = daily["n"] >= (24 * 60 // BAR_MIN) * 0.9      # a truncated first day isn't a real prior day
    prev = daily[["day_high", "day_low"]].where(full_day).shift(1)
    return df["_date"].map(prev["day_high"]), df["_date"].map(prev["day_low"])


def _session_ranges(df: pd.DataFrame):
    minute = df["_dt"].dt.hour * 60 + df["_dt"].dt.minute
    for name, start in SESSIONS.items():
        in_or = (minute >= start) & (minute < start + OR_MINUTES)
        grp = df[in_or].groupby("_date").agg(hi=("high", "max"), lo=("low", "min"), n=("high", "size"))
        grp = grp[grp["n"] == OR_MINUTES // BAR_MIN]
        formed = minute >= start + OR_MINUTES          # the range exists only once it has closed
        df[f"or_high_{name}"] = df["_date"].map(grp["hi"]).where(formed)
        df[f"or_low_{name}"] = df["_date"].map(grp["lo"]).where(formed)
        df[f"in_window_{name}"] = ((minute >= start + OR_MINUTES)
                                   & (minute < start + SESSION_WINDOW_MIN)
                                   & df[f"or_high_{name}"].notna())


def prepare(bars: pd.DataFrame) -> pd.DataFrame:
    """`bars`: columns timestamp (ms, bar open, UTC), open, high, low, close,
    volume — oldest first, closed bars only."""
    df = bars.reset_index(drop=True).copy()
    df = add_session_features(df)
    df = add_vwap(df)
    df["atr15_pct"] = _atr15_pct(df)
    df["pdh"], df["pdl"] = _prior_day_levels(df)
    _session_ranges(df)
    return df
