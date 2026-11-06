"""
Tests for the intraday feature layer (backend/data/intraday_features.py).
Verifies VWAP correctness, session flags, opening-range timing, and the
no-look-ahead property (features at bar t use only data up to t).
"""
import numpy as np
import pandas as pd
import pytest

from backend.data.intraday_features import (
    add_session_features, add_vwap, add_opening_range, build_intraday_features,
)


def _intraday_df(days=2, bars_per_day=288, bar_min=5, seed=0):
    """Synthetic 5m OHLCV spanning a few UTC days, starting at 00:00 UTC."""
    rng = np.random.default_rng(seed)
    n = days * bars_per_day
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    ts = [(start + pd.Timedelta(minutes=bar_min * i)).value // 10**6 for i in range(n)]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


class TestSession:
    def test_hour_and_session_flags(self):
        df = add_session_features(_intraday_df())
        assert df["hour"].between(0, 23).all()
        # teatime flag only at hour 16
        assert (df.loc[df["hour"] == 16, "sess_teatime"] == 1).all()
        assert (df.loc[df["hour"] != 16, "sess_teatime"] == 0).all()

    def test_weekend_flag(self):
        df = add_session_features(_intraday_df(days=8))
        # 2024-01-06 is a Saturday → weekend rows exist
        assert df["is_weekend"].sum() > 0
        assert set(df["is_weekend"].unique()).issubset({0.0, 1.0})


class TestVWAP:
    def test_vwap_matches_manual_first_day(self):
        df = add_session_features(_intraday_df(days=1))
        df = add_vwap(df)
        # manual session VWAP at last bar of the day
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        manual = (tp * df["volume"]).sum() / df["volume"].sum()
        assert df["vwap"].iloc[-1] == pytest.approx(manual, rel=1e-9)

    def test_vwap_resets_each_day(self):
        df = add_session_features(_intraday_df(days=2))
        df = add_vwap(df)
        # first bar of day-2: VWAP equals that bar's typical price (fresh session)
        day2_start = df[df["_date"] == sorted(df["_date"].unique())[1]].index[0]
        tp0 = (df["high"] + df["low"] + df["close"]).iloc[day2_start] / 3.0
        assert df["vwap"].iloc[day2_start] == pytest.approx(tp0, rel=1e-6)


class TestOpeningRange:
    def test_or_nan_during_range_then_set(self):
        df = add_session_features(_intraday_df(days=1))
        df = add_opening_range(df, or_minutes=30, bar_minutes=5)  # 6 bars
        # first 6 bars of the session have NaN OR (range still forming)
        assert df["or_high"].iloc[:6].isna().all()
        # after that it's populated and constant within the day
        post = df["or_high"].iloc[6:].dropna()
        assert len(post) > 0 and post.nunique() == 1

    def test_or_high_is_range_max(self):
        df = add_session_features(_intraday_df(days=1))
        df = add_opening_range(df, or_minutes=30, bar_minutes=5)
        expected = df["high"].iloc[:6].max()
        assert df["or_high"].iloc[6] == pytest.approx(expected)


class TestNoLookAhead:
    def test_features_causal(self):
        """Truncating the future must not change features at an earlier bar."""
        full = build_intraday_features(_intraday_df(days=2), or_minutes=30, bar_minutes=5)
        cut = 200
        trunc = build_intraday_features(_intraday_df(days=2).iloc[:cut].copy(),
                                        or_minutes=30, bar_minutes=5)
        for col in ("vwap", "vwap_dev", "trend_up", "atr", "or_high", "above_vwap"):
            a = full[col].iloc[cut - 1]
            b = trunc[col].iloc[cut - 1]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b, rel=1e-6, nan_ok=False), f"{col} leaks future"
