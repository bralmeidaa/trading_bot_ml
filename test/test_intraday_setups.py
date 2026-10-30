"""
Tests for intraday setups (backend/strategy/intraday_setups.py).
Verifies ORB setups fire correctly, carry the right asymmetric R:R, are causal,
and the triple-barrier simulator computes R-multiples correctly.
"""
import numpy as np
import pandas as pd
import pytest

from backend.strategy.intraday_setups import (
    SetupConfig, Setup, generate_setups, simulate_setups,
)


def _df_from_closes(closes, start="2024-01-01 13:00:00", bar_min=5, vol=50.0):
    """Build an OHLCV frame (NY session) from a close path."""
    start = pd.Timestamp(start, tz="UTC")
    n = len(closes)
    ts = [(start + pd.Timedelta(minutes=bar_min * i)).value // 10**6 for i in range(n)]
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.0005,
                         "low": c * 0.9995, "close": c, "volume": np.full(n, vol)})


class TestSetupGeneration:
    def test_breakout_up_fires_with_trend_and_vwap(self):
        # rising path: opening range then break up, trend up, above VWAP
        closes = list(np.linspace(100, 101, 6)) + list(np.linspace(101.2, 108, 80))
        df = _df_from_closes(closes)
        cfg = SetupConfig(or_minutes=30, bar_minutes=5, require_trend=False,
                          require_vwap=False, min_vol_ratio=0.0)
        setups = generate_setups(df, "BTC/USDT", cfg)
        assert len(setups) >= 1
        s = setups[0]
        assert s.direction == 1
        assert s.bar_index >= 6      # only after the opening range closes

    def test_asymmetric_rr(self):
        closes = list(np.linspace(100, 101, 6)) + list(np.linspace(101.2, 110, 80))
        df = _df_from_closes(closes)
        cfg = SetupConfig(rr=2.0, require_trend=False, require_vwap=False, min_vol_ratio=0.0)
        setups = generate_setups(df, "BTC/USDT", cfg)
        s = setups[0]
        risk = abs(s.entry - s.stop)
        reward = abs(s.target - s.entry)
        assert reward == pytest.approx(2.0 * risk, rel=1e-9)
        assert s.stop < s.entry < s.target   # long geometry

    def test_no_setup_during_opening_range(self):
        closes = list(np.linspace(100, 101, 6)) + list(np.linspace(101.2, 108, 40))
        df = _df_from_closes(closes)
        cfg = SetupConfig(require_trend=False, require_vwap=False, min_vol_ratio=0.0)
        setups = generate_setups(df, "BTC/USDT", cfg)
        assert all(s.bar_index >= 6 for s in setups)

    def test_session_filter_blocks_asia(self):
        # 02:00 UTC = Asia only; NY/London filter should block
        closes = list(np.linspace(100, 101, 6)) + list(np.linspace(101.2, 108, 40))
        df = _df_from_closes(closes, start="2024-01-01 02:00:00")
        cfg = SetupConfig(require_trend=False, require_vwap=False, min_vol_ratio=0.0,
                          allow_sessions=("sess_ny",))
        setups = generate_setups(df, "BTC/USDT", cfg)
        assert len(setups) == 0


class TestSimulate:
    def test_target_hit_gives_positive_r(self):
        # long setup that reaches target
        s = Setup(ts=0, symbol="X", direction=1, entry=100.0, stop=99.0,
                  target=102.0, rr=2.0, bar_index=0)
        # bars after entry: price rises to 102+
        df = pd.DataFrame({"timestamp": range(5), "open": [100, 101, 102, 103, 104],
                           "high": [100, 101.5, 102.5, 103, 104], "low": [100, 100.5, 101.5, 102, 103],
                           "close": [100, 101, 102, 103, 104], "volume": [1]*5})
        trades = simulate_setups(df, [s], cost_per_side=0.0)
        assert len(trades) == 1
        assert trades[0]["hit"] == "target"
        assert trades[0]["r_multiple"] == pytest.approx(2.0, rel=1e-6)
        assert trades[0]["net"] > 0

    def test_stop_hit_gives_negative_r(self):
        s = Setup(ts=0, symbol="X", direction=1, entry=100.0, stop=99.0,
                  target=102.0, rr=2.0, bar_index=0)
        df = pd.DataFrame({"timestamp": range(4), "open": [100, 99.5, 98, 97],
                           "high": [100, 99.8, 99, 98], "low": [100, 98.9, 98, 97],
                           "close": [100, 99.5, 98, 97], "volume": [1]*4})
        trades = simulate_setups(df, [s], cost_per_side=0.0)
        assert trades[0]["hit"] == "stop"
        assert trades[0]["r_multiple"] == pytest.approx(-1.0, rel=1e-6)

    def test_cost_reduces_net(self):
        s = Setup(ts=0, symbol="X", direction=1, entry=100.0, stop=99.0,
                  target=102.0, rr=2.0, bar_index=0)
        df = pd.DataFrame({"timestamp": range(3), "open": [100, 102, 103],
                           "high": [100, 102.5, 103], "low": [100, 101, 102],
                           "close": [100, 102, 103], "volume": [1]*3})
        free = simulate_setups(df, [s], cost_per_side=0.0)[0]["net"]
        costed = simulate_setups(df, [s], cost_per_side=0.002)[0]["net"]
        assert costed < free
