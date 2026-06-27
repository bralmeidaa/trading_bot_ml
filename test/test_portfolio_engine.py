"""
Unit tests for the cross-sectional portfolio engine (paper trading).
Tests the pure logic (no network): weight computation, mark-to-market,
rebalance cost, kill-switch, market-neutrality.
"""
import numpy as np
import pandas as pd
import pytest

from backend.strategy.portfolio_engine import (
    CrossSectionalPortfolioEngine, PortfolioConfig,
)

SYMS = ["BTC/USDT"] + [f"C{i}" for i in range(7)]


def _engine(**kw):
    cfg = PortfolioConfig(universe=SYMS, lookback=10, rebalance_days=12, k=2,
                          btc_filter=False, total_capital=1000.0, **kw)
    return CrossSectionalPortfolioEngine(cfg)


def _panel(n=60, seed=1):
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.01, 0.01, len(SYMS))   # C-low worst, last best
    cols = {s: 100 * np.exp(np.cumsum(rng.normal(drifts[j], 0.01, n)))
            for j, s in enumerate(SYMS)}
    idx = np.arange(n) * 86400000
    close = pd.DataFrame(cols, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=SYMS)
    return close, vol


class TestComputeWeights:
    def test_market_neutral(self):
        eng = _engine()
        close, vol = _panel()
        w = eng.compute_weights(close, vol)
        assert w.sum() == pytest.approx(0.0, abs=1e-9)
        assert (w > 0).sum() == 2 and (w < 0).sum() == 2   # k=2 each side

    def test_regime_off_returns_flat(self):
        # btc_filter on; force BTC downtrend → flat
        cfg = PortfolioConfig(universe=SYMS, lookback=10, k=2, btc_filter=True,
                              total_capital=1000.0)
        eng = CrossSectionalPortfolioEngine(cfg)
        n = 60
        idx = np.arange(n) * 86400000
        cols = {s: np.linspace(100, 110, n) for s in SYMS}
        cols["BTC/USDT"] = np.linspace(110, 70, n)   # strong downtrend
        close = pd.DataFrame(cols, index=idx)
        vol = pd.DataFrame(1e6, index=idx, columns=SYMS)
        w = eng.compute_weights(close, vol)
        assert w.abs().sum() == pytest.approx(0.0, abs=1e-9)


class TestMarkToMarket:
    def test_no_change_first_mark(self):
        eng = _engine()
        eng.state.weights = {"C6": 0.5, "C0": -0.5}
        r = eng.mark_to_market({"C6": 100, "C0": 100})
        assert r == 0.0 and eng.state.equity == 1000.0   # first mark sets baseline

    def test_long_gain_short_flat(self):
        eng = _engine()
        eng.state.weights = {"C6": 0.5, "C0": -0.5}
        eng.mark_to_market({"C6": 100, "C0": 100})       # baseline
        r = eng.mark_to_market({"C6": 110, "C0": 100})   # long +10%, short flat
        assert r == pytest.approx(0.05)                  # 0.5*0.10
        assert eng.state.equity == pytest.approx(1050.0)

    def test_market_neutral_cancels(self):
        eng = _engine()
        eng.state.weights = {"C6": 0.5, "C0": -0.5}
        eng.mark_to_market({"C6": 100, "C0": 100})
        r = eng.mark_to_market({"C6": 110, "C0": 110})   # both +10% → neutral nets 0
        assert r == pytest.approx(0.0, abs=1e-12)


class TestRebalanceCost:
    def test_turnover_charges_cost(self):
        eng = _engine(cost_per_side=0.001)
        w = pd.Series({"C6": 0.5, "C5": 0.5, "C0": -0.5, "C1": -0.5})
        eq0 = eng.state.equity
        legs = eng.apply_rebalance(w)
        # turnover = sum|new-old| = 2.0 (from flat) → cost = 2.0*0.001
        assert eng.state.equity == pytest.approx(eq0 * (1 - 2.0 * 0.001))
        assert len(legs) == 4
        assert eng.state.rebalances == 1

    def test_no_turnover_no_cost(self):
        eng = _engine()
        w = pd.Series({"C6": 0.5, "C5": 0.5, "C0": -0.5, "C1": -0.5})
        eng.apply_rebalance(w)
        eq = eng.state.equity
        eng.apply_rebalance(w)   # same weights → zero turnover
        assert eng.state.equity == pytest.approx(eq)


class TestConfigSanity:
    def test_history_days_exceeds_build_panel_min_bars(self):
        """Regression: history_days must be > build_panel's default min_bars (200),
        else every symbol is dropped and the panel comes back empty (0 rebalances)."""
        from backend.data.universe import build_panel
        import inspect
        default_min_bars = inspect.signature(build_panel).parameters["min_bars"].default
        cfg = PortfolioConfig(universe=SYMS)
        assert cfg.history_days >= default_min_bars


class TestKillSwitch:
    def test_trips_on_drawdown(self):
        eng = _engine(max_drawdown_kill=0.20)
        eng.state.peak_equity = 1000.0
        eng.state.equity = 850.0     # -15%
        assert not eng.killswitch_tripped()
        eng.state.equity = 750.0     # -25%
        assert eng.killswitch_tripped()


class TestApiCompat:
    def test_pnl_and_positions(self):
        eng = _engine()
        w = pd.Series({"C6": 0.5, "C5": 0.5, "C0": -0.5, "C1": -0.5})
        eng.apply_rebalance(w)
        eng.state.equity = 1100.0
        assert eng.total_pnl == pytest.approx(100.0)
        pos = eng.positions()
        assert len(pos) == 4
        assert all("side" in p and "notional" in p for p in pos)
