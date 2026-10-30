"""
Unit tests for the funding-rate carry strategy core
(backend/strategy/funding_carry.py). Pure functions, no network.

Locks in the invariants:
  - never selects a symbol with funding <= min_funding (no forced "pay" legs)
  - top-m selection, equal weight, fewer-than-m eligible still fully funded
  - no look-ahead: weights decided at t pay off with funding fixed at t+1
  - cost monotonically reduces net return
  - a book that never holds anything earns exactly 0 (no spurious cost/P&L)
"""
import numpy as np
import pandas as pd
import pytest

from backend.strategy.funding_carry import select_top_m, simulate, metrics, walk_forward

SYMS = [f"C{i}" for i in range(5)]


class TestSelectTopM:
    def test_top_m_equal_weight(self):
        row = pd.Series({"A": 0.001, "B": 0.0005, "C": 0.0002, "D": -0.0001, "E": 0.0})
        w = select_top_m(row, m=2)
        assert w["A"] == pytest.approx(0.5) and w["B"] == pytest.approx(0.5)
        assert w["C"] == 0.0 and w["D"] == 0.0 and w["E"] == 0.0

    def test_never_selects_non_positive_funding(self):
        row = pd.Series({"A": -0.001, "B": 0.0, "C": -0.0005})
        w = select_top_m(row, m=3, min_funding=0.0)
        assert (w == 0).all()

    def test_fewer_than_m_eligible_still_fully_weighted(self):
        row = pd.Series({"A": 0.001, "B": -0.0002, "C": -0.0001})
        w = select_top_m(row, m=3)   # only A is eligible
        assert w["A"] == pytest.approx(1.0)
        assert w.sum() == pytest.approx(1.0)

    def test_nan_excluded(self):
        row = pd.Series({"A": 0.001, "B": np.nan, "C": 0.0005})
        w = select_top_m(row, m=2)
        assert w["B"] == 0.0
        assert w.sum() == pytest.approx(1.0)


class TestSimulate:
    def test_no_lookahead_payoff_uses_next_fixing(self):
        """Position selected from funding at t must be paid funding.iloc[t+1],
        not funding.iloc[t] — a flipping sign series makes this unambiguous."""
        idx = range(4)
        funding = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01]}, index=idx)
        port = simulate(funding, m=1, cost_per_side=0.0)
        # t=0: select A (funding[0]=+0.01>0) -> payoff at t=1 is funding[1]=-0.01
        assert port.iloc[1] == pytest.approx(-0.01)
        # t=1: funding[1]=-0.01 -> not eligible -> flat -> payoff at t=2 is 0
        assert port.iloc[2] == pytest.approx(0.0)

    def test_never_holding_earns_zero(self):
        funding = pd.DataFrame({"A": [-0.01] * 10, "B": [-0.02] * 10})
        port = simulate(funding, m=2, cost_per_side=0.0015)
        assert port.abs().sum() == pytest.approx(0.0)

    def test_higher_cost_reduces_net_when_rotating(self):
        rng = np.random.default_rng(0)
        funding = pd.DataFrame({s: rng.normal(0.0003, 0.0005, 60) for s in SYMS})
        lo = simulate(funding, m=2, cost_per_side=0.0005)
        hi = simulate(funding, m=2, cost_per_side=0.0100)
        assert hi.sum() < lo.sum()

    def test_constant_positive_funding_no_turnover_cost_after_first(self):
        """Stable top-m membership → only the initial entry pays cost."""
        funding = pd.DataFrame({"A": [0.001] * 20, "B": [0.0005] * 20, "C": [-0.001] * 20})
        port = simulate(funding, m=2, cost_per_side=0.0015)
        # steady state net payoff per period (after period 0) ~ (0.001+0.0005)/2
        assert port.iloc[5:].mean() == pytest.approx(0.00075, abs=1e-9)

    def test_empty_on_too_short_series(self):
        funding = pd.DataFrame({"A": [0.001, 0.002]})
        assert simulate(funding, m=1).empty

    def test_rebalance_holds_weights_between_reevaluations(self):
        """rebalance=3 must only re-pick the book every 3rd fixing, holding the
        prior weights (and paying no extra turnover cost) in between."""
        funding = pd.DataFrame({
            "A": [0.001, -0.001, 0.001, 0.001, -0.001, 0.001],
            "B": [-0.001, 0.001, -0.001, -0.001, 0.001, -0.001],
        })
        port = simulate(funding, m=1, cost_per_side=0.0015, rebalance=3)
        # at t=1,2 the book must still hold A (chosen at t=0), despite B looking
        # better at t=1 — so payoff at t=2 (from weights set at t=1==held-A) is
        # funding["A"][2] = +0.001, not funding["B"][2] = -0.001
        assert port.iloc[2] == pytest.approx(0.001)

    def test_rebalance_one_matches_default(self):
        rng = np.random.default_rng(2)
        funding = pd.DataFrame({s: rng.normal(0.0003, 0.0005, 50) for s in SYMS})
        a = simulate(funding, m=2, cost_per_side=0.0015)
        b = simulate(funding, m=2, cost_per_side=0.0015, rebalance=1)
        pd.testing.assert_series_equal(a, b)


class TestMetrics:
    def test_sharpe_sign_matches_mean(self):
        pos = pd.Series(np.full(100, 0.0005))
        neg = pd.Series(np.full(100, -0.0005))
        assert metrics(pos)["sharpe"] > 0
        assert metrics(neg)["sharpe"] < 0

    def test_empty_is_zero(self):
        assert metrics(pd.Series(dtype=float))["n"] == 0


class TestWalkForward:
    def test_runs_and_reports_folds(self):
        rng = np.random.default_rng(1)
        funding = pd.DataFrame({s: rng.normal(0.0003, 0.0005, 200) for s in SYMS})
        agg = walk_forward(funding, n_folds=3, m=2)
        assert agg is not None
        assert agg["n_folds"] >= 1
        assert "sharpe" in agg and "pos_folds" in agg
