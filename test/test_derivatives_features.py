"""
Unit tests for backend/data/derivatives_features.py — pure functions only
(fetch_funding_history / fetch_funding_panel / fetch_open_interest_coverage
are network I/O and are exercised by the research harness, not here).

Locks in:
  - point-in-time alignment of funding rate onto a panel index (no look-ahead)
  - cross-sectional z-score shape/NaN handling
  - BTC-dominance proxy sign and the no-BTC fallback
"""
import numpy as np
import pandas as pd
import pytest

from backend.data.derivatives_features import (
    align_funding_to_panel, cross_sectional_zscore, btc_dominance_proxy,
)

SYMS = [f"C{i}" for i in range(5)]


def _panel_index(n=20, step_ms=3_600_000):
    return pd.Index(np.arange(n) * step_ms, name="timestamp")


class TestAlignFundingToPanel:
    def test_backward_fill_no_lookahead(self):
        idx = _panel_index(10, step_ms=3_600_000)  # hourly bars, 10 bars
        # funding fixings every ~3.5h (not aligned to panel bars)
        fr = pd.DataFrame({
            "timestamp": [0, 12_600_000, 25_200_000],  # 0h, 3.5h, 7h
            "funding": [0.0001, 0.0005, -0.0002],
        })
        aligned = align_funding_to_panel({"C0": fr}, idx)
        # bar at t=2h (7_200_000ms) must see the 0h fixing (0.0001), NOT a future one
        assert aligned["C0"].loc[7_200_000] == pytest.approx(0.0001)
        # bar at t=4h must see the 3.5h fixing (0.0005)
        assert aligned["C0"].loc[14_400_000] == pytest.approx(0.0005)
        # bar before any fixing is NaN (nothing known yet)
        idx2 = pd.Index([-1000], name="timestamp")
        aligned2 = align_funding_to_panel({"C0": fr}, idx2)
        assert aligned2["C0"].isna().all()

    def test_missing_symbol_absent_from_output(self):
        idx = _panel_index()
        aligned = align_funding_to_panel({}, idx)
        assert aligned.empty or aligned.shape[1] == 0

    def test_multiple_symbols_independent_columns(self):
        idx = _panel_index(5, step_ms=3_600_000)
        fr_a = pd.DataFrame({"timestamp": [0], "funding": [0.01]})
        fr_b = pd.DataFrame({"timestamp": [0], "funding": [-0.02]})
        aligned = align_funding_to_panel({"A": fr_a, "B": fr_b}, idx)
        assert set(aligned.columns) == {"A", "B"}
        assert (aligned["A"] == 0.01).all()
        assert (aligned["B"] == -0.02).all()


class TestCrossSectionalZscore:
    def test_row_mean_zero(self):
        panel = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 5)), columns=SYMS)
        z = cross_sectional_zscore(panel)
        assert z.mean(axis=1).abs().max() < 1e-9

    def test_clip_bounds(self):
        panel = pd.DataFrame({"A": [0.0, 0.0], "B": [100.0, -100.0]})
        z = cross_sectional_zscore(panel, clip=2.0)
        assert z.abs().max().max() <= 2.0 + 1e-9

    def test_constant_row_is_nan_not_inf(self):
        panel = pd.DataFrame({"A": [1.0], "B": [1.0], "C": [1.0]})  # std=0
        z = cross_sectional_zscore(panel)
        assert not np.isinf(z.values).any()


class TestBtcDominanceProxy:
    def test_nan_without_btc_column(self):
        close = pd.DataFrame({"ETH/USDT": [100, 101, 102]})
        proxy = btc_dominance_proxy(close)
        assert proxy.isna().all()

    def test_positive_when_btc_outperforms_basket(self):
        n = 40
        idx = np.arange(n)
        btc = 100 * (1.01 ** idx)          # BTC trending up strongly
        alt = 100 * (1.001 ** idx)         # basket barely moving
        close = pd.DataFrame({"BTC/USDT": btc, "ETH/USDT": alt, "SOL/USDT": alt})
        proxy = btc_dominance_proxy(close)
        assert proxy.iloc[-1] > 0

    def test_negative_when_basket_outperforms_btc(self):
        n = 40
        idx = np.arange(n)
        btc = 100 * (1.001 ** idx)
        alt = 100 * (1.02 ** idx)
        close = pd.DataFrame({"BTC/USDT": btc, "ETH/USDT": alt, "SOL/USDT": alt})
        proxy = btc_dominance_proxy(close)
        assert proxy.iloc[-1] < 0
