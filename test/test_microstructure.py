"""Tests for backend/data/microstructure.py pure functions (no network)."""
import numpy as np
import pandas as pd
import pytest

from backend.data.microstructure import (
    forward_return, microprice_dev, decay_correlations, threshold_backtest,
    maker_viability,
)


class TestForwardReturn:
    def test_basic(self):
        mid = pd.Series([100.0, 101.0, 102.0, 103.0])
        fr = forward_return(mid, 1)
        assert fr.iloc[0] == pytest.approx(0.01)
        assert pd.isna(fr.iloc[-1])   # no t+1 for last bar

    def test_horizon_2(self):
        mid = pd.Series([100.0, 101.0, 110.0])
        fr = forward_return(mid, 2)
        assert fr.iloc[0] == pytest.approx(0.10)


class TestMicropriceDev:
    def test_sign(self):
        df = pd.DataFrame({"microprice": [101.0, 99.0], "mid": [100.0, 100.0]})
        d = microprice_dev(df)
        assert d.iloc[0] > 0 and d.iloc[1] < 0


class TestDecayCorrelations:
    def test_perfect_predictor_high_corr(self):
        # construct: signal at t equals next-bar return → corr ~1 at h=1
        n = 500
        rng = np.random.default_rng(0)
        fwd = rng.normal(0, 0.001, n)
        mid = 100 * np.cumprod(1 + np.concatenate([[0], fwd[:-1]]))
        df = pd.DataFrame({"timestamp": range(n), "mid": mid,
                           "imbalance_top20": fwd})   # signal = the coming return
        corrs = decay_correlations(df, [1], "imbalance_top20")
        assert corrs[1] > 0.9

    def test_noise_near_zero(self):
        n = 500
        rng = np.random.default_rng(1)
        mid = 100 * np.cumprod(1 + rng.normal(0, 0.001, n))
        df = pd.DataFrame({"timestamp": range(n), "mid": mid,
                           "imbalance_top20": rng.normal(0, 1, n)})   # random signal
        corrs = decay_correlations(df, [1, 5], "imbalance_top20")
        assert abs(corrs[1]) < 0.2 and abs(corrs[5]) < 0.2


class TestThresholdBacktest:
    def test_cost_reduces_net(self):
        rng = np.random.default_rng(2)
        imb = pd.Series(rng.normal(0, 1, 500))
        fwd = pd.Series(rng.normal(0, 0.001, 500))
        lo = threshold_backtest(imb, fwd, 0.9, cost_bps=2)
        hi = threshold_backtest(imb, fwd, 0.9, cost_bps=8)
        assert hi["net_bps"] < lo["net_bps"]
        assert hi["gross_bps"] == pytest.approx(lo["gross_bps"])  # cost doesn't touch gross

    def test_predictive_signal_positive_gross(self):
        # imbalance sign matches forward return → positive gross
        rng = np.random.default_rng(3)
        imb = pd.Series(rng.normal(0, 1, 500))
        fwd = np.sign(imb) * pd.Series(np.abs(rng.normal(0, 0.001, 500)))
        r = threshold_backtest(imb, fwd, 0.8, cost_bps=0)
        assert r["gross_bps"] > 0 and r["win_rate"] > 0.9

    def test_too_few_returns_none(self):
        assert threshold_backtest(pd.Series([0.1, 0.2]), pd.Series([0.0, 0.0]), 0.9, 2) is None


class TestMakerViability:
    def test_thin_spread_gross_below_taker_floor(self):
        # ETH-like: spread ~0, gross +2bps, but taker floor = 0 + 2*2 = 4bps
        spread = pd.Series([0.05, 0.06, 0.07])
        v = maker_viability(spread, gross_bps=2.0, taker_fee_bps_side=2.0)
        assert v["taker_floor_bps"] == pytest.approx(4.06, abs=0.01)
        assert v["beats_taker"] is False          # 2 < 4.06
        assert v["beats_maker_optimistic"] is True  # 2 > half-spread 0.03

    def test_strong_signal_beats_taker(self):
        spread = pd.Series([0.10, 0.10, 0.10])
        v = maker_viability(spread, gross_bps=10.0, taker_fee_bps_side=1.0)
        assert v["beats_taker"] is True           # 10 > 0.1 + 2 = 2.1

    def test_wide_spread_weak_signal_fails_both(self):
        spread = pd.Series([1.3, 1.3, 1.4])   # SOL-like
        v = maker_viability(spread, gross_bps=-0.1)
        assert v["beats_taker"] is False and v["beats_maker_optimistic"] is False
