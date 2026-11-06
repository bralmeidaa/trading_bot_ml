"""
AdvancedRiskManager (Kelly Criterion) tests.

Invariants:
  - Falls back to base_risk_pct with no trade history
  - Kelly decreases position size after a losing streak
  - Drawdown protection reduces size near max_drawdown
  - Volatility multiplier scales inversely with vol
  - Position size is always positive and within [min, max] bounds
"""
import numpy as np
import pandas as pd
import pytest


class TestKellyCriterion:
    def test_fallback_with_no_trades(self, risk_manager):
        result = risk_manager._kelly_risk_pct()
        assert result == risk_manager.params.base_risk_pct, (
            "With no trade history, should fall back to base_risk_pct"
        )

    def test_fallback_with_insufficient_trades(self, risk_manager):
        for _ in range(risk_manager.params.kelly_min_trades - 1):
            risk_manager.record_trade(0.01)
        result = risk_manager._kelly_risk_pct()
        assert result == risk_manager.params.base_risk_pct

    def test_kelly_positive_after_win_streak(self, risk_manager):
        for _ in range(20):
            risk_manager.record_trade(0.02)   # all wins
        result = risk_manager._kelly_risk_pct()
        assert result > 0

    def test_kelly_capped_at_max_risk(self, risk_manager):
        for _ in range(50):
            risk_manager.record_trade(0.05)   # large wins
        result = risk_manager._kelly_risk_pct()
        assert result <= risk_manager.params.max_risk_pct

    def test_kelly_floored_at_min_risk(self, risk_manager):
        for _ in range(50):
            risk_manager.record_trade(-0.05)   # all losses
        result = risk_manager._kelly_risk_pct()
        assert result >= risk_manager.params.min_risk_pct


class TestPositionSizing:
    def test_position_size_positive(self, risk_manager):
        size = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=14.5, confidence=0.7
        )
        assert size > 0

    def test_position_size_zero_on_zero_stop_distance(self, risk_manager):
        size = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=15.0, confidence=0.7
        )
        assert size == 0.0

    def test_drawdown_reduces_size(self, risk_manager):
        base = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=14.5,
            current_equity=1000.0, peak_equity=1000.0
        )
        reduced = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=14.5,
            current_equity=910.0, peak_equity=1000.0   # 9% drawdown
        )
        assert reduced < base, "Position size must decrease when near max drawdown"

    def test_higher_confidence_increases_size(self, risk_manager):
        low = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=14.5, confidence=0.3
        )
        high = risk_manager.calculate_position_size(
            capital=1000.0, entry_price=15.0, stop_loss=14.5, confidence=0.9
        )
        assert high > low

    def test_volatile_market_decreases_size(self, risk_manager):
        calm = pd.Series(np.random.normal(0, 0.001, 30))  # low vol
        volatile = pd.Series(np.random.normal(0, 0.02, 30))  # high vol
        size_calm = risk_manager.calculate_position_size(
            1000.0, 15.0, 14.5, returns=calm
        )
        size_volatile = risk_manager.calculate_position_size(
            1000.0, 15.0, 14.5, returns=volatile
        )
        assert size_calm > size_volatile


class TestCostApplication:
    def test_buy_cost_raises_price(self, risk_manager):
        raw = 15.0
        effective = risk_manager.apply_costs(raw, 'buy')
        assert effective > raw

    def test_sell_cost_lowers_price(self, risk_manager):
        raw = 15.0
        effective = risk_manager.apply_costs(raw, 'sell')
        assert effective < raw

    def test_round_trip_cost_is_symmetric(self, risk_manager):
        raw = 15.0
        buy_price = risk_manager.apply_costs(raw, 'buy')
        sell_price = risk_manager.apply_costs(raw, 'sell')
        # Round-trip should deduct ~2× (commission + slippage)
        expected_total_cost = 2 * (risk_manager.params.commission_pct + risk_manager.params.slippage_pct)
        actual_cost = (buy_price - sell_price) / raw
        assert abs(actual_cost - expected_total_cost) < 0.001


class TestTradeRecording:
    def test_record_accumulates(self, risk_manager):
        initial = len(risk_manager._results)
        risk_manager.record_trade(0.01)
        risk_manager.record_trade(-0.02)
        assert len(risk_manager._results) == initial + 2

    def test_old_results_pruned(self, risk_manager):
        for _ in range(250):
            risk_manager.record_trade(0.01)
        max_history = risk_manager.params.kelly_lookback * 4
        assert len(risk_manager._results) <= max_history
