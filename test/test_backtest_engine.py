"""
WalkForwardEngine tests — verify backtesting logic, cost simulation,
and metric calculations.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestWalkForwardEngine:
    @pytest.fixture
    def engine(self):
        from backend.backtesting.engine import WalkForwardEngine, BacktestConfig
        return WalkForwardEngine(BacktestConfig(n_splits=3, min_test_bars=50))

    def test_returns_backtest_result(self, engine, large_ohlcv, signal_gen):
        from backend.backtesting.engine import BacktestResult
        result = engine.run(large_ohlcv.copy(), signal_gen)
        assert isinstance(result, BacktestResult)

    def test_empty_result_on_tiny_data(self, engine, signal_gen, sample_ohlcv):
        tiny = sample_ohlcv.iloc[:30]
        result = engine.run(tiny.copy(), signal_gen)
        assert result.n_trades == 0
        assert result.is_valid is False

    def test_result_has_all_fields(self, engine, large_ohlcv, signal_gen):
        result = engine.run(large_ohlcv.copy(), signal_gen)
        for field in ('total_return', 'sharpe_ratio', 'sortino_ratio',
                      'max_drawdown', 'win_rate', 'profit_factor', 'n_trades',
                      'calmar_ratio', 'fold_results', 'is_valid'):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_max_drawdown_non_negative(self, engine, large_ohlcv, signal_gen):
        result = engine.run(large_ohlcv.copy(), signal_gen)
        assert result.max_drawdown >= 0

    def test_win_rate_in_range(self, engine, large_ohlcv, signal_gen):
        result = engine.run(large_ohlcv.copy(), signal_gen)
        assert 0.0 <= result.win_rate <= 1.0

    def test_is_valid_requires_sharpe_above_threshold(self, engine, large_ohlcv, signal_gen):
        result = engine.run(large_ohlcv.copy(), signal_gen)
        if result.is_valid:
            assert result.sharpe_ratio > 0.8
            assert result.max_drawdown < 0.25
        else:
            assert result.sharpe_ratio <= 0.8 or result.max_drawdown >= 0.25

    def test_fold_results_populated(self, engine, large_ohlcv, signal_gen):
        result = engine.run(large_ohlcv.copy(), signal_gen)
        assert isinstance(result.fold_results, list)


class TestCostSimulation:
    def test_apply_costs_buy(self):
        from backend.backtesting.engine import WalkForwardEngine, BacktestConfig
        engine = WalkForwardEngine(BacktestConfig())
        raw_price = 15.0
        effective = engine._apply_costs(raw_price, 'buy')
        assert effective > raw_price  # buying is more expensive

    def test_apply_costs_sell(self):
        from backend.backtesting.engine import WalkForwardEngine, BacktestConfig
        engine = WalkForwardEngine(BacktestConfig())
        raw_price = 15.0
        effective = engine._apply_costs(raw_price, 'sell')
        assert effective < raw_price  # selling gets less

    def test_costs_are_symmetric(self):
        from backend.backtesting.engine import WalkForwardEngine, BacktestConfig
        cfg = BacktestConfig(commission_pct=0.001, slippage_pct=0.0005)
        engine = WalkForwardEngine(cfg)
        p = 100.0
        buy = engine._apply_costs(p, 'buy')
        sell = engine._apply_costs(p, 'sell')
        total_cost_pct = (buy - sell) / p
        expected = 2 * (cfg.commission_pct + cfg.slippage_pct)
        assert abs(total_cost_pct - expected) < 1e-9


class TestMetricsCalculation:
    def test_metrics_from_all_wins(self):
        from backend.backtesting.engine import WalkForwardEngine
        engine = WalkForwardEngine()
        from backend.backtesting.engine import TradeRecord
        trades = [TradeRecord(0, 3, 1, 15.0, 15.3, 0.02) for _ in range(20)]
        result = engine._metrics(trades, [])
        assert result.win_rate == 1.0
        assert result.total_return > 0

    def test_metrics_from_all_losses(self):
        from backend.backtesting.engine import WalkForwardEngine
        engine = WalkForwardEngine()
        from backend.backtesting.engine import TradeRecord
        trades = [TradeRecord(0, 3, 1, 15.0, 14.7, -0.02) for _ in range(20)]
        result = engine._metrics(trades, [])
        assert result.win_rate == 0.0
        assert result.total_return < 0

    def test_empty_trades_returns_empty_result(self):
        from backend.backtesting.engine import WalkForwardEngine
        engine = WalkForwardEngine()
        result = engine._metrics([], [])
        assert result.n_trades == 0
        assert result.total_return == 0.0
