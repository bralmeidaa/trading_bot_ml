"""Pure tests for backend.agent.stats and backend.agent.market_data."""
import pytest

from backend.agent.market_data import atr_pct_from_ohlcv, default_bracket
from backend.agent.stats import trade_stats


def _t(pnl, entry=100.0, stop=98.0, qty=1.0, reason="stop"):
    return {"pnl": pnl, "entry_price": entry, "stop_loss": stop, "quantity": qty, "reason": reason}


class TestTradeStats:
    def test_empty(self):
        s = trade_stats([])
        assert s["n"] == 0 and s["win_rate"] is None and s["expectancy_r"] is None

    def test_basic_numbers(self):
        s = trade_stats([_t(4.0, reason="target"), _t(-2.0), _t(-2.0), _t(1.0, reason="llm: exit")])
        assert s["n"] == 4 and s["wins"] == 2 and s["losses"] == 2
        assert s["net_pnl"] == pytest.approx(1.0)
        assert s["profit_factor"] == pytest.approx(5.0 / 4.0)
        assert s["expectancy_usd"] == pytest.approx(0.25)
        # risk per trade = (100-98)×1 = 2 → R = 2, -1, -1, 0.5
        assert s["expectancy_r"] == pytest.approx(0.125)
        assert s["exits"] == {"target": 1, "stop": 2, "llm": 1}

    def test_trades_without_stop_excluded_from_r(self):
        s = trade_stats([_t(1.0, stop=0.0), _t(-2.0)])
        assert s["expectancy_r"] == pytest.approx(-1.0)


class TestAtr:
    def test_atr_pct_of_flat_bars(self):
        rows = [[i, 100, 101, 99, 100, 1] for i in range(20)]
        assert atr_pct_from_ohlcv(rows) == pytest.approx(0.02)

    def test_atr_needs_two_rows(self):
        assert atr_pct_from_ohlcv([[0, 1, 1, 1, 1, 1]]) is None

    def test_default_bracket_clamps_to_floor_and_ceiling(self):
        low = default_bracket(0.001, 1.5, 2.5, min_stop_pct=0.012, max_stop_pct=0.08)
        assert low["stop_pct"] == 0.012
        high = default_bracket(0.2, 1.5, 2.5, min_stop_pct=0.012, max_stop_pct=0.08)
        assert high["stop_pct"] == 0.08
        none = default_bracket(None, 1.5, 2.5, min_stop_pct=0.012, max_stop_pct=0.08)
        assert none["stop_pct"] == 0.012
        assert none["target_pct"] == pytest.approx(0.012 * 2.5 / 1.5)
