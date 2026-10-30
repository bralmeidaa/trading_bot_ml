"""Tests for backend.agent.paper_execution — fill math + persistence round-trip."""
import pytest

from backend.agent.paper_execution import (
    COMMISSION, SLIPPAGE, AgentTrade, PaperBroker, compute_pnl,
    fill_entry_price, fill_exit_price,
)


class FakeExchange:
    def __init__(self, prices):
        self.prices = dict(prices)

    def fetch_ticker(self, symbol):
        return {"last": self.prices[symbol]}


class TestFillMath:
    def test_long_entry_costs_more_than_raw(self):
        assert fill_entry_price(100.0, direction=1) == pytest.approx(100.0 * (1 + COMMISSION + SLIPPAGE))

    def test_short_entry_fills_below_raw(self):
        assert fill_entry_price(100.0, direction=-1) == pytest.approx(100.0 * (1 - COMMISSION - SLIPPAGE))

    def test_long_exit_nets_less_than_raw(self):
        assert fill_exit_price(100.0, direction=1) == pytest.approx(100.0 * (1 - COMMISSION - SLIPPAGE))

    def test_short_exit_costs_more_than_raw(self):
        assert fill_exit_price(100.0, direction=-1) == pytest.approx(100.0 * (1 + COMMISSION + SLIPPAGE))

    def test_round_trip_always_loses_to_cost_on_flat_price(self):
        """Entering and exiting at the SAME raw price must be a net loss —
        commission+slippage are charged on both legs, never a wash."""
        entry = fill_entry_price(100.0, direction=1)
        exit_ = fill_exit_price(100.0, direction=1)
        pnl = compute_pnl(entry, exit_, quantity=1.0, direction=1)
        assert pnl["pnl"] < 0

    def test_compute_pnl_long_profit(self):
        pnl = compute_pnl(entry_price=100.0, exit_price=110.0, quantity=2.0, direction=1)
        assert pnl["pnl_pct"] == pytest.approx(0.10)
        assert pnl["pnl"] == pytest.approx(20.0)

    def test_compute_pnl_short_profit(self):
        pnl = compute_pnl(entry_price=100.0, exit_price=90.0, quantity=2.0, direction=-1)
        assert pnl["pnl_pct"] == pytest.approx(0.10)
        assert pnl["pnl"] == pytest.approx(20.0)


class TestPaperBroker:
    def test_place_order_opens_long_position(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 50000.0}), total_capital=1000.0)
        trade = broker.place_order("BTC/USDT", "buy", notional_usd=100.0, rationale="test")
        assert trade.symbol == "BTC/USDT"
        assert trade.direction == 1
        assert trade.status == "open"
        assert "BTC/USDT" in broker.open_trades

    def test_place_order_opens_short_position(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 50000.0}), total_capital=1000.0)
        trade = broker.place_order("BTC/USDT", "sell", notional_usd=100.0, rationale="test")
        assert trade.direction == -1

    def test_close_position_realizes_pnl_into_equity(self):
        exchange = FakeExchange({"BTC/USDT": 100.0})
        broker = PaperBroker(exchange, total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", notional_usd=100.0, rationale="test")
        exchange.prices["BTC/USDT"] = 200.0   # price doubles
        closed = broker.close_position("BTC/USDT", reason="manual")
        assert closed is not None
        assert closed.status == "closed"
        assert closed.pnl > 0
        assert broker.equity == pytest.approx(1000.0 + closed.pnl)
        assert "BTC/USDT" not in broker.open_trades

    def test_close_position_on_empty_symbol_returns_none(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 100.0}), total_capital=1000.0)
        assert broker.close_position("BTC/USDT", reason="x") is None

    def test_close_all_flattens_every_open_position(self):
        exchange = FakeExchange({"BTC/USDT": 100.0, "ETH/USDT": 50.0})
        broker = PaperBroker(exchange, total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", notional_usd=100.0, rationale="t")
        broker.place_order("ETH/USDT", "buy", notional_usd=100.0, rationale="t")
        closed = broker.close_all("kill_switch")
        assert len(closed) == 2
        assert broker.open_trades == {}

    def test_mark_to_market_reflects_unrealized_move(self):
        exchange = FakeExchange({"BTC/USDT": 100.0})
        broker = PaperBroker(exchange, total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", notional_usd=100.0, rationale="t")
        exchange.prices["BTC/USDT"] = 150.0
        equity = broker.mark_to_market()
        assert equity > 1000.0   # unrealized gain reflected without closing

    def test_mark_to_market_tolerates_ticker_failure(self):
        class FlakyExchange(FakeExchange):
            def fetch_ticker(self, symbol):
                raise RuntimeError("network down")

        broker = PaperBroker(FlakyExchange({"BTC/USDT": 100.0}), total_capital=1000.0)
        broker.open_trades["BTC/USDT"] = AgentTrade(
            id="x", symbol="BTC/USDT", direction=1, entry_time=0,
            entry_price=100.0, quantity=1.0, stop_loss=0.0, take_profit=0.0,
        )
        equity = broker.mark_to_market()   # must not raise
        assert equity == pytest.approx(1000.0)   # unrealized skipped on failure

    def test_open_notional_reflects_open_positions(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 100.0}), total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", notional_usd=50.0, rationale="t")
        notional = broker.open_notional()
        assert notional["BTC/USDT"] == pytest.approx(50.0, rel=0.05)   # ~ (fees shift it slightly)

    def test_positions_shape(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 100.0}), total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", notional_usd=50.0, rationale="t")
        positions = broker.positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC/USDT"
        assert positions[0]["side"] == "long"


class TestAgentTradePersistenceCompat:
    def test_agent_trade_saves_via_existing_trade_repository(self, fresh_db):
        """AgentTrade must be duck-type compatible with TradeRepository.save()
        — zero changes needed to the existing persistence code."""
        from backend.persistence.repository import TradeRepository

        trade = AgentTrade(
            id="test_agent_trade_1", symbol="BTC/USDT", direction=1,
            entry_time=1700000000000, entry_price=50000.0, quantity=0.002,
            stop_loss=0.0, take_profit=0.0, exit_time=1700003600000,
            exit_price=51000.0, pnl=2.0, pnl_pct=0.02, status="closed",
            reason="manual_stop", bot_id="ai_agent",
        )
        repo = TradeRepository()
        repo.save(trade)
        recent = repo.get_recent_trades(limit=10)
        assert any(t["id"] == "test_agent_trade_1" and t["bot_id"] == "ai_agent" for t in recent)
