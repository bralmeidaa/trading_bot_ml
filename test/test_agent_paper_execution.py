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


class TestPositionSemantics:
    def test_same_side_order_adds_with_weighted_average_entry(self):
        exchange = FakeExchange({"BTC/USDT": 100.0})
        broker = PaperBroker(exchange, total_capital=1000.0)
        first = broker.place_order("BTC/USDT", "buy", notional_usd=50.0, rationale="a")
        exchange.prices["BTC/USDT"] = 200.0
        second = broker.place_order("BTC/USDT", "buy", notional_usd=50.0, rationale="b")
        assert second.id == first.id                     # same position, not a replacement
        assert len(broker.open_trades) == 1
        e1, e2 = fill_entry_price(100.0, 1), fill_entry_price(200.0, 1)
        q1, q2 = 50.0 / e1, 50.0 / e2
        assert second.quantity == pytest.approx(q1 + q2)
        assert second.entry_price == pytest.approx((q1 * e1 + q2 * e2) / (q1 + q2))

    def test_opposite_side_order_is_refused_and_position_untouched(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 100.0}), total_capital=1000.0)
        trade = broker.place_order("BTC/USDT", "buy", notional_usd=50.0, rationale="a")
        qty = trade.quantity
        with pytest.raises(ValueError):
            broker.place_order("BTC/USDT", "sell", notional_usd=50.0, rationale="flip")
        assert broker.open_trades["BTC/USDT"].direction == 1
        assert broker.open_trades["BTC/USDT"].quantity == qty

    def test_bracket_set_on_open_long_and_short(self):
        broker = PaperBroker(FakeExchange({"BTC/USDT": 100.0, "ETH/USDT": 100.0}), total_capital=1000.0)
        long_t = broker.place_order("BTC/USDT", "buy", 50.0, "l", stop_pct=0.02, target_pct=0.04)
        short_t = broker.place_order("ETH/USDT", "sell", 50.0, "s", stop_pct=0.02, target_pct=0.04)
        assert long_t.stop_loss == pytest.approx(long_t.entry_price * 0.98)
        assert long_t.take_profit == pytest.approx(long_t.entry_price * 1.04)
        assert short_t.stop_loss == pytest.approx(short_t.entry_price * 1.02)
        assert short_t.take_profit == pytest.approx(short_t.entry_price * 0.96)

    def test_failed_price_fetch_on_close_keeps_the_position(self):
        exchange = FakeExchange({"BTC/USDT": 100.0})
        broker = PaperBroker(exchange, total_capital=1000.0)
        broker.place_order("BTC/USDT", "buy", 50.0, "a")
        del exchange.prices["BTC/USDT"]                   # fetch_ticker now raises KeyError
        with pytest.raises(KeyError):
            broker.close_position("BTC/USDT", reason="x")
        assert "BTC/USDT" in broker.open_trades


class TestExitTriggers:
    def _broker_with(self, side, price=100.0):
        exchange = FakeExchange({"BTC/USDT": price})
        broker = PaperBroker(exchange, total_capital=1000.0)
        broker.place_order("BTC/USDT", side, 50.0, "t", stop_pct=0.02, target_pct=0.04)
        return broker

    def test_long_stop(self):
        broker = self._broker_with("buy")
        closed = broker.check_exits({"BTC/USDT": 97.0})
        assert [t.reason for t in closed] == ["stop"]
        assert closed[0].pnl < 0
        assert "BTC/USDT" in broker.last_stop_at

    def test_long_target(self):
        broker = self._broker_with("buy")
        closed = broker.check_exits({"BTC/USDT": 106.0})
        assert [t.reason for t in closed] == ["target"]
        assert closed[0].pnl > 0
        assert "BTC/USDT" not in broker.last_stop_at

    def test_short_stop_and_target(self):
        broker = self._broker_with("sell")
        assert [t.reason for t in broker.check_exits({"BTC/USDT": 103.0})] == ["stop"]
        broker = self._broker_with("sell")
        assert [t.reason for t in broker.check_exits({"BTC/USDT": 95.0})] == ["target"]

    def test_inside_bracket_stays_open(self):
        broker = self._broker_with("buy")
        assert broker.check_exits({"BTC/USDT": 101.0}) == []
        assert "BTC/USDT" in broker.open_trades

    def test_missing_price_is_skipped(self):
        broker = self._broker_with("buy")
        assert broker.check_exits({}) == []

    def test_equity_after_exit_includes_realized(self):
        broker = self._broker_with("buy")
        closed = broker.check_exits({"BTC/USDT": 106.0})
        assert broker.mark_to_market({}) == pytest.approx(1000.0 + closed[0].pnl)

    def test_fetch_prices_prefers_fetch_tickers(self):
        class TickersExchange(FakeExchange):
            def fetch_ticker(self, symbol):
                raise AssertionError("should use fetch_tickers")

            def fetch_tickers(self, symbols):
                return {s: {"last": self.prices[s]} for s in symbols}

        broker = PaperBroker(TickersExchange({"BTC/USDT": 1.0, "ETH/USDT": 2.0}), total_capital=1.0)
        assert broker.fetch_prices(["BTC/USDT", "ETH/USDT"]) == {"BTC/USDT": 1.0, "ETH/USDT": 2.0}


class TestPersistenceAndRestore:
    def test_open_add_close_are_persisted_and_restorable(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        exchange = FakeExchange({"BTC/USDT": 100.0, "ETH/USDT": 50.0})
        broker = PaperBroker(exchange, total_capital=1000.0, trade_repo=repo, bot_id="bot_x")
        broker.place_order("BTC/USDT", "buy", 50.0, "a", stop_pct=0.02, target_pct=0.04)
        broker.place_order("BTC/USDT", "buy", 50.0, "add")
        broker.place_order("ETH/USDT", "sell", 50.0, "b", stop_pct=0.02, target_pct=0.04)
        exchange.prices["ETH/USDT"] = 40.0
        closed = broker.close_position("ETH/USDT", reason="llm: done")

        open_rows = repo.get_open_trades("bot_x")
        closed_rows = repo.get_closed_trades("bot_x")
        assert [r["symbol"] for r in open_rows] == ["BTC/USDT"]
        assert open_rows[0]["quantity"] == pytest.approx(broker.open_trades["BTC/USDT"].quantity)
        assert open_rows[0]["stop_loss"] > 0
        assert [r["symbol"] for r in closed_rows] == ["ETH/USDT"]
        assert closed_rows[0]["pnl"] == pytest.approx(closed.pnl)
        assert repo.get_open_trades("other_bot") == []

        restored = PaperBroker(exchange, total_capital=1000.0, trade_repo=repo, bot_id="bot_x")
        restored.restore([AgentTrade(**r) for r in open_rows], [AgentTrade(**r) for r in closed_rows])
        assert set(restored.open_trades) == {"BTC/USDT"}
        assert restored.realized_pnl() == pytest.approx(closed.pnl)
        assert restored.equity == pytest.approx(1000.0 + closed.pnl)

    def test_agent_trades_do_not_seed_legacy_kelly(self, fresh_db):
        from backend.persistence.repository import TradeRepository
        repo = TradeRepository()
        repo.save(AgentTrade(id="a1", symbol="BTC/USDT", direction=1, entry_time=1, entry_price=1.0,
                             quantity=1.0, stop_loss=0.0, take_profit=0.0, exit_time=2,
                             exit_price=1.1, pnl=0.1, pnl_pct=0.1, status="closed",
                             bot_id="ai_agent_v2"))
        assert repo.get_recent_pnl_pcts() == []


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
