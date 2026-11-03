"""
Tests for backend.agent.engine.AIAgentEngine — the LLM is always mocked
(never a real network/API call). What's exercised here is the deterministic
scaffolding around the LLM: kill-switches gating whether the LLM is even
called, guardrail enforcement on any proposed order (via the REAL
backend.agent.guardrails + backend.agent.tools wiring, not re-implemented in
the mock), decision logging, and graceful degradation on LLM errors.

No pytest-asyncio marker needed (none is used elsewhere in this repo) —
engine._cycle() is driven via asyncio.run() from a plain sync test function.
"""
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backend.agent.engine import AgentConfig, AIAgentEngine
from backend.agent.llm_client import ChatResult
from backend.agent.paper_execution import AgentTrade


class FakeExchange:
    def __init__(self, prices):
        self.prices = dict(prices)

    def fetch_ticker(self, symbol):
        return {"last": self.prices.get(symbol, 100.0), "percentage": 0.0,
               "high": 100.0, "low": 100.0, "quoteVolume": 1_000_000.0}


class FakeDecisionRepo:
    """Captures save() calls for inspection instead of hitting a real DB —
    persistence round-trip fidelity is covered separately in
    test_agent_persistence.py."""

    def __init__(self):
        self.calls = []

    def save(self, **kwargs):
        self.calls.append(kwargs)


def _stub_that_calls_tool(tool_name, tool_args, error=None):
    """Build a chat_with_tools stub that invokes the REAL tool_dispatch
    handler it's given — this exercises the actual guardrail wiring, not a
    re-implementation of it in the test."""

    def _stub(messages, tools, tool_dispatch, max_tool_turns=4):
        if error:
            return ChatResult(error=error)
        handler = tool_dispatch[tool_name]
        output = handler(tool_args)
        return ChatResult(text="", tool_trace=[{"name": tool_name, "args": tool_args, "output": output}],
                          prompt_tokens=10, completion_tokens=5, latency_ms=50)

    return _stub


def _make_engine(mock_llm_client, fake_exchange, decision_repo=None, **cfg_kwargs):
    cfg = AgentConfig(**cfg_kwargs)
    return AIAgentEngine(cfg, exchange=fake_exchange, llm_client=mock_llm_client,
                         decision_repo=decision_repo or FakeDecisionRepo())


def _inject_realized_loss(engine, pnl: float):
    """broker.mark_to_market() recomputes equity from total_capital + closed
    trade pnls every cycle — so to simulate a prior loss, inject a closed
    AgentTrade rather than poking .equity directly (which _cycle's first
    mark_to_market call would just overwrite)."""
    engine.broker.closed_trades.append(AgentTrade(
        id="seed", symbol="BTC/USDT", direction=1, entry_time=0,
        entry_price=100.0, quantity=1.0, stop_loss=0.0, take_profit=0.0,
        pnl=pnl, status="closed",
    ))


class TestKillSwitchesBlockLLM:
    def test_drawdown_tripped_halts_without_calling_llm(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo,
                              max_drawdown_kill=0.20, total_capital=1000.0)
        _inject_realized_loss(engine, -300.0)   # equity -> 700 after mark_to_market, 30% dd

        asyncio.run(engine._cycle())

        assert len(mock_llm_client.calls) == 0, "LLM must never be called once drawdown is tripped"
        assert engine.state.halted is True
        assert decision_repo.calls[-1]["halted_reason"] == "drawdown_kill_switch"

    def test_llm_call_budget_exceeded_skips_llm_this_cycle(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo, max_llm_calls_per_day=5)
        engine.state.llm_calls_today = 5

        asyncio.run(engine._cycle())

        assert len(mock_llm_client.calls) == 0
        assert engine.state.halted is False   # NOT a permanent halt, just this cycle
        assert decision_repo.calls[-1]["halted_reason"] == "llm_call_budget"

    def test_daily_trade_cap_skips_llm(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo, max_trades_per_day=3)
        engine.state.trades_today = 3

        asyncio.run(engine._cycle())

        assert len(mock_llm_client.calls) == 0
        assert decision_repo.calls[-1]["halted_reason"] == "daily_trade_cap"

    def test_daily_loss_limit_skips_llm(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo,
                              daily_loss_limit=0.05, max_drawdown_kill=0.20, total_capital=1000.0)
        engine._daily_start_equity = 1000.0
        _inject_realized_loss(engine, -60.0)   # equity -> 940, -6% today (past 5% limit, under 20% dd)

        asyncio.run(engine._cycle())

        assert len(mock_llm_client.calls) == 0
        assert decision_repo.calls[-1]["halted_reason"] == "daily_loss_limit"


class TestValidOrderFlow:
    def test_valid_order_executes_and_is_logged(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo)
        mock_llm_client.chat_with_tools = _stub_that_calls_tool(
            "place_paper_order",
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 50.0, "rationale": "test buy"},
        )

        asyncio.run(engine._cycle())

        assert "BTC/USDT" in engine.broker.open_trades
        assert engine.state.trades_today == 1
        logged = decision_repo.calls[-1]
        assert logged["halted_reason"] is None
        assert "BTC/USDT" in logged["tool_calls_executed"]

    def test_hold_is_logged_without_opening_a_position(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo)
        mock_llm_client.chat_with_tools = _stub_that_calls_tool("hold", {"rationale": "no signal"})

        asyncio.run(engine._cycle())

        assert engine.broker.open_trades == {}
        assert engine.state.trades_today == 0
        assert decision_repo.calls[-1]["rationale"] == "no signal"


class TestRejectedOrderFlow:
    def test_oversized_order_rejected_not_executed(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo, max_position_pct=0.10)
        mock_llm_client.chat_with_tools = _stub_that_calls_tool(
            "place_paper_order",
            {"symbol": "BTC/USDT", "side": "buy", "notional_usd": 999.0, "rationale": "too big"},
        )

        asyncio.run(engine._cycle())

        assert engine.broker.open_trades == {}
        assert engine.state.trades_today == 0
        logged = decision_repo.calls[-1]
        assert "max_position_pct" in logged["guardrail_rejections"]

    def test_hallucinated_symbol_rejected(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo)
        mock_llm_client.chat_with_tools = _stub_that_calls_tool(
            "place_paper_order",
            {"symbol": "DOGE/USDT", "side": "buy", "notional_usd": 10.0, "rationale": "moon"},
        )

        asyncio.run(engine._cycle())

        assert engine.broker.open_trades == {}
        assert "allow-list" in decision_repo.calls[-1]["guardrail_rejections"]


class TestNewsStalenessTracking:
    def test_item_seen_for_first_time_is_zero_cycles_old(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        engine.state.cycle_count = 1

        out = engine._annotate_news([{"url": "https://x/1", "title": "hack"}])

        assert out[0]["cycles_since_first_seen"] == 0

    def test_item_seen_in_a_later_cycle_ages(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        engine.state.cycle_count = 1
        engine._annotate_news([{"url": "https://x/1", "title": "hack"}])

        engine.state.cycle_count = 4
        out = engine._annotate_news([{"url": "https://x/1", "title": "hack"}])

        assert out[0]["cycles_since_first_seen"] == 3

    def test_item_dropping_out_of_window_is_pruned_from_tracker(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        engine.state.cycle_count = 1
        engine._annotate_news([{"url": "https://x/old", "title": "old"}])

        engine.state.cycle_count = 2
        engine._annotate_news([{"url": "https://x/new", "title": "new"}])  # old no longer in the digest

        assert "https://x/old" not in engine.state.news_first_seen_cycle
        assert "https://x/new" in engine.state.news_first_seen_cycle

    def test_item_without_url_falls_back_to_title_as_key(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        engine.state.cycle_count = 1
        engine._annotate_news([{"title": "hack", "url": ""}])

        engine.state.cycle_count = 2
        out = engine._annotate_news([{"title": "hack", "url": ""}])

        assert out[0]["cycles_since_first_seen"] == 1


class TestBusinessHoursCadence:
    TZ = ZoneInfo("America/Sao_Paulo")

    def test_weekday_inside_window_is_business_hours(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        wednesday_noon = datetime(2026, 11, 4, 12, 0, tzinfo=self.TZ)   # 2026-11-04 is a Wednesday
        assert engine.is_business_hours(wednesday_noon) is True

    def test_weekday_before_opening_is_off_hours(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        wednesday_5am = datetime(2026, 11, 4, 5, 0, tzinfo=self.TZ)
        assert engine.is_business_hours(wednesday_5am) is False

    def test_weekday_after_closing_is_off_hours(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        wednesday_8pm = datetime(2026, 11, 4, 20, 0, tzinfo=self.TZ)
        assert engine.is_business_hours(wednesday_8pm) is False

    def test_saturday_is_off_hours_even_during_window(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange)
        saturday_noon = datetime(2026, 11, 7, 12, 0, tzinfo=self.TZ)
        assert engine.is_business_hours(saturday_noon) is False

    def test_current_cycle_interval_switches_with_business_hours(self, mock_llm_client, fake_exchange):
        engine = _make_engine(mock_llm_client, fake_exchange,
                              cycle_interval_sec=3600, off_hours_cycle_interval_sec=14400)
        engine.is_business_hours = lambda now=None: True
        assert engine.current_cycle_interval() == 3600
        engine.is_business_hours = lambda now=None: False
        assert engine.current_cycle_interval() == 14400


class TestLLMErrorDegradesGracefully:
    def test_azure_error_does_not_crash_cycle(self, mock_llm_client, fake_exchange):
        decision_repo = FakeDecisionRepo()
        engine = _make_engine(mock_llm_client, fake_exchange, decision_repo)
        mock_llm_client.chat_with_tools = _stub_that_calls_tool(
            "hold", {}, error="connection timeout",
        )

        asyncio.run(engine._cycle())   # must not raise

        assert engine.state.halted is False
        assert "connection timeout" in decision_repo.calls[-1]["halted_reason"]
        assert engine.broker.open_trades == {}
