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
