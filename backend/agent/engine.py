"""
AI agent decision loop (Phase 0 — paper trading only).

Mirrors the structure of backend/strategy/portfolio_engine.py: a dataclass
config + state, a `start()` asyncio loop, blocking I/O via asyncio.to_thread,
and a `_cycle()` that's the one place per-tick logic lives. The crucial
difference is non-determinism (an LLM decides), so EVERY cycle is logged to
AgentDecisionLog regardless of outcome (hold / executed / rejected / halted)
— that log is the only way to audit "why did it do that" after the fact.

Guardrails (backend.agent.guardrails) are checked BEFORE calling the LLM
(saves cost on a halted/capped day) and the ONE proposed-order path
(place_paper_order tool) is re-validated again at execution time — the model
never gets a direct line to PaperBroker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from backend.agent import guardrails
from backend.agent.llm_client import AgentLLMClient
from backend.agent.paper_execution import PaperBroker
from backend.agent.tools import TOOL_SCHEMAS, build_tool_dispatch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cautious crypto paper-trading assistant. Each \
cycle you may look at recent price action and news, then either place ONE \
paper order or hold. Only trade symbols you're given tools for. Always call \
`hold` with a rationale if you see no clear reason to trade — doing nothing \
is a normal, good outcome. Never invent prices, news, or positions — only \
use what the tools return. Keep position sizes modest; you do not need to \
use your full size allowance."""


@dataclass
class AgentConfig:
    universe: List[str] = field(default_factory=lambda: list(guardrails.SYMBOL_ALLOWLIST))
    total_capital: float = 500.0            # separate dedicated paper book from the $1500 cross-sectional one
    cycle_interval_sec: int = 3600          # hourly — see plan for rationale (not HFT, caps LLM cost)
    max_position_pct: float = 0.15
    max_open_positions: int = 3
    daily_loss_limit: float = 0.05
    max_drawdown_kill: float = 0.20
    max_trades_per_day: int = 10
    max_llm_calls_per_day: int = 150
    news_lookback_hours: int = 24


@dataclass
class AgentState:
    halted: bool = False              # permanent (drawdown kill-switch)
    cycle_count: int = 0
    trades_today: int = 0
    llm_calls_today: int = 0
    tokens_today: int = 0


class AIAgentEngine:
    """Paper-trading engine. API-compatible read attributes exposed."""

    def __init__(self, config: AgentConfig, exchange, llm_client: AgentLLMClient,
                equity_repo=None, daily_repo=None, decision_repo=None, trade_repo=None):
        self.config = config
        self.state = AgentState()
        self.exchange = exchange
        self.llm_client = llm_client
        self.broker = PaperBroker(exchange, config.total_capital)
        self.equity_repo = equity_repo
        self.daily_repo = daily_repo
        self.decision_repo = decision_repo
        self.trade_repo = trade_repo
        self._last_reset_date = datetime.now().date()
        self._daily_start_equity = config.total_capital

    # ── API-compatible read properties ──────────────────────────────────
    @property
    def total_pnl(self) -> float:
        return self.broker.equity - self.config.total_capital

    @property
    def daily_pnl(self) -> float:
        return self.broker.equity - self._daily_start_equity

    def positions(self) -> List[dict]:
        return self.broker.positions()

    # ── guardrail orchestration (pure checks, I/O-free) ─────────────────
    def pre_llm_block_reason(self) -> Optional[str]:
        """Checked BEFORE calling the LLM each cycle — returns why the model
        should not be called this cycle, or None if it's fine to proceed."""
        if self.state.halted:
            return "engine_halted"
        if guardrails.drawdown_tripped(self.broker.peak_equity, self.broker.equity,
                                       self.config.max_drawdown_kill):
            return "drawdown_kill_switch"
        if guardrails.daily_loss_tripped(self.daily_pnl, self._daily_start_equity,
                                         self.config.daily_loss_limit):
            return "daily_loss_limit"
        if guardrails.daily_trade_cap_exceeded(self.state.trades_today, self.config.max_trades_per_day):
            return "daily_trade_cap"
        if guardrails.llm_call_budget_exceeded(self.state.llm_calls_today, self.config.max_llm_calls_per_day):
            return "llm_call_budget"
        return None

    def _order_guardrail_check(self, payload: dict):
        return guardrails.validate_order_request(
            payload, equity=self.broker.equity, open_notional=self.broker.open_notional(),
            max_position_pct=self.config.max_position_pct,
            max_open_positions=self.config.max_open_positions,
        )

    # ── live loop ────────────────────────────────────────────────────────
    async def start(self):
        logger.info(f"🤖 AI agent engine starting (PAPER, {len(self.config.universe)} symbols, "
                    f"cycle every {self.config.cycle_interval_sec}s)")
        try:
            while not self.state.halted:
                await self._cycle()
                await asyncio.sleep(self.config.cycle_interval_sec)
        except asyncio.CancelledError:
            logger.info("AI agent engine cancelled")
        except Exception as exc:
            logger.critical(f"💥 AI agent engine error: {exc}")

    async def _cycle(self):
        self._check_daily_reset()
        self.state.cycle_count += 1

        await asyncio.to_thread(self.broker.mark_to_market)
        self._record_equity()

        # Permanent kill-switch check happens here too (independent of the
        # pre-LLM block reason, which is informational for the log) — halts
        # the loop itself, not just this cycle, and flattens all positions.
        if guardrails.drawdown_tripped(self.broker.peak_equity, self.broker.equity,
                                       self.config.max_drawdown_kill):
            logger.critical(f"🛑 AI agent kill-switch: drawdown ≥ {self.config.max_drawdown_kill:.0%}. Halting.")
            closed = await asyncio.to_thread(self.broker.close_all, "drawdown_kill_switch")
            self._persist_closed_trades(closed)
            self.state.halted = True
            self._log_decision(halted_reason="drawdown_kill_switch")
            return

        block_reason = self.pre_llm_block_reason()
        if block_reason:
            self._log_decision(halted_reason=block_reason)
            return

        await self._run_llm_turn()

    async def _run_llm_turn(self):
        guardrail_check = self._order_guardrail_check
        dispatch = build_tool_dispatch(self.config.universe, self.exchange, self.broker, guardrail_check)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Cycle #{self.state.cycle_count}. Tradeable symbols: {self.config.universe}. "
                f"Decide whether to place one paper order or hold. Use the tools to "
                f"check prices, news and your current positions before deciding.")},
        ]
        result = await asyncio.to_thread(
            self.llm_client.chat_with_tools, messages, TOOL_SCHEMAS, dispatch)

        self.state.llm_calls_today += 1
        self.state.tokens_today += result.prompt_tokens + result.completion_tokens

        executed, rejections = [], []
        for call in result.tool_trace:
            if call["name"] == "place_paper_order":
                if call["output"].get("executed"):
                    executed.append(call)
                    self.state.trades_today += 1
                elif call["output"].get("rejected_reason"):
                    rejections.append(call)

        self._log_decision(
            raw_llm_output=result.text, tool_trace=result.tool_trace,
            rejections=rejections, prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens, latency_ms=result.latency_ms,
            halted_reason=f"llm_error: {result.error}" if result.error else None,
        )

    # ── persistence / logging ───────────────────────────────────────────
    def _record_equity(self):
        if self.equity_repo:
            try:
                self.equity_repo.save(self.broker.equity, self.total_pnl, self.daily_pnl,
                                      len(self.broker.open_trades))
            except Exception:
                pass

    def _persist_closed_trades(self, trades):
        if not self.trade_repo:
            return
        for trade in trades:
            try:
                self.trade_repo.save(trade)
            except Exception as exc:
                logger.error(f"Failed to persist agent trade {trade.id}: {exc}")

    def _check_daily_reset(self):
        today = datetime.now().date()
        if today != self._last_reset_date:
            if self.daily_repo:
                closed_today = [t for t in self.broker.closed_trades]
                wins = sum(1 for t in closed_today if (t.pnl or 0) > 0)
                losses = sum(1 for t in closed_today if (t.pnl or 0) <= 0)
                try:
                    self.daily_repo.save(self._last_reset_date.isoformat(), self.daily_pnl,
                                         self.state.trades_today, wins, losses,
                                         self.state.llm_calls_today, self.state.tokens_today)
                except Exception:
                    pass
            self._daily_start_equity = self.broker.equity
            self._last_reset_date = today
            self.state.trades_today = 0
            self.state.llm_calls_today = 0
            self.state.tokens_today = 0

    def _log_decision(self, raw_llm_output: str = "", tool_trace: Optional[list] = None,
                      rejections: Optional[list] = None, prompt_tokens: int = 0,
                      completion_tokens: int = 0, latency_ms: int = 0,
                      halted_reason: Optional[str] = None):
        if not self.decision_repo:
            return
        try:
            self.decision_repo.save(
                timestamp=datetime.utcnow(), cycle_id=self.state.cycle_count,
                symbols_considered=json.dumps(self.config.universe),
                context_snapshot=json.dumps({"equity": self.broker.equity,
                                             "positions": self.broker.positions()}),
                raw_llm_output=raw_llm_output,
                tool_calls_requested=json.dumps(tool_trace or [], default=str),
                tool_calls_executed=json.dumps([t for t in (tool_trace or [])
                                                if t.get("output", {}).get("executed")], default=str),
                guardrail_rejections=json.dumps(rejections or [], default=str),
                rationale=self._extract_rationale(tool_trace),
                model_name=getattr(self.llm_client.cfg, "deployment", None),
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                latency_ms=latency_ms, halted_reason=halted_reason,
            )
        except Exception as exc:
            logger.error(f"Failed to persist agent decision log: {exc}")

    @staticmethod
    def _extract_rationale(tool_trace: Optional[list]) -> Optional[str]:
        if not tool_trace:
            return None
        for call in reversed(tool_trace):
            if call["name"] in ("place_paper_order", "hold"):
                return call.get("args", {}).get("rationale")
        return None
