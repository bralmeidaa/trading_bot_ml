"""
AI agent decision loop (paper trading only).

Mirrors the structure of backend/strategy/portfolio_engine.py: a dataclass
config + state, a `start()` asyncio loop, blocking I/O via asyncio.to_thread,
and a `_cycle()` that's the one place per-tick LLM logic lives. EVERY LLM
cycle is logged to AgentDecisionLog regardless of outcome — that log is the
only way to audit "why did it do that" after the fact.

Three loops run side by side:
- the LLM cycle decides entries and discretionary exits. It runs on a
  heartbeat (cycle_interval_sec) and is woken earlier by market events;
- the exit monitor (every ~30s, no LLM) enforces each position's stop-loss and
  take-profit, marks to market and checks the drawdown kill-switch, so
  protection doesn't wait for the next LLM call;
- the event loop (when a BinanceStream is attached) watches for sharp price
  moves, liquidation bursts, fresh headlines and positions nearing their stop,
  and wakes the LLM cycle — debounced per event and rate-limited by
  min_llm_gap_sec and the daily LLM budget.

Guardrails (backend.agent.guardrails) are checked BEFORE calling the LLM and
again on every proposed order — the model never gets a direct line to
PaperBroker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

from backend.agent import guardrails
from backend.agent.llm_client import AgentLLMClient
from backend.agent.market_data import AtrCache, default_bracket
from backend.agent.news_sources import digest_for_symbols
from backend.agent.paper_execution import ROUND_TRIP_COST, AgentTrade, PaperBroker
from backend.agent.tools import TOOL_SCHEMAS, build_tool_dispatch
from backend.stream import events as ev

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cautious crypto paper-trading assistant. Each \
cycle you may look at recent price action, news and your open positions, then \
take at most ONE action: open/add a position (place_paper_order), exit one \
(close_paper_position), or hold. Only trade symbols you're given tools for. \
Always call `hold` with a rationale if you see no clear reason to act — doing \
nothing is a normal, good outcome. Never invent prices, news, or positions — \
only use what the tools return.

Exits: every position has an automatic stop-loss and take-profit, enforced \
every ~30 seconds without you. You don't need to babysit them. Close a \
position yourself only when the reason you entered no longer holds. To switch \
sides on a symbol, close first; an order against an open position is \
rejected. A same-side order adds to the position.

Risk: each trade risks at most ~1% of equity (position size × stop distance). \
BTC, ETH and SOL move together, so being long (or short) all three is ONE \
bet, and the guardrails cap net directional exposure. Prefer one well-founded \
position over several correlated ones. Every round trip costs ~0.3% in fees \
and slippage, so a trade needs an expected move clearly larger than that.

Each news item from get_news_digest includes `cycles_since_first_seen`. 0 \
means genuinely new this cycle. If it's greater than 0, that story is \
already known/priced-in — do not use it alone, again, to open a new \
position or add to an existing one. Only act on an aging story if you also \
cite fresh evidence from get_market_context (the current price move) \
showing the market is still actively reacting, and say so explicitly in \
the rationale. Simply restating an old headline is not a valid rationale \
for a new trade.

Cycles run frequently during business hours. Short, tactical trades — \
opened and closed within the same day, including ones that go against the \
slower multi-day narrative — are a valid use of this cadence, as long as \
they're grounded in the current price snapshot, not in stale news."""


@dataclass
class AgentConfig:
    universe: List[str] = field(default_factory=lambda: list(guardrails.SYMBOL_ALLOWLIST))
    total_capital: float = 500.0            # separate dedicated paper book from the $1500 cross-sectional one
    bot_id: str = "ai_agent_v2"             # v1 ledger was unreliable (positions were overwritten), v2 starts clean
    cycle_interval_sec: int = 3600
    # Only the LLM cadence slows down outside business hours (token cost);
    # the exit monitor below runs 24/7 regardless.
    off_hours_cycle_interval_sec: int = 4 * 3600
    business_hours_start: int = 6
    business_hours_end: int = 19
    business_days: tuple = (0, 1, 2, 3, 4)   # Mon=0 .. Fri=4
    timezone: str = "America/Sao_Paulo"
    exit_check_interval_sec: int = 30
    max_position_pct: float = 0.15
    max_open_positions: int = 3
    max_risk_per_trade: float = 0.01
    max_net_exposure_pct: float = 0.30
    # Stops tighter than 4× the round-trip cost turn fees into >25% of the
    # risk per trade — the reason the earlier 5m intraday research was NO-GO.
    min_stop_pct: float = 4 * ROUND_TRIP_COST
    max_stop_pct: float = 0.08
    stop_atr_mult: float = 1.5
    target_atr_mult: float = 2.5
    atr_timeframe: str = "15m"
    stop_cooldown_sec: int = 1800
    daily_loss_limit: float = 0.05
    max_drawdown_kill: float = 0.20
    max_trades_per_day: int = 10
    max_llm_calls_per_day: int = 150
    max_tool_turns: int = 6
    # event-driven wake-ups (only active with a stream attached)
    event_check_interval_sec: int = 5
    min_llm_gap_sec: int = 120
    event_cooldown_sec: int = 900
    move_window_min: int = 5
    move_atr_mult: float = 1.5
    liq_bucket_sec: int = 300
    liq_burst_mult: float = 3.0
    liq_min_usd: float = 250_000.0
    near_stop_r: float = 0.3
    news_poll_sec: int = 180


@dataclass
class AgentState:
    halted: bool = False              # permanent (drawdown kill-switch)
    cycle_count: int = 0
    trades_today: int = 0
    llm_calls_today: int = 0
    tokens_today: int = 0
    news_first_seen_cycle: dict = field(default_factory=dict)  # url/title -> cycle first seen
    last_exit_check: Optional[float] = None


class AIAgentEngine:
    """Paper-trading engine. API-compatible read attributes exposed."""

    def __init__(self, config: AgentConfig, exchange, llm_client: AgentLLMClient,
                equity_repo=None, daily_repo=None, decision_repo=None, trade_repo=None,
                stream=None):
        self.config = config
        self.stream = stream
        self._wake: Optional[asyncio.Queue] = None
        self._event_last_fired: dict = {}
        self._news_seen: Optional[set] = None
        self._news_last_poll = 0.0
        self._last_llm_at = 0.0
        self._event_task: Optional[asyncio.Task] = None
        self._wake_reasons: List[str] = []
        self.events_fired = 0
        self.state = AgentState()
        self.exchange = exchange
        self.llm_client = llm_client
        self.broker = PaperBroker(exchange, config.total_capital, trade_repo=trade_repo,
                                  bot_id=config.bot_id)
        self.atr = AtrCache(exchange, timeframe=config.atr_timeframe)
        self.equity_repo = equity_repo
        self.daily_repo = daily_repo
        self.decision_repo = decision_repo
        self.trade_repo = trade_repo
        self._tz = ZoneInfo(config.timezone)
        self._last_reset_date = datetime.now(self._tz).date()
        self._daily_start_equity = config.total_capital
        self._exit_task: Optional[asyncio.Task] = None

    # ── API-compatible read properties ──────────────────────────────────
    @property
    def total_pnl(self) -> float:
        return self.broker.equity - self.config.total_capital

    @property
    def daily_pnl(self) -> float:
        return self.broker.equity - self._daily_start_equity

    def positions(self) -> List[dict]:
        return self.broker.positions()

    def closed_today(self) -> List[AgentTrade]:
        start = datetime.now(self._tz).replace(hour=0, minute=0, second=0, microsecond=0)
        start_ms = int(start.timestamp() * 1000)
        return [t for t in self.broker.closed_trades if (t.exit_time or 0) >= start_ms]

    # ── restore after restart ───────────────────────────────────────────
    def restore(self) -> int:
        """Reload this bot's open positions and realized PnL from the DB.
        Returns the number of open positions restored. Blocking (DB I/O)."""
        if not self.trade_repo:
            return 0
        try:
            open_rows = self.trade_repo.get_open_trades(self.config.bot_id)
            closed_rows = self.trade_repo.get_closed_trades(self.config.bot_id)
        except Exception as exc:
            logger.error(f"AI agent restore failed, starting with an empty book: {exc}")
            return 0
        self.broker.restore([AgentTrade(**r) for r in open_rows],
                            [AgentTrade(**r) for r in closed_rows])
        self._daily_start_equity = (self.config.total_capital + self.broker.realized_pnl()
                                    - sum(t.pnl or 0.0 for t in self.closed_today()))
        logger.info(f"🤖 AI agent restored {len(open_rows)} open / {len(closed_rows)} closed trades "
                    f"(realized ${self.broker.realized_pnl():+.2f})")
        return len(open_rows)

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

    def _annotate_news(self, items: List[dict]) -> List[dict]:
        """Tag each news item with how many cycles ago it was first seen, so
        the model can tell a genuinely new headline from one it has already
        reacted to in a prior cycle. Prunes entries no longer in `items` —
        they've aged out of the source's lookback window already."""
        cycle = self.state.cycle_count
        tracker = self.state.news_first_seen_cycle
        out = []
        kept_keys = set()
        for item in items:
            key = item.get("url") or item.get("title")
            first_seen = tracker.setdefault(key, cycle)
            kept_keys.add(key)
            item = dict(item)
            item["cycles_since_first_seen"] = cycle - first_seen
            out.append(item)
        self.state.news_first_seen_cycle = {k: v for k, v in tracker.items() if k in kept_keys}
        return out

    def _prepare_order(self, args: dict) -> dict:
        """Fill the ATR-based default stop/target when the model omits them.
        Model-supplied values are left as-is for the guardrails to accept or
        reject (never silently clamped)."""
        args = dict(args)
        if args.get("stop_pct") is None:
            bracket = default_bracket(self.atr.atr_pct(args.get("symbol")),
                                      self.config.stop_atr_mult, self.config.target_atr_mult,
                                      self.config.min_stop_pct, self.config.max_stop_pct)
            args["stop_pct"] = bracket["stop_pct"]
            if args.get("target_pct") is None:
                args["target_pct"] = bracket["target_pct"]
        elif args.get("target_pct") is None and isinstance(args["stop_pct"], (int, float)):
            args["target_pct"] = args["stop_pct"] * (self.config.target_atr_mult / self.config.stop_atr_mult)
        return args

    def _cooldown_remaining(self, symbol: Optional[str]) -> float:
        last = self.broker.last_stop_at.get(symbol) if symbol else None
        if last is None:
            return 0.0
        return max(0.0, self.config.stop_cooldown_sec - (time.time() - last))

    def _order_guardrail_check(self, payload: dict):
        symbol = payload.get("symbol") if isinstance(payload, dict) else None
        return guardrails.validate_order_request(
            payload, equity=self.broker.equity, open_notional=self.broker.open_notional(),
            max_position_pct=self.config.max_position_pct,
            max_open_positions=self.config.max_open_positions,
            signed_notional=self.broker.signed_notional(),
            max_net_exposure_pct=self.config.max_net_exposure_pct,
            max_risk_pct=self.config.max_risk_per_trade,
            min_stop_pct=self.config.min_stop_pct,
            max_stop_pct=self.config.max_stop_pct,
            cooldown_remaining_sec=self._cooldown_remaining(symbol),
        )

    # ── live loops ───────────────────────────────────────────────────────
    def is_business_hours(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(self._tz)
        return (now.weekday() in self.config.business_days
                and self.config.business_hours_start <= now.hour < self.config.business_hours_end)

    def current_cycle_interval(self) -> int:
        return (self.config.cycle_interval_sec if self.is_business_hours()
                else self.config.off_hours_cycle_interval_sec)

    async def start(self):
        logger.info(f"🤖 AI agent engine starting (PAPER, {len(self.config.universe)} symbols, "
                    f"LLM heartbeat {self.config.cycle_interval_sec}s business hours / "
                    f"{self.config.off_hours_cycle_interval_sec}s off-hours, exits every "
                    f"{self.config.exit_check_interval_sec}s, events "
                    f"{'on' if self.stream else 'off'})")
        self._wake = asyncio.Queue()
        self._exit_task = asyncio.create_task(self._exit_monitor_loop())
        if self.stream:
            self._event_task = asyncio.create_task(self._event_loop())
        reasons: List[ev.WakeEvent] = []
        try:
            while not self.state.halted:
                try:
                    await self._cycle(reasons)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"AI agent cycle failed, continuing: {exc}")
                reasons = await self._wait_for_wake(self.current_cycle_interval())
        except asyncio.CancelledError:
            logger.info("AI agent engine cancelled")
        finally:
            for task in (self._exit_task, self._event_task):
                if task:
                    task.cancel()

    async def _wait_for_wake(self, timeout: float) -> List[ev.WakeEvent]:
        """Block until the heartbeat timeout or an event. Events respect
        min_llm_gap_sec; everything queued by then is handled in one cycle."""
        try:
            first = await asyncio.wait_for(self._wake.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return []
        gap = self.config.min_llm_gap_sec - (time.time() - self._last_llm_at)
        if gap > 0:
            await asyncio.sleep(gap)
        reasons = [first]
        while not self._wake.empty():
            reasons.append(self._wake.get_nowait())
        return reasons

    def _debounced(self, event: ev.WakeEvent, now: float) -> bool:
        last = self._event_last_fired.get(event.key)
        if last is not None and now - last < self.config.event_cooldown_sec:
            return True
        self._event_last_fired[event.key] = now
        return False

    def _detect_events(self) -> List[ev.WakeEvent]:
        """Blocking (ATR and news may hit REST): evaluate every detector."""
        c, s = self.config, self.stream
        found: List[ev.WakeEvent] = []
        for sym in c.universe:
            last = s.price(sym, max_age_sec=60)
            move = ev.price_move(sym, s.closes(sym), last, self.atr.atr_pct(sym),
                                 c.move_window_min, c.move_atr_mult)
            if move:
                found.append(move)
            burst = ev.liquidation_burst(sym, s.liquidation_buckets(sym, c.liq_bucket_sec, 25),
                                         c.liq_burst_mult, c.liq_min_usd)
            if burst:
                found.append(burst)
        for trade in list(self.broker.open_trades.values()):
            near = ev.near_stop(trade.symbol, trade.direction, trade.entry_price, trade.stop_loss,
                                s.price(trade.symbol, max_age_sec=60), c.near_stop_r)
            if near:
                found.append(near)
        now = time.time()
        if now - self._news_last_poll >= c.news_poll_sec:
            self._news_last_poll = now
            items = digest_for_symbols(c.universe)
            if self._news_seen is None:      # first poll only seeds: old headlines aren't events
                self._news_seen = {i.get("url") or i.get("title") for i in items}
            else:
                found.extend(ev.fresh_news(items, self._news_seen))
        return found

    async def _event_loop(self):
        while not self.state.halted:
            try:
                found = await asyncio.to_thread(self._detect_events)
                now = time.time()
                for event in found:
                    if not self._debounced(event, now):
                        self.events_fired += 1
                        logger.info(f"⚡ AI agent wake event: {event.detail}")
                        self._wake.put_nowait(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"AI agent event loop error: {exc}")
            await asyncio.sleep(self.config.event_check_interval_sec)

    async def _exit_monitor_loop(self):
        while not self.state.halted:
            try:
                await self._exit_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"AI agent exit monitor error: {exc}")
            await asyncio.sleep(self.config.exit_check_interval_sec)

    def _exit_check_sync(self) -> List[AgentTrade]:
        symbols = list(self.broker.open_trades.keys())
        prices = self.stream.prices(symbols, max_age_sec=60) if self.stream else {}
        missing = [s for s in symbols if s not in prices]
        if missing:     # stream down or stale → REST fallback
            prices.update(self.broker.fetch_prices(missing))
        closed = self.broker.check_exits(prices)
        self.broker.mark_to_market(prices)
        return closed

    async def _exit_check(self):
        """One pass of the no-LLM protection loop."""
        if self.broker.open_trades:
            await asyncio.to_thread(self._exit_check_sync)
        self.state.last_exit_check = time.time()
        await self._halt_if_drawdown()

    async def _halt_if_drawdown(self) -> bool:
        if self.state.halted or not guardrails.drawdown_tripped(
                self.broker.peak_equity, self.broker.equity, self.config.max_drawdown_kill):
            return False
        logger.critical(f"🛑 AI agent kill-switch: drawdown ≥ {self.config.max_drawdown_kill:.0%}. Halting.")
        await asyncio.to_thread(self.broker.close_all, "drawdown_kill_switch")
        self.state.halted = True
        self._log_decision(halted_reason="drawdown_kill_switch")
        return True

    async def _cycle(self, wake_reasons: Optional[List[ev.WakeEvent]] = None):
        self._check_daily_reset()
        self.state.cycle_count += 1
        self._wake_reasons = [e.detail for e in (wake_reasons or [])]

        await asyncio.to_thread(self.broker.mark_to_market)
        self._record_equity()

        if await self._halt_if_drawdown():
            return

        block_reason = self.pre_llm_block_reason()
        if block_reason:
            self._log_decision(halted_reason=block_reason)
            return

        await self._run_llm_turn()

    async def _run_llm_turn(self):
        dispatch = build_tool_dispatch(self.config.universe, self.exchange, self.broker,
                                       self._order_guardrail_check,
                                       news_tracker=self._annotate_news,
                                       order_preparer=self._prepare_order)
        if self._wake_reasons:
            trigger = ("You were woken early by market events: "
                       + "; ".join(self._wake_reasons)
                       + ". Check whether they matter for a trade or an open position. "
                         "Reacting is optional — an event alone is not a reason to trade.")
        else:
            trigger = "Scheduled check (no market event)."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Cycle #{self.state.cycle_count}. {trigger} Tradeable symbols: "
                f"{self.config.universe}. Equity ${self.broker.equity:.2f}. Check prices, news "
                f"and your current positions (with their stops/targets) before deciding: "
                f"open/add, close, or hold.")},
        ]
        self._last_llm_at = time.time()
        result = await asyncio.to_thread(
            self.llm_client.chat_with_tools, messages, TOOL_SCHEMAS, dispatch,
            max_tool_turns=self.config.max_tool_turns)

        self.state.llm_calls_today += 1
        self.state.tokens_today += result.prompt_tokens + result.completion_tokens

        rejections = []
        for call in result.tool_trace:
            if call["name"] in ("place_paper_order", "close_paper_position"):
                if call["output"].get("executed"):
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
        """Trades are persisted by the broker on every change; kept so callers
        that close positions outside the broker's normal path stay idempotent."""
        if not self.trade_repo:
            return
        for trade in trades:
            try:
                self.trade_repo.save(trade)
            except Exception as exc:
                logger.error(f"Failed to persist agent trade {trade.id}: {exc}")

    def _check_daily_reset(self):
        today = datetime.now(self._tz).date()
        if today != self._last_reset_date:
            if self.daily_repo:
                day_start = datetime.combine(self._last_reset_date, datetime.min.time(), self._tz)
                day_end = datetime.combine(today, datetime.min.time(), self._tz)
                lo, hi = int(day_start.timestamp() * 1000), int(day_end.timestamp() * 1000)
                closed = [t for t in self.broker.closed_trades if lo <= (t.exit_time or 0) < hi]
                wins = sum(1 for t in closed if (t.pnl or 0) > 0)
                losses = sum(1 for t in closed if (t.pnl or 0) <= 0)
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
                                             "positions": self.broker.positions(),
                                             "wake_reasons": self._wake_reasons}),
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
            if call["name"] in ("place_paper_order", "close_paper_position", "hold"):
                return call.get("args", {}).get("rationale")
        return None
