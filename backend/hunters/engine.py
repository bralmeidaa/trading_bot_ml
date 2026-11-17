"""
Hunter engine: deterministic strategies propose, the LLM judges, the paper
broker executes — on its own book (bot_id hunter_judge_v1), separate from the
free-form LLM agent so the two can be compared side by side.

Every 5-minute bar close:
  1. refresh 5m bars (REST, incremental) and compute features;
  2. run every hunter on the latest closed bar;
  3. shadow-only hunters (the control) go straight to the shadow ledger;
  4. every live candidate goes to ONE LLM call that approves / halves /
     vetoes each — funded or not, so the judge is measured on all of them;
  5. an approved candidate is executed only if the tournament allocator
     gives its hunter capital (risk scale > 0, earned by the forward record
     of the judge's approved picks; NO-GO in replay → none until then) and
     it passes the book's gates (risk, exposure, cooldown, one position per
     symbol);
  6. every candidate, whatever happened, is followed in the shadow ledger
     with the judge's verdict.

A separate 30s loop enforces stops/targets/time limits on the real book,
advances the shadow ledger and checks the drawdown kill-switch.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from backend.agent import guardrails
from backend.agent.paper_execution import ROUND_TRIP_COST, AgentTrade, PaperBroker
from backend.agent.tools import build_tool_dispatch
from backend.allocator.thompson import AllocatorConfig, allocate
from backend.hunters.base import Candidate, Hunter, ScanContext
from backend.hunters.features import prepare
from backend.hunters.judge import run_judge
from backend.hunters.shadow import TAKEN, ShadowBook
from backend.hunters.strategies import default_hunters

logger = logging.getLogger(__name__)

HUNTER_UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "LTC/USDT",
    "ATOM/USDT", "UNI/USDT", "AAVE/USDT", "NEAR/USDT", "DOGE/USDT",
]
BAR_MS = 5 * 60 * 1000


@dataclass
class HunterConfig:
    universe: List[str] = field(default_factory=lambda: list(HUNTER_UNIVERSE))
    total_capital: float = 500.0
    bot_id: str = "hunter_judge_v1"
    bar_history: int = 600                  # 50h of 5m bars: today + a full prior day
    scan_delay_sec: int = 15                # after the bar closes, so Binance has it
    exit_check_interval_sec: int = 30
    max_position_pct: float = 0.15
    max_open_positions: int = 4
    max_risk_per_trade: float = 0.01
    max_net_exposure_pct: float = 0.30
    min_stop_pct: float = 4 * ROUND_TRIP_COST
    max_stop_pct: float = 0.08
    max_hold_sec: int = 24 * 3600
    stop_cooldown_sec: int = 1800
    candidate_cooldown_sec: int = 3600      # same hunter+symbol at most once an hour
    daily_loss_limit: float = 0.05
    max_drawdown_kill: float = 0.20
    max_llm_calls_per_day: int = 60
    max_tool_turns: int = 6
    # tournament allocator (backend/allocator)
    alloc_min_trades: int = 30
    alloc_explore_scale: float = 0.5
    alloc_demote_below: float = 0.5


@dataclass
class HunterState:
    halted: bool = False
    scans: int = 0
    candidates_today: int = 0
    executed_today: int = 0
    llm_calls_today: int = 0
    tokens_today: int = 0
    day: Optional[str] = None
    last_scan_at: Optional[float] = None
    last_scan_summary: dict = field(default_factory=dict)
    last_exit_check: Optional[float] = None


class HunterEngine:
    def __init__(self, config: HunterConfig, exchange, llm_client, trade_repo=None,
                 shadow_repo=None, stream=None, hunters: Optional[List[Hunter]] = None,
                 notifier: Optional[Callable[[str, str], None]] = None):
        self.config = config
        self.state = HunterState()
        self.exchange = exchange
        self.llm_client = llm_client
        self.trade_repo = trade_repo
        self.shadow_repo = shadow_repo
        self.stream = stream
        self.hunters = hunters if hunters is not None else default_hunters()
        self.broker = PaperBroker(exchange, config.total_capital, trade_repo=trade_repo,
                                  bot_id=config.bot_id)
        self.shadow = ShadowBook(shadow_repo, max_hold_sec=config.max_hold_sec)
        self.bars: Dict[str, pd.DataFrame] = {}
        self._last_candidate_ms: Dict[str, int] = {}
        self._daily_start_equity = config.total_capital
        self._news_seen_call: Dict[str, int] = {}
        self._exit_task: Optional[asyncio.Task] = None
        self.notifier = notifier                       # (subject, body) — email in prod
        self.allocation: Dict[str, dict] = {}
        self.replay_verdicts: Dict[str, str] = {}
        self.lessons: List[str] = []                   # student lessons for the judge prompt
        self._closed_shadow: List[dict] = []           # in-memory evidence when there's no repo

    # ── read helpers ─────────────────────────────────────────────────────
    @property
    def total_pnl(self) -> float:
        return self.broker.equity - self.config.total_capital

    def restore(self) -> int:
        n = 0
        if self.trade_repo:
            try:
                open_rows = self.trade_repo.get_open_trades(self.config.bot_id)
                closed_rows = self.trade_repo.get_closed_trades(self.config.bot_id)
                self.broker.restore([AgentTrade(**r) for r in open_rows],
                                    [AgentTrade(**r) for r in closed_rows])
                n = len(open_rows)
            except Exception as exc:
                logger.error(f"hunter restore (trades) failed: {exc}")
        if self.shadow_repo:
            try:
                self.shadow.restore(self.shadow_repo.get_open())
            except Exception as exc:
                logger.error(f"hunter restore (shadow) failed: {exc}")
        self._daily_start_equity = self.broker.equity
        self.refresh_allocation(notify=False)
        return n

    # ── tournament ───────────────────────────────────────────────────────
    def _evidence(self) -> Dict[str, List[float]]:
        """R of every closed shadow candidate the JUDGE APPROVED, per hunter —
        executed or not. What trades live is "hunter filtered by the judge",
        so that is what earns capital (a raw-negative hunter can still earn
        it if the judge's picks are positive)."""
        rows = self._closed_shadow
        if self.shadow_repo:
            try:
                rows = self.shadow_repo.get_closed()
            except Exception as exc:
                logger.warning(f"allocator: shadow read failed: {exc}")
        out: Dict[str, List[float]] = {}
        for r in rows:
            if r.get("r_multiple") is not None and r.get("decision") in TAKEN:
                out.setdefault(r["hunter"], []).append(float(r["r_multiple"]))
        return out

    def set_replay_verdicts(self, report: Optional[dict]):
        """{hunter: GO / NO-GO / INSUFFICIENT} from a replay report: a seeded
        hunter that is NO-GO in replay gets no exploration capital."""
        self.replay_verdicts = {name: rep.get("verdict") for name, rep in
                                ((report or {}).get("report") or {}).items()}
        self.refresh_allocation(notify=False)

    def refresh_allocation(self, notify: bool = True) -> Dict[str, dict]:
        """Recompute every hunter's tournament status and risk scale from its
        closed shadow trades. Status changes are announced via `notifier`."""
        cfg = AllocatorConfig(min_trades=self.config.alloc_min_trades,
                              explore_scale=self.config.alloc_explore_scale,
                              demote_below=self.config.alloc_demote_below)
        meta = {h.name: {"live": h.live, "variant": h.parent is not None,
                         "replay": self.replay_verdicts.get(h.name)} for h in self.hunters}
        new = allocate(self._evidence(), meta, cfg)
        changes = [(name, self.allocation[name]["status"], row)
                   for name, row in new.items()
                   if name in self.allocation and self.allocation[name]["status"] != row["status"]]
        self.allocation = new
        if notify and changes and self.notifier:
            lines = [f"{name}: {old} -> {row['status']} (n={row['n']}, mean {row['mean_r']}R, "
                     f"P(edge>0)={row['p_positive']}, risk scale {row['risk_scale']})"
                     for name, old, row in changes]
            try:
                self.notifier(f"[hunters] tournament: {len(changes)} status change(s)",
                              "\n".join(lines))
            except Exception as exc:
                logger.warning(f"allocator notice failed: {exc}")
        return new

    def risk_scale(self, hunter: Hunter) -> float:
        row = self.allocation.get(hunter.name)
        if row is not None:
            return float(row["risk_scale"])
        if (not hunter.live or hunter.parent is not None
                or self.replay_verdicts.get(hunter.name) == "NO-GO"):
            return 0.0
        return self.config.alloc_explore_scale

    def set_variants(self, variants: List[Hunter]):
        """Swap the student's variants in (seeded hunters stay)."""
        self.hunters = [h for h in self.hunters if h.parent is None] + list(variants)
        self.refresh_allocation(notify=False)

    # ── market data ──────────────────────────────────────────────────────
    def _fetch_bars(self, symbol: str, now_ms: int) -> pd.DataFrame:
        cached = self.bars.get(symbol)
        if cached is None or cached.empty:
            rows = self.exchange.fetch_ohlcv(symbol, "5m", limit=self.config.bar_history)
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            since = int(cached["timestamp"].iloc[-1])
            rows = self.exchange.fetch_ohlcv(symbol, "5m", since=since, limit=100)
            new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = (pd.concat([cached, new]).drop_duplicates("timestamp", keep="last")
                  .sort_values("timestamp"))
        df = df[df["timestamp"] + BAR_MS <= now_ms]            # closed bars only
        df = df.tail(self.config.bar_history).reset_index(drop=True)
        self.bars[symbol] = df
        return df

    def _prices(self, symbols: List[str]) -> Dict[str, float]:
        symbols = list(dict.fromkeys(symbols))
        prices = self.stream.prices(symbols, max_age_sec=60) if self.stream else {}
        missing = [s for s in symbols if s not in prices]
        if missing:
            prices.update(self.broker.fetch_prices(missing))
        return prices

    def _scan_context(self) -> ScanContext:
        liq = None
        if self.stream is not None:
            def liq(symbol, side, bucket_sec, n):
                return self.stream.liquidation_buckets(symbol, bucket_sec, n, side=side)
        return ScanContext(min_stop_pct=self.config.min_stop_pct,
                           max_stop_pct=self.config.max_stop_pct, liquidations=liq)

    # ── gates ────────────────────────────────────────────────────────────
    def _block_reason(self) -> Optional[str]:
        if self.state.halted:
            return "halted"
        if guardrails.daily_loss_tripped(self.broker.equity - self._daily_start_equity,
                                         self._daily_start_equity, self.config.daily_loss_limit):
            return "daily_loss_limit"
        return None

    def _sized_payload(self, c: Candidate, factor: float = 1.0) -> dict:
        eq = self.broker.equity
        notional = min(self.config.max_risk_per_trade * eq / c.stop_pct,
                       self.config.max_position_pct * eq) * factor
        return {"symbol": c.symbol, "side": c.side, "notional_usd": math.floor(notional * 100) / 100,
                "stop_pct": c.stop_pct, "target_pct": c.target_pct}

    def _gate(self, c: Candidate, payload: dict) -> Optional[str]:
        if c.symbol in self.broker.open_trades:
            return "symbol already has an open position in this book"
        last_stop = self.broker.last_stop_at.get(c.symbol)
        cooldown = (max(0.0, self.config.stop_cooldown_sec - (time.time() - last_stop))
                    if last_stop else 0.0)
        ok, reason = guardrails.validate_order_request(
            payload, equity=self.broker.equity, open_notional=self.broker.open_notional(),
            max_position_pct=self.config.max_position_pct,
            max_open_positions=self.config.max_open_positions,
            signed_notional=self.broker.signed_notional(),
            max_net_exposure_pct=self.config.max_net_exposure_pct,
            max_risk_pct=self.config.max_risk_per_trade,
            min_stop_pct=self.config.min_stop_pct, max_stop_pct=self.config.max_stop_pct,
            cooldown_remaining_sec=cooldown, allowlist=self.config.universe)
        return None if ok else reason

    # ── one scan (blocking: REST + LLM; run via asyncio.to_thread) ──────
    def scan_once(self, now_ms: Optional[int] = None) -> dict:
        now_ms = now_ms or int(time.time() * 1000)
        self._roll_day(now_ms)
        ctx = self._scan_context()
        found: List[Candidate] = []
        errors = 0
        for sym in self.config.universe:
            try:
                raw = self._fetch_bars(sym, now_ms)
                if len(raw) < 100:
                    continue
                df = prepare(raw)
            except Exception as exc:
                errors += 1
                logger.warning(f"hunters: bars/features failed for {sym}: {exc}")
                continue
            for h in self.hunters:
                try:
                    c = h.scan(sym, df, ctx)
                except Exception as exc:
                    logger.warning(f"hunter {h.name} failed on {sym}: {exc}")
                    continue
                if c is None or self.shadow.has(c.id):
                    continue
                key = f"{c.hunter}:{c.symbol}"
                if now_ms - self._last_candidate_ms.get(key, 0) < self.config.candidate_cooldown_sec * 1000:
                    continue
                self._last_candidate_ms[key] = now_ms
                found.append(c)

        summary = {"candidates": len(found), "shadow_only": 0, "judged": 0, "unfunded": 0,
                   "gated": 0, "executed": 0, "errors": errors}
        if not found:
            return self._finish_scan(summary)
        self.state.candidates_today += len(found)
        prices = self._prices([c.symbol for c in found])

        # Every live candidate is judged, funded or not: the judge's verdict is
        # recorded as-is (that's what the tournament allocates on), and only
        # execution depends on tournament capital and the book's gates.
        to_judge: List[Candidate] = []
        for c in found:
            if not c.live:
                self.shadow.open(c, "shadow_only", "shadow-only hunter",
                                 prices.get(c.symbol, c.ref_price), now_ms)
                summary["shadow_only"] += 1
            else:
                to_judge.append(c)
        if not to_judge:
            return self._finish_scan(summary)

        verdicts = self._judge(to_judge)
        summary["judged"] = len(to_judge)
        block = self._block_reason()
        scales = {h.name: self.risk_scale(h) for h in self.hunters}
        for c in to_judge:
            decision, rationale = verdicts[c.id]
            price = prices.get(c.symbol, c.ref_price)
            trade_id = None
            if decision in TAKEN:
                scale = scales.get(c.hunter, 0.0)
                payload = self._sized_payload(c, scale * (0.5 if decision == "reduced" else 1.0))
                if scale <= 0:
                    st = self.allocation.get(c.hunter, {}).get("status", "unallocated")
                    rationale = f"[not executed: no tournament capital ({st})] {rationale}"
                    summary["unfunded"] += 1
                elif (reason := block or self._gate(c, payload)):   # earlier fills change exposure
                    rationale = f"[not executed: {reason}] {rationale}"
                    summary["gated"] += 1
                else:
                    try:
                        trade = self.broker.place_order(
                            c.symbol, c.side, payload["notional_usd"],
                            f"[{c.hunter}] {c.rationale} | judge: {rationale}",
                            stop_pct=c.stop_pct, target_pct=c.target_pct)
                        trade_id = trade.id
                        summary["executed"] += 1
                        self.state.executed_today += 1
                    except Exception as exc:
                        rationale = f"[not executed: {exc}] {rationale}"
            self.shadow.open(c, decision, rationale, price, now_ms, live_trade_id=trade_id)
        return self._finish_scan(summary)

    def _finish_scan(self, summary: dict) -> dict:
        self.state.scans += 1
        self.state.last_scan_at = time.time()
        self.state.last_scan_summary = summary
        if summary.get("candidates"):
            logger.info(f"🎯 hunters scan: {summary}")
        return summary

    def _judge(self, candidates: List[Candidate]) -> Dict[str, tuple]:
        if self.state.llm_calls_today >= self.config.max_llm_calls_per_day:
            return {c.id: ("budget", "daily LLM budget exhausted") for c in candidates}
        handlers = build_tool_dispatch(self.config.universe, self.exchange, self.broker,
                                       guardrail_check=lambda p: (False, "read-only"),
                                       news_tracker=self._annotate_news)
        verdicts, result = run_judge(self.llm_client, candidates, handlers,
                                     max_tool_turns=self.config.max_tool_turns,
                                     lessons=self.lessons)
        self.state.llm_calls_today += 1
        self.state.tokens_today += (result.prompt_tokens or 0) + (result.completion_tokens or 0)
        return verdicts

    def _annotate_news(self, items: List[dict]) -> List[dict]:
        """Same freshness tag the agent uses, counted in scans (5 min each)."""
        call = self.state.scans
        out, keep = [], {}
        for item in items:
            key = item.get("url") or item.get("title")
            first = self._news_seen_call.get(key, call)
            keep[key] = first
            out.append({**item, "cycles_since_first_seen": call - first})
        self._news_seen_call = keep
        return out

    def _roll_day(self, now_ms: int):
        day = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).date().isoformat()
        if day != self.state.day:
            if self.state.day is not None:
                self._daily_start_equity = self.broker.equity
                self.refresh_allocation()
            self.state.day = day
            self.state.candidates_today = self.state.executed_today = 0
            self.state.llm_calls_today = self.state.tokens_today = 0

    # ── exits (no LLM) ───────────────────────────────────────────────────
    def exit_check_once(self, now_ms: Optional[int] = None) -> List[AgentTrade]:
        now_ms = now_ms or int(time.time() * 1000)
        symbols = list(self.broker.open_trades) + [r["symbol"] for r in self.shadow.open_rows.values()]
        if not symbols:
            return []
        prices = self._prices(symbols)
        closed = self.broker.check_exits(prices)
        for sym, trade in list(self.broker.open_trades.items()):
            if now_ms - trade.entry_time >= self.config.max_hold_sec * 1000 and sym in prices:
                t = self.broker.close_position(sym, "time", raw_price=prices[sym])
                if t:
                    closed.append(t)
        for sym in set(r["symbol"] for r in self.shadow.open_rows.values()):
            if sym in prices:
                done = self.shadow.update(sym, now_ms, price=prices[sym])
                if not self.shadow_repo:
                    self._closed_shadow.extend(done)
        self.broker.mark_to_market({s: p for s, p in prices.items() if s in self.broker.open_trades})
        return closed

    # ── loops ────────────────────────────────────────────────────────────
    async def start(self):
        logger.info(f"🎯 hunter engine starting ({len(self.hunters)} hunters × "
                    f"{len(self.config.universe)} symbols, book {self.config.bot_id})")
        self._exit_task = asyncio.create_task(self._exit_loop())
        try:
            while not self.state.halted:
                now = time.time()
                next_close = (int(now * 1000) // BAR_MS + 1) * BAR_MS / 1000
                await asyncio.sleep(max(1.0, next_close - now + self.config.scan_delay_sec))
                try:
                    await asyncio.to_thread(self.scan_once)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"hunter scan failed, continuing: {exc}")
        except asyncio.CancelledError:
            logger.info("hunter engine cancelled")
        finally:
            if self._exit_task:
                self._exit_task.cancel()

    async def _exit_loop(self):
        while not self.state.halted:
            try:
                await asyncio.to_thread(self.exit_check_once)
                self.state.last_exit_check = time.time()
                if guardrails.drawdown_tripped(self.broker.peak_equity, self.broker.equity,
                                               self.config.max_drawdown_kill):
                    logger.critical("🛑 hunter book kill-switch: drawdown limit hit. Halting.")
                    await asyncio.to_thread(self.broker.close_all, "drawdown_kill_switch")
                    self.state.halted = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"hunter exit loop error: {exc}")
            await asyncio.sleep(self.config.exit_check_interval_sec)

    def status(self) -> dict:
        s = self.state
        realized = self.broker.realized_pnl()
        return {
            "bot_id": self.config.bot_id, "halted": s.halted,
            "equity": round(self.broker.equity, 2), "total_pnl": round(self.total_pnl, 2),
            "realized_pnl": round(realized, 2),
            "positions": self.broker.positions(),
            "shadow_open": len(self.shadow.open_rows),
            "scans": s.scans, "candidates_today": s.candidates_today,
            "executed_today": s.executed_today, "judge_calls_today": s.llm_calls_today,
            "tokens_today": s.tokens_today,
            "last_scan_at": (datetime.utcfromtimestamp(s.last_scan_at).isoformat()
                             if s.last_scan_at else None),
            "last_scan": s.last_scan_summary,
            "hunters": [{"name": h.name, "version": h.version, "live": h.live, "parent": h.parent,
                         "tournament": self.allocation.get(h.name)} for h in self.hunters],
            "lessons": self.lessons,
            "universe": self.config.universe,
        }
