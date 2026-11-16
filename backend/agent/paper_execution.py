"""
Paper execution for the AI agent — synthesizes fills against REAL fetched
prices (not a synthetic weight-only ledger like portfolio_engine.py), same
commission+slippage model already used in production_trading_system.py's
paper path, so the agent's paper track record is comparable in kind to the
legacy per-symbol bot's.

Pure fill math (entry/exit price, pnl, exit triggers) is network-free and
unit-tested. I/O (fetch_ticker/fetch_tickers) is meant to be called via
asyncio.to_thread by the engine. The broker is touched from two threads (the
LLM turn and the exit monitor), so every state change holds `self._lock`.

Every open/add/close is persisted immediately through `trade_repo` (upsert by
trade id), so a restart can rebuild the book with `restore()` instead of
silently dropping open positions.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COMMISSION = 0.001   # 0.1% — same assumption as production_trading_system.py
SLIPPAGE = 0.0005     # 0.05%
ROUND_TRIP_COST = 2 * (COMMISSION + SLIPPAGE)


@dataclass
class AgentTrade:
    """Field-for-field compatible with production_trading_system.Trade, so it
    persists through the existing TradeRepository.save()."""
    id: str
    symbol: str
    direction: int          # 1 long / -1 short
    entry_time: int         # ms epoch
    entry_price: float
    quantity: float
    stop_loss: float        # 0.0 = no stop
    take_profit: float      # 0.0 = no target
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"
    reason: Optional[str] = None
    bot_id: str = "ai_agent"


def fill_entry_price(raw_price: float, direction: int) -> float:
    """Commission+slippage always work AGAINST the trader: buying costs more,
    selling nets less."""
    if direction == 1:
        return raw_price * (1 + COMMISSION + SLIPPAGE)
    return raw_price * (1 - COMMISSION - SLIPPAGE)


def fill_exit_price(raw_price: float, direction: int) -> float:
    """Exit is the opposite side of entry: closing a long = selling (nets
    less); closing a short = buying (costs more)."""
    if direction == 1:
        return raw_price * (1 - COMMISSION - SLIPPAGE)
    return raw_price * (1 + COMMISSION + SLIPPAGE)


def compute_pnl(entry_price: float, exit_price: float, quantity: float,
                direction: int) -> Dict[str, float]:
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    pnl = quantity * entry_price * pnl_pct
    return {"pnl": pnl, "pnl_pct": pnl_pct}


def bracket_levels(entry_price: float, direction: int, stop_pct: Optional[float],
                   target_pct: Optional[float]) -> Dict[str, float]:
    """Stop/target prices from distances expressed as a fraction of entry."""
    stop = entry_price * (1 - direction * stop_pct) if stop_pct else 0.0
    target = entry_price * (1 + direction * target_pct) if target_pct else 0.0
    return {"stop_loss": stop, "take_profit": target}


def exit_trigger(trade: AgentTrade, price: float) -> Optional[str]:
    """'stop' / 'target' if `price` crossed the trade's bracket, else None.
    Stop wins if both are somehow crossed (conservative)."""
    if trade.direction == 1:
        if trade.stop_loss and price <= trade.stop_loss:
            return "stop"
        if trade.take_profit and price >= trade.take_profit:
            return "target"
    else:
        if trade.stop_loss and price >= trade.stop_loss:
            return "stop"
        if trade.take_profit and price <= trade.take_profit:
            return "target"
    return None


class PaperBroker:
    """Dedicated paper book for the AI agent (own capital, own ledger)."""

    def __init__(self, exchange, total_capital: float, trade_repo=None,
                 bot_id: str = "ai_agent"):
        self.exchange = exchange
        self.total_capital = total_capital
        self.trade_repo = trade_repo
        self.bot_id = bot_id
        self.equity = total_capital
        self.peak_equity = total_capital
        self.open_trades: Dict[str, AgentTrade] = {}   # one position per symbol
        self.closed_trades: List[AgentTrade] = []
        self.last_prices: Dict[str, float] = {}
        self.last_stop_at: Dict[str, float] = {}       # symbol -> epoch sec of last stop-out
        self._lock = threading.RLock()

    # ── read helpers ─────────────────────────────────────────────────────
    def open_notional(self) -> Dict[str, float]:
        with self._lock:
            return {t.symbol: abs(t.quantity * t.entry_price) for t in self.open_trades.values()}

    def signed_notional(self) -> Dict[str, float]:
        with self._lock:
            return {t.symbol: t.direction * abs(t.quantity * t.entry_price)
                    for t in self.open_trades.values()}

    def positions(self) -> List[dict]:
        with self._lock:
            out = []
            for t in self.open_trades.values():
                last = self.last_prices.get(t.symbol)
                unreal = (compute_pnl(t.entry_price, last, t.quantity, t.direction)["pnl"]
                          if last else None)
                out.append({
                    "symbol": t.symbol, "side": "long" if t.direction == 1 else "short",
                    "entry_price": round(t.entry_price, 6),
                    "quantity": round(t.quantity, 8),
                    "notional": round(abs(t.quantity * t.entry_price), 2),
                    "stop_loss": round(t.stop_loss, 6) if t.stop_loss else None,
                    "take_profit": round(t.take_profit, 6) if t.take_profit else None,
                    "unrealized_pnl": round(unreal, 4) if unreal is not None else None,
                })
            return out

    def realized_pnl(self) -> float:
        with self._lock:
            return sum(t.pnl or 0.0 for t in self.closed_trades)

    # ── persistence ──────────────────────────────────────────────────────
    def _persist(self, trade: AgentTrade):
        if not self.trade_repo:
            return
        try:
            self.trade_repo.save(trade)
        except Exception as exc:
            logger.error(f"Failed to persist agent trade {trade.id}: {exc}")

    def restore(self, open_trades: List[AgentTrade], closed_trades: List[AgentTrade]):
        """Rebuild the book after a restart. Equity is recomputed on the next
        mark_to_market; until then it reflects realized PnL only."""
        with self._lock:
            self.open_trades = {t.symbol: t for t in open_trades}
            self.closed_trades = list(closed_trades)
            self.equity = self.total_capital + self.realized_pnl()
            self.peak_equity = max(self.total_capital, self.equity)

    # ── I/O (call via asyncio.to_thread) ────────────────────────────────
    def fetch_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Last price per symbol; one fetch_tickers call when the exchange
        supports it, per-symbol fallback otherwise. Failed symbols are omitted."""
        if not symbols:
            return {}
        prices: Dict[str, float] = {}
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            for sym in symbols:
                last = (tickers.get(sym) or {}).get("last")
                if last:
                    prices[sym] = float(last)
        except Exception:
            for sym in symbols:
                try:
                    prices[sym] = float(self.exchange.fetch_ticker(sym)["last"])
                except Exception:
                    continue
        return prices

    def place_order(self, symbol: str, side: str, notional_usd: float, rationale: str,
                    stop_pct: Optional[float] = None,
                    target_pct: Optional[float] = None) -> AgentTrade:
        """Open a position, or add to an existing SAME-side one (weighted
        average entry, same trade id). Opposite-side orders are refused here —
        exiting is close_position's job. Guardrails must have run already."""
        direction = 1 if side.lower() == "buy" else -1
        raw_price = float(self.exchange.fetch_ticker(symbol)["last"])
        entry_price = fill_entry_price(raw_price, direction)
        quantity = notional_usd / entry_price
        now_ms = int(time.time() * 1000)

        with self._lock:
            self.last_prices[symbol] = raw_price
            existing = self.open_trades.get(symbol)
            if existing and existing.direction != direction:
                raise ValueError(f"{symbol} has an open {'long' if existing.direction == 1 else 'short'}; "
                                 f"close it before trading the other side")
            if existing:
                total_qty = existing.quantity + quantity
                avg_entry = (existing.quantity * existing.entry_price
                             + quantity * entry_price) / total_qty
                existing.quantity = total_qty
                existing.entry_price = avg_entry
                if stop_pct or target_pct:
                    existing.stop_loss = bracket_levels(avg_entry, direction, stop_pct, None)["stop_loss"] \
                        if stop_pct else existing.stop_loss
                    existing.take_profit = bracket_levels(avg_entry, direction, None, target_pct)["take_profit"] \
                        if target_pct else existing.take_profit
                trade = existing
                action = "added to"
            else:
                levels = bracket_levels(entry_price, direction, stop_pct, target_pct)
                trade = AgentTrade(
                    id=f"{symbol.replace('/', '')}_{now_ms}",
                    symbol=symbol, direction=direction,
                    entry_time=now_ms, entry_price=entry_price, quantity=quantity,
                    stop_loss=levels["stop_loss"], take_profit=levels["take_profit"],
                    reason=rationale[:500] if rationale else None, bot_id=self.bot_id,
                )
                self.open_trades[symbol] = trade
                action = "opened"
            self._persist(trade)
        logger.info(f"🤖 AI agent {action} {side} {symbol} ${notional_usd:.2f} @ ${entry_price:.4f} "
                    f"(stop {trade.stop_loss:.4f} / target {trade.take_profit:.4f}) — "
                    f"{rationale[:120] if rationale else ''}")
        return trade

    def close_position(self, symbol: str, reason: str,
                       raw_price: Optional[float] = None) -> Optional[AgentTrade]:
        if symbol not in self.open_trades:
            return None
        if raw_price is None:
            raw_price = float(self.exchange.fetch_ticker(symbol)["last"])   # before popping: a failed fetch keeps the position
        with self._lock:
            trade = self.open_trades.pop(symbol, None)
            if trade is None:
                return None
            exit_price = fill_exit_price(raw_price, trade.direction)
            pnl_info = compute_pnl(trade.entry_price, exit_price, trade.quantity, trade.direction)
            trade.exit_time = int(time.time() * 1000)
            trade.exit_price = exit_price
            trade.pnl = pnl_info["pnl"]
            trade.pnl_pct = pnl_info["pnl_pct"]
            trade.status = "closed"
            trade.reason = reason[:500] if reason else reason
            self.closed_trades.append(trade)
            self.equity = (self.total_capital + self.realized_pnl()
                           + self._unrealized(self.last_prices))
            self.peak_equity = max(self.peak_equity, self.equity)
            if reason == "stop":
                self.last_stop_at[symbol] = time.time()
            self._persist(trade)
        logger.info(f"🤖 AI agent closed {symbol}: pnl=${trade.pnl:+.2f} "
                    f"({trade.pnl_pct:+.2%}) reason={reason}")
        return trade

    def close_all(self, reason: str) -> List[AgentTrade]:
        closed = []
        for sym in list(self.open_trades.keys()):
            try:
                t = self.close_position(sym, reason)
            except Exception as exc:
                logger.error(f"close_all: failed to close {sym}: {exc}")
                continue
            if t is not None:
                closed.append(t)
        return closed

    def check_exits(self, prices: Dict[str, float]) -> List[AgentTrade]:
        """Close every open position whose stop or target was crossed at the
        given prices. Fills at the observed price, not at the stop level —
        a gap past the stop costs what it would cost live."""
        to_close = []
        with self._lock:
            for sym, trade in self.open_trades.items():
                price = prices.get(sym)
                if price is None:
                    continue
                trigger = exit_trigger(trade, price)
                if trigger:
                    to_close.append((sym, trigger, price))
        closed = []
        for sym, trigger, price in to_close:
            t = self.close_position(sym, trigger, raw_price=price)
            if t is not None:
                closed.append(t)
        return closed

    def _unrealized(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for trade in self.open_trades.values():
            price = prices.get(trade.symbol)
            if price is None:
                continue
            total += compute_pnl(trade.entry_price, price, trade.quantity, trade.direction)["pnl"]
        return total

    def mark_to_market(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Recompute equity from realized + unrealized. Fetches prices if not
        given; a symbol whose price is unavailable contributes 0 unrealized."""
        if prices is None:
            prices = self.fetch_prices(list(self.open_trades.keys()))
        with self._lock:
            self.last_prices.update(prices)
            self.equity = self.total_capital + self.realized_pnl() + self._unrealized(prices)
            self.peak_equity = max(self.peak_equity, self.equity)
            return self.equity
