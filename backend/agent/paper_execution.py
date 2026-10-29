"""
Paper execution for the AI agent — synthesizes fills against REAL fetched
prices (not a synthetic weight-only ledger like portfolio_engine.py), same
commission+slippage model already used in production_trading_system.py's
paper path, so the agent's paper track record is comparable in kind to the
legacy per-symbol bot's.

Pure fill math (entry/exit price, pnl) is network-free and unit-tested.
I/O (fetch_ticker) is isolated in mark_to_market()/place_order(), both meant
to be called via asyncio.to_thread by the engine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COMMISSION = 0.001   # 0.1% — same assumption as production_trading_system.py
SLIPPAGE = 0.0005     # 0.05%


@dataclass
class AgentTrade:
    """Field-for-field compatible with production_trading_system.Trade, so a
    closed trade can be persisted via the EXISTING TradeRepository.save()
    (backend/persistence/repository.py) with zero changes to that code —
    bot_id="ai_agent" is the free-text discriminator it already supports."""
    id: str
    symbol: str
    direction: int          # 1 long / -1 short
    entry_time: int         # ms epoch
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"
    reason: Optional[str] = None
    bot_id: str = "ai_agent"


def fill_entry_price(raw_price: float, direction: int) -> float:
    """Commission+slippage always work AGAINST the trader: buying costs more,
    selling nets less, regardless of direction. Entry: buy to go long (pay up),
    sell-short-style entry to go short still means selling into the market so
    it fills a bit lower (same cost direction as a real market order)."""
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


class PaperBroker:
    """Dedicated paper book for the AI agent — separate from the
    cross-sectional engine's equity entirely (own capital, own ledger)."""

    def __init__(self, exchange, total_capital: float):
        self.exchange = exchange
        self.total_capital = total_capital
        self.equity = total_capital
        self.peak_equity = total_capital
        self.open_trades: Dict[str, AgentTrade] = {}   # keyed by symbol (one position/symbol)
        self.closed_trades: List[AgentTrade] = []

    # ── read helpers ─────────────────────────────────────────────────────
    def open_notional(self) -> Dict[str, float]:
        return {t.symbol: abs(t.quantity * t.entry_price) for t in self.open_trades.values()}

    def positions(self) -> List[dict]:
        return [
            {"symbol": t.symbol, "side": "long" if t.direction == 1 else "short",
             "entry_price": round(t.entry_price, 6),
             "quantity": round(t.quantity, 8),
             "notional": round(abs(t.quantity * t.entry_price), 2)}
            for t in self.open_trades.values()
        ]

    # ── I/O (call via asyncio.to_thread) ────────────────────────────────
    def place_order(self, symbol: str, side: str, notional_usd: float,
                    rationale: str) -> AgentTrade:
        """Open (or add to) a position at the current fetched price. Assumes
        the caller already ran validate_order_request() — this does not
        re-validate guardrails, it just executes the fill."""
        direction = 1 if side.lower() == "buy" else -1
        raw_price = float(self.exchange.fetch_ticker(symbol)["last"])
        entry_price = fill_entry_price(raw_price, direction)
        quantity = notional_usd / entry_price

        trade = AgentTrade(
            id=f"{symbol.replace('/', '')}_{int(time.time() * 1000)}",
            symbol=symbol, direction=direction,
            entry_time=int(time.time() * 1000), entry_price=entry_price,
            quantity=quantity, stop_loss=0.0, take_profit=0.0,
            reason=rationale[:500] if rationale else None,
        )
        self.open_trades[symbol] = trade
        logger.info(f"🤖 AI agent paper order: {side} {symbol} ${notional_usd:.2f} "
                    f"@ ${entry_price:.4f} — {rationale[:120] if rationale else ''}")
        return trade

    def close_position(self, symbol: str, reason: str) -> Optional[AgentTrade]:
        trade = self.open_trades.pop(symbol, None)
        if trade is None:
            return None
        raw_price = float(self.exchange.fetch_ticker(symbol)["last"])
        exit_price = fill_exit_price(raw_price, trade.direction)
        pnl_info = compute_pnl(trade.entry_price, exit_price, trade.quantity, trade.direction)
        trade.exit_time = int(time.time() * 1000)
        trade.exit_price = exit_price
        trade.pnl = pnl_info["pnl"]
        trade.pnl_pct = pnl_info["pnl_pct"]
        trade.status = "closed"
        trade.reason = reason
        self.equity += trade.pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.closed_trades.append(trade)
        logger.info(f"🤖 AI agent closed {symbol}: pnl=${trade.pnl:+.2f} "
                    f"({trade.pnl_pct:+.2%}) reason={reason}")
        return trade

    def close_all(self, reason: str) -> List[AgentTrade]:
        return [t for sym in list(self.open_trades.keys())
               if (t := self.close_position(sym, reason)) is not None]

    def mark_to_market(self) -> float:
        """Update equity from unrealized moves on open positions (does not
        close anything). Returns the current equity."""
        unrealized = 0.0
        for trade in self.open_trades.values():
            try:
                price = float(self.exchange.fetch_ticker(trade.symbol)["last"])
            except Exception:
                continue
            pnl_info = compute_pnl(trade.entry_price, price, trade.quantity, trade.direction)
            unrealized += pnl_info["pnl"]
        realized_equity = self.total_capital + sum(t.pnl or 0.0 for t in self.closed_trades)
        self.equity = realized_equity + unrealized
        self.peak_equity = max(self.peak_equity, self.equity)
        return self.equity
