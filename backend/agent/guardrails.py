"""
Guardrails for the AI trading agent — pure functions, no I/O, no network.

These are independent of the LLM's own "judgment": every tool call the model
makes is validated here BEFORE it is allowed to touch PaperBroker, and every
cycle checks the kill-switches here BEFORE the LLM is even called (saves cost
on a halted day). Never trust free-form model output as an order directly —
validate_order_request() is the single choke point every proposed trade must
pass through.
"""
from __future__ import annotations

from typing import Dict, Tuple

# v1 basket — the model can never propose a symbol outside this list.
SYMBOL_ALLOWLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

VALID_SIDES = ("buy", "sell")


def validate_order_request(payload: dict, equity: float, open_notional: Dict[str, float],
                           max_position_pct: float = 0.15,
                           max_open_positions: int = 3) -> Tuple[bool, str]:
    """
    Validate a proposed order BEFORE any paper fill. Rejects (doesn't clamp)
    so a rejection is always visible/loggable rather than silently resized.

    `payload`: dict with "symbol", "side", "notional_usd" (as the model's tool
    call would supply, already JSON-decoded).
    `open_notional`: {symbol: abs(notional)} of currently open positions,
    used to enforce max_open_positions without double-counting a symbol
    the agent is already in (adding to an existing position doesn't count
    as a NEW position for the count cap, but still respects the size cap).
    """
    if not isinstance(payload, dict):
        return False, "payload is not a dict"

    symbol = payload.get("symbol")
    side = payload.get("side")
    notional = payload.get("notional_usd")

    if not isinstance(symbol, str) or symbol not in SYMBOL_ALLOWLIST:
        return False, f"symbol '{symbol}' not in allow-list {SYMBOL_ALLOWLIST}"
    if not isinstance(side, str) or side.lower() not in VALID_SIDES:
        return False, f"side '{side}' must be one of {VALID_SIDES}"
    if not isinstance(notional, (int, float)) or isinstance(notional, bool):
        return False, f"notional_usd must be a number, got {type(notional).__name__}"
    if notional <= 0:
        return False, "notional_usd must be positive"
    if equity <= 0:
        return False, "equity is non-positive — cannot size a new position"

    max_notional = equity * max_position_pct
    if notional > max_notional:
        return False, (f"notional ${notional:.2f} exceeds max_position_pct cap "
                       f"(${max_notional:.2f} = {max_position_pct:.0%} of equity ${equity:.2f})")

    is_new_symbol = symbol not in open_notional or abs(open_notional[symbol]) < 1e-9
    if is_new_symbol and len(open_notional) >= max_open_positions:
        return False, (f"already at max_open_positions ({max_open_positions}) and "
                       f"'{symbol}' would be a new position")

    return True, "ok"


def daily_loss_tripped(daily_pnl: float, start_equity: float, limit_pct: float = 0.05) -> bool:
    """True once today's loss reaches `limit_pct` of the day's starting equity."""
    if start_equity <= 0:
        return False
    return (-daily_pnl / start_equity) >= limit_pct


def drawdown_tripped(peak_equity: float, equity: float, limit_pct: float = 0.20) -> bool:
    """True once equity has fallen `limit_pct` from its running peak."""
    if peak_equity <= 0:
        return False
    dd = (peak_equity - equity) / peak_equity
    return dd >= limit_pct


def daily_trade_cap_exceeded(trades_today: int, cap: int = 10) -> bool:
    return trades_today >= cap


def llm_call_budget_exceeded(calls_today: int, cap: int = 150) -> bool:
    return calls_today >= cap
