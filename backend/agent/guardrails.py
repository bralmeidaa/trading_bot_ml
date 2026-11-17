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

from typing import Dict, List, Optional, Tuple

# v1 basket — the model can never propose a symbol outside this list.
SYMBOL_ALLOWLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

VALID_SIDES = ("buy", "sell")


def validate_order_request(payload: dict, equity: float, open_notional: Dict[str, float],
                           max_position_pct: float = 0.15,
                           max_open_positions: int = 3,
                           signed_notional: Optional[Dict[str, float]] = None,
                           max_net_exposure_pct: Optional[float] = None,
                           max_risk_pct: Optional[float] = None,
                           min_stop_pct: Optional[float] = None,
                           max_stop_pct: Optional[float] = None,
                           cooldown_remaining_sec: float = 0.0,
                           allowlist: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate a proposed order BEFORE any paper fill. Rejects (doesn't clamp)
    so a rejection is always visible/loggable rather than silently resized.

    `payload`: dict with "symbol", "side", "notional_usd" and optionally
    "stop_pct"/"target_pct" (fractions of entry).
    `open_notional`: {symbol: abs(notional)} of currently open positions.
    `signed_notional`: {symbol: +notional long / -notional short}; when given,
    an opposite-side order on an open symbol is rejected (exits go through the
    close tool), the size cap applies to the cumulative position, and
    `max_net_exposure_pct` caps |Σ long − Σ short| so three same-direction
    positions on correlated coins can't add up to one oversized bet.
    The optional checks are skipped when their limit is None.
    """
    if not isinstance(payload, dict):
        return False, "payload is not a dict"

    symbol = payload.get("symbol")
    side = payload.get("side")
    notional = payload.get("notional_usd")

    allowed = allowlist or SYMBOL_ALLOWLIST
    if not isinstance(symbol, str) or symbol not in allowed:
        return False, f"symbol '{symbol}' not in allow-list {allowed}"
    if not isinstance(side, str) or side.lower() not in VALID_SIDES:
        return False, f"side '{side}' must be one of {VALID_SIDES}"
    if not isinstance(notional, (int, float)) or isinstance(notional, bool):
        return False, f"notional_usd must be a number, got {type(notional).__name__}"
    if notional <= 0:
        return False, "notional_usd must be positive"
    if equity <= 0:
        return False, "equity is non-positive — cannot size a new position"

    direction = 1 if side.lower() == "buy" else -1
    existing_signed = (signed_notional or {}).get(symbol, 0.0)
    if signed_notional is not None and existing_signed * direction < 0:
        return False, (f"'{symbol}' has an open position on the other side — call "
                       f"close_paper_position first; orders never flip a position")

    if cooldown_remaining_sec > 0:
        return False, (f"'{symbol}' stopped out recently — cooldown, "
                       f"{cooldown_remaining_sec / 60:.0f} min left")

    max_notional = equity * max_position_pct
    position_after = abs(existing_signed) + notional if signed_notional is not None else notional
    if position_after > max_notional:
        return False, (f"position ${position_after:.2f} would exceed max_position_pct cap "
                       f"(${max_notional:.2f} = {max_position_pct:.0%} of equity ${equity:.2f})")

    is_new_symbol = symbol not in open_notional or abs(open_notional[symbol]) < 1e-9
    if is_new_symbol and len(open_notional) >= max_open_positions:
        return False, (f"already at max_open_positions ({max_open_positions}) and "
                       f"'{symbol}' would be a new position")

    stop_pct = payload.get("stop_pct")
    target_pct = payload.get("target_pct")
    if stop_pct is not None:
        if not isinstance(stop_pct, (int, float)) or isinstance(stop_pct, bool) or stop_pct <= 0:
            return False, "stop_pct must be a positive number"
        if min_stop_pct is not None and stop_pct < min_stop_pct:
            return False, (f"stop_pct {stop_pct:.2%} is tighter than the minimum {min_stop_pct:.2%} "
                           f"(round-trip cost would eat too much of the risk)")
        if max_stop_pct is not None and stop_pct > max_stop_pct:
            return False, f"stop_pct {stop_pct:.2%} is wider than the maximum {max_stop_pct:.2%}"
        if target_pct is not None and (not isinstance(target_pct, (int, float))
                                       or isinstance(target_pct, bool) or target_pct < stop_pct):
            return False, "target_pct must be a number at least as large as stop_pct (reward ≥ risk)"
        if max_risk_pct is not None:
            risk = position_after * stop_pct
            max_risk = equity * max_risk_pct
            if risk > max_risk:
                return False, (f"risk ${risk:.2f} (position × stop) exceeds max_risk_pct "
                               f"(${max_risk:.2f} = {max_risk_pct:.1%} of equity)")

    if signed_notional is not None and max_net_exposure_pct is not None:
        net_after = sum(signed_notional.values()) + direction * notional
        max_net = equity * max_net_exposure_pct
        if abs(net_after) > max_net:
            return False, (f"net directional exposure would be ${net_after:+.2f}, above the "
                           f"${max_net:.2f} cap ({max_net_exposure_pct:.0%} of equity) — BTC/ETH/SOL "
                           f"move together, so same-side positions are one bet")

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
