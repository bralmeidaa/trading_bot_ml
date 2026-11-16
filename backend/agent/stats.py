"""Closed-trade statistics for the agent's paper book. Pure: takes trade
dicts (TradeRepository.get_closed_trades shape), no I/O."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional


def _initial_risk(t: dict) -> Optional[float]:
    """USD at risk at entry (distance to stop × quantity), None without a stop.
    Uses the final average entry, so it's approximate for scaled-in trades."""
    stop = t.get("stop_loss") or 0.0
    if not stop:
        return None
    return abs(t["entry_price"] - stop) * t["quantity"]


def _exit_kind(reason: Optional[str]) -> str:
    reason = reason or ""
    if reason in ("stop", "target"):
        return reason
    if reason.startswith("llm"):
        return "llm"
    return reason or "unknown"


def trade_stats(trades: List[dict]) -> dict:
    pnls = [t.get("pnl") or 0.0 for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    r_multiples = [(t.get("pnl") or 0.0) / risk for t in trades
                   if (risk := _initial_risk(t))]
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else None,
        "net_pnl": sum(pnls),
        "avg_win": gross_win / len(wins) if wins else None,
        "avg_loss": -gross_loss / len(losses) if losses else None,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "expectancy_usd": sum(pnls) / n if n else None,
        "expectancy_r": sum(r_multiples) / len(r_multiples) if r_multiples else None,
        "exits": dict(Counter(_exit_kind(t.get("reason")) for t in trades)),
    }
