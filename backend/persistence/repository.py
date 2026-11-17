"""
Repository layer: thin CRUD wrappers around the ORM models.

All public methods open their own session and close it when done,
so callers don't need to manage sessions.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.exc import IntegrityError

from .database import get_session
from .models import (
    TradeRecord, EquitySnapshot, DailyStats,
    AgentEquitySnapshot, AgentDailyStats, AgentDecisionLog, ShadowTrade,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _ms_to_dt(ms: Optional[int]) -> Optional[datetime]:
    """Convert a millisecond Unix timestamp to a naive UTC datetime."""
    if ms is None:
        return None
    return datetime.utcfromtimestamp(ms / 1000)


# ══════════════════════════════════════════════════════════════════════════
# Trade repository
# ══════════════════════════════════════════════════════════════════════════

class TradeRepository:
    """Persist and query closed trades."""

    def save(self, trade) -> None:
        """Insert or update a Trade dataclass instance in the DB."""
        session = get_session()
        try:
            record = TradeRecord(
                id          = trade.id,
                bot_id      = getattr(trade, "bot_id", ""),
                symbol      = trade.symbol,
                direction   = trade.direction,
                entry_time  = _ms_to_dt(trade.entry_time),
                exit_time   = _ms_to_dt(trade.exit_time),
                entry_price = trade.entry_price,
                exit_price  = trade.exit_price,
                quantity    = trade.quantity,
                stop_loss   = trade.stop_loss,
                take_profit = trade.take_profit,
                pnl         = trade.pnl,
                pnl_pct     = trade.pnl_pct,
                status      = trade.status,
                reason      = trade.reason,
            )
            session.merge(record)   # INSERT OR UPDATE (upsert via primary key)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save trade {trade.id}: {exc}")
        finally:
            session.close()

    def get_recent_pnl_pcts(self, limit: int = 200) -> List[float]:
        """Return pnl_pct values for the most recent `limit` closed trades.
        Used to seed the AdvancedRiskManager (Kelly Criterion) on startup.
        """
        session = get_session()
        try:
            rows = (
                session.query(TradeRecord.pnl_pct)
                .filter(
                    TradeRecord.status == "closed",
                    TradeRecord.pnl_pct.isnot(None),
                    # the AI agent shares this table; its trades must not seed the legacy Kelly sizing
                    ~TradeRecord.bot_id.like("ai_agent%"),
                )
                .order_by(TradeRecord.exit_time.desc())
                .limit(limit)
                .all()
            )
            # Return in chronological order (oldest first) for Kelly accumulation
            return [r.pnl_pct for r in reversed(rows)]
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 100) -> List[dict]:
        """Return the last `limit` closed trades as plain dicts (for API/dashboard)."""
        session = get_session()
        try:
            rows = (
                session.query(TradeRecord)
                .filter(TradeRecord.status == "closed")
                .order_by(TradeRecord.exit_time.desc())
                .limit(limit)
                .all()
            )
            return [_trade_to_dict(r) for r in rows]
        finally:
            session.close()

    def count(self) -> int:
        session = get_session()
        try:
            return session.query(TradeRecord).filter(TradeRecord.status == "closed").count()
        finally:
            session.close()

    def get_open_trades(self, bot_id: str) -> List[dict]:
        """Open trades of one bot, as AgentTrade-compatible kwargs (ms epochs)."""
        return self._query_bot(bot_id, "open")

    def get_closed_trades(self, bot_id: str, since: Optional[datetime] = None) -> List[dict]:
        """Closed trades of one bot, oldest exit first, as AgentTrade kwargs."""
        return self._query_bot(bot_id, "closed", since)

    def _query_bot(self, bot_id: str, status: str, since: Optional[datetime] = None) -> List[dict]:
        session = get_session()
        try:
            q = session.query(TradeRecord).filter(TradeRecord.bot_id == bot_id,
                                                  TradeRecord.status == status)
            if since is not None:
                q = q.filter(TradeRecord.exit_time >= since)
            q = q.order_by(TradeRecord.exit_time.asc() if status == "closed"
                           else TradeRecord.entry_time.asc())
            return [_trade_to_kwargs(r) for r in q.all()]
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════════════════
# Equity repository
# ══════════════════════════════════════════════════════════════════════════

class EquityRepository:
    """Persist equity curve snapshots."""

    def save(self, equity: float, total_pnl: float, daily_pnl: float,
             active_trades: int = 0) -> None:
        session = get_session()
        try:
            snap = EquitySnapshot(
                timestamp     = datetime.utcnow(),
                equity        = equity,
                total_pnl     = total_pnl,
                daily_pnl     = daily_pnl,
                active_trades = active_trades,
            )
            session.add(snap)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save equity snapshot: {exc}")
        finally:
            session.close()

    def get_curve(self, limit: int = 500) -> List[dict]:
        """Return the last `limit` snapshots for the dashboard."""
        session = get_session()
        try:
            rows = (
                session.query(EquitySnapshot)
                .order_by(EquitySnapshot.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "timestamp":     r.timestamp.isoformat(),
                    "equity":        r.equity,
                    "total_pnl":     r.total_pnl,
                    "daily_pnl":     r.daily_pnl,
                    "active_trades": r.active_trades,
                }
                for r in reversed(rows)
            ]
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════════════════
# Daily stats repository
# ══════════════════════════════════════════════════════════════════════════

class DailyStatsRepository:
    """Persist per-day performance summaries."""

    def save(self, date_str: str, pnl: float, trades: int,
             wins: int, losses: int) -> None:
        # date is a UNIQUE column but not the primary key → can't use merge().
        # Implement explicit check-then-insert-or-update pattern.
        session = get_session()
        try:
            existing = session.query(DailyStats).filter_by(date=date_str).first()
            if existing:
                existing.pnl    = pnl
                existing.trades = trades
                existing.wins   = wins
                existing.losses = losses
            else:
                session.add(DailyStats(date=date_str, pnl=pnl, trades=trades,
                                       wins=wins, losses=losses))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save daily stats for {date_str}: {exc}")
        finally:
            session.close()

    def get_history(self, limit: int = 90) -> List[dict]:
        """Return the last `limit` days for the dashboard."""
        session = get_session()
        try:
            rows = (
                session.query(DailyStats)
                .order_by(DailyStats.date.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "date":    r.date,
                    "pnl":     r.pnl,
                    "trades":  r.trades,
                    "wins":    r.wins,
                    "losses":  r.losses,
                }
                for r in reversed(rows)
            ]
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════════════════
# AI agent repositories — dedicated tables, see models.py for why they're
# separate from EquityRepository/DailyStatsRepository above.
# ══════════════════════════════════════════════════════════════════════════

class AgentEquityRepository:
    """Persist the AI agent's dedicated equity curve."""

    def save(self, equity: float, total_pnl: float, daily_pnl: float,
             open_positions: int = 0) -> None:
        session = get_session()
        try:
            snap = AgentEquitySnapshot(
                timestamp=datetime.utcnow(), equity=equity, total_pnl=total_pnl,
                daily_pnl=daily_pnl, open_positions=open_positions,
            )
            session.add(snap)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save agent equity snapshot: {exc}")
        finally:
            session.close()

    def get_curve(self, limit: int = 500) -> List[dict]:
        session = get_session()
        try:
            rows = (
                session.query(AgentEquitySnapshot)
                .order_by(AgentEquitySnapshot.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {"timestamp": r.timestamp.isoformat(), "equity": r.equity,
                 "total_pnl": r.total_pnl, "daily_pnl": r.daily_pnl,
                 "open_positions": r.open_positions}
                for r in reversed(rows)
            ]
        finally:
            session.close()


class AgentDailyStatsRepository:
    """Persist the AI agent's per-day performance + LLM usage summary."""

    def save(self, date_str: str, pnl: float, trades: int, wins: int, losses: int,
             llm_calls: int = 0, tokens_used: int = 0) -> None:
        session = get_session()
        try:
            existing = session.query(AgentDailyStats).filter_by(date=date_str).first()
            if existing:
                existing.pnl, existing.trades = pnl, trades
                existing.wins, existing.losses = wins, losses
                existing.llm_calls, existing.tokens_used = llm_calls, tokens_used
            else:
                session.add(AgentDailyStats(date=date_str, pnl=pnl, trades=trades,
                                            wins=wins, losses=losses, llm_calls=llm_calls,
                                            tokens_used=tokens_used))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save agent daily stats for {date_str}: {exc}")
        finally:
            session.close()

    def get_history(self, limit: int = 90) -> List[dict]:
        session = get_session()
        try:
            rows = (
                session.query(AgentDailyStats)
                .order_by(AgentDailyStats.date.desc())
                .limit(limit)
                .all()
            )
            return [
                {"date": r.date, "pnl": r.pnl, "trades": r.trades, "wins": r.wins,
                 "losses": r.losses, "llm_calls": r.llm_calls, "tokens_used": r.tokens_used}
                for r in reversed(rows)
            ]
        finally:
            session.close()


class AgentDecisionLogRepository:
    """Persist the AI agent's per-cycle decision audit trail."""

    def save(self, timestamp: datetime, cycle_id: int, symbols_considered: Optional[str] = None,
             context_snapshot: Optional[str] = None, raw_llm_output: Optional[str] = None,
             tool_calls_requested: Optional[str] = None, tool_calls_executed: Optional[str] = None,
             guardrail_rejections: Optional[str] = None, rationale: Optional[str] = None,
             model_name: Optional[str] = None, prompt_tokens: int = 0, completion_tokens: int = 0,
             latency_ms: int = 0, halted_reason: Optional[str] = None) -> None:
        session = get_session()
        try:
            session.add(AgentDecisionLog(
                timestamp=timestamp, cycle_id=cycle_id, symbols_considered=symbols_considered,
                context_snapshot=context_snapshot, raw_llm_output=raw_llm_output,
                tool_calls_requested=tool_calls_requested, tool_calls_executed=tool_calls_executed,
                guardrail_rejections=guardrail_rejections, rationale=rationale,
                model_name=model_name, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens, latency_ms=latency_ms,
                halted_reason=halted_reason,
            ))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save agent decision log: {exc}")
        finally:
            session.close()

    def get_recent(self, limit: int = 50) -> List[dict]:
        session = get_session()
        try:
            rows = (
                session.query(AgentDecisionLog)
                .order_by(AgentDecisionLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {"timestamp": r.timestamp.isoformat(), "cycle_id": r.cycle_id,
                 "symbols_considered": r.symbols_considered, "context_snapshot": r.context_snapshot,
                 "raw_llm_output": r.raw_llm_output, "tool_calls_requested": r.tool_calls_requested,
                 "tool_calls_executed": r.tool_calls_executed,
                 "guardrail_rejections": r.guardrail_rejections, "rationale": r.rationale,
                 "model_name": r.model_name, "prompt_tokens": r.prompt_tokens,
                 "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms,
                 "halted_reason": r.halted_reason}
                for r in rows
            ]
        finally:
            session.close()


# ══════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════

class ShadowTradeRepository:
    """Upsert/read shadow trades (dicts with ShadowTrade column names; times
    as ms epochs)."""

    _TIME_COLS = ("created_at", "exit_time")

    def save(self, row: dict) -> None:
        session = get_session()
        try:
            data = {k: (_ms_to_dt(v) if k in self._TIME_COLS else v) for k, v in row.items()
                    if hasattr(ShadowTrade, k)}
            session.merge(ShadowTrade(**data))
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error(f"Failed to save shadow trade {row.get('id')}: {exc}")
        finally:
            session.close()

    def get_open(self) -> List[dict]:
        return self._query(status="open")

    def get_closed(self, since: Optional[datetime] = None) -> List[dict]:
        return self._query(status="closed", since=since)

    def _query(self, status: str, since: Optional[datetime] = None) -> List[dict]:
        session = get_session()
        try:
            q = session.query(ShadowTrade).filter(ShadowTrade.status == status)
            if since is not None:
                q = q.filter(ShadowTrade.created_at >= since)
            rows = q.order_by(ShadowTrade.created_at.asc()).all()
            return [{c.name: (_dt_to_ms(getattr(r, c.name)) if c.name in self._TIME_COLS
                              else getattr(r, c.name)) for c in ShadowTrade.__table__.columns}
                    for r in rows]
        finally:
            session.close()


def _dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    from calendar import timegm
    return timegm(dt.timetuple()) * 1000 + dt.microsecond // 1000


def _trade_to_kwargs(r: TradeRecord) -> dict:
    return {
        "id": r.id, "symbol": r.symbol, "direction": r.direction,
        "entry_time": _dt_to_ms(r.entry_time), "entry_price": r.entry_price,
        "quantity": r.quantity, "stop_loss": r.stop_loss or 0.0,
        "take_profit": r.take_profit or 0.0, "exit_time": _dt_to_ms(r.exit_time),
        "exit_price": r.exit_price, "pnl": r.pnl, "pnl_pct": r.pnl_pct,
        "status": r.status, "reason": r.reason, "bot_id": r.bot_id,
    }


def _trade_to_dict(r: TradeRecord) -> dict:
    return {
        "id":          r.id,
        "bot_id":      r.bot_id,
        "symbol":      r.symbol,
        "direction":   r.direction,
        "entry_time":  r.entry_time.isoformat() if r.entry_time else None,
        "exit_time":   r.exit_time.isoformat()  if r.exit_time  else None,
        "entry_price": r.entry_price,
        "exit_price":  r.exit_price,
        "quantity":    r.quantity,
        "pnl":         r.pnl,
        "pnl_pct":     r.pnl_pct,
        "status":      r.status,
        "reason":      r.reason,
    }
