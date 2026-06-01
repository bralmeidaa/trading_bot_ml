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
from .models import TradeRecord, EquitySnapshot, DailyStats

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
# Helper
# ══════════════════════════════════════════════════════════════════════════

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
