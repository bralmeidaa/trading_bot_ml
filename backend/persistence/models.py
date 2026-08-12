"""
SQLAlchemy ORM models for trade persistence.

Tables:
  trades            — every closed trade (main audit trail)
  equity_snapshots  — equity curve sampled every ~10 min
  daily_stats       — per-day PnL summary (written at midnight reset)
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    """One closed trade.  Primary key = trade ID set by ProductionTradingSystem."""
    __tablename__ = "trades"

    id          = Column(String,   primary_key=True)
    bot_id      = Column(String,   nullable=False, index=True)
    symbol      = Column(String,   nullable=False, index=True)
    direction   = Column(Integer,  nullable=False)          # 1 long / -1 short

    entry_time  = Column(DateTime, nullable=False, index=True)
    exit_time   = Column(DateTime, nullable=True)

    entry_price = Column(Float,    nullable=False)
    exit_price  = Column(Float,    nullable=True)
    quantity    = Column(Float,    nullable=False)
    stop_loss   = Column(Float,    nullable=False)
    take_profit = Column(Float,    nullable=False)

    pnl         = Column(Float,    nullable=True)
    pnl_pct     = Column(Float,    nullable=True)

    status      = Column(String,   default="open")          # open / closed / cancelled
    reason      = Column(String,   nullable=True)           # stop_loss / take_profit / shutdown

    created_at  = Column(DateTime, default=datetime.utcnow)


class EquitySnapshot(Base):
    """Periodic equity curve sample (one row per ~10 min)."""
    __tablename__ = "equity_snapshots"

    id            = Column(Integer,  primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    equity        = Column(Float,    nullable=False)
    total_pnl     = Column(Float,    nullable=False)
    daily_pnl     = Column(Float,    nullable=False)
    active_trades = Column(Integer,  default=0)


class DailyStats(Base):
    """One row per trading day, written at midnight UTC reset."""
    __tablename__ = "daily_stats"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    date        = Column(String,  nullable=False, index=True)  # ISO-8601, e.g. "2025-06-01"
    pnl         = Column(Float,   nullable=False)
    trades      = Column(Integer, default=0)
    wins        = Column(Integer, default=0)
    losses      = Column(Integer, default=0)

    __table_args__ = (UniqueConstraint("date", name="uq_daily_stats_date"),)
