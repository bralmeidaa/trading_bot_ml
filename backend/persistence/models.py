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
    Column, DateTime, Float, Integer, String, Text, UniqueConstraint
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


# ── AI agent (Azure AI Foundry, paper trading) — dedicated tables ──────────
# Deliberately NOT reusing EquitySnapshot/DailyStats above: those two tables
# already have no strategy discriminator and are silently shared between the
# two pre-existing engines (production_trading_system.py / portfolio_engine.py)
# — adding a third writer would make that worse. New, additive tables instead
# (created via init_db()'s create_all(), non-destructive). Closed trades DO
# reuse TradeRecord/TradeRepository unchanged, via bot_id="ai_agent".

class AgentEquitySnapshot(Base):
    """Periodic equity curve sample for the AI agent's dedicated paper book."""
    __tablename__ = "agent_equity_snapshots"

    id            = Column(Integer,  primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, nullable=False, index=True)
    equity        = Column(Float,    nullable=False)
    total_pnl     = Column(Float,    nullable=False)
    daily_pnl     = Column(Float,    nullable=False)
    open_positions = Column(Integer, default=0)


class AgentDailyStats(Base):
    """One row per trading day for the AI agent, written at midnight reset."""
    __tablename__ = "agent_daily_stats"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    date        = Column(String,  nullable=False, index=True)
    pnl         = Column(Float,   nullable=False)
    trades      = Column(Integer, default=0)
    wins        = Column(Integer, default=0)
    losses      = Column(Integer, default=0)
    llm_calls   = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)

    __table_args__ = (UniqueConstraint("date", name="uq_agent_daily_stats_date"),)


class AgentDecisionLog(Base):
    """Durable WHAT-it-saw / WHY-it-decided log — the single most important
    table for this feature, since the agent's behavior is non-deterministic.
    Every cycle writes a row here regardless of outcome (hold / executed /
    rejected / halted), so a human can audit "why did it do that" after the
    fact without needing to re-run anything."""
    __tablename__ = "agent_decision_log"

    id                    = Column(Integer,  primary_key=True, autoincrement=True)
    timestamp             = Column(DateTime, nullable=False, index=True)
    cycle_id              = Column(Integer,  nullable=False, index=True)
    symbols_considered    = Column(Text,     nullable=True)   # JSON list
    context_snapshot      = Column(Text,     nullable=True)   # JSON: equity+positions at cycle start
    raw_llm_output        = Column(Text,     nullable=True)
    tool_calls_requested  = Column(Text,     nullable=True)   # JSON trace (every tool call made)
    tool_calls_executed   = Column(Text,     nullable=True)   # JSON: subset that actually filled
    guardrail_rejections  = Column(Text,     nullable=True)   # JSON, if any order was rejected
    rationale             = Column(Text,     nullable=True)
    model_name            = Column(String,   nullable=True)
    prompt_tokens         = Column(Integer,  default=0)
    completion_tokens     = Column(Integer,  default=0)
    latency_ms            = Column(Integer,  default=0)
    halted_reason         = Column(String,   nullable=True)   # set if the cycle skipped the LLM entirely
