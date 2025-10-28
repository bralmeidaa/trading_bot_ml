#!/usr/bin/env python3
"""
Database Models for Trading Bot ML
SQLAlchemy models for MySQL HeatWave integration
"""

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, 
    Enum, JSON, Date, Time, ForeignKey, Index, UniqueConstraint, Numeric
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import Enum as PyEnum
from decimal import Decimal
import json

Base = declarative_base()

# Enums
class BotStatus(PyEnum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class TradeStatus(PyEnum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class LogLevel(PyEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"

class SettingType(PyEnum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"

# =====================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =====================================================
class GlobalConfig(Base):
    __tablename__ = 'global_configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    total_capital = Column(Numeric(15, 2), nullable=False, default=10000.00)
    max_concurrent_trades = Column(Integer, nullable=False, default=4)
    daily_loss_limit = Column(Numeric(5, 4), nullable=False, default=0.0500)
    daily_profit_target = Column(Numeric(5, 4), nullable=False, default=0.0300)
    emergency_stop_drawdown = Column(Numeric(5, 4), nullable=False, default=0.0800)
    paper_trading = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    updated_by = Column(String(100), default='system')
    is_active = Column(Boolean, nullable=False, default=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'total_capital': float(self.total_capital),
            'max_concurrent_trades': self.max_concurrent_trades,
            'daily_loss_limit': float(self.daily_loss_limit),
            'daily_profit_target': float(self.daily_profit_target),
            'emergency_stop_drawdown': float(self.emergency_stop_drawdown),
            'paper_trading': self.paper_trading,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'updated_by': self.updated_by,
            'is_active': self.is_active
        }

# =====================================================
# 2. BOTS DE TRADING
# =====================================================
class TradingBot(Base):
    __tablename__ = 'trading_bots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String(50), nullable=False, unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    capital_allocation = Column(Numeric(5, 4), nullable=False)
    max_risk_per_trade = Column(Numeric(5, 4), nullable=False)
    confidence_threshold = Column(Numeric(5, 4), nullable=False)
    stop_loss_pct = Column(Numeric(5, 4), nullable=False)
    take_profit_pct = Column(Numeric(5, 4), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    status = Column(Enum(BotStatus), default=BotStatus.STOPPED, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(15, 8), default=0.00000000)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    trades = relationship("Trade", back_populates="bot", cascade="all, delete-orphan")
    signals = relationship("TradingSignal", back_populates="bot", cascade="all, delete-orphan")
    daily_performance = relationship("DailyPerformance", back_populates="bot", cascade="all, delete-orphan")
    equity_curve = relationship("EquityCurve", back_populates="bot", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            'id': self.id,
            'bot_id': self.bot_id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'capital_allocation': float(self.capital_allocation),
            'max_risk_per_trade': float(self.max_risk_per_trade),
            'confidence_threshold': float(self.confidence_threshold),
            'stop_loss_pct': float(self.stop_loss_pct),
            'take_profit_pct': float(self.take_profit_pct),
            'enabled': self.enabled,
            'status': self.status.value if self.status else 'stopped',
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': float(self.total_pnl),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# =====================================================
# 3. TRADES
# =====================================================
class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), nullable=False, unique=True, index=True)
    bot_id = Column(String(50), ForeignKey('trading_bots.bot_id', ondelete='CASCADE'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(Integer, nullable=False, index=True)  # 1=long, -1=short
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Numeric(20, 8), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    stop_loss = Column(Numeric(20, 8), nullable=False)
    take_profit = Column(Numeric(20, 8), nullable=False)
    exit_time = Column(DateTime, nullable=True, index=True)
    exit_price = Column(Numeric(20, 8), nullable=True)
    pnl = Column(Numeric(15, 8), nullable=True, index=True)
    pnl_pct = Column(Numeric(8, 4), nullable=True)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN, index=True)
    exit_reason = Column(String(100), nullable=True)
    confidence = Column(Numeric(5, 4), nullable=True)
    signal_strength = Column(Numeric(5, 4), nullable=True)
    fees = Column(Numeric(15, 8), default=0.00000000)
    slippage = Column(Numeric(8, 4), default=0.0000)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    bot = relationship("TradingBot", back_populates="trades")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'bot_id': self.bot_id,
            'symbol': self.symbol,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_price': float(self.entry_price),
            'quantity': float(self.quantity),
            'stop_loss': float(self.stop_loss),
            'take_profit': float(self.take_profit),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'pnl_pct': float(self.pnl_pct) if self.pnl_pct else None,
            'status': self.status.value if self.status else 'open',
            'exit_reason': self.exit_reason,
            'confidence': float(self.confidence) if self.confidence else None,
            'signal_strength': float(self.signal_strength) if self.signal_strength else None,
            'fees': float(self.fees),
            'slippage': float(self.slippage),
            'meta_data': self.meta_data,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# =====================================================
# 4. LOGS DO SISTEMA
# =====================================================
class SystemLog(Base):
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.current_timestamp(), index=True)
    level = Column(Enum(LogLevel), nullable=False, index=True)
    source = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    bot_id = Column(String(50), ForeignKey('trading_bots.bot_id', ondelete='SET NULL'), nullable=True, index=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id', ondelete='SET NULL'), nullable=True, index=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level.value if self.level else 'INFO',
            'source': self.source,
            'message': self.message,
            'bot_id': self.bot_id,
            'trade_id': self.trade_id,
            'meta_data': self.meta_data,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# =====================================================
# 5. SINAIS DE TRADING
# =====================================================
class TradingSignal(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), nullable=False, unique=True, index=True)
    bot_id = Column(String(50), ForeignKey('trading_bots.bot_id', ondelete='CASCADE'), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(Integer, nullable=False)  # 1=long, -1=short
    strength = Column(Numeric(5, 4), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Numeric(20, 8), nullable=False)
    stop_loss = Column(Numeric(20, 8), nullable=False)
    take_profit = Column(Numeric(20, 8), nullable=False)
    executed = Column(Boolean, default=False, index=True)
    trade_id = Column(String(100), ForeignKey('trades.trade_id', ondelete='SET NULL'), nullable=True)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    bot = relationship("TradingBot", back_populates="signals")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'signal_id': self.signal_id,
            'bot_id': self.bot_id,
            'symbol': self.symbol,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'strength': float(self.strength),
            'confidence': float(self.confidence),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'entry_price': float(self.entry_price),
            'stop_loss': float(self.stop_loss),
            'take_profit': float(self.take_profit),
            'executed': self.executed,
            'trade_id': self.trade_id,
            'meta_data': self.meta_data,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# =====================================================
# 6. PERFORMANCE DIÁRIA
# =====================================================
class DailyPerformance(Base):
    __tablename__ = 'daily_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    bot_id = Column(String(50), ForeignKey('trading_bots.bot_id', ondelete='CASCADE'), nullable=True, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(15, 8), default=0.00000000, index=True)
    gross_profit = Column(Numeric(15, 8), default=0.00000000)
    gross_loss = Column(Numeric(15, 8), default=0.00000000)
    win_rate = Column(Numeric(5, 4), default=0.0000, index=True)
    profit_factor = Column(Numeric(8, 4), default=0.0000)
    avg_win = Column(Numeric(15, 8), default=0.00000000)
    avg_loss = Column(Numeric(15, 8), default=0.00000000)
    max_drawdown = Column(Numeric(8, 4), default=0.0000)
    sharpe_ratio = Column(Numeric(8, 4), default=0.0000)
    total_fees = Column(Numeric(15, 8), default=0.00000000)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relationships
    bot = relationship("TradingBot", back_populates="daily_performance")
    
    __table_args__ = (
        UniqueConstraint('date', 'bot_id', name='unique_daily_bot'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'bot_id': self.bot_id,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': float(self.total_pnl),
            'gross_profit': float(self.gross_profit),
            'gross_loss': float(self.gross_loss),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'avg_win': float(self.avg_win),
            'avg_loss': float(self.avg_loss),
            'max_drawdown': float(self.max_drawdown),
            'sharpe_ratio': float(self.sharpe_ratio),
            'total_fees': float(self.total_fees),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# =====================================================
# 7. EQUITY CURVE
# =====================================================
class EquityCurve(Base):
    __tablename__ = 'equity_curve'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    bot_id = Column(String(50), ForeignKey('trading_bots.bot_id', ondelete='CASCADE'), nullable=True, index=True)
    balance = Column(Numeric(15, 8), nullable=False)
    equity = Column(Numeric(15, 8), nullable=False, index=True)
    drawdown = Column(Numeric(8, 4), default=0.0000, index=True)
    drawdown_pct = Column(Numeric(8, 4), default=0.0000)
    total_trades = Column(Integer, default=0)
    open_trades = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relationships
    bot = relationship("TradingBot", back_populates="equity_curve")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'bot_id': self.bot_id,
            'balance': float(self.balance),
            'equity': float(self.equity),
            'drawdown': float(self.drawdown),
            'drawdown_pct': float(self.drawdown_pct),
            'total_trades': self.total_trades,
            'open_trades': self.open_trades,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# =====================================================
# 8. CONFIGURAÇÕES DO SISTEMA
# =====================================================
class SystemSetting(Base):
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    setting_key = Column(String(100), nullable=False, unique=True, index=True)
    setting_value = Column(Text, nullable=False)
    setting_type = Column(Enum(SettingType), nullable=False, default=SettingType.STRING)
    description = Column(Text, nullable=True)
    category = Column(String(50), default='general', index=True)
    is_sensitive = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    def get_typed_value(self) -> Any:
        """Retorna o valor convertido para o tipo correto."""
        if self.setting_type == SettingType.INTEGER:
            return int(self.setting_value)
        elif self.setting_type == SettingType.FLOAT:
            return float(self.setting_value)
        elif self.setting_type == SettingType.BOOLEAN:
            return self.setting_value.lower() in ('true', '1', 'yes', 'on')
        elif self.setting_type == SettingType.JSON:
            return json.loads(self.setting_value)
        else:
            return self.setting_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'setting_key': self.setting_key,
            'setting_value': self.get_typed_value(),
            'setting_type': self.setting_type.value if self.setting_type else 'string',
            'description': self.description,
            'category': self.category,
            'is_sensitive': self.is_sensitive,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# =====================================================
# 9. RESTRIÇÕES DE HORÁRIO DE TRADING
# =====================================================
class TradingRestriction(Base):
    __tablename__ = 'trading_restrictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    start_day_of_week = Column(Integer, nullable=False)  # 0=Sunday, 1=Monday, ..., 6=Saturday
    start_time = Column(Time, nullable=False)
    end_day_of_week = Column(Integer, nullable=False)
    end_time = Column(Time, nullable=False)
    timezone = Column(String(50), default='America/Sao_Paulo')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'start_day_of_week': self.start_day_of_week,
            'start_time': self.start_time.strftime('%H:%M:%S') if self.start_time else None,
            'end_day_of_week': self.end_day_of_week,
            'end_time': self.end_time.strftime('%H:%M:%S') if self.end_time else None,
            'timezone': self.timezone,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def get_day_name(self, day_of_week: int) -> str:
        """Converter número do dia para nome"""
        days = ['Domingo', 'Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado']
        return days[day_of_week] if 0 <= day_of_week <= 6 else 'Inválido'
    
    def get_formatted_period(self) -> str:
        """Retornar período formatado para exibição"""
        start_day = self.get_day_name(self.start_day_of_week)
        end_day = self.get_day_name(self.end_day_of_week)
        start_time = self.start_time.strftime('%H:%M') if self.start_time else '00:00'
        end_time = self.end_time.strftime('%H:%M') if self.end_time else '00:00'
        
        if self.start_day_of_week == self.end_day_of_week:
            return f"{start_day} das {start_time} às {end_time}"
        else:
            return f"{start_day} {start_time} até {end_day} {end_time}"

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def create_all_tables(engine):
    """Cria todas as tabelas no banco de dados."""
    Base.metadata.create_all(engine)

def get_session_factory(engine):
    """Retorna uma factory de sessões."""
    return sessionmaker(bind=engine)

# Índices adicionais para otimização
Index('idx_trades_bot_status', Trade.bot_id, Trade.status)
Index('idx_trades_entry_time_desc', Trade.entry_time.desc())
Index('idx_logs_timestamp_level', SystemLog.timestamp, SystemLog.level)
Index('idx_signals_bot_executed', TradingSignal.bot_id, TradingSignal.executed)
Index('idx_equity_timestamp_bot', EquityCurve.timestamp, EquityCurve.bot_id)