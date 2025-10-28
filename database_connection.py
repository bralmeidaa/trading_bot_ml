#!/usr/bin/env python3
"""
Database Connection Manager for MySQL HeatWave
Handles connection pooling, session management, and database operations
"""

import os
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import pymysql
from urllib.parse import quote_plus
import time

from database_models import (
    Base, GlobalConfig, TradingBot, Trade, SystemLog, 
    TradingSignal, DailyPerformance, EquityCurve, SystemSetting,
    BotStatus, TradeStatus, LogLevel
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gerenciador de conexão com MySQL HeatWave."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connection_string = None
        self._is_connected = False
        
    def initialize(self, connection_string: Optional[str] = None) -> bool:
        """
        Inicializa a conexão com o banco de dados.
        
        Args:
            connection_string: String de conexão MySQL. Se None, usa variáveis de ambiente.
            
        Returns:
            bool: True se conectado com sucesso
        """
        try:
            if connection_string:
                self._connection_string = connection_string
            else:
                self._connection_string = self._build_connection_string()
            
            # Criar engine com pool de conexões
            self.engine = create_engine(
                self._connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,  # Reciclar conexões a cada hora
                echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
            )
            
            # Testar conexão
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Criar session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self._is_connected = True
            logger.info("✅ Conexão com MySQL HeatWave estabelecida com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com MySQL HeatWave: {e}")
            self._is_connected = False
            return False
    
    def _build_connection_string(self) -> str:
        """Constrói a string de conexão a partir das variáveis de ambiente."""
        
        # Variáveis de ambiente obrigatórias
        required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Variáveis de ambiente obrigatórias não encontradas: {missing_vars}")
        
        # Construir string de conexão
        host = os.getenv('DB_HOST')
        port = os.getenv('DB_PORT', '3306')
        user = os.getenv('DB_USER')
        password = quote_plus(os.getenv('DB_PASSWORD'))
        database = os.getenv('DB_NAME')
        
        # Parâmetros adicionais para MySQL HeatWave
        params = {
            'charset': 'utf8mb4',
            'autocommit': 'false',
            'connect_timeout': '60',
            'read_timeout': '60',
            'write_timeout': '60'
        }
        
        # SSL se especificado
        if os.getenv('DB_SSL', 'false').lower() == 'true':
            params['ssl_disabled'] = 'false'
            if os.getenv('DB_SSL_CA'):
                params['ssl_ca'] = os.getenv('DB_SSL_CA')
        
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?{param_string}"
        
        logger.info(f"🔗 String de conexão construída para: {user}@{host}:{port}/{database}")
        return connection_string
    
    @contextmanager
    def get_session(self):
        """Context manager para sessões do banco de dados."""
        if not self._is_connected:
            raise RuntimeError("Banco de dados não conectado. Chame initialize() primeiro.")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erro na sessão do banco: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self) -> bool:
        """Cria todas as tabelas no banco de dados."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("✅ Tabelas criadas/verificadas com sucesso")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao criar tabelas: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica a saúde da conexão com o banco."""
        try:
            with self.get_session() as session:
                # Teste básico de conectividade
                result = session.execute(text("SELECT 1 as health_check")).fetchone()
                
                # Estatísticas das tabelas
                tables_info = {}
                for table_name in ['trades', 'trading_bots', 'system_logs']:
                    try:
                        count_result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()
                        tables_info[table_name] = count_result[0] if count_result else 0
                    except:
                        tables_info[table_name] = "N/A"
                
                return {
                    'status': 'healthy',
                    'connected': True,
                    'tables': tables_info,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Health check falhou: {e}")
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def clear_all_data(self) -> bool:
        """
        CUIDADO: Limpa todos os dados das tabelas principais.
        Usado pelo botão "Zerar Histórico".
        """
        try:
            with self.get_session() as session:
                # Ordem importante devido às foreign keys
                tables_to_clear = [
                    'system_logs',
                    'trading_signals', 
                    'daily_performance',
                    'equity_curve',
                    'trades'
                ]
                
                for table in tables_to_clear:
                    session.execute(text(f"TRUNCATE TABLE {table}"))
                    logger.info(f"🗑️ Tabela {table} limpa")
                
                # Reset dos contadores dos bots
                session.execute(text("""
                    UPDATE trading_bots 
                    SET total_trades = 0, 
                        winning_trades = 0, 
                        total_pnl = 0.00000000,
                        updated_at = CURRENT_TIMESTAMP
                """))
                
                logger.info("✅ Histórico de trades e logs limpo com sucesso")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erro ao limpar dados: {e}")
            return False
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas gerais de trading."""
        try:
            with self.get_session() as session:
                # Usar a view criada no schema.sql
                result = session.execute(text("SELECT * FROM v_trading_stats")).fetchone()
                
                if result:
                    return {
                        'total_trades': result.total_trades or 0,
                        'winning_trades': result.winning_trades or 0,
                        'losing_trades': result.losing_trades or 0,
                        'win_rate': float(result.win_rate or 0),
                        'total_pnl': float(result.total_pnl or 0),
                        'avg_win': float(result.avg_win or 0),
                        'avg_loss': float(result.avg_loss or 0),
                        'gross_profit': float(result.gross_profit or 0),
                        'gross_loss': float(result.gross_loss or 0),
                        'total_fees': float(result.total_fees or 0),
                        'profit_factor': float(result.gross_profit / result.gross_loss) if result.gross_loss and result.gross_loss > 0 else 0
                    }
                else:
                    return self._empty_stats()
                    
        except Exception as e:
            logger.error(f"Erro ao buscar estatísticas: {e}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas vazias."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'total_fees': 0.0,
            'profit_factor': 0.0
        }
    
    def get_bot_stats(self) -> List[Dict[str, Any]]:
        """Retorna estatísticas por bot."""
        try:
            with self.get_session() as session:
                # Usar a view criada no schema.sql
                results = session.execute(text("SELECT * FROM v_bot_stats")).fetchall()
                
                return [
                    {
                        'bot_id': row.bot_id,
                        'symbol': row.symbol,
                        'timeframe': row.timeframe,
                        'bot_status': row.bot_status,
                        'total_trades': row.total_trades or 0,
                        'winning_trades': row.winning_trades or 0,
                        'win_rate': float(row.win_rate or 0),
                        'total_pnl': float(row.total_pnl or 0),
                        'avg_win': float(row.avg_win or 0),
                        'avg_loss': float(row.avg_loss or 0),
                        'total_fees': float(row.total_fees or 0),
                        'last_trade': row.last_trade.isoformat() if row.last_trade else None
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Erro ao buscar estatísticas dos bots: {e}")
            return []
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Retorna trades ativos."""
        try:
            with self.get_session() as session:
                # Usar a view criada no schema.sql
                results = session.execute(text("SELECT * FROM v_active_trades")).fetchall()
                
                return [
                    {
                        'trade_id': row.trade_id,
                        'bot_id': row.bot_id,
                        'symbol': row.symbol,
                        'direction': 'LONG' if row.direction == 1 else 'SHORT',
                        'entry_time': row.entry_time.isoformat() if row.entry_time else None,
                        'entry_price': float(row.entry_price),
                        'stop_loss': float(row.stop_loss),
                        'take_profit': float(row.take_profit),
                        'quantity': float(row.quantity),
                        'duration_minutes': row.duration_minutes or 0,
                        'confidence': float(row.confidence) if row.confidence else None,
                        'signal_strength': float(row.signal_strength) if row.signal_strength else None
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Erro ao buscar trades ativos: {e}")
            return []
    
    def get_equity_curve_data(self, days: int = 30, bot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna dados da equity curve."""
        try:
            with self.get_session() as session:
                query = """
                    SELECT timestamp, bot_id, equity, drawdown_pct, total_trades
                    FROM equity_curve 
                    WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
                """
                params = [days]
                
                if bot_id:
                    query += " AND bot_id = %s"
                    params.append(bot_id)
                
                query += " ORDER BY timestamp ASC"
                
                results = session.execute(text(query), params).fetchall()
                
                return [
                    {
                        'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                        'bot_id': row.bot_id,
                        'equity': float(row.equity),
                        'drawdown_pct': float(row.drawdown_pct or 0),
                        'total_trades': row.total_trades or 0
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Erro ao buscar equity curve: {e}")
            return []

# Instância global do gerenciador
db_manager = DatabaseManager()

# Funções de conveniência
def get_db_session():
    """Retorna um context manager para sessão do banco."""
    return db_manager.get_session()

def initialize_database(connection_string: Optional[str] = None) -> bool:
    """Inicializa a conexão com o banco de dados."""
    return db_manager.initialize(connection_string)

def create_database_tables() -> bool:
    """Cria todas as tabelas no banco."""
    return db_manager.create_tables()