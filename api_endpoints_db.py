#!/usr/bin/env python3
"""
Database API Endpoints
Novos endpoints para integração com MySQL HeatWave
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text, desc, asc, and_, or_

from database_connection import get_db_session, db_manager
from database_models import (
    Trade, TradingBot, SystemLog, TradingSignal, 
    DailyPerformance, EquityCurve, GlobalConfig, SystemSetting,
    TradingRestriction, TradeStatus, BotStatus, LogLevel
)

logger = logging.getLogger(__name__)

# Router para endpoints do banco de dados
db_router = APIRouter(prefix="/api/db", tags=["database"])

# =====================================================
# MODELOS PYDANTIC PARA REQUESTS/RESPONSES
# =====================================================

class TradeCreate(BaseModel):
    trade_id: str
    bot_id: str
    symbol: str
    direction: int = Field(..., ge=-1, le=1, description="1=long, -1=short")
    entry_price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    signal_strength: Optional[float] = Field(None, ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None

class TradeUpdate(BaseModel):
    exit_price: Optional[float] = Field(None, gt=0)
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: Optional[str] = Field(None, regex="^(open|closed|cancelled)$")
    exit_reason: Optional[str] = None
    fees: Optional[float] = Field(None, ge=0)
    slippage: Optional[float] = None

class LogCreate(BaseModel):
    level: str = Field(..., regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|SUCCESS)$")
    source: str = Field(..., max_length=100)
    message: str = Field(..., max_length=5000)
    bot_id: Optional[str] = None
    trade_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BotStatusUpdate(BaseModel):
    status: str = Field(..., regex="^(stopped|running|paused|error)$")

class TimeRangeQuery(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    days: Optional[int] = Field(None, ge=1, le=365)

# =====================================================
# ENDPOINTS DE TRADES
# =====================================================

@db_router.post("/trades", response_model=Dict[str, Any])
async def create_trade(trade_data: TradeCreate):
    """Cria um novo trade no banco de dados."""
    try:
        with get_db_session() as session:
            # Verificar se o trade_id já existe
            existing_trade = session.query(Trade).filter(Trade.trade_id == trade_data.trade_id).first()
            if existing_trade:
                raise HTTPException(status_code=400, detail=f"Trade {trade_data.trade_id} já existe")
            
            # Criar novo trade
            new_trade = Trade(
                trade_id=trade_data.trade_id,
                bot_id=trade_data.bot_id,
                symbol=trade_data.symbol,
                direction=trade_data.direction,
                entry_time=datetime.now(),
                entry_price=trade_data.entry_price,
                quantity=trade_data.quantity,
                stop_loss=trade_data.stop_loss,
                take_profit=trade_data.take_profit,
                confidence=trade_data.confidence,
                signal_strength=trade_data.signal_strength,
                metadata=trade_data.metadata,
                status=TradeStatus.OPEN
            )
            
            session.add(new_trade)
            session.flush()  # Para obter o ID
            
            logger.info(f"✅ Trade {trade_data.trade_id} criado para bot {trade_data.bot_id}")
            return {"success": True, "trade_id": new_trade.trade_id, "id": new_trade.id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao criar trade: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.put("/trades/{trade_id}", response_model=Dict[str, Any])
async def update_trade(trade_id: str, trade_update: TradeUpdate):
    """Atualiza um trade existente."""
    try:
        with get_db_session() as session:
            trade = session.query(Trade).filter(Trade.trade_id == trade_id).first()
            if not trade:
                raise HTTPException(status_code=404, detail=f"Trade {trade_id} não encontrado")
            
            # Atualizar campos fornecidos
            update_data = trade_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(trade, field):
                    setattr(trade, field, value)
            
            # Se está fechando o trade, definir exit_time
            if trade_update.status == "closed" and not trade.exit_time:
                trade.exit_time = datetime.now()
            
            logger.info(f"✅ Trade {trade_id} atualizado")
            return {"success": True, "trade_id": trade_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar trade: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/trades", response_model=List[Dict[str, Any]])
async def get_trades(
    bot_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Busca trades com filtros opcionais."""
    try:
        with get_db_session() as session:
            query = session.query(Trade)
            
            # Aplicar filtros
            if bot_id:
                query = query.filter(Trade.bot_id == bot_id)
            if status:
                query = query.filter(Trade.status == status)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            # Ordenar por data de entrada (mais recente primeiro)
            query = query.order_by(desc(Trade.entry_time))
            
            # Paginação
            trades = query.offset(offset).limit(limit).all()
            
            return [trade.to_dict() for trade in trades]
            
    except Exception as e:
        logger.error(f"❌ Erro ao buscar trades: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/trades/active", response_model=List[Dict[str, Any]])
async def get_active_trades():
    """Retorna todos os trades ativos usando a view otimizada."""
    try:
        return db_manager.get_active_trades()
    except Exception as e:
        logger.error(f"❌ Erro ao buscar trades ativos: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# ENDPOINTS DE LOGS
# =====================================================

@db_router.post("/logs", response_model=Dict[str, Any])
async def create_log(log_data: LogCreate):
    """Cria um novo log no banco de dados."""
    try:
        with get_db_session() as session:
            new_log = SystemLog(
                level=LogLevel(log_data.level),
                source=log_data.source,
                message=log_data.message,
                bot_id=log_data.bot_id,
                trade_id=log_data.trade_id,
                metadata=log_data.metadata,
                timestamp=datetime.now()
            )
            
            session.add(new_log)
            session.flush()
            
            return {"success": True, "log_id": new_log.id}
            
    except Exception as e:
        logger.error(f"❌ Erro ao criar log: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/logs", response_model=List[Dict[str, Any]])
async def get_logs(
    level: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    bot_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),  # Últimas 24h por padrão, máximo 1 semana
    limit: int = Query(100, ge=1, le=1000)
):
    """Busca logs com filtros opcionais."""
    try:
        with get_db_session() as session:
            # Filtro de tempo
            time_filter = datetime.now() - timedelta(hours=hours)
            
            query = session.query(SystemLog).filter(SystemLog.timestamp >= time_filter)
            
            # Aplicar filtros
            if level:
                query = query.filter(SystemLog.level == level)
            if source:
                query = query.filter(SystemLog.source.like(f"%{source}%"))
            if bot_id:
                query = query.filter(SystemLog.bot_id == bot_id)
            
            # Ordenar por timestamp (mais recente primeiro)
            query = query.order_by(desc(SystemLog.timestamp))
            
            logs = query.limit(limit).all()
            
            return [log.to_dict() for log in logs]
            
    except Exception as e:
        logger.error(f"❌ Erro ao buscar logs: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# ENDPOINTS DE ESTATÍSTICAS
# =====================================================

@db_router.get("/stats/general", response_model=Dict[str, Any])
async def get_general_stats():
    """Retorna estatísticas gerais de trading."""
    try:
        return db_manager.get_trading_stats()
    except Exception as e:
        logger.error(f"❌ Erro ao buscar estatísticas gerais: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/stats/bots", response_model=List[Dict[str, Any]])
async def get_bot_stats():
    """Retorna estatísticas por bot."""
    try:
        return db_manager.get_bot_stats()
    except Exception as e:
        logger.error(f"❌ Erro ao buscar estatísticas dos bots: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/stats/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    bot_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """Retorna métricas de performance calculadas."""
    try:
        with get_db_session() as session:
            # Calcular período
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            # Chamar stored procedure para calcular métricas
            session.execute(
                text("CALL CalculatePerformanceMetrics(:bot_id, :start_date, :end_date)"),
                {"bot_id": bot_id, "start_date": start_date, "end_date": end_date}
            )
            
            # Buscar resultado da performance diária
            query = session.query(DailyPerformance).filter(
                DailyPerformance.date == end_date
            )
            
            if bot_id:
                query = query.filter(DailyPerformance.bot_id == bot_id)
            
            performance = query.first()
            
            if performance:
                return performance.to_dict()
            else:
                return {
                    "date": end_date.isoformat(),
                    "bot_id": bot_id,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": 0.0
                }
                
    except Exception as e:
        logger.error(f"❌ Erro ao calcular métricas de performance: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# ENDPOINTS DE EQUITY CURVE
# =====================================================

@db_router.get("/equity-curve", response_model=List[Dict[str, Any]])
async def get_equity_curve(
    bot_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
):
    """Retorna dados da equity curve."""
    try:
        return db_manager.get_equity_curve_data(days=days, bot_id=bot_id)
    except Exception as e:
        logger.error(f"❌ Erro ao buscar equity curve: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.post("/equity-curve", response_model=Dict[str, Any])
async def add_equity_point(
    bot_id: Optional[str] = None,
    balance: float = Field(..., gt=0),
    equity: float = Field(..., gt=0),
    drawdown: float = Field(0.0, ge=0),
    total_trades: int = Field(0, ge=0),
    open_trades: int = Field(0, ge=0)
):
    """Adiciona um ponto na equity curve."""
    try:
        with get_db_session() as session:
            # Calcular drawdown percentual
            drawdown_pct = (drawdown / equity * 100) if equity > 0 else 0
            
            new_point = EquityCurve(
                timestamp=datetime.now(),
                bot_id=bot_id,
                balance=balance,
                equity=equity,
                drawdown=drawdown,
                drawdown_pct=drawdown_pct,
                total_trades=total_trades,
                open_trades=open_trades
            )
            
            session.add(new_point)
            session.flush()
            
            return {"success": True, "point_id": new_point.id}
            
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar ponto na equity curve: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# ENDPOINTS DE ADMINISTRAÇÃO
# =====================================================

@db_router.post("/admin/clear-history", response_model=Dict[str, Any])
async def clear_trading_history(confirm: bool = Query(False)):
    """
    CUIDADO: Limpa todo o histórico de trades e logs.
    Requer confirmação explícita.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Confirmação necessária. Use ?confirm=true para confirmar a operação."
        )
    
    try:
        success = db_manager.clear_all_data()
        if success:
            logger.warning("🗑️ HISTÓRICO LIMPO - Todos os trades e logs foram removidos")
            return {
                "success": True, 
                "message": "Histórico de trades e logs limpo com sucesso",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Falha ao limpar histórico")
            
    except Exception as e:
        logger.error(f"❌ Erro ao limpar histórico: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/admin/health", response_model=Dict[str, Any])
async def database_health_check():
    """Verifica a saúde da conexão com o banco de dados."""
    try:
        return db_manager.health_check()
    except Exception as e:
        logger.error(f"❌ Health check falhou: {e}")
        return {
            "status": "error",
            "connected": False,
            "error": str(e),
            "timestamp": datetime.now().timestamp()
        }

@db_router.post("/admin/cleanup", response_model=Dict[str, Any])
async def cleanup_old_data(days_to_keep: int = Query(30, ge=7, le=365)):
    """Limpa dados antigos do banco (logs e equity curve)."""
    try:
        with get_db_session() as session:
            # Chamar stored procedure de limpeza
            session.execute(
                text("CALL CleanOldData(:days_to_keep)"),
                {"days_to_keep": days_to_keep}
            )
            
            logger.info(f"🧹 Limpeza automática executada - mantidos últimos {days_to_keep} dias")
            return {
                "success": True,
                "message": f"Dados antigos limpos (mantidos últimos {days_to_keep} dias)",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ Erro na limpeza automática: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# ENDPOINTS DE BOTS
# =====================================================

@db_router.get("/bots", response_model=List[Dict[str, Any]])
async def get_trading_bots():
    """Retorna todos os bots de trading."""
    try:
        with get_db_session() as session:
            bots = session.query(TradingBot).all()
            return [bot.to_dict() for bot in bots]
            
    except Exception as e:
        logger.error(f"❌ Erro ao buscar bots: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.put("/bots/{bot_id}/status", response_model=Dict[str, Any])
async def update_bot_status(bot_id: str, status_update: BotStatusUpdate):
    """Atualiza o status de um bot."""
    try:
        with get_db_session() as session:
            bot = session.query(TradingBot).filter(TradingBot.bot_id == bot_id).first()
            if not bot:
                raise HTTPException(status_code=404, detail=f"Bot {bot_id} não encontrado")
            
            bot.status = BotStatus(status_update.status)
            
            logger.info(f"✅ Status do bot {bot_id} atualizado para {status_update.status}")
            return {"success": True, "bot_id": bot_id, "new_status": status_update.status}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar status do bot: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# =====================================================
# MIDDLEWARE PARA INICIALIZAÇÃO DO BANCO
# =====================================================

async def ensure_database_connection():
    """Garante que a conexão com o banco está ativa."""
    if not db_manager._is_connected:
        success = db_manager.initialize()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Banco de dados não disponível. Verifique a configuração."
            )

# =====================================================
# TRADING RESTRICTIONS ENDPOINTS
# =====================================================

class TradingRestrictionCreate(BaseModel):
    name: str = Field(..., description="Nome da restrição")
    description: Optional[str] = Field(None, description="Descrição da restrição")
    start_day_of_week: int = Field(..., ge=0, le=6, description="Dia da semana inicial (0=Domingo)")
    start_time: str = Field(..., description="Horário inicial (HH:MM:SS)")
    end_day_of_week: int = Field(..., ge=0, le=6, description="Dia da semana final (0=Domingo)")
    end_time: str = Field(..., description="Horário final (HH:MM:SS)")
    timezone: str = Field(default="America/Sao_Paulo", description="Timezone")
    is_active: bool = Field(default=True, description="Se a restrição está ativa")

class TradingRestrictionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    start_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    start_time: Optional[str] = None
    end_day_of_week: Optional[int] = Field(None, ge=0, le=6)
    end_time: Optional[str] = None
    timezone: Optional[str] = None
    is_active: Optional[bool] = None

@db_router.get("/restrictions", response_model=List[Dict[str, Any]])
async def get_trading_restrictions(
    active_only: bool = Query(False, description="Retornar apenas restrições ativas")
):
    """Listar todas as restrições de horário de trading."""
    try:
        with get_db_session() as session:
            query = session.query(TradingRestriction)
            
            if active_only:
                query = query.filter(TradingRestriction.is_active == True)
            
            restrictions = query.order_by(TradingRestriction.start_day_of_week, TradingRestriction.start_time).all()
            
            result = []
            for restriction in restrictions:
                restriction_dict = restriction.to_dict()
                restriction_dict['formatted_period'] = restriction.get_formatted_period()
                result.append(restriction_dict)
            
            logger.info(f"✅ Retornadas {len(result)} restrições de trading")
            return result
            
    except Exception as e:
        logger.error(f"❌ Erro ao buscar restrições: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.post("/restrictions", response_model=Dict[str, Any])
async def create_trading_restriction(restriction: TradingRestrictionCreate):
    """Criar nova restrição de horário de trading."""
    try:
        from datetime import time
        
        with get_db_session() as session:
            # Converter strings de tempo para objetos time
            start_time = time.fromisoformat(restriction.start_time)
            end_time = time.fromisoformat(restriction.end_time)
            
            new_restriction = TradingRestriction(
                name=restriction.name,
                description=restriction.description,
                start_day_of_week=restriction.start_day_of_week,
                start_time=start_time,
                end_day_of_week=restriction.end_day_of_week,
                end_time=end_time,
                timezone=restriction.timezone,
                is_active=restriction.is_active
            )
            
            session.add(new_restriction)
            session.commit()
            session.refresh(new_restriction)
            
            result = new_restriction.to_dict()
            result['formatted_period'] = new_restriction.get_formatted_period()
            
            logger.info(f"✅ Restrição criada: {restriction.name}")
            return result
            
    except ValueError as e:
        logger.error(f"❌ Formato de horário inválido: {e}")
        raise HTTPException(status_code=400, detail=f"Formato de horário inválido: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Erro ao criar restrição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.put("/restrictions/{restriction_id}", response_model=Dict[str, Any])
async def update_trading_restriction(restriction_id: int, restriction_update: TradingRestrictionUpdate):
    """Atualizar restrição de horário existente."""
    try:
        from datetime import time
        
        with get_db_session() as session:
            restriction = session.query(TradingRestriction).filter(TradingRestriction.id == restriction_id).first()
            if not restriction:
                raise HTTPException(status_code=404, detail=f"Restrição {restriction_id} não encontrada")
            
            # Atualizar campos fornecidos
            if restriction_update.name is not None:
                restriction.name = restriction_update.name
            if restriction_update.description is not None:
                restriction.description = restriction_update.description
            if restriction_update.start_day_of_week is not None:
                restriction.start_day_of_week = restriction_update.start_day_of_week
            if restriction_update.start_time is not None:
                restriction.start_time = time.fromisoformat(restriction_update.start_time)
            if restriction_update.end_day_of_week is not None:
                restriction.end_day_of_week = restriction_update.end_day_of_week
            if restriction_update.end_time is not None:
                restriction.end_time = time.fromisoformat(restriction_update.end_time)
            if restriction_update.timezone is not None:
                restriction.timezone = restriction_update.timezone
            if restriction_update.is_active is not None:
                restriction.is_active = restriction_update.is_active
            
            session.commit()
            session.refresh(restriction)
            
            result = restriction.to_dict()
            result['formatted_period'] = restriction.get_formatted_period()
            
            logger.info(f"✅ Restrição {restriction_id} atualizada")
            return result
            
    except ValueError as e:
        logger.error(f"❌ Formato de horário inválido: {e}")
        raise HTTPException(status_code=400, detail=f"Formato de horário inválido: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar restrição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.delete("/restrictions/{restriction_id}")
async def delete_trading_restriction(restriction_id: int):
    """Deletar restrição de horário."""
    try:
        with get_db_session() as session:
            restriction = session.query(TradingRestriction).filter(TradingRestriction.id == restriction_id).first()
            if not restriction:
                raise HTTPException(status_code=404, detail=f"Restrição {restriction_id} não encontrada")
            
            restriction_name = restriction.name
            session.delete(restriction)
            session.commit()
            
            logger.info(f"✅ Restrição '{restriction_name}' deletada")
            return {"success": True, "message": f"Restrição '{restriction_name}' deletada com sucesso"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erro ao deletar restrição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@db_router.get("/restrictions/check", response_model=Dict[str, Any])
async def check_trading_allowed(
    check_time: Optional[str] = Query(None, description="Horário para verificar (ISO format)")
):
    """Verificar se trading está permitido em um horário específico."""
    try:
        from datetime import datetime
        import pytz
        
        # Usar horário atual se não especificado
        if check_time:
            check_datetime = datetime.fromisoformat(check_time.replace('Z', '+00:00'))
        else:
            check_datetime = datetime.now()
        
        # Converter para timezone do Brasil
        brazil_tz = pytz.timezone('America/Sao_Paulo')
        if check_datetime.tzinfo is None:
            check_datetime = brazil_tz.localize(check_datetime)
        else:
            check_datetime = check_datetime.astimezone(brazil_tz)
        
        day_of_week = check_datetime.weekday()  # 0=Monday, converter para 0=Sunday
        day_of_week = (day_of_week + 1) % 7  # Converter para 0=Sunday
        time_of_day = check_datetime.time()
        
        with get_db_session() as session:
            # Buscar restrições ativas
            restrictions = session.query(TradingRestriction).filter(
                TradingRestriction.is_active == True
            ).all()
            
            blocked_by = []
            
            for restriction in restrictions:
                is_blocked = False
                
                # Verificar se está no período de restrição
                if restriction.start_day_of_week == restriction.end_day_of_week:
                    # Restrição no mesmo dia
                    if (day_of_week == restriction.start_day_of_week and 
                        restriction.start_time <= time_of_day <= restriction.end_time):
                        is_blocked = True
                else:
                    # Restrição que atravessa dias
                    if restriction.start_day_of_week < restriction.end_day_of_week:
                        # Caso normal (ex: Sexta 18:00 a Segunda 08:00)
                        if ((day_of_week == restriction.start_day_of_week and time_of_day >= restriction.start_time) or
                            (day_of_week == restriction.end_day_of_week and time_of_day <= restriction.end_time) or
                            (restriction.start_day_of_week < day_of_week < restriction.end_day_of_week)):
                            is_blocked = True
                    else:
                        # Caso especial: atravessa a semana (ex: Sábado 07:00 a Domingo 20:00)
                        if ((day_of_week == restriction.start_day_of_week and time_of_day >= restriction.start_time) or
                            (day_of_week == restriction.end_day_of_week and time_of_day <= restriction.end_time) or
                            (day_of_week >= restriction.start_day_of_week or day_of_week <= restriction.end_day_of_week)):
                            is_blocked = True
                
                if is_blocked:
                    blocked_by.append({
                        'id': restriction.id,
                        'name': restriction.name,
                        'description': restriction.description,
                        'formatted_period': restriction.get_formatted_period()
                    })
            
            is_allowed = len(blocked_by) == 0
            
            result = {
                'is_trading_allowed': is_allowed,
                'check_time': check_datetime.isoformat(),
                'day_of_week': day_of_week,
                'time_of_day': time_of_day.strftime('%H:%M:%S'),
                'blocked_by': blocked_by,
                'total_active_restrictions': len(restrictions)
            }
            
            logger.info(f"✅ Trading {'permitido' if is_allowed else 'bloqueado'} em {check_datetime}")
            return result
            
    except Exception as e:
        logger.error(f"❌ Erro ao verificar restrições: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Adicionar dependency para todos os endpoints
for route in db_router.routes:
    if hasattr(route, 'dependencies'):
        route.dependencies.append(Depends(ensure_database_connection))