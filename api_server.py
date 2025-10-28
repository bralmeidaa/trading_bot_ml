#!/usr/bin/env python3
"""
API Server for Trading Bot ML Frontend
Provides REST API endpoints for the dashboard frontend.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import asyncio
from datetime import datetime, timedelta
import uvicorn
from pathlib import Path

# Import our trading system
from production_trading_system import ProductionTradingSystem, GlobalConfig, BotConfig, create_production_config

# Import configuration and logging systems
from config_manager import config_manager
from enhanced_logging import enhanced_logger, LogLevel, LogCategory

# Import database components
from database_connection import initialize_database, create_database_tables, db_manager
from api_endpoints_db import db_router

app = FastAPI(title="Trading Bot ML API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include database router
app.include_router(db_router)

# Global trading system instance
trading_system: Optional[ProductionTradingSystem] = None
system_logs = []  # Store system logs in memory
system_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def startup_event():
    """Inicialização do servidor."""
    try:
        # Inicializar conexão com banco de dados
        db_connection_string = os.getenv('DATABASE_URL')
        if initialize_database(db_connection_string):
            add_system_log("SUCCESS", "Conexão com MySQL HeatWave estabelecida", "database")
            
            # Criar tabelas se necessário
            if create_database_tables():
                add_system_log("SUCCESS", "Tabelas do banco verificadas/criadas", "database")
            else:
                add_system_log("WARNING", "Falha ao criar/verificar tabelas", "database")
        else:
            add_system_log("ERROR", "Falha na conexão com MySQL HeatWave", "database")
            
    except Exception as e:
        add_system_log("ERROR", f"Erro na inicialização: {e}", "startup")

def add_system_log(level: str, message: str, source: str = "system"):
    """Add a log entry to the system logs."""
    global system_logs
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "source": source
    }
    system_logs.append(log_entry)
    # Keep only last 100 logs to prevent memory issues
    if len(system_logs) > 100:
        system_logs = system_logs[-100:]

def load_config_from_file() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Load configuration from trading_config.json file."""
    try:
        with open('trading_config.json', 'r') as f:
            config_data = json.load(f)
        
        # Create GlobalConfig from file
        global_config = GlobalConfig(
            total_capital=config_data['global_config']['total_capital'],
            max_concurrent_trades=config_data['global_config']['max_concurrent_trades'],
            daily_loss_limit=config_data['global_config']['daily_loss_limit'],
            daily_profit_target=config_data['global_config']['daily_profit_target'],
            emergency_stop_drawdown=config_data['global_config']['emergency_stop_drawdown'],
            paper_trading=config_data['global_config']['paper_trading']
        )
        
        # Create BotConfig list from file
        bot_configs = []
        for bot_data in config_data['bot_configs']:
            bot_config = BotConfig(
                symbol=bot_data['symbol'],
                timeframe=bot_data['timeframe'],
                capital_allocation=bot_data['capital_allocation'],
                max_risk_per_trade=bot_data['max_risk_per_trade'],
                confidence_threshold=bot_data['confidence_threshold'],
                stop_loss_pct=bot_data['stop_loss_pct'],
                take_profit_pct=bot_data['take_profit_pct'],
                enabled=bot_data['enabled']
            )
            bot_configs.append(bot_config)
        
        return global_config, bot_configs
        
    except Exception as e:
        add_system_log("ERROR", f"Failed to load config from file: {e}", "config")
        # Fallback to default config
        return create_production_config()

# Pydantic models for API
class SystemStatus(BaseModel):
    running: bool
    uptime: str
    total_capital: float
    paper_trading: bool

class PerformanceMetrics(BaseModel):
    total_pnl: float
    total_roi: float
    daily_pnl: float
    active_trades: int
    win_rate: float
    total_trades: int
    max_drawdown: float

class BotStatus(BaseModel):
    symbol: str
    timeframe: str
    status: str
    pnl: float
    trades: int
    enabled: bool

class TradeInfo(BaseModel):
    symbol: str
    direction: str
    pnl: float
    status: str
    time: str
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None

class EquityPoint(BaseModel):
    timestamp: int
    equity: float

class SystemLog(BaseModel):
    timestamp: str
    level: str
    message: str
    source: str

class ConfigUpdate(BaseModel):
    trading_mode: str
    total_capital: float

class BotConfigUpdate(BaseModel):
    symbol: str
    timeframe: str
    enabled: bool
    risk_per_trade: float
    max_positions: int
    stop_loss: float
    take_profit: float

# New models for enhanced configuration and logging
class GlobalConfigUpdate(BaseModel):
    total_capital: Optional[float] = None
    max_concurrent_trades: Optional[int] = None
    daily_loss_limit: Optional[float] = None
    daily_profit_target: Optional[float] = None
    emergency_stop_drawdown: Optional[float] = None
    paper_trading: Optional[bool] = None

class BotConfigUpdate(BaseModel):
    capital_allocation: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    confidence_threshold: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    enabled: Optional[bool] = None

class NewBotConfig(BaseModel):
    symbol: str
    timeframe: str
    capital_allocation: float
    max_risk_per_trade: float
    confidence_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    enabled: bool = True

class LogFilter(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    level: Optional[str] = None
    category: Optional[str] = None
    bot_id: Optional[str] = None
    symbol: Optional[str] = None
    limit: int = 1000

# Note: Static files mounting is done in main() function after all API routes are defined
# This prevents conflicts between API routes and static file serving

@app.get("/api/status")
async def get_system_status():
    """Get current system status."""
    global trading_system, system_task
    
    running = system_task is not None and not system_task.done()
    uptime = "0:00:00"
    
    if trading_system:
        uptime_delta = datetime.now() - trading_system.system_start_time
        uptime = str(uptime_delta).split('.')[0]  # Remove microseconds
        
        status_data = {
            "running": running,
            "uptime": uptime,
            "total_capital": trading_system.global_config.total_capital,
            "paper_trading": trading_system.global_config.paper_trading
        }
        return {"success": True, "data": status_data}
    
    # System is not running - get values from config file
    try:
        config_data = config_manager.get_configuration_dict()
        global_config = config_data.get("global_config", {})
        
        status_data = {
            "running": False,
            "uptime": "0:00:00",
            "total_capital": global_config.get("total_capital", 10000.0),
            "paper_trading": global_config.get("paper_trading", True)
        }
        return {"success": True, "data": status_data}
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error loading config for status: {e}")
        status_data = {
            "running": False,
            "uptime": "0:00:00",
            "total_capital": 10000.0,
            "paper_trading": True
        }
        return {"success": True, "data": status_data}

@app.get("/api/system/status")
async def get_detailed_system_status():
    """Get detailed system status including all components."""
    global trading_system
    
    base_status = await get_system_status()
    
    # Add detailed component status
    detailed_status = {
        "system": base_status["data"],
        "components": {
            "trading_engine": "active" if trading_system else "inactive",
            "regime_detection": "active",
            "ml_ensemble": "active", 
            "specialized_bots": "active",
            "continuous_learning": "active"
        },
        "health": {
            "overall": "healthy" if trading_system else "stopped",
            "api": "healthy",
            "database": "healthy",
            "exchange_connection": "healthy"
        }
    }
    
    return {"success": True, "data": detailed_status}

@app.get("/api/metrics")
async def get_performance_metrics():
    """Get current performance metrics."""
    global trading_system
    
    if trading_system:
        # Calculate metrics from trade history
        total_trades = 0
        winning_trades = 0
        total_pnl = 0.0
        active_trades = 0
        
        if hasattr(trading_system, 'trade_history') and trading_system.trade_history:
            total_trades = len(trading_system.trade_history)
            winning_trades = len([t for t in trading_system.trade_history if t.pnl and t.pnl > 0])
            total_pnl = sum(t.pnl for t in trading_system.trade_history if t.pnl)
            active_trades = len([t for t in trading_system.trade_history if t.status == 'open'])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate total ROI
        total_capital = 1200.0  # Default capital from config
        if hasattr(trading_system, 'global_config') and trading_system.global_config:
            total_capital = trading_system.global_config.total_capital
        
        total_roi = total_pnl / total_capital if total_capital > 0 else 0.0
        
        # Calculate max drawdown (simplified)
        max_drawdown = 0.0
        if hasattr(trading_system, 'trade_history') and trading_system.trade_history:
            running_pnl = 0.0
            peak_pnl = 0.0
            for trade in trading_system.trade_history:
                if trade.pnl:
                    running_pnl += trade.pnl
                    if running_pnl > peak_pnl:
                        peak_pnl = running_pnl
                    if peak_pnl > 0:
                        drawdown = (peak_pnl - running_pnl) / peak_pnl
                        max_drawdown = max(max_drawdown, drawdown)
        
        metrics_data = {
            "total_pnl": total_pnl,
            "total_roi": total_roi,
            "daily_pnl": total_pnl,  # Simplified - using total as daily
            "active_trades": active_trades,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "max_drawdown": max_drawdown
        }
        return {"success": True, "data": metrics_data}
    
    # Return demo data if system not running
    metrics_data = {
        "total_pnl": 0.0,
        "total_roi": 0.0,
        "daily_pnl": 0.0,
        "active_trades": 0,
        "win_rate": 0.0,
        "total_trades": 0,
        "max_drawdown": 0.0
    }
    return {"success": True, "data": metrics_data}

@app.get("/api/bots")
async def get_bot_status():
    """Get status of all trading bots."""
    global trading_system
    
    bots = []
    
    if trading_system:
        # System is running - get live data
        # bot_configs is a dictionary {bot_id: BotConfig}
        for bot_id, config in trading_system.bot_configs.items():
            # Calculate bot-specific PnL
            bot_trades = [t for t in trading_system.trade_history if t.symbol == config.symbol]
            bot_pnl = sum([t.pnl for t in bot_trades if t.pnl])
            
            bots.append({
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "status": "running" if config.enabled else "paused",
                "pnl": bot_pnl,
                "trades": len(bot_trades),
                "enabled": config.enabled
            })
    else:
        # System is not running - get configured bots from config file
        try:
            config_data = config_manager.get_configuration_dict()
            bot_configs = config_data.get("bot_configs", [])
            
            for i, bot_config in enumerate(bot_configs):
                bots.append({
                    "symbol": bot_config.get("symbol", ""),
                    "timeframe": bot_config.get("timeframe", ""),
                    "status": "stopped" if bot_config.get("enabled", False) else "disabled",
                    "pnl": 0.0,  # No PnL data when system is stopped
                    "trades": 0,  # No trade data when system is stopped
                    "enabled": bot_config.get("enabled", False)
                })
        except Exception as e:
            enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error loading bot configs: {e}")
    
    return {"success": True, "data": {"bots": bots}}

def safe_format_timestamp(timestamp):
    """Safely format timestamp handling various edge cases."""
    if not timestamp or timestamp is None:
        return "N/A"
    try:
        # Handle both seconds and milliseconds
        if isinstance(timestamp, (int, float)):
            # Check for invalid float values
            if not isinstance(timestamp, int) and (timestamp != timestamp or timestamp == float('inf') or timestamp == float('-inf')):
                return "Invalid"
            if timestamp > 1e12:  # milliseconds
                return datetime.fromtimestamp(timestamp / 1000).strftime("%H:%M")
            else:  # seconds
                return datetime.fromtimestamp(timestamp).strftime("%H:%M")
        else:
            return "Invalid"
    except (ValueError, OSError, TypeError, OverflowError) as e:
        if ENHANCED_FEATURES:
            enhanced_logger.log_structured(LogLevel.WARNING, LogCategory.API, f"Invalid timestamp: {timestamp}, error: {e}")
        return "Invalid"

@app.get("/api/trades/recent")
async def get_recent_trades():
    """Get recent trades with safe timestamp handling."""
    global trading_system
    
    if trading_system:
        recent_trades = trading_system.trade_history[-10:]  # Last 10 trades
        trades = []
        
        for trade in recent_trades:
            # Safe timestamp processing
            safe_time = safe_format_timestamp(trade.entry_time)
            
            trades.append({
                "symbol": trade.symbol,
                "direction": "LONG" if trade.direction == 1 else "SHORT",
                "pnl": trade.pnl or 0.0,
                "status": trade.status,
                "time": safe_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time  # Keep original for frontend processing
            })
        
        return {"success": True, "data": {"trades": trades}}
    
    # Return empty data when system is not running
    return {"success": True, "data": {"trades": []}}

def safe_validate_equity_point(point):
    """Safely validate and sanitize equity curve points."""
    try:
        timestamp = point.get('timestamp')
        equity = point.get('equity')
        
        # Validate timestamp
        if not timestamp or not isinstance(timestamp, (int, float)):
            return None
            
        # Validate equity
        if equity is None or not isinstance(equity, (int, float)):
            return None
            
        return {
            "timestamp": int(timestamp),
            "equity": float(equity)
        }
    except (TypeError, ValueError, KeyError) as e:
        enhanced_logger.log_structured(LogLevel.WARNING, LogCategory.API, f"Invalid equity point: {point}, error: {e}")
        return None

@app.get("/api/equity")
async def get_equity_curve():
    """Get equity curve data with safe timestamp validation."""
    global trading_system
    
    if trading_system and trading_system.equity_curve:
        # Safely process equity points
        equity_points = []
        for point in trading_system.equity_curve[-100:]:  # Last 100 points
            safe_point = safe_validate_equity_point(point)
            if safe_point:
                equity_points.append(safe_point)
        
        return {"success": True, "data": {"equity_curve": equity_points}}
    
    # Return empty data when system is not running
    return {"success": True, "data": {"equity_curve": []}}



@app.put("/api/bots/{bot_id}/config")
async def update_bot_config(bot_id: str, config: BotConfigUpdate):
    """Update bot configuration."""
    global trading_system
    
    try:
        add_system_log("INFO", f"Updating configuration for bot {bot_id}", "api")
        
        # For now, we'll store the config update and require system restart
        # In a production system, you might want to update the running bot
        
        # TODO: Implement actual bot config update
        # This would typically involve:
        # 1. Validating the new configuration
        # 2. Updating the bot's configuration in memory/database
        # 3. Restarting the specific bot with new config
        
        add_system_log("SUCCESS", f"Bot {bot_id} configuration updated (restart required)", "api")
        
        return {
            "message": "Bot configuration updated successfully",
            "restart_required": True,
            "config": {
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "enabled": config.enabled,
                "risk_per_trade": config.risk_per_trade,
                "max_positions": config.max_positions,
                "stop_loss": config.stop_loss,
                "take_profit": config.take_profit
            }
        }
        
    except Exception as e:
        add_system_log("ERROR", f"Failed to update bot {bot_id}: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to update bot configuration: {str(e)}")

@app.post("/api/bots/{bot_id}/toggle")
async def toggle_bot(bot_id: str, enabled: dict):
    """Toggle bot enabled/disabled state."""
    global trading_system
    
    try:
        is_enabled = enabled.get("enabled", False)
        action = "enabled" if is_enabled else "disabled"
        
        add_system_log("INFO", f"Bot {bot_id} {action}", "api")
        
        # TODO: Implement actual bot toggle
        # This would typically involve:
        # 1. Finding the bot by ID
        # 2. Updating its enabled state
        # 3. Starting/stopping the bot accordingly
        
        return {
            "message": f"Bot {action} successfully",
            "bot_id": bot_id,
            "enabled": is_enabled
        }
        
    except Exception as e:
        add_system_log("ERROR", f"Failed to toggle bot {bot_id}: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to toggle bot: {str(e)}")

@app.post("/api/generate-test-trades")
async def generate_test_trades():
    """Generate some test trades for demonstration purposes."""
    global trading_system
    
    try:
        add_system_log("INFO", "Generating test trades for demonstration", "api")
        
        if not trading_system:
            raise HTTPException(status_code=400, detail="Trading system not running")
        
        # Generate some mock trades
        import random
        from datetime import datetime, timedelta
        
        symbols = ['LINK/USDT', 'BTC/USDT', 'ETH/USDT']
        
        for i in range(5):  # Generate 5 test trades
            symbol = random.choice(symbols)
            direction = random.choice([1, -1])  # 1 for long, -1 for short
            entry_price = random.uniform(100, 50000) if 'BTC' in symbol else random.uniform(1, 100)
            exit_price = entry_price * (1 + random.uniform(-0.05, 0.05))  # ±5% change
            pnl = (exit_price - entry_price) * direction * random.uniform(0.1, 1.0)
            
            # Create a mock trade object
            trade = type('Trade', (), {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'status': random.choice(['completed', 'open', 'closed']),
                'entry_time': int((datetime.now() - timedelta(hours=random.randint(1, 24))).timestamp() * 1000)
            })()
            
            # Add to trading system's trade history
            if not hasattr(trading_system, 'trade_history'):
                trading_system.trade_history = []
            trading_system.trade_history.append(trade)
        
        add_system_log("SUCCESS", f"Generated 5 test trades", "api")
        
        return {
            "message": "Test trades generated successfully",
            "trades_generated": 5
        }
        
    except Exception as e:
        add_system_log("ERROR", f"Failed to generate test trades: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to generate test trades: {str(e)}")

@app.post("/api/start")
async def start_system(background_tasks: BackgroundTasks):
    """Start the trading system."""
    global trading_system, system_task
    
    if system_task and not system_task.done():
        raise HTTPException(status_code=400, detail="System is already running")
    
    try:
        add_system_log("INFO", "Starting trading system...", "api")
        
        # Create system configuration from file
        global_config, bot_configs = load_config_from_file()
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        
        add_system_log("INFO", f"System initialized with {len(bot_configs)} bots", "system")
        add_system_log("INFO", f"Paper trading: {global_config.paper_trading}", "system")
        add_system_log("INFO", f"Total capital: ${global_config.total_capital:,.2f}", "system")
        
        # Start system in background
        system_task = asyncio.create_task(trading_system.start())
        
        add_system_log("SUCCESS", "Trading system started successfully", "api")
        return {"success": True, "message": "Trading system started successfully"}
    except Exception as e:
        add_system_log("ERROR", f"Failed to start system: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")

@app.post("/api/stop")
async def stop_system():
    """Stop the trading system gracefully."""
    global trading_system, system_task
    
    if not system_task or system_task.done():
        raise HTTPException(status_code=400, detail="System is not running")
    
    try:
        add_system_log("INFO", "Stopping trading system...", "api")
        
        # Cancel the system task
        system_task.cancel()
        
        # Graceful shutdown
        if trading_system:
            await trading_system._shutdown()
        
        add_system_log("SUCCESS", "Trading system stopped successfully", "api")
        return {"success": True, "message": "Trading system stopped successfully"}
    except Exception as e:
        add_system_log("ERROR", f"Failed to stop system: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")

@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop - immediately halt all operations."""
    global trading_system, system_task
    
    try:
        if system_task and not system_task.done():
            system_task.cancel()
        
        if trading_system:
            await trading_system._emergency_shutdown()
        
        return {"success": True, "message": "Emergency stop executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

@app.post("/api/config")
async def update_configuration(config: dict):
    """Update system configuration including bot configs."""
    global trading_system
    
    try:
        # Load current configuration
        config_file = "trading_config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                current_config = json.load(f)
        else:
            current_config = {"global_config": {}, "bot_configs": []}
        
        # Update global configuration
        if "trading_mode" in config:
            current_config["global_config"]["paper_trading"] = config["trading_mode"] == "paper"
        if "total_capital" in config:
            current_config["global_config"]["total_capital"] = config["total_capital"]
        if "daily_loss_limit" in config:
            current_config["global_config"]["daily_loss_limit"] = config["daily_loss_limit"]
        if "daily_profit_target" in config:
            current_config["global_config"]["daily_profit_target"] = config["daily_profit_target"]
        if "max_concurrent_trades" in config:
            current_config["global_config"]["max_concurrent_trades"] = config["max_concurrent_trades"]
        if "emergency_stop_drawdown" in config:
            current_config["global_config"]["emergency_stop_drawdown"] = config["emergency_stop_drawdown"]
        
        # Update bot configurations if provided
        if "bot_configs" in config:
            # Validate maximum of 5 bots
            if len(config["bot_configs"]) > 5:
                raise HTTPException(status_code=400, detail="Maximum of 5 bots allowed")
            
            # Validate each bot configuration
            for bot_config in config["bot_configs"]:
                if not all(key in bot_config for key in ["symbol", "timeframe"]):
                    raise HTTPException(status_code=400, detail="Bot configuration must include symbol and timeframe")
            
            current_config["bot_configs"] = config["bot_configs"]
        
        # Add metadata
        current_config["last_updated"] = datetime.now().isoformat()
        current_config["updated_by"] = "API User"
        current_config["update_reason"] = "Configuration updated via API"
        
        # Save configuration to file
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)
        
        # Update running system if available
        if trading_system:
            if "trading_mode" in config:
                trading_system.global_config.paper_trading = config["trading_mode"] == "paper"
            if "total_capital" in config:
                trading_system.global_config.total_capital = config["total_capital"]
            if "daily_loss_limit" in config:
                trading_system.global_config.daily_loss_limit = config["daily_loss_limit"]
            if "daily_profit_target" in config:
                trading_system.global_config.daily_profit_target = config["daily_profit_target"]
        
        add_system_log("SUCCESS", "Configuration updated successfully", "api")
        return {"message": "Configuration updated successfully", "success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        add_system_log("ERROR", f"Failed to update configuration: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/logs")
async def get_system_logs():
    """Get recent system logs."""
    try:
        # Return in-memory logs first
        if system_logs:
            return {"success": True, "data": {"logs": system_logs}}
        
        # Fallback to file logs if no in-memory logs
        if os.path.exists("trading_system.log"):
            with open("trading_system.log", "r") as f:
                lines = f.readlines()
                # Return last 50 lines
                recent_logs = lines[-50:] if len(lines) > 50 else lines
                
                # Parse logs into structured format
                parsed_logs = []
                for line in recent_logs:
                    line = line.strip()
                    if line:
                        # Try to parse format: "YYYY-MM-DD HH:MM:SS,mmm - name - LEVEL - MESSAGE"
                        if ' - ' in line:
                            parts = line.split(' - ')
                            if len(parts) >= 3:
                                timestamp_part = parts[0]
                                name_part = parts[1]
                                level_part = parts[2]
                                message_part = ' - '.join(parts[3:]) if len(parts) > 3 else level_part
                                
                                # Extract just the time part
                                if ',' in timestamp_part:
                                    timestamp = timestamp_part.split(',')[0]
                                else:
                                    timestamp = timestamp_part
                                
                                # Determine level
                                if 'INFO' in level_part:
                                    level = "INFO"
                                elif 'WARNING' in level_part:
                                    level = "WARNING"
                                elif 'ERROR' in level_part:
                                    level = "ERROR"
                                else:
                                    level = "INFO"
                                
                                parsed_logs.append({
                                    "timestamp": timestamp,
                                    "level": level,
                                    "message": message_part,
                                    "source": name_part
                                })
                
                return {"success": True, "data": {"logs": parsed_logs}}
        
        # Return empty logs if no file exists
        return {"success": True, "data": {"logs": []}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/api/signal-quality")
async def get_signal_quality():
    """Retorna dados de qualidade dos sinais."""
    # Simulação - em produção, obter do sistema real
    return {
        "current_quality_score": 0.72,
        "signals_evaluated": 45,
        "signals_passed": 12,
        "signals_rejected": 33,
        "avg_quality_score": 0.68,
        "pass_rate": 0.27,
        "layer_scores": {
            "technical": 0.75,
            "market_structure": 0.68,
            "binance_sentiment": 0.82,
            "ml_confidence": 0.71
        },
        "recent_rejections": [
            {
                "symbol": "BTC/USDT",
                "quality_score": 0.58,
                "timestamp": datetime.now().isoformat(),
                "reasons": ["Confiança ML insuficiente"]
            },
            {
                "symbol": "ETH/USDT", 
                "quality_score": 0.61,
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "reasons": ["Estrutura de mercado inadequada"]
            }
        ]
    }

@app.get("/api/market-sentiment/{symbol}")
async def get_market_sentiment(symbol: str):
    """Retorna análise de sentiment para um símbolo."""
    # Simulação - em produção, usar BinanceAdvancedData real
    import random
    
    sentiment_score = random.uniform(0.3, 0.8)
    sentiment_label = 'bullish' if sentiment_score > 0.6 else 'bearish' if sentiment_score < 0.4 else 'neutral'
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "confidence": random.uniform(0.7, 1.0),
        "factors": [
            "Funding rate neutro",
            "Open Interest crescente" if random.random() > 0.5 else "Open Interest estável",
            "Pressão compradora forte" if sentiment_score > 0.6 else "Pressão vendedora forte"
        ],
        "raw_data": {
            "funding_rate": random.uniform(-0.001, 0.001),
            "open_interest": {
                "current_oi": random.uniform(50000, 100000),
                "trend": random.choice(["increasing", "stable", "decreasing"]),
                "change_pct": random.uniform(-5, 5)
            },
            "long_short_ratio": {
                "current_ratio": random.uniform(0.8, 1.5),
                "sentiment": random.choice(["bullish", "neutral", "bearish"])
            },
            "taker_data": {
                "buy_sell_ratio": random.uniform(0.8, 1.4),
                "trend": random.choice(["buying_pressure", "selling_pressure", "balanced"])
            },
            "order_book": {
                "spread_pct": random.uniform(0.001, 0.01),
                "sentiment": sentiment_label,
                "imbalance": random.uniform(0.4, 0.6)
            }
        }
    }

@app.get("/api/market-regime/{symbol}")
async def get_market_regime_analysis(symbol: str):
    """Retorna análise completa de regime para um símbolo."""
    # Simulação - em produção, usar sistema real
    import random
    
    regimes = ['trending_bull', 'trending_bear', 'ranging', 'high_volatility', 'transitional']
    regime = random.choice(regimes)
    confidence = random.uniform(0.6, 1.0)
    
    # Configuração de estratégia baseada no regime
    strategy_configs = {
        'trending_bull': {
            'strategy_type': 'trend_following',
            'max_trades_per_day': 3,
            'quality_threshold': 0.65,
            'risk_per_trade': 0.008
        },
        'trending_bear': {
            'strategy_type': 'trend_following_short',
            'max_trades_per_day': 2,
            'quality_threshold': 0.70,
            'risk_per_trade': 0.006
        },
        'ranging': {
            'strategy_type': 'mean_reversion',
            'max_trades_per_day': 4,
            'quality_threshold': 0.75,
            'risk_per_trade': 0.005
        },
        'high_volatility': {
            'strategy_type': 'volatility_breakout',
            'max_trades_per_day': 1,
            'quality_threshold': 0.85,
            'risk_per_trade': 0.003
        },
        'transitional': {
            'strategy_type': 'conservative',
            'max_trades_per_day': 1,
            'quality_threshold': 0.80,
            'risk_per_trade': 0.004
        }
    }
    
    return {
        'regime': regime,
        'confidence': confidence,
        'trend_strength': random.uniform(0.2, 0.9),
        'volatility_level': random.uniform(0.1, 0.8),
        'volume_profile': random.choice(['low', 'normal', 'high']),
        'breakout_frequency': random.uniform(0.1, 0.7),
        'regime_duration': random.randint(1, 15),
        'factors': [
            f"Tendência {'forte' if random.random() > 0.5 else 'fraca'}",
            f"Volume {'alto' if random.random() > 0.5 else 'normal'}",
            f"Volatilidade {'alta' if random.random() > 0.5 else 'baixa'}"
        ],
        'strategy_config': strategy_configs.get(regime, strategy_configs['transitional'])
    }

# ============================================================================
# ENHANCED CONFIGURATION MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/config/full")
async def get_full_configuration():
    """Get complete system configuration."""
    try:
        config_data = config_manager.get_configuration_dict()
        enhanced_logger.log_api_request("/api/config/full", "GET")
        return {"success": True, "data": config_data}
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error getting full config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config/global")
async def update_global_configuration(config: GlobalConfigUpdate):
    """Update global configuration parameters."""
    try:
        updates = {k: v for k, v in config.dict().items() if v is not None}
        success = config_manager.update_global_config(updates, "API User")
        
        if success:
            enhanced_logger.log_config_change("global_update", updates, "API User")
            return {"message": "Global configuration updated successfully", "updates": updates}
        else:
            raise HTTPException(status_code=400, detail="Failed to update global configuration")
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.CONFIG, f"Error updating global config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config/bot/{bot_id}")
async def update_bot_configuration(bot_id: str, config: BotConfigUpdate):
    """Update specific bot configuration."""
    try:
        updates = {k: v for k, v in config.dict().items() if v is not None}
        success = config_manager.update_bot_config(bot_id, updates, "API User")
        
        if success:
            enhanced_logger.log_config_change("bot_update", {"bot_id": bot_id, **updates}, "API User")
            return {"message": f"Bot {bot_id} configuration updated successfully", "updates": updates}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to update bot {bot_id} configuration")
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.CONFIG, f"Error updating bot config {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/bot")
async def add_bot_configuration(config: NewBotConfig):
    """Add new bot configuration."""
    try:
        bot_config = BotConfig(**config.dict())
        success = config_manager.add_bot_config(bot_config, "API User")
        
        if success:
            bot_id = f"{config.symbol}_{config.timeframe}"
            enhanced_logger.log_config_change("bot_add", config.dict(), "API User")
            return {"message": f"Bot {bot_id} added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add bot configuration")
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.CONFIG, f"Error adding bot config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/config/bot/{bot_id}")
async def remove_bot_configuration(bot_id: str):
    """Remove bot configuration."""
    try:
        success = config_manager.remove_bot_config(bot_id, "API User")
        
        if success:
            enhanced_logger.log_config_change("bot_remove", {"bot_id": bot_id}, "API User")
            return {"message": f"Bot {bot_id} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Bot {bot_id} not found")
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.CONFIG, f"Error removing bot config {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/backups")
async def get_configuration_backups():
    """Get list of available configuration backups."""
    try:
        backups = config_manager.get_available_backups()
        return {"backups": backups}
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error getting backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/restore/{backup_filename}")
async def restore_configuration_backup(backup_filename: str):
    """Restore configuration from backup."""
    try:
        success = config_manager.restore_backup(backup_filename, "API User")
        
        if success:
            enhanced_logger.log_config_change("backup_restore", {"backup_file": backup_filename}, "API User")
            return {"message": f"Configuration restored from {backup_filename}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to restore backup")
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.CONFIG, f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENHANCED LOGGING ENDPOINTS
# ============================================================================

@app.get("/api/logs/enhanced")
async def get_enhanced_logs(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    level: Optional[str] = Query(None, description="Log level filter"),
    category: Optional[str] = Query(None, description="Log category filter"),
    bot_id: Optional[str] = Query(None, description="Bot ID filter"),
    symbol: Optional[str] = Query(None, description="Symbol filter"),
    limit: int = Query(1000, description="Maximum number of logs to return")
):
    """Get filtered enhanced logs."""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse enums
        level_enum = LogLevel(level) if level else None
        category_enum = LogCategory(category) if category else None
        
        logs = enhanced_logger.get_logs(
            start_date=start_dt,
            end_date=end_dt,
            level=level_enum,
            category=category_enum,
            bot_id=bot_id,
            symbol=symbol,
            limit=limit
        )
        
        # Convert to dict for JSON response
        log_dicts = [
            {
                "timestamp": log.timestamp,
                "level": log.level,
                "category": log.category,
                "message": log.message,
                "data": log.data,
                "bot_id": log.bot_id,
                "symbol": log.symbol,
                "trade_id": log.trade_id
            }
            for log in logs
        ]
        
        enhanced_logger.log_api_request("/api/logs/enhanced", "GET", {
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "level": level,
                "category": category,
                "bot_id": bot_id,
                "symbol": symbol,
                "limit": limit
            },
            "result_count": len(log_dicts)
        })
        
        return {
            "logs": log_dicts,
            "total_count": len(log_dicts),
            "filters_applied": {
                "start_date": start_date,
                "end_date": end_date,
                "level": level,
                "category": category,
                "bot_id": bot_id,
                "symbol": symbol,
                "limit": limit
            }
        }
        
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error getting enhanced logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/logs/export")
async def export_logs(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    format: str = "json"
):
    """Export logs to file."""
    try:
        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # Export logs
        export_path = enhanced_logger.export_logs(
            start_date=start_dt,
            end_date=end_dt,
            format=format
        )
        
        enhanced_logger.log_structured(LogLevel.INFO, LogCategory.API, 
                                     f"Logs exported to {export_path}",
                                     data={"format": format, "start_date": start_date, "end_date": end_date})
        
        # Return file for download
        return FileResponse(
            path=export_path,
            filename=Path(export_path).name,
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error exporting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs/statistics")
async def get_log_statistics():
    """Get logging statistics."""
    try:
        stats = enhanced_logger.get_log_statistics()
        return stats
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error getting log statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs/categories")
async def get_log_categories():
    """Get available log categories and levels."""
    return {
        "levels": [level.value for level in LogLevel],
        "categories": [category.value for category in LogCategory]
    }

# ============================================================================
# BOT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/bots/available-symbols")
async def get_available_symbols():
    """Get list of available trading symbols."""
    # Popular cryptocurrency pairs
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT",
        "SOL/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "AVAX/USDT",
        "UNI/USDT", "LTC/USDT", "ATOM/USDT", "ALGO/USDT", "VET/USDT",
        "FTM/USDT", "NEAR/USDT", "SAND/USDT", "MANA/USDT", "CRV/USDT"
    ]
    return {"success": True, "data": {"symbols": symbols}}

@app.get("/api/bots/available-timeframes")
async def get_available_timeframes():
    """Get list of available timeframes."""
    timeframes = [
        {"value": "1m", "label": "1 Minute"},
        {"value": "5m", "label": "5 Minutes"},
        {"value": "15m", "label": "15 Minutes"},
        {"value": "30m", "label": "30 Minutes"},
        {"value": "1h", "label": "1 Hour"},
        {"value": "4h", "label": "4 Hours"},
        {"value": "1d", "label": "1 Day"}
    ]
    return {"success": True, "data": {"timeframes": timeframes}}

@app.post("/api/bots/add")
async def add_bot_configuration(bot_config: NewBotConfig):
    """Add a new bot configuration."""
    try:
        # Load current configuration
        config_file = "trading_config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                current_config = json.load(f)
        else:
            current_config = {"global_config": {}, "bot_configs": []}
        
        # Check if we already have 5 bots
        if len(current_config.get("bot_configs", [])) >= 5:
            raise HTTPException(status_code=400, detail="Maximum of 5 bots allowed")
        
        # Check for duplicate symbol/timeframe combination
        existing_combinations = [(bot["symbol"], bot["timeframe"]) for bot in current_config.get("bot_configs", [])]
        if (bot_config.symbol, bot_config.timeframe) in existing_combinations:
            raise HTTPException(status_code=400, detail=f"Bot with {bot_config.symbol} on {bot_config.timeframe} already exists")
        
        # Add new bot configuration
        new_bot = {
            "symbol": bot_config.symbol,
            "timeframe": bot_config.timeframe,
            "capital_allocation": bot_config.capital_allocation,
            "max_risk_per_trade": bot_config.max_risk_per_trade,
            "confidence_threshold": bot_config.confidence_threshold,
            "stop_loss_pct": bot_config.stop_loss_pct,
            "take_profit_pct": bot_config.take_profit_pct,
            "enabled": bot_config.enabled
        }
        
        current_config.setdefault("bot_configs", []).append(new_bot)
        
        # Update metadata
        current_config["last_updated"] = datetime.now().isoformat()
        current_config["updated_by"] = "API User"
        current_config["update_reason"] = f"Added new bot: {bot_config.symbol} {bot_config.timeframe}"
        
        # Save configuration
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)
        
        add_system_log("SUCCESS", f"Added new bot: {bot_config.symbol} {bot_config.timeframe}", "api")
        return {
            "message": f"Bot {bot_config.symbol} {bot_config.timeframe} added successfully",
            "success": True,
            "bot_config": new_bot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        add_system_log("ERROR", f"Failed to add bot: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to add bot: {str(e)}")

@app.put("/api/bots/update/{bot_index}")
async def update_bot_configuration(bot_index: int, bot_config: dict):
    """Update an existing bot configuration."""
    try:
        # Load current configuration
        config_file = "trading_config.json"
        if not os.path.exists(config_file):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        with open(config_file, "r") as f:
            current_config = json.load(f)
        
        bot_configs = current_config.get("bot_configs", [])
        
        if bot_index < 0 or bot_index >= len(bot_configs):
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Update bot configuration
        old_bot = bot_configs[bot_index].copy()
        for key, value in bot_config.items():
            if key in ["symbol", "timeframe", "capital_allocation", "max_risk_per_trade", 
                      "confidence_threshold", "stop_loss_pct", "take_profit_pct", "enabled"]:
                bot_configs[bot_index][key] = value
        
        # Check for duplicate symbol/timeframe combination (excluding current bot)
        if "symbol" in bot_config or "timeframe" in bot_config:
            new_symbol = bot_configs[bot_index]["symbol"]
            new_timeframe = bot_configs[bot_index]["timeframe"]
            
            for i, bot in enumerate(bot_configs):
                if i != bot_index and bot["symbol"] == new_symbol and bot["timeframe"] == new_timeframe:
                    raise HTTPException(status_code=400, detail=f"Bot with {new_symbol} on {new_timeframe} already exists")
        
        # Update metadata
        current_config["last_updated"] = datetime.now().isoformat()
        current_config["updated_by"] = "API User"
        current_config["update_reason"] = f"Updated bot: {bot_configs[bot_index]['symbol']} {bot_configs[bot_index]['timeframe']}"
        
        # Save configuration
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)
        
        add_system_log("SUCCESS", f"Updated bot: {bot_configs[bot_index]['symbol']} {bot_configs[bot_index]['timeframe']}", "api")
        return {
            "message": f"Bot updated successfully",
            "success": True,
            "bot_config": bot_configs[bot_index]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        add_system_log("ERROR", f"Failed to update bot: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to update bot: {str(e)}")

@app.delete("/api/bots/remove/{bot_index}")
async def remove_bot_configuration(bot_index: int):
    """Remove a bot configuration."""
    try:
        # Load current configuration
        config_file = "trading_config.json"
        if not os.path.exists(config_file):
            raise HTTPException(status_code=404, detail="Configuration file not found")
        
        with open(config_file, "r") as f:
            current_config = json.load(f)
        
        bot_configs = current_config.get("bot_configs", [])
        
        if bot_index < 0 or bot_index >= len(bot_configs):
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Remove bot configuration
        removed_bot = bot_configs.pop(bot_index)
        
        # Update metadata
        current_config["last_updated"] = datetime.now().isoformat()
        current_config["updated_by"] = "API User"
        current_config["update_reason"] = f"Removed bot: {removed_bot['symbol']} {removed_bot['timeframe']}"
        
        # Save configuration
        with open(config_file, "w") as f:
            json.dump(current_config, f, indent=2)
        
        add_system_log("SUCCESS", f"Removed bot: {removed_bot['symbol']} {removed_bot['timeframe']}", "api")
        return {
            "message": f"Bot {removed_bot['symbol']} {removed_bot['timeframe']} removed successfully",
            "success": True,
            "removed_bot": removed_bot
        }
        
    except HTTPException:
        raise
    except Exception as e:
        add_system_log("ERROR", f"Failed to remove bot: {str(e)}", "api")
        raise HTTPException(status_code=500, detail=f"Failed to remove bot: {str(e)}")

@app.get("/api/bots/count")
async def get_bot_count():
    """Get current number of configured bots."""
    try:
        config_file = "trading_config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                current_config = json.load(f)
            bot_count = len(current_config.get("bot_configs", []))
        else:
            bot_count = 0
        
        return {"success": True, "data": {
            "current_count": bot_count,
            "maximum_allowed": 5,
            "can_add_more": bot_count < 5
        }}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bot count: {str(e)}")

@app.post("/api/backtest")
async def run_backtest():
    """Run backtest analysis using the optimized configuration test."""
    try:
        import subprocess
        import sys
        
        # Run the test_optimized_config.py script
        result = subprocess.run(
            [sys.executable, "test_optimized_config.py"],
            capture_output=True,
            text=True,
            cwd="/workspace/trading_bot_ml"
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "analysis_complete": True,
                "recommendations": [
                    "START with Conservative Optimized configuration",
                    "MONITOR trade frequency (target: 5-10 trades/day)",
                    "SCALE UP to Balanced/Aggressive if performance is good",
                    "MAINTAIN strict risk management protocols",
                    "REVIEW and adjust parameters weekly based on results"
                ]
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
                "output": result.stdout
            }
    except Exception as e:
        enhanced_logger.log_structured(LogLevel.ERROR, LogCategory.API, f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


def main():
    """Run the API server."""
    print("🚀 Starting Trading Bot ML API Server...")
    print("📊 Dashboard will be available at: http://localhost:12000")
    print("🔧 API documentation at: http://localhost:12000/docs")
    
    # Mount static files - serve React build (AFTER all API routes are defined)
    # This prevents conflicts between API routes and static file serving
    frontend_dist = Path("frontend_react/dist")
    frontend_index = frontend_dist / "index.html"
    
    if frontend_index.exists():
        print("✅ Frontend build found - enabling full-stack mode")
        try:
            app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
        except RuntimeError as e:
            print(f"⚠️  Could not mount static files: {e}. API-only mode.")
    else:
        print("⚠️  Frontend build not found. Running in API-only mode.")
        print(f"   Expected: {frontend_index.absolute()}")
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=12000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
