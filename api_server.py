#!/usr/bin/env python3
"""
API Server for Trading Bot ML Frontend
Provides REST API endpoints for the dashboard frontend.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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

app = FastAPI(title="Trading Bot ML API", version="1.0.0")

# Global trading system instance
trading_system: Optional[ProductionTradingSystem] = None
system_task: Optional[asyncio.Task] = None

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

class ConfigUpdate(BaseModel):
    trading_mode: str
    total_capital: float
    daily_loss_limit: float
    daily_profit_target: float

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

# Mount static files - serve React build
#app.mount("/", StaticFiles(directory="frontend_react/dist", html=True), name="static")

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status."""
    global trading_system, system_task
    
    running = system_task is not None and not system_task.done()
    uptime = "0:00:00"
    
    if trading_system:
        uptime_delta = datetime.now() - trading_system.system_start_time
        uptime = str(uptime_delta).split('.')[0]  # Remove microseconds
        
        return SystemStatus(
            running=running,
            uptime=uptime,
            total_capital=trading_system.global_config.total_capital,
            paper_trading=trading_system.global_config.paper_trading
        )
    
    return SystemStatus(
        running=False,
        uptime="0:00:00",
        total_capital=10000.0,
        paper_trading=True
    )

@app.get("/api/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get current performance metrics."""
    global trading_system
    
    if trading_system:
        # Calculate win rate from trade history
        winning_trades = len([t for t in trading_system.trade_history if t.pnl and t.pnl > 0])
        total_trades = len(trading_system.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate max drawdown from equity curve
        max_drawdown = 0.0
        if trading_system.equity_curve:
            equity_values = [point['equity'] for point in trading_system.equity_curve]
            peak = equity_values[0]
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return PerformanceMetrics(
            total_pnl=trading_system.total_pnl,
            total_roi=trading_system.total_pnl / trading_system.global_config.total_capital,
            daily_pnl=trading_system.daily_pnl,
            active_trades=len(trading_system.active_trades),
            win_rate=win_rate,
            total_trades=total_trades,
            max_drawdown=max_drawdown
        )
    
    # Return demo data if system not running
    return PerformanceMetrics(
        total_pnl=0.0,
        total_roi=0.0,
        daily_pnl=0.0,
        active_trades=0,
        win_rate=0.0,
        total_trades=0,
        max_drawdown=0.0
    )

@app.get("/api/bots", response_model=Dict[str, List[BotStatus]])
async def get_bot_status():
    """Get status of all trading bots."""
    global trading_system
    
    if trading_system:
        bots = []
        for i, config in enumerate(trading_system.bot_configs):
            # Calculate bot-specific PnL
            bot_trades = [t for t in trading_system.trade_history if t.symbol == config.symbol]
            bot_pnl = sum([t.pnl for t in bot_trades if t.pnl])
            
            bots.append(BotStatus(
                symbol=config.symbol,
                timeframe=config.timeframe,
                status="running" if config.enabled else "paused",
                pnl=bot_pnl,
                trades=len(bot_trades),
                enabled=config.enabled
            ))
        
        return {"bots": bots}
    
    # Return empty data when system is not running
    return {"bots": []}

@app.get("/api/trades/recent", response_model=Dict[str, List[TradeInfo]])
async def get_recent_trades():
    """Get recent trades."""
    global trading_system
    
    if trading_system:
        recent_trades = trading_system.trade_history[-10:]  # Last 10 trades
        trades = []
        
        for trade in recent_trades:
            trades.append(TradeInfo(
                symbol=trade.symbol,
                direction="LONG" if trade.direction == 1 else "SHORT",
                pnl=trade.pnl or 0.0,
                status=trade.status,
                time=datetime.fromtimestamp(trade.entry_time / 1000).strftime("%H:%M"),
                entry_price=trade.entry_price,
                exit_price=trade.exit_price
            ))
        
        return {"trades": trades}
    
    # Return empty data when system is not running
    return {"trades": []}

@app.get("/api/equity", response_model=Dict[str, List[EquityPoint]])
async def get_equity_curve():
    """Get equity curve data."""
    global trading_system
    
    if trading_system and trading_system.equity_curve:
        equity_points = [
            EquityPoint(timestamp=point['timestamp'], equity=point['equity'])
            for point in trading_system.equity_curve[-100:]  # Last 100 points
        ]
        return {"equity_curve": equity_points}
    
    # Return empty data when system is not running
    return {"equity_curve": []}

@app.post("/api/start")
async def start_system(background_tasks: BackgroundTasks):
    """Start the trading system."""
    global trading_system, system_task
    
    if system_task and not system_task.done():
        raise HTTPException(status_code=400, detail="System is already running")
    
    try:
        # Create system configuration
        global_config, bot_configs = create_production_config()
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        
        # Start system in background
        system_task = asyncio.create_task(trading_system.start())
        
        return {"message": "Trading system started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")

@app.post("/api/stop")
async def stop_system():
    """Stop the trading system gracefully."""
    global trading_system, system_task
    
    if not system_task or system_task.done():
        raise HTTPException(status_code=400, detail="System is not running")
    
    try:
        # Cancel the system task
        system_task.cancel()
        
        # Graceful shutdown
        if trading_system:
            await trading_system._shutdown()
        
        return {"message": "Trading system stopped successfully"}
    except Exception as e:
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
        
        return {"message": "Emergency stop executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emergency stop failed: {str(e)}")

@app.post("/api/config")
async def update_configuration(config: ConfigUpdate):
    """Update system configuration."""
    global trading_system
    
    try:
        if trading_system:
            # Update global configuration
            trading_system.global_config.total_capital = config.total_capital
            trading_system.global_config.daily_loss_limit = config.daily_loss_limit
            trading_system.global_config.daily_profit_target = config.daily_profit_target
            trading_system.global_config.paper_trading = config.trading_mode == "paper"
        
        # Save configuration to file
        config_data = {
            "trading_mode": config.trading_mode,
            "total_capital": config.total_capital,
            "daily_loss_limit": config.daily_loss_limit,
            "daily_profit_target": config.daily_profit_target,
            "updated_at": datetime.now().isoformat()
        }
        
        with open("system_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        return {"message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/api/logs")
async def get_system_logs():
    """Get recent system logs."""
    try:
        if os.path.exists("trading_system.log"):
            with open("trading_system.log", "r") as f:
                lines = f.readlines()
                # Return last 100 lines
                recent_logs = lines[-100:] if len(lines) > 100 else lines
                return {"logs": [line.strip() for line in recent_logs]}
        
        return {"logs": ["No logs available"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
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
        return config_data
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
    return {"error": "Endpoint not found"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}


# Mount static files - serve React build
app.mount("/", StaticFiles(directory="frontend_react/dist", html=True), name="static")

def main():
    """Run the API server."""
    print("🚀 Starting Trading Bot ML API Server...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    
    # Ensure React build directory exists
    Path("frontend_react/dist").mkdir(parents=True, exist_ok=True)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
