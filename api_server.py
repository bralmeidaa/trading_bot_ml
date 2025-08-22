#!/usr/bin/env python3
"""
API Server for Trading Bot ML Frontend
Provides REST API endpoints for the dashboard frontend.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard page."""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

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
    
    # Return demo data
    return {
        "bots": [
            BotStatus(symbol="LINK/USDT", timeframe="5m", status="running", pnl=156.78, trades=3, enabled=True),
            BotStatus(symbol="LINK/USDT", timeframe="1m", status="running", pnl=89.45, trades=8, enabled=True),
            BotStatus(symbol="ADA/USDT", timeframe="1m", status="running", pnl=234.12, trades=5, enabled=True),
            BotStatus(symbol="ADA/USDT", timeframe="5m", status="paused", pnl=-45.67, trades=2, enabled=False)
        ]
    }

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
    
    # Return demo data
    return {
        "trades": [
            TradeInfo(symbol="LINK/USDT", direction="LONG", pnl=45.67, status="closed", time="10:30"),
            TradeInfo(symbol="ADA/USDT", direction="SHORT", pnl=-23.45, status="closed", time="10:15"),
            TradeInfo(symbol="LINK/USDT", direction="LONG", pnl=78.90, status="open", time="10:00"),
            TradeInfo(symbol="ADA/USDT", direction="LONG", pnl=34.56, status="closed", time="09:45"),
            TradeInfo(symbol="LINK/USDT", direction="SHORT", pnl=-12.34, status="closed", time="09:30")
        ]
    }

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
    
    # Return demo data
    now = datetime.now()
    equity_points = []
    equity = 10000.0
    
    for i in range(50):
        timestamp = int((now - timedelta(minutes=i*5)).timestamp() * 1000)
        equity += (0.5 - __import__('random').random()) * 50
        equity_points.append(EquityPoint(timestamp=timestamp, equity=max(equity, 9000)))
    
    return {"equity_curve": list(reversed(equity_points))}

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

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error"}

def main():
    """Run the API server."""
    print("🚀 Starting Trading Bot ML API Server...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    
    # Ensure frontend directory exists
    Path("frontend").mkdir(exist_ok=True)
    
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