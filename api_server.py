#!/usr/bin/env python3
"""
API Server for Trading Bot ML Frontend
Provides REST API endpoints for the dashboard frontend.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
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

# CORS — allows the frontend (served at any port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class NewBotConfig(BaseModel):
    symbol: str
    timeframe: str
    capital_allocation: float
    max_risk_per_trade: float
    confidence_threshold: float = 0.65
    stop_loss_pct: float = 0.018
    take_profit_pct: float = 0.035
    enabled: bool = True

class BotConfigUpdate(BaseModel):
    capital_allocation: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    confidence_threshold: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    enabled: Optional[bool] = None

# Static file paths
_REACT_DIST = Path("frontend_react/dist")
_LEGACY_FRONTEND = Path("frontend")

# Vite builds assets into dist/assets/ and the generated index.html references
# them as /assets/... (absolute path from root). Mount that directory directly
# so the browser can resolve them without a /static prefix.
if (_REACT_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_REACT_DIST / "assets")), name="assets")
elif _LEGACY_FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_LEGACY_FRONTEND)), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the React SPA entry point."""
    for candidate in [_REACT_DIST / "index.html", _LEGACY_FRONTEND / "index.html"]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Dashboard not found — run <code>npm run build</code> inside frontend_react/</h1>",
        status_code=404,
    )

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
        # bot_configs is a dict {bot_id: BotConfig} — must iterate .values()
        for config in trading_system.bot_configs.values():
            bot_trades = [t for t in trading_system.trade_history if t.symbol == config.symbol]
            bot_pnl = sum(t.pnl for t in bot_trades if t.pnl is not None)

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

@app.get("/api/config")
async def get_configuration():
    """Return current system configuration so the form can be pre-populated."""
    # Try to load saved config file first, then fall back to running system
    if os.path.exists("system_config.json"):
        try:
            with open("system_config.json", "r") as f:
                return json.load(f)
        except Exception:
            pass

    if trading_system:
        return {
            "trading_mode": "paper" if trading_system.global_config.paper_trading else "live",
            "total_capital": trading_system.global_config.total_capital,
            "daily_loss_limit": trading_system.global_config.daily_loss_limit * 100,
            "daily_profit_target": trading_system.global_config.daily_profit_target * 100,
        }

    return {
        "trading_mode": "paper",
        "total_capital": 1200.0,
        "daily_loss_limit": 4.0,
        "daily_profit_target": 2.5,
    }


@app.get("/api/daily-stats")
async def get_daily_stats():
    """Return daily PnL history for the dashboard."""
    # Try DB-backed stats first
    try:
        from backend.persistence.repository import DailyStatsRepository
        repo = DailyStatsRepository()
        return {"daily_stats": repo.get_history(limit=30)}
    except Exception:
        pass

    if trading_system and trading_system.daily_stats:
        return {"daily_stats": trading_system.daily_stats[-30:]}

    return {"daily_stats": []}


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

# ── Bot management endpoints ────────────────────────────────────────────────

AVAILABLE_SYMBOLS = ["LINK/USDT", "BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

@app.get("/api/bots/available-symbols")
async def get_available_symbols():
    return {"symbols": AVAILABLE_SYMBOLS}

@app.get("/api/bots/available-timeframes")
async def get_available_timeframes():
    return {"timeframes": AVAILABLE_TIMEFRAMES}

@app.get("/api/bots/count")
async def get_bot_count():
    count = len(trading_system.bot_configs) if trading_system else 0
    return {"count": count}

@app.get("/api/bots/{bot_id:path}/config")
async def get_bot_config(bot_id: str):
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not running")
    config = trading_system.bot_configs.get(bot_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    from dataclasses import asdict
    return asdict(config)

@app.post("/api/bots/{bot_id:path}/toggle")
async def toggle_bot(bot_id: str, body: dict = None):
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not running")
    config = trading_system.bot_configs.get(bot_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    # Use explicit value if provided, else flip
    if body and "enabled" in body:
        config.enabled = bool(body["enabled"])
    else:
        config.enabled = not config.enabled
    return {"bot_id": bot_id, "enabled": config.enabled}

@app.post("/api/bots/add")
async def add_bot(new_bot: NewBotConfig):
    global trading_system
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not running")
    bot_id = f"{new_bot.symbol}_{new_bot.timeframe}"
    if bot_id in trading_system.bot_configs:
        raise HTTPException(status_code=409, detail=f"Bot '{bot_id}' already exists")
    from production_trading_system import BotConfig, OptimizedSignalGenerator
    config = BotConfig(
        symbol=new_bot.symbol, timeframe=new_bot.timeframe,
        capital_allocation=new_bot.capital_allocation,
        max_risk_per_trade=new_bot.max_risk_per_trade,
        confidence_threshold=new_bot.confidence_threshold,
        stop_loss_pct=new_bot.stop_loss_pct,
        take_profit_pct=new_bot.take_profit_pct,
        enabled=new_bot.enabled,
    )
    trading_system.bot_configs[bot_id] = config
    trading_system.signal_generators[bot_id] = OptimizedSignalGenerator(new_bot.symbol, new_bot.timeframe)
    return {"message": f"Bot '{bot_id}' added", "bot_id": bot_id}

@app.put("/api/bots/update/{bot_index}")
async def update_bot(bot_index: int, update: BotConfigUpdate):
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not running")
    bots = list(trading_system.bot_configs.items())
    if bot_index < 0 or bot_index >= len(bots):
        raise HTTPException(status_code=404, detail=f"Bot index {bot_index} out of range")
    bot_id, config = bots[bot_index]
    for field, value in update.model_dump(exclude_none=True).items():
        setattr(config, field, value)
    return {"message": f"Bot '{bot_id}' updated"}

@app.delete("/api/bots/remove/{bot_index}")
async def remove_bot(bot_index: int):
    if not trading_system:
        raise HTTPException(status_code=503, detail="Trading system not running")
    bots = list(trading_system.bot_configs.keys())
    if bot_index < 0 or bot_index >= len(bots):
        raise HTTPException(status_code=404, detail=f"Bot index {bot_index} out of range")
    bot_id = bots[bot_index]
    del trading_system.bot_configs[bot_id]
    trading_system.signal_generators.pop(bot_id, None)
    return {"message": f"Bot '{bot_id}' removed"}

# ── Additional config and metrics endpoints ──────────────────────────────────

@app.get("/api/config/full")
async def get_full_config():
    """Return complete system configuration (global + all bots)."""
    if trading_system:
        from dataclasses import asdict
        return {
            "global_config": asdict(trading_system.global_config),
            "bot_configs": [
                {"bot_id": bid, **asdict(cfg)}
                for bid, cfg in trading_system.bot_configs.items()
            ],
        }
    # Defaults when system not running
    _, bot_cfgs = create_production_config()
    from production_trading_system import GlobalConfig
    from dataclasses import asdict
    global_cfg = GlobalConfig(total_capital=1200.0, paper_trading=True)
    return {
        "global_config": asdict(global_cfg),
        "bot_configs": [asdict(c) for c in bot_cfgs],
    }

@app.get("/api/performance-metrics")
async def get_performance_metrics_extended():
    """Extended performance metrics (alias of /api/metrics with extra fields)."""
    base = await get_performance_metrics()
    # Enrich with additional fields expected by React components
    data = base.model_dump() if hasattr(base, 'model_dump') else dict(base)
    data["sharpe_ratio"] = 0.0
    data["profit_factor"] = 0.0
    if trading_system and len(trading_system.trade_history) >= 2:
        returns = [t.pnl_pct for t in trading_system.trade_history if t.pnl_pct is not None]
        if returns:
            import numpy as np
            arr = np.array(returns)
            vol = arr.std() * (252 ** 0.5)
            ann_ret = arr.mean() * 252
            data["sharpe_ratio"] = float(ann_ret / vol) if vol > 0 else 0.0
            wins = arr[arr > 0]
            losses = arr[arr <= 0]
            if len(losses) > 0 and abs(losses.mean()) > 0:
                data["profit_factor"] = float((wins.mean() * len(wins)) / (abs(losses.mean()) * len(losses)))
    return data

@app.post("/api/backtest")
async def run_backtest_endpoint(background_tasks: BackgroundTasks):
    """Trigger the walk-forward backtest in the background."""
    import subprocess, sys
    def _run():
        subprocess.run(
            [sys.executable, "run_backtest.py", "--days", "90", "--splits", "3",
             "--output", "backtest_results.json"],
            capture_output=True
        )
    background_tasks.add_task(_run)
    return {"message": "Backtest started — results will be written to backtest_results.json"}

@app.get("/api/backtest/results")
async def get_backtest_results():
    """Return the latest backtest results if available."""
    if os.path.exists("backtest_results.json"):
        with open("backtest_results.json") as f:
            return json.load(f)
    return {"message": "No backtest results yet — POST /api/backtest to run one"}

# ── SPA catch-all — must be LAST route ─────────────────────────────────────
# Returns index.html for any path that didn't match an API route.
# This enables React Router client-side navigation (e.g. /dashboard, /settings).

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str):
    """Fallback for React Router deep links."""
    # Never intercept API or asset routes (safety net — they're registered first)
    if full_path.startswith("api/") or full_path.startswith("assets/"):
        raise HTTPException(status_code=404, detail="Not found")
    index = _REACT_DIST / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend not built")


# ── Error handlers ─────────────────────────────────────────────────────────

# Error handlers — must return Response objects, not plain dicts
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found"})

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

def main():
    """Run the API server."""
    print("🚀 Starting Trading Bot ML API Server...")
    print("📊 Dashboard will be available at: http://localhost:12000")
    print("🔧 API documentation at: http://localhost:12000/docs")
    
    # Ensure frontend directory exists
    Path("frontend").mkdir(exist_ok=True)
    
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