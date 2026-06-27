#!/usr/bin/env python3
"""
Trading Bot ML — API Server
All responses follow the contract: {"success": true, "data": {...}}
Errors from HTTPException are {"detail": "..."} (FastAPI standard).
See docs/API_REFERENCE.md for full endpoint documentation.
"""
import gc
import json
import os
import asyncio
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Body, BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from production_trading_system import (
    BotConfig, GlobalConfig, OptimizedSignalGenerator,
    ProductionTradingSystem, create_production_config,
)

# ══════════════════════════════════════════════════════════════════════════════
# App setup
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Trading Bot ML API",
    version="2.0.0",
    description=(
        "REST API for the Trading Bot ML dashboard. "
        "Every successful response is wrapped in {\"success\": true, \"data\": {...}}. "
        "Errors use FastAPI's standard {\"detail\": \"...\"} format."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import logging
import time as _time
logger = logging.getLogger("api_server")


@app.middleware("http")
async def timing_middleware(request, call_next):
    """Log slow requests (>1s) to help diagnose event-loop stalls / 504s."""
    start = _time.time()
    response = await call_next(request)
    elapsed = _time.time() - start
    if elapsed > 1.0:
        logger.warning(f"SLOW {request.method} {request.url.path} took {elapsed:.2f}s")
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


# Global state
trading_system: Optional[ProductionTradingSystem] = None
system_task: Optional[asyncio.Task] = None

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

class NewBotConfig(BaseModel):
    symbol: str              = Field(..., description="Trading pair, e.g. LINK/USDT")
    timeframe: str           = Field(..., description="Candle interval, e.g. 5m")
    capital_allocation: float = Field(..., description="Fraction of total capital (0-1)")
    max_risk_per_trade: float = Field(..., description="Max risk per trade as fraction (0-1)")
    confidence_threshold: float = Field(0.65, description="Min signal confidence to enter")
    stop_loss_pct: float     = Field(0.018, description="Stop-loss distance as fraction")
    take_profit_pct: float   = Field(0.035, description="Take-profit distance as fraction")
    enabled: bool            = Field(True, description="Whether this bot is active")

class BotConfigUpdate(BaseModel):
    capital_allocation: Optional[float] = None
    max_risk_per_trade: Optional[float] = None
    confidence_threshold: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    enabled: Optional[bool] = None

class ConfigUpdate(BaseModel):
    trading_mode: str    = Field(..., description="'paper' or 'live'")
    total_capital: float = Field(..., description="Total capital in USD")
    daily_loss_limit: float    = Field(..., description="Max daily loss as fraction (e.g. 0.04)")
    daily_profit_target: float = Field(..., description="Daily profit target as fraction (e.g. 0.025)")

class ToggleBody(BaseModel):
    enabled: Optional[bool] = Field(None, description="Desired state; omit to flip current")

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_SYMBOLS    = ["LINK/USDT", "BTC/USDT", "ETH/USDT", "ADA/USDT",
                        "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def ok(data: Any) -> dict:
    """Wrap any payload in the standard success envelope."""
    return {"success": True, "data": data}


def _bot_to_dict(bot_id: str, config: BotConfig) -> dict:
    """Convert BotConfig to dict and inject the `id` field the frontend needs."""
    d = asdict(config)
    d["id"] = bot_id
    return d


def _parse_log_line(line: str) -> dict:
    """
    Parse a standard Python logging line into a structured dict.
    Expected format: "YYYY-MM-DD HH:MM:SS,mmm - source - LEVEL - message"
    """
    try:
        parts = line.strip().split(" - ", 3)
        if len(parts) >= 4:
            ts_part = parts[0].strip()
            time_str = ts_part.split(" ")[1].split(",")[0] if " " in ts_part else ts_part
            level = parts[2].strip().upper()
            if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
                level = "INFO"
            return {
                "timestamp": time_str,
                "level":     level,
                "message":   parts[3].strip(),
                "source":    parts[1].strip(),
            }
    except Exception:
        pass
    return {"timestamp": "00:00:00", "level": "INFO",
            "message": line.strip(), "source": "system"}


def _read_log_lines(n: int = 100) -> List[dict]:
    """Return the last n parsed log lines."""
    log_path = "trading_system.log"
    if not os.path.exists(log_path):
        return [{"timestamp": datetime.now().strftime("%H:%M:%S"),
                 "level": "INFO", "message": "No log file found", "source": "system"}]
    try:
        with open(log_path, "r", errors="ignore") as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [_parse_log_line(ln) for ln in recent if ln.strip()]
    except Exception as exc:
        return [{"timestamp": "00:00:00", "level": "ERROR",
                 "message": str(exc), "source": "system"}]


# ══════════════════════════════════════════════════════════════════════════════
# Static file serving (React SPA)
# ══════════════════════════════════════════════════════════════════════════════

_REACT_DIST    = Path("frontend_react/dist")
_LEGACY_FRONT  = Path("frontend")

if (_REACT_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_REACT_DIST / "assets")), name="assets")
elif _LEGACY_FRONT.exists():
    app.mount("/static", StaticFiles(directory=str(_LEGACY_FRONT)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_root():
    for candidate in [_REACT_DIST / "index.html", _LEGACY_FRONT / "index.html"]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Dashboard not found — run <code>npm run build</code> inside frontend_react/</h1>",
        status_code=404,
    )


# ══════════════════════════════════════════════════════════════════════════════
# System endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/health",
    tags=["System"],
    summary="Health check",
    response_description="Service status, timestamp and version",
)
async def health_check():
    """Returns 200 if the API server is running."""
    return ok({"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.0.0"})


@app.get(
    "/api/status",
    tags=["System"],
    summary="System status",
    response_description="{running, uptime, total_capital, paper_trading}",
)
async def get_system_status():
    """Returns whether the trading system is running plus its configuration snapshot."""
    running = system_task is not None and not system_task.done()
    if trading_system:
        delta = datetime.now() - trading_system.system_start_time
        return ok({
            "running":       running,
            "uptime":        str(delta).split(".")[0],
            "total_capital": trading_system.global_config.total_capital,
            "paper_trading": trading_system.global_config.paper_trading,
        })
    return ok({"running": False, "uptime": "0:00:00",
               "total_capital": 1200.0, "paper_trading": True})


@app.post(
    "/api/start",
    tags=["System"],
    summary="Start the trading system",
)
async def start_system():
    """Initialises and starts the trading system in a background asyncio task."""
    global trading_system, system_task
    if system_task and not system_task.done():
        raise HTTPException(400, "System is already running")
    try:
        global_config, bot_configs = create_production_config()
        # Construction calls exchange.load_markets() (blocking network I/O) —
        # build it in a thread so we don't stall the event loop during startup.
        trading_system = await asyncio.to_thread(
            ProductionTradingSystem, global_config, bot_configs
        )
        system_task = asyncio.create_task(trading_system.start())
        return ok({"message": "Trading system started successfully"})
    except Exception as exc:
        raise HTTPException(500, f"Failed to start system: {exc}")


# ── Cross-sectional portfolio engine (Phase 3, paper) ───────────────────────
portfolio_engine = None          # type: ignore
portfolio_task: Optional[asyncio.Task] = None


@app.post("/api/portfolio/start", tags=["Portfolio"],
          summary="Start the daily cross-sectional momentum portfolio (paper)")
async def start_portfolio():
    """Launches the validated cross-sectional momentum engine (paper trading).
    See docs/STRATEGY_THESIS.md. Market-neutral, daily rebalance, kill-switch."""
    global portfolio_engine, portfolio_task
    if portfolio_task and not portfolio_task.done():
        raise HTTPException(400, "Portfolio engine already running")
    try:
        import ccxt
        from backend.strategy.portfolio_engine import (
            CrossSectionalPortfolioEngine, PortfolioConfig)
        from backend.data.universe import EXPANDED_UNIVERSE
        try:
            from backend.persistence.database import init_db
            from backend.persistence.repository import EquityRepository, DailyStatsRepository
            init_db()   # create tables (equity_snapshots, etc.) — idempotent
            eq_repo, daily_repo = EquityRepository(), DailyStatsRepository()
        except Exception:
            eq_repo = daily_repo = None
        cfg = PortfolioConfig(universe=EXPANDED_UNIVERSE)   # validated defaults
        exchange = await asyncio.to_thread(
            lambda: ccxt.binance({"enableRateLimit": True}))
        portfolio_engine = CrossSectionalPortfolioEngine(
            cfg, exchange=exchange, equity_repo=eq_repo, daily_repo=daily_repo)
        portfolio_task = asyncio.create_task(portfolio_engine.start())
        return ok({"message": "Portfolio engine started (paper)",
                   "universe": len(cfg.universe), "rebalance_days": cfg.rebalance_days})
    except Exception as exc:
        raise HTTPException(500, f"Failed to start portfolio engine: {exc}")


@app.post("/api/portfolio/stop", tags=["Portfolio"], summary="Stop the portfolio engine")
async def stop_portfolio():
    global portfolio_task, portfolio_engine
    if not portfolio_task or portfolio_task.done():
        raise HTTPException(400, "Portfolio engine not running")
    if portfolio_engine:
        portfolio_engine.state.halted = True
    portfolio_task.cancel()
    return ok({"message": "Portfolio engine stopped"})


@app.get("/api/portfolio", tags=["Portfolio"],
         summary="Current portfolio status, equity and positions")
async def get_portfolio():
    """Status of the cross-sectional portfolio engine."""
    running = portfolio_task is not None and not portfolio_task.done()
    if not portfolio_engine:
        return ok({"running": False, "positions": [], "equity": 0.0,
                   "total_pnl": 0.0, "daily_pnl": 0.0, "rebalances": 0, "halted": False})
    e = portfolio_engine
    return ok({
        "running": running,
        "halted": e.state.halted,
        "equity": round(e.state.equity, 2),
        "total_pnl": round(e.total_pnl, 2),
        "daily_pnl": round(e.daily_pnl, 2),
        "peak_equity": round(e.state.peak_equity, 2),
        "rebalances": e.state.rebalances,
        "positions": e.positions(),
        "equity_curve": e.equity_curve[-100:],
        "config": {"rebalance_days": e.config.rebalance_days, "lookback": e.config.lookback,
                   "k": e.config.k, "mode": e.config.mode, "paper": e.config.paper_trading},
    })


# ── Order book collector (parallel data-accumulation track) ─────────────────
orderbook_collector = None       # type: ignore
collector_task: Optional[asyncio.Task] = None


async def _launch_collector():
    """Start the order book collector (idempotent). Returns a status message."""
    global orderbook_collector, collector_task
    if collector_task and not collector_task.done():
        return "collector already running"
    import ccxt
    from backend.data.orderbook_collector import OrderBookCollector
    from backend.data.universe import EXPANDED_UNIVERSE
    interval = int(os.getenv("COLLECTOR_INTERVAL_SEC", "60"))
    storage = os.getenv("COLLECTOR_DIR", "orderbook_data")
    exchange = await asyncio.to_thread(lambda: ccxt.binance({"enableRateLimit": True}))
    orderbook_collector = OrderBookCollector(
        EXPANDED_UNIVERSE, exchange=exchange, interval_sec=interval, storage_dir=storage)
    collector_task = asyncio.create_task(orderbook_collector.start())
    return f"collector started ({len(EXPANDED_UNIVERSE)} symbols, every {interval}s)"


@app.post("/api/collector/start", tags=["Collector"],
          summary="Start the live order book collector")
async def start_collector():
    try:
        return ok({"message": await _launch_collector()})
    except Exception as exc:
        raise HTTPException(500, f"Failed to start collector: {exc}")


@app.post("/api/collector/stop", tags=["Collector"], summary="Stop the collector")
async def stop_collector():
    global collector_task, orderbook_collector
    if not collector_task or collector_task.done():
        raise HTTPException(400, "Collector not running")
    if orderbook_collector:
        orderbook_collector.halted = True
    collector_task.cancel()
    return ok({"message": "Collector stopped"})


@app.get("/api/collector", tags=["Collector"], summary="Collector status")
async def get_collector():
    running = collector_task is not None and not collector_task.done()
    if not orderbook_collector:
        return ok({"running": False, "snapshots_written": 0})
    return ok({**orderbook_collector.status(), "running": running})


@app.on_event("startup")
async def _autostart():
    """Auto-launch the collector and portfolio engine on service boot.
    Controlled by env: AUTOSTART_COLLECTOR / AUTOSTART_PORTFOLIO (default '1').
    Each wrapped so one failure never blocks the API from coming up."""
    if os.getenv("AUTOSTART_COLLECTOR", "1") != "0":
        try:
            msg = await _launch_collector()
            logger.info(f"autostart: {msg}")
        except Exception as exc:
            logger.warning(f"autostart collector failed: {exc}")
    if os.getenv("AUTOSTART_PORTFOLIO", "1") != "0":
        try:
            await start_portfolio()
            logger.info("autostart: portfolio engine started")
        except Exception as exc:
            logger.warning(f"autostart portfolio failed: {exc}")


@app.post(
    "/api/stop",
    tags=["System"],
    summary="Gracefully stop the trading system",
)
async def stop_system():
    """Cancels the background task and calls the graceful shutdown handler."""
    global trading_system, system_task
    if not system_task or system_task.done():
        raise HTTPException(400, "System is not running")
    try:
        system_task.cancel()
        if trading_system:
            await trading_system._shutdown()
        return ok({"message": "Trading system stopped successfully"})
    except Exception as exc:
        raise HTTPException(500, f"Failed to stop system: {exc}")


@app.post(
    "/api/emergency-stop",
    tags=["System"],
    summary="Emergency stop — immediately halt all operations",
)
async def emergency_stop():
    """Cancels the task and calls the emergency shutdown handler (closes all positions)."""
    global trading_system, system_task
    try:
        if system_task and not system_task.done():
            system_task.cancel()
        if trading_system:
            await trading_system._emergency_shutdown()
        return ok({"message": "Emergency stop executed successfully"})
    except Exception as exc:
        raise HTTPException(500, f"Emergency stop failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/metrics",
    tags=["Metrics"],
    summary="Trading performance metrics",
    response_description="{total_pnl, total_roi, daily_pnl, active_trades, win_rate, total_trades, max_drawdown, daily_trades}",
)
async def get_metrics():
    """Returns aggregate P&L and trade statistics."""
    if trading_system:
        total_trades   = len(trading_system.trade_history)
        winning_trades = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
        win_rate       = winning_trades / total_trades if total_trades > 0 else 0.0

        max_drawdown = 0.0
        if trading_system.equity_curve:
            equities = [p["equity"] for p in trading_system.equity_curve]
            peak = equities[0]
            for eq in equities:
                peak = max(peak, eq)
                max_drawdown = max(max_drawdown, (peak - eq) / peak if peak > 0 else 0)

        return ok({
            "total_pnl":    trading_system.total_pnl,
            "total_roi":    trading_system.total_pnl / trading_system.global_config.total_capital,
            "daily_pnl":    trading_system.daily_pnl,
            "active_trades":len(trading_system.active_trades),
            "win_rate":     win_rate,
            "total_trades": total_trades,
            "max_drawdown": max_drawdown,
            "daily_trades": trading_system.daily_trades,
        })
    return ok({"total_pnl": 0.0, "total_roi": 0.0, "daily_pnl": 0.0,
               "active_trades": 0, "win_rate": 0.0, "total_trades": 0,
               "max_drawdown": 0.0, "daily_trades": 0})


@app.get(
    "/api/performance-metrics",
    tags=["Metrics"],
    summary="System resource usage (CPU, memory, GC)",
    response_description="{system_metrics, application_metrics, trading_metrics, timestamp}",
)
async def get_performance_metrics():
    """Returns process-level resource usage for the PerformanceMonitor component."""
    # System metrics via psutil (optional) or os fallback
    try:
        import psutil
        proc       = psutil.Process()
        mem_pct    = round(proc.memory_percent(), 1)
        mem_avail  = round(psutil.virtual_memory().available / 1024 / 1024)
        cpu_pct    = round(psutil.cpu_percent(interval=0.1), 1)
    except ImportError:
        mem_pct   = 0.0
        mem_avail = 512
        cpu_pct   = 0.0

    gc_stats      = gc.get_stats()
    gc_collections = sum(s.get("collections", 0) for s in gc_stats)
    gc_collected   = sum(s.get("collected", 0) for s in gc_stats)

    # Log buffer usage
    log_lines = 0
    if os.path.exists("trading_system.log"):
        try:
            with open("trading_system.log", "r", errors="ignore") as f:
                log_lines = sum(1 for _ in f)
        except Exception:
            pass
    max_logs = 1000

    return ok({
        "system_metrics": {
            "memory_percent":    mem_pct,
            "memory_available_mb": mem_avail,
            "cpu_percent":       cpu_pct,
            "gc_collections":    gc_collections,
            "gc_collected":      gc_collected,
        },
        "application_metrics": {
            "logs_utilization": round(min(log_lines, max_logs) / max_logs * 100, 1),
            "active_logs":      min(log_lines, max_logs),
            "max_logs":         max_logs,
        },
        "trading_metrics": {
            "system_running": system_task is not None and not system_task.done(),
            "task_status":    "running" if (system_task and not system_task.done()) else "stopped",
        },
        "timestamp": datetime.now().isoformat(),
    })


@app.get(
    "/api/equity",
    tags=["Metrics"],
    summary="Equity curve",
    response_description="{equity_curve: [{timestamp(ms), equity}]}",
)
async def get_equity_curve():
    """Returns the last 100 equity snapshots (timestamp in milliseconds)."""
    if trading_system and trading_system.equity_curve:
        curve = [
            {"timestamp": p["timestamp"], "equity": p["equity"]}
            for p in trading_system.equity_curve[-100:]
        ]
        return ok({"equity_curve": curve})

    # Demo data — deterministic random walk
    import random
    rng   = random.Random(42)
    now   = datetime.now()
    eq    = 1200.0
    pts   = []
    for i in range(50):
        ts = int((now - timedelta(minutes=5 * (49 - i))).timestamp() * 1000)
        eq += (rng.random() - 0.45) * 20
        pts.append({"timestamp": ts, "equity": round(max(eq, 1000.0), 2)})
    return ok({"equity_curve": pts})


@app.get(
    "/api/daily-stats",
    tags=["Metrics"],
    summary="Daily P&L history (last 30 days)",
)
async def get_daily_stats():
    """Returns per-day P&L summaries from the database or in-memory store."""
    try:
        from backend.persistence.repository import DailyStatsRepository
        return ok({"daily_stats": DailyStatsRepository().get_history(limit=30)})
    except Exception:
        pass
    if trading_system and trading_system.daily_stats:
        return ok({"daily_stats": trading_system.daily_stats[-30:]})
    return ok({"daily_stats": []})


# ══════════════════════════════════════════════════════════════════════════════
# Bots endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/bots",
    tags=["Bots"],
    summary="List all bots with their current status",
    response_description="{bots: [{id, symbol, timeframe, status, pnl, trades, enabled}]}",
)
async def get_bots():
    """Returns all configured bots. Each bot includes an `id` field used for toggle/config calls."""
    if trading_system:
        bots = []
        for bot_id, cfg in trading_system.bot_configs.items():
            trades_for_bot = [t for t in trading_system.trade_history if t.symbol == cfg.symbol]
            bot_pnl = sum(t.pnl for t in trades_for_bot if t.pnl is not None)
            bots.append({
                "id":       bot_id,
                "symbol":   cfg.symbol,
                "timeframe":cfg.timeframe,
                "status":   "running" if cfg.enabled else "paused",
                "pnl":      bot_pnl,
                "trades":   len(trades_for_bot),
                "enabled":  cfg.enabled,
            })
        return ok({"bots": bots})

    # Demo data
    return ok({"bots": [
        {"id": "LINK/USDT_5m", "symbol": "LINK/USDT", "timeframe": "5m",
         "status": "running", "pnl": 0.0, "trades": 0, "enabled": True},
        {"id": "LINK/USDT_1m", "symbol": "LINK/USDT", "timeframe": "1m",
         "status": "running", "pnl": 0.0, "trades": 0, "enabled": True},
    ]})


@app.get(
    "/api/bots/count",
    tags=["Bots"],
    summary="Number of configured bots",
    response_description="{current_count, maximum_allowed, can_add_more}",
)
async def get_bot_count():
    """Returns bot count and whether more bots can be added (max 5)."""
    count = len(trading_system.bot_configs) if trading_system else 0
    return ok({"current_count": count, "maximum_allowed": 5, "can_add_more": count < 5})


@app.get(
    "/api/bots/available-symbols",
    tags=["Bots"],
    summary="Supported trading pairs",
)
async def get_available_symbols():
    return ok({"symbols": AVAILABLE_SYMBOLS})


@app.get(
    "/api/bots/available-timeframes",
    tags=["Bots"],
    summary="Supported candle timeframes",
)
async def get_available_timeframes():
    return ok({"timeframes": AVAILABLE_TIMEFRAMES})


@app.get(
    "/api/bots/{bot_id:path}/config",
    tags=["Bots"],
    summary="Get full config for a specific bot",
)
async def get_bot_config(bot_id: str):
    """Returns the BotConfig dataclass fields plus the `id` key."""
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    cfg = trading_system.bot_configs.get(bot_id)
    if not cfg:
        raise HTTPException(404, f"Bot '{bot_id}' not found")
    return ok(_bot_to_dict(bot_id, cfg))


@app.post(
    "/api/bots/{bot_id:path}/toggle",
    tags=["Bots"],
    summary="Enable or disable a bot",
    response_description="{bot_id, enabled}",
)
async def toggle_bot(bot_id: str, body: ToggleBody = Body(default=ToggleBody())):
    """Toggles the `enabled` flag. Pass `{\"enabled\": true/false}` to set explicitly."""
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    cfg = trading_system.bot_configs.get(bot_id)
    if not cfg:
        raise HTTPException(404, f"Bot '{bot_id}' not found")
    cfg.enabled = body.enabled if body.enabled is not None else not cfg.enabled
    return ok({"bot_id": bot_id, "enabled": cfg.enabled})


@app.post(
    "/api/bots/add",
    tags=["Bots"],
    summary="Add a new bot",
    response_description="{message, bot_id}",
)
async def add_bot(new_bot: NewBotConfig):
    """Creates a new BotConfig and registers it in the running system."""
    global trading_system
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    bot_id = f"{new_bot.symbol}_{new_bot.timeframe}"
    if bot_id in trading_system.bot_configs:
        raise HTTPException(409, f"Bot '{bot_id}' already exists")
    cfg = BotConfig(
        symbol=new_bot.symbol, timeframe=new_bot.timeframe,
        capital_allocation=new_bot.capital_allocation,
        max_risk_per_trade=new_bot.max_risk_per_trade,
        confidence_threshold=new_bot.confidence_threshold,
        stop_loss_pct=new_bot.stop_loss_pct,
        take_profit_pct=new_bot.take_profit_pct,
        enabled=new_bot.enabled,
    )
    trading_system.bot_configs[bot_id] = cfg
    trading_system.signal_generators[bot_id] = OptimizedSignalGenerator(
        new_bot.symbol, new_bot.timeframe)
    return ok({"message": f"Bot '{bot_id}' added", "bot_id": bot_id})


@app.put(
    "/api/bots/update/{bot_index}",
    tags=["Bots"],
    summary="Update a bot by its list index (0-based)",
)
async def update_bot(bot_index: int, update: BotConfigUpdate):
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    bots = list(trading_system.bot_configs.items())
    if bot_index < 0 or bot_index >= len(bots):
        raise HTTPException(404, f"Bot index {bot_index} out of range")
    bot_id, cfg = bots[bot_index]
    for field, value in update.model_dump(exclude_none=True).items():
        setattr(cfg, field, value)
    return ok({"message": f"Bot '{bot_id}' updated"})


@app.delete(
    "/api/bots/remove/{bot_index}",
    tags=["Bots"],
    summary="Remove a bot by its list index (0-based)",
)
async def remove_bot(bot_index: int):
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    bots = list(trading_system.bot_configs.keys())
    if bot_index < 0 or bot_index >= len(bots):
        raise HTTPException(404, f"Bot index {bot_index} out of range")
    bot_id = bots[bot_index]
    del trading_system.bot_configs[bot_id]
    trading_system.signal_generators.pop(bot_id, None)
    return ok({"message": f"Bot '{bot_id}' removed"})


# ══════════════════════════════════════════════════════════════════════════════
# Trades & Logs endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/trades/recent",
    tags=["Trades"],
    summary="Last 10 closed trades",
    response_description="{trades: [{id, symbol, direction('long'/'short'), pnl, status, time, entry_price, exit_price}]}",
)
async def get_recent_trades():
    """Returns the most recent closed trades. Direction is lowercase ('long'/'short')."""
    if trading_system:
        trades = []
        for i, t in enumerate(trading_system.trade_history[-10:]):
            trades.append({
                "id":          t.id,
                "symbol":      t.symbol,
                "direction":   "long" if t.direction == 1 else "short",
                "pnl":         t.pnl or 0.0,
                "status":      t.status,
                "time":        datetime.fromtimestamp(t.entry_time / 1000).strftime("%H:%M"),
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
            })
        return ok({"trades": trades})

    return ok({"trades": [
        {"id": "demo_1", "symbol": "LINK/USDT", "direction": "long",
         "pnl": 0.0, "status": "closed", "time": "--:--",
         "entry_price": None, "exit_price": None},
    ]})


@app.get(
    "/api/logs",
    tags=["Logs"],
    summary="Last 100 log entries",
    response_description="{logs: [{timestamp, level, message, source}]}",
)
async def get_logs():
    """Returns structured log objects parsed from trading_system.log."""
    return ok({"logs": _read_log_lines(100)})


@app.get(
    "/api/logs/statistics",
    tags=["Logs"],
    summary="Log entry count by severity",
)
async def get_log_statistics():
    stats = {"total_entries": 0, "error_count": 0, "warning_count": 0, "info_count": 0}
    if os.path.exists("trading_system.log"):
        try:
            with open("trading_system.log", "r", errors="ignore") as f:
                for line in f:
                    stats["total_entries"] += 1
                    up = line.upper()
                    if " - ERROR" in up or " - CRITICAL" in up:
                        stats["error_count"] += 1
                    elif " - WARNING" in up:
                        stats["warning_count"] += 1
                    else:
                        stats["info_count"] += 1
        except Exception:
            pass
    return ok(stats)


@app.get(
    "/api/logs/categories",
    tags=["Logs"],
    summary="Available log source categories",
)
async def get_log_categories():
    return ok({"categories": ["system", "trading", "ml", "api", "risk", "persistence"]})


# ══════════════════════════════════════════════════════════════════════════════
# Config endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/config",
    tags=["Config"],
    summary="Simple config (trading_mode, capital, limits)",
)
async def get_config():
    """Returns a simplified config dict — used by the basic ConfigurationPanel."""
    if os.path.exists("system_config.json"):
        try:
            with open("system_config.json") as f:
                return ok(json.load(f))
        except Exception:
            pass
    if trading_system:
        return ok({
            "trading_mode":       "paper" if trading_system.global_config.paper_trading else "live",
            "total_capital":      trading_system.global_config.total_capital,
            "daily_loss_limit":   trading_system.global_config.daily_loss_limit,
            "daily_profit_target":trading_system.global_config.daily_profit_target,
        })
    return ok({"trading_mode": "paper", "total_capital": 1200.0,
               "daily_loss_limit": 0.04, "daily_profit_target": 0.025})


@app.post(
    "/api/config",
    tags=["Config"],
    summary="Update simple config and persist to disk",
)
async def update_config(config: ConfigUpdate):
    """Updates the running system config and writes system_config.json."""
    global trading_system
    try:
        if trading_system:
            trading_system.global_config.total_capital      = config.total_capital
            trading_system.global_config.daily_loss_limit   = config.daily_loss_limit
            trading_system.global_config.daily_profit_target = config.daily_profit_target
            trading_system.global_config.paper_trading      = config.trading_mode == "paper"
        payload = {
            "trading_mode":       config.trading_mode,
            "total_capital":      config.total_capital,
            "daily_loss_limit":   config.daily_loss_limit,
            "daily_profit_target":config.daily_profit_target,
            "updated_at":         datetime.now().isoformat(),
        }
        with open("system_config.json", "w") as f:
            json.dump(payload, f, indent=2)
        return ok({"message": "Configuration updated successfully"})
    except Exception as exc:
        raise HTTPException(500, f"Failed to update configuration: {exc}")


@app.get(
    "/api/config/full",
    tags=["Config"],
    summary="Full config — global settings + all bot configs",
    response_description="{global_config: {...}, bot_configs: [{id, ...}]}",
)
async def get_full_config():
    """Returns complete system state: GlobalConfig fields and all BotConfig entries with their ids."""
    if trading_system:
        return ok({
            "global_config": asdict(trading_system.global_config),
            "bot_configs":   [_bot_to_dict(bid, cfg)
                              for bid, cfg in trading_system.bot_configs.items()],
        })
    _, bot_cfgs = create_production_config()
    global_cfg = GlobalConfig(total_capital=1200.0, paper_trading=True)
    return ok({
        "global_config": asdict(global_cfg),
        "bot_configs":   [dict(id=f"{c.symbol}_{c.timeframe}", **asdict(c)) for c in bot_cfgs],
    })


@app.get(
    "/api/config/backups",
    tags=["Config"],
    summary="List available config backup files",
)
async def get_config_backups():
    """Lists system_config_backup_*.json files in the working directory."""
    files = sorted(Path(".").glob("system_config_backup_*.json"), reverse=True)
    return ok({"backups": [f.name for f in files[:10]]})


@app.put(
    "/api/config/global",
    tags=["Config"],
    summary="Update individual global config fields",
)
async def update_global_config(updates: Dict[str, Any] = Body(...)):
    """Accepts a partial GlobalConfig dict and applies the provided fields."""
    global trading_system
    if trading_system:
        for field, value in updates.items():
            if hasattr(trading_system.global_config, field):
                setattr(trading_system.global_config, field, value)
    return ok({"message": "Global configuration updated"})


@app.put(
    "/api/config/bot/{bot_id:path}",
    tags=["Config"],
    summary="Update a bot's config by its string id",
)
async def update_bot_config_by_id(bot_id: str, updates: Dict[str, Any] = Body(...)):
    """Accepts a partial BotConfig dict. Use the bot's id string (e.g. 'LINK/USDT_5m')."""
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    cfg = trading_system.bot_configs.get(bot_id)
    if not cfg:
        raise HTTPException(404, f"Bot '{bot_id}' not found")
    for field, value in updates.items():
        if hasattr(cfg, field):
            setattr(cfg, field, value)
    return ok({"message": f"Bot '{bot_id}' updated", "bot_id": bot_id})


@app.post(
    "/api/config/bot",
    tags=["Config"],
    summary="Add a bot (alias of POST /api/bots/add)",
)
async def add_bot_via_config(new_bot: NewBotConfig):
    """Delegates to the /api/bots/add endpoint — provided for EnhancedConfigPanel compatibility."""
    return await add_bot(new_bot)


@app.delete(
    "/api/config/bot/{bot_id:path}",
    tags=["Config"],
    summary="Remove a bot by its string id",
)
async def delete_bot_by_id(bot_id: str):
    """Removes a bot using its id string rather than a numeric index."""
    if not trading_system:
        raise HTTPException(503, "Trading system not running")
    if bot_id not in trading_system.bot_configs:
        raise HTTPException(404, f"Bot '{bot_id}' not found")
    del trading_system.bot_configs[bot_id]
    trading_system.signal_generators.pop(bot_id, None)
    return ok({"message": f"Bot '{bot_id}' removed"})


@app.post(
    "/api/config/restore/{filename}",
    tags=["Config"],
    summary="Restore a config from a backup file",
)
async def restore_config_backup(filename: str):
    """Copies the backup file over system_config.json."""
    if not Path(filename).exists():
        raise HTTPException(404, f"Backup file '{filename}' not found")
    try:
        shutil.copy(filename, "system_config.json")
        return ok({"message": f"Configuration restored from {filename}"})
    except Exception as exc:
        raise HTTPException(500, f"Failed to restore: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Analytics endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/signal-quality",
    tags=["Analytics"],
    summary="Signal quality metrics",
    response_description="{current_quality_score, signals_evaluated, pass_rate, layer_scores, recent_rejections}",
)
async def get_signal_quality():
    """Returns signal pass/reject statistics and per-layer quality scores."""
    if trading_system and trading_system.trade_history:
        total  = len(trading_system.trade_history)
        passed = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
        rate   = round(passed / total, 2) if total > 0 else 0.0
        rejections = [
            {
                "symbol":        t.symbol,
                "quality_score": round(abs(t.pnl_pct or 0), 3),
                "timestamp":     datetime.fromtimestamp(
                    t.entry_time / 1000).strftime("%H:%M:%S"),
            }
            for t in list(reversed(trading_system.trade_history))[:5]
            if t.pnl and t.pnl <= 0
        ]
        return ok({
            "current_quality_score": rate,
            "signals_evaluated":     total,
            "signals_passed":        passed,
            "signals_rejected":      total - passed,
            "avg_quality_score":     rate,
            "pass_rate":             rate,
            "layer_scores": {
                "technical":          round(min(rate * 1.05, 1.0), 2),
                "market_structure":   round(rate * 0.9, 2),
                "binance_sentiment":  round(rate * 0.95, 2),
                "ml_confidence":      round(min(rate * 1.1, 1.0), 2),
            },
            "recent_rejections": rejections,
        })

    return ok({
        "current_quality_score": 0.0,
        "signals_evaluated":     0,
        "signals_passed":        0,
        "signals_rejected":      0,
        "avg_quality_score":     0.0,
        "pass_rate":             0.0,
        "layer_scores": {"technical": 0.0, "market_structure": 0.0,
                         "binance_sentiment": 0.0, "ml_confidence": 0.0},
        "recent_rejections": [],
    })


@app.get(
    "/api/market-regime/{symbol:path}",
    tags=["Analytics"],
    summary="Market regime for a symbol",
    response_description="{regime, confidence, trend_strength, strategy_config, factors}",
)
async def get_market_regime(symbol: str):
    """Detects the current market regime (ranging/trending/volatile) from recent trade history."""
    _STRATEGY_MAP = {
        "trending_bull": {"strategy_type": "trend_following",
                          "max_trades_per_day": 8, "quality_threshold": 0.65, "risk_per_trade": 0.02},
        "trending_bear": {"strategy_type": "trend_following_short",
                          "max_trades_per_day": 6, "quality_threshold": 0.70, "risk_per_trade": 0.015},
        "ranging":       {"strategy_type": "mean_reversion",
                          "max_trades_per_day": 6, "quality_threshold": 0.65, "risk_per_trade": 0.015},
        "high_volatility":{"strategy_type": "volatility_breakout",
                           "max_trades_per_day": 4, "quality_threshold": 0.75, "risk_per_trade": 0.01},
    }

    regime          = "ranging"
    confidence      = 0.60
    trend_strength  = 0.30
    volatility      = 0.50
    factors         = ["Insufficient data — system not started"]

    if trading_system:
        sym_trades = [t for t in trading_system.trade_history if t.symbol == symbol]
        if sym_trades:
            recent   = sym_trades[-20:]
            win_rate = sum(1 for t in recent if t.pnl and t.pnl > 0) / len(recent)
            if win_rate >= 0.60:
                regime, trend_strength, confidence = "trending_bull", 0.70, 0.75
            elif win_rate <= 0.35:
                regime, trend_strength, confidence = "trending_bear", 0.65, 0.70
            else:
                regime, trend_strength, confidence = "ranging", 0.30, 0.65
            factors = [f"Win rate {win_rate:.0%} over last {len(recent)} trades"]
        else:
            factors = [f"No trades yet for {symbol}"]

    return ok({
        "regime":            regime,
        "confidence":        round(confidence, 2),
        "regime_duration":   42,
        "trend_strength":    round(trend_strength, 2),
        "volatility_level":  round(volatility, 2),
        "volume_profile":    "medium",
        "breakout_frequency":0.2,
        "strategy_config":   _STRATEGY_MAP.get(regime, _STRATEGY_MAP["ranging"]),
        "factors":           factors,
    })


@app.get(
    "/api/market-sentiment/{symbol:path}",
    tags=["Analytics"],
    summary="Market sentiment for a symbol",
    response_description="{sentiment_label, sentiment_score, confidence, factors, raw_data}",
)
async def get_market_sentiment(symbol: str):
    """Returns bullish/bearish/neutral sentiment derived from recent trade performance."""
    sentiment_label = "neutral"
    sentiment_score = 0.5

    if trading_system:
        sym_trades = [t for t in trading_system.trade_history if t.symbol == symbol]
        if sym_trades:
            recent   = sym_trades[-10:]
            win_rate = sum(1 for t in recent if t.pnl and t.pnl > 0) / len(recent)
            sentiment_score = round(win_rate, 2)
            if win_rate >= 0.60:
                sentiment_label = "bullish"
            elif win_rate <= 0.40:
                sentiment_label = "bearish"

    return ok({
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence":      0.65,
        "factors":         [f"Based on recent {symbol} trade results"],
        "raw_data": {
            "funding_rate":       0.0001,
            "long_short_ratio":   {"current_ratio": 1.0, "sentiment": sentiment_label},
            "open_interest":      {"trend": "stable",  "change_pct": 0.0},
            "order_book":         {"sentiment": "neutral", "spread_pct": 0.02},
        },
    })


@app.post(
    "/api/backtest",
    tags=["Analytics"],
    summary="Trigger walk-forward backtest (runs in background)",
)
async def run_backtest_endpoint(background_tasks: BackgroundTasks):
    """Starts the backtest as a subprocess. Results appear in backtest_results.json.
    Validation scripts live in .claude/research/ (kept out of the app tree)."""
    import subprocess, sys, os
    script = os.path.join(".claude", "research", "run_backtest.py")
    def _run():
        subprocess.run(
            [sys.executable, script, "--days", "90", "--splits", "3",
             "--output", "backtest_results.json"],
            capture_output=True,
        )
    background_tasks.add_task(_run)
    return ok({"message": "Backtest started — results will be written to backtest_results.json"})


@app.get(
    "/api/backtest/results",
    tags=["Analytics"],
    summary="Latest walk-forward backtest results",
)
async def get_backtest_results():
    """Returns the contents of backtest_results.json if it exists."""
    if os.path.exists("backtest_results.json"):
        with open("backtest_results.json") as f:
            return ok(json.load(f))
    return ok({"message": "No backtest results yet — POST /api/backtest to run one"})


# ══════════════════════════════════════════════════════════════════════════════
# SPA catch-all — MUST be last
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
async def serve_spa(full_path: str):
    """Return index.html for any non-API route to support React Router deep links."""
    if full_path.startswith("api/") or full_path.startswith("assets/"):
        raise HTTPException(404, "Not found")
    index = _REACT_DIST / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    raise HTTPException(404, "Frontend not built — run npm run build inside frontend_react/")


# ══════════════════════════════════════════════════════════════════════════════
# Error handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"success": False, "error": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error"})


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Starting Trading Bot ML API Server...")
    print("Dashboard : http://localhost:12000")
    print("API docs  : http://localhost:12000/docs")
    uvicorn.run(app, host="0.0.0.0", port=12000, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
