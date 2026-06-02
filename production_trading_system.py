#!/usr/bin/env python3
"""
Production Trading System
Ready-to-use implementation of the optimized profitable trading strategies.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import asyncio
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass, asdict
from enum import Enum
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import logging
warnings.filterwarnings('ignore')

# Persistence layer — optional; system runs without it (in-memory only)
try:
    from backend.persistence.database import init_db
    from backend.persistence.repository import (
        TradeRepository, EquityRepository, DailyStatsRepository
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for individual trading bot."""
    symbol: str
    timeframe: str
    capital_allocation: float  # Percentage of total capital
    max_risk_per_trade: float  # Percentage risk per trade
    confidence_threshold: float  # Minimum confidence for trades
    stop_loss_pct: float
    take_profit_pct: float
    enabled: bool = True


@dataclass
class GlobalConfig:
    """Global system configuration."""
    total_capital: float = 10000.0
    max_concurrent_trades: int = 4
    daily_loss_limit: float = 0.05  # 5%
    daily_profit_target: float = 0.03  # 3%
    emergency_stop_drawdown: float = 0.08  # 8%
    paper_trading: bool = True  # Start in paper trading mode


@dataclass
class TradeSignal:
    """Trading signal with metadata."""
    symbol: str
    direction: int  # 1 for long, -1 for short
    strength: float
    confidence: float
    timestamp: int
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any]


@dataclass
class Trade:
    """Individual trade record."""
    id: str
    symbol: str
    direction: int
    entry_time: int
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"  # open, closed, cancelled
    reason: Optional[str] = None
    bot_id: str = ""


class ProductionTradingSystem:
    """Main production trading system."""
    
    def __init__(self, global_config: GlobalConfig, bot_configs: List[BotConfig]):
        self.global_config = global_config
        self.bot_configs = {f"{config.symbol}_{config.timeframe}": config for config in bot_configs}
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'sandbox': global_config.paper_trading,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # System state
        self.active_trades: Dict[str, Trade] = {}
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.daily_trades = 0
        self.system_start_time = datetime.now()
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.daily_stats = []
        
        # Peak equity for drawdown tracking (used by risk manager)
        self._peak_equity = global_config.total_capital

        # Signal generators for each bot
        self.signal_generators = {}
        for bot_id, config in self.bot_configs.items():
            self.signal_generators[bot_id] = OptimizedSignalGenerator(config.symbol, config.timeframe)

        # Risk managers (Kelly Criterion + volatility sizing) — loaded from backend if available
        self.risk_managers: Dict[str, Any] = {}
        try:
            from backend.core.risk import AdvancedRiskManager, RiskParams
            for bot_id in self.bot_configs:
                self.risk_managers[bot_id] = AdvancedRiskManager(RiskParams())
            logger.info("AdvancedRiskManager loaded (Kelly Criterion active)")
        except ImportError:
            logger.info("backend.core.risk not found — using default position sizing")

        # Persistence repositories
        self._trade_repo: Optional[Any] = None
        self._equity_repo: Optional[Any] = None
        self._daily_repo: Optional[Any] = None
        self._equity_snapshot_counter = 0   # write equity to DB every 20 calls (~10 min)
        self._daily_wins = 0
        self._daily_losses = 0
        if _DB_AVAILABLE:
            try:
                init_db()
                self._trade_repo  = TradeRepository()
                self._equity_repo = EquityRepository()
                self._daily_repo  = DailyStatsRepository()
                self._load_history_from_db()
                logger.info("Database persistence enabled")
            except Exception as exc:
                logger.warning(f"DB init failed ({exc}) — running in-memory only")

        logger.info(f"Production Trading System initialized with {len(bot_configs)} bots")
        logger.info(f"Paper Trading: {global_config.paper_trading}")
        logger.info(f"Total Capital: ${global_config.total_capital:,.2f}")
    
    async def start(self):
        """Start the trading system."""
        logger.info("🚀 Starting Production Trading System...")
        await self._initialize_models()

        try:
            while True:
                cycle_start = time.time()
                # Check if we need to reset daily stats
                self._check_daily_reset()

                # Check emergency stops
                if self._check_emergency_stops():
                    logger.critical("🛑 Emergency stop triggered! Shutting down system.")
                    break

                # Process each bot
                for bot_id, config in self.bot_configs.items():
                    if not config.enabled:
                        continue

                    try:
                        await self._process_bot(bot_id, config)
                    except Exception as e:
                        logger.error(f"Error processing bot {bot_id}: {e}")

                # Update system metrics
                self._update_metrics()

                # Log system status
                self._log_system_status()

                # Heartbeat: how long the cycle took (detects loop-blocking issues)
                cycle_elapsed = time.time() - cycle_start
                if cycle_elapsed > 5:
                    logger.warning(f"⏱️ Slow trading cycle: {cycle_elapsed:.1f}s")
                else:
                    logger.debug(f"Cycle completed in {cycle_elapsed:.2f}s")

                # Wait before next iteration
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("👋 Shutting down trading system...")
            await self._shutdown()
        except Exception as e:
            logger.critical(f"💥 Critical system error: {e}")
            await self._emergency_shutdown()
    
    async def _process_bot(self, bot_id: str, config: BotConfig):
        """Process individual bot logic."""
        try:
            # Get current market data — run blocking CCXT call off the event loop
            # so it never starves the FastAPI server sharing this loop.
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, config.symbol, config.timeframe, None, 200
            )
            if not ohlcv or len(ohlcv) < 100:
                return

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            current_price = df.iloc[-1]['close']

            # Generate signals — ML training inside is CPU-bound; offload to a thread
            signal_generator = self.signal_generators[bot_id]
            signals = await asyncio.to_thread(signal_generator.generate_signals, df)

            if not signals:
                logger.debug(f"[{bot_id}] No signal this cycle")
                return
            
            # Get the latest signal
            latest_signal = signals[-1]
            logger.info(
                f"[{bot_id}] Signal: dir={latest_signal.direction} "
                f"conf={latest_signal.confidence:.2f} strength={latest_signal.strength:.2f} "
                f"@ ${current_price:.4f}"
            )

            # Check if we should enter a new trade
            if self._should_enter_trade(bot_id, config, latest_signal):
                await self._enter_trade(bot_id, config, latest_signal, current_price)
            
            # Check existing trades for this bot
            bot_trades = [trade for trade in self.active_trades.values() 
                         if trade.symbol == config.symbol]
            
            for trade in bot_trades:
                await self._check_trade_exit(trade, current_price)
                
        except Exception as e:
            logger.error(f"Error in _process_bot for {bot_id}: {e}")
    
    def _should_enter_trade(self, bot_id: str, config: BotConfig, signal: TradeSignal) -> bool:
        """Determine if we should enter a new trade."""
        # Check confidence threshold
        if signal.confidence < config.confidence_threshold:
            return False
        
        # Check if we already have a trade for this symbol
        existing_trades = [trade for trade in self.active_trades.values() 
                          if trade.symbol == config.symbol]
        if existing_trades:
            return False
        
        # Check global trade limits
        if len(self.active_trades) >= self.global_config.max_concurrent_trades:
            return False
        
        # Check daily limits
        if self.daily_pnl <= -self.global_config.daily_loss_limit * self.global_config.total_capital:
            logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        if self.daily_pnl >= self.global_config.daily_profit_target * self.global_config.total_capital:
            logger.info(f"Daily profit target reached: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    async def _enter_trade(self, bot_id: str, config: BotConfig, signal: TradeSignal, current_price: float):
        """Enter a new trade."""
        try:
            # Position sizing: use AdvancedRiskManager (Kelly) if available, else fixed-risk fallback
            risk_mgr = self.risk_managers.get(bot_id)
            if risk_mgr:
                current_equity = self.global_config.total_capital + self.total_pnl
                hist_returns = pd.Series(
                    [t.pnl_pct for t in self.trade_history if t.pnl_pct is not None]
                )
                position_size = risk_mgr.calculate_position_size(
                    capital=self.global_config.total_capital * config.capital_allocation,
                    entry_price=current_price,
                    stop_loss=signal.stop_loss,
                    confidence=signal.confidence,
                    current_equity=current_equity,
                    peak_equity=self._peak_equity,
                    returns=hist_returns if len(hist_returns) >= 10 else None,
                )
            else:
                risk_amount = self.global_config.total_capital * config.capital_allocation * config.max_risk_per_trade
                stop_distance = abs(current_price - signal.stop_loss) / current_price
                position_size = risk_amount / (stop_distance * current_price) if stop_distance > 0 else 0

            if position_size <= 0:
                return

            # Create trade record
            trade_id = f"{config.symbol}_{int(time.time())}"
            trade = Trade(
                id=trade_id,
                symbol=config.symbol,
                direction=signal.direction,
                entry_time=int(time.time() * 1000),
                entry_price=current_price,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                bot_id=bot_id,
            )

            # Execute trade (paper trading or real)
            if self.global_config.paper_trading:
                # Simulate real execution costs: 0.1% commission + 0.05% slippage
                commission = 0.001
                slippage = 0.0005
                trade.entry_price = current_price * (1 + commission + slippage)
                self.active_trades[trade_id] = trade
                logger.info(
                    f"📝 Paper Trade Entered: {config.symbol} {signal.direction} "
                    f"@ ${trade.entry_price:.4f} (raw: ${current_price:.4f})"
                )
            else:
                # Real trading - place actual order
                order_type = 'market'
                side = 'buy' if signal.direction == 1 else 'sell'
                
                order = self.exchange.create_order(
                    symbol=config.symbol,
                    type=order_type,
                    side=side,
                    amount=position_size,
                    price=None  # Market order
                )
                
                if order['status'] == 'filled':
                    trade.entry_price = order['average']
                    self.active_trades[trade_id] = trade
                    logger.info(f"💰 Real Trade Entered: {config.symbol} {signal.direction} @ ${trade.entry_price:.4f}")
                else:
                    logger.error(f"Failed to enter trade: {order}")
            
        except Exception as e:
            logger.error(f"Error entering trade: {e}")
    
    async def _check_trade_exit(self, trade: Trade, current_price: float):
        """Check if trade should be exited."""
        try:
            should_exit = False
            exit_reason = None
            
            # Check stop loss
            if ((trade.direction == 1 and current_price <= trade.stop_loss) or
                (trade.direction == -1 and current_price >= trade.stop_loss)):
                should_exit = True
                exit_reason = "stop_loss"
            
            # Check take profit
            elif ((trade.direction == 1 and current_price >= trade.take_profit) or
                  (trade.direction == -1 and current_price <= trade.take_profit)):
                should_exit = True
                exit_reason = "take_profit"
            
            if should_exit:
                await self._exit_trade(trade, current_price, exit_reason)
                
        except Exception as e:
            logger.error(f"Error checking trade exit: {e}")
    
    async def _exit_trade(self, trade: Trade, exit_price: float, reason: str):
        """Exit an existing trade."""
        try:
            # Simulate execution costs for paper trading (same round-trip model as entry)
            if self.global_config.paper_trading:
                commission = 0.001
                slippage = 0.0005
                # Long exits via sell → price is reduced; short exits via buy → price is raised
                if trade.direction == 1:
                    exit_price = exit_price * (1 - commission - slippage)
                else:
                    exit_price = exit_price * (1 + commission + slippage)

            # Calculate PnL
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * trade.direction
            pnl = trade.quantity * trade.entry_price * pnl_pct

            # Update trade record
            trade.exit_time = int(time.time() * 1000)
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.pnl_pct = pnl_pct
            trade.status = "closed"
            trade.reason = reason

            # Execute exit (paper trading or real)
            if self.global_config.paper_trading:
                logger.info(f"📝 Paper Trade Exited: {trade.symbol} PnL: ${pnl:.2f} ({pnl_pct:.2%}) - {reason}")
            else:
                # Real trading - place exit order
                side = 'sell' if trade.direction == 1 else 'buy'
                
                order = self.exchange.create_order(
                    symbol=trade.symbol,
                    type='market',
                    side=side,
                    amount=trade.quantity,
                    price=None
                )
                
                if order['status'] == 'filled':
                    trade.exit_price = order['average']
                    # Recalculate PnL with actual exit price
                    pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * trade.direction
                    pnl = trade.quantity * trade.entry_price * pnl_pct
                    trade.pnl = pnl
                    trade.pnl_pct = pnl_pct
                    
                    logger.info(f"💰 Real Trade Exited: {trade.symbol} PnL: ${pnl:.2f} ({pnl_pct:.2%}) - {reason}")
            
            # Update system metrics
            self.daily_pnl += pnl
            self.total_pnl += pnl
            self.daily_trades += 1

            # Update peak equity and feed risk manager for Kelly Criterion
            current_equity = self.global_config.total_capital + self.total_pnl
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity
            risk_mgr = self.risk_managers.get(trade.bot_id)
            if risk_mgr and pnl_pct is not None:
                risk_mgr.record_trade(float(pnl_pct))

            # Track daily win/loss counts for daily stats
            if pnl > 0:
                self._daily_wins += 1
            else:
                self._daily_losses += 1

            # Persist to database
            if self._trade_repo:
                try:
                    self._trade_repo.save(trade)
                except Exception as exc:
                    logger.warning(f"Could not persist trade {trade.id}: {exc}")

            # Move to trade history
            self.trade_history.append(trade)
            del self.active_trades[trade.id]

        except Exception as e:
            logger.error(f"Error exiting trade: {e}")
    
    def _check_daily_reset(self):
        """Check if we need to reset daily statistics."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            date_str = self.last_reset_date.isoformat()
            logger.info(
                f"📊 Daily Summary [{date_str}] — "
                f"PnL: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}, "
                f"Wins: {self._daily_wins}, Losses: {self._daily_losses}"
            )

            # Persist daily stats
            self.daily_stats.append({
                'date': date_str,
                'pnl': self.daily_pnl,
                'trades': self.daily_trades,
                'wins': self._daily_wins,
                'losses': self._daily_losses,
            })
            if self._daily_repo:
                try:
                    self._daily_repo.save(
                        date_str, self.daily_pnl, self.daily_trades,
                        self._daily_wins, self._daily_losses
                    )
                except Exception as exc:
                    logger.warning(f"Could not persist daily stats: {exc}")

            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self._daily_wins = 0
            self._daily_losses = 0
            self.last_reset_date = current_date
    
    def _check_emergency_stops(self) -> bool:
        """Check if emergency stops should be triggered."""
        # Check total drawdown
        if self.total_pnl <= -self.global_config.emergency_stop_drawdown * self.global_config.total_capital:
            logger.critical(f"🚨 Emergency drawdown stop triggered: ${self.total_pnl:.2f}")
            return True
        
        return False
    
    def _update_metrics(self):
        """Update system performance metrics."""
        current_equity = self.global_config.total_capital + self.total_pnl

        self.equity_curve.append({
            'timestamp': int(time.time() * 1000),
            'equity': current_equity,
            'active_trades': len(self.active_trades),
            'daily_pnl': self.daily_pnl
        })

        # Keep only last 1000 points in memory
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]

        # Persist equity snapshot every 20 calls (~10 minutes at 30s cadence)
        self._equity_snapshot_counter += 1
        if self._equity_repo and self._equity_snapshot_counter % 20 == 0:
            try:
                self._equity_repo.save(
                    equity=current_equity,
                    total_pnl=self.total_pnl,
                    daily_pnl=self.daily_pnl,
                    active_trades=len(self.active_trades),
                )
            except Exception as exc:
                logger.warning(f"Could not persist equity snapshot: {exc}")
    
    def _log_system_status(self):
        """Log current system status."""
        current_equity = self.global_config.total_capital + self.total_pnl
        total_return = self.total_pnl / self.global_config.total_capital
        
        logger.info(f"💼 System Status - Equity: ${current_equity:.2f} ({total_return:.2%}), "
                   f"Active Trades: {len(self.active_trades)}, Daily PnL: ${self.daily_pnl:.2f}")
    
    async def _shutdown(self):
        """Graceful shutdown."""
        logger.info("🔄 Graceful shutdown initiated...")
        
        # Close all active trades
        for trade in list(self.active_trades.values()):
            try:
                # Get current price for exit
                ticker = self.exchange.fetch_ticker(trade.symbol)
                current_price = ticker['last']
                await self._exit_trade(trade, current_price, "system_shutdown")
            except Exception as e:
                logger.error(f"Error closing trade during shutdown: {e}")
        
        # Save final state
        self._save_system_state()
        logger.info("✅ System shutdown complete")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown."""
        logger.critical("🚨 Emergency shutdown initiated!")
        
        # Try to close all positions immediately
        for trade in list(self.active_trades.values()):
            try:
                if not self.global_config.paper_trading:
                    side = 'sell' if trade.direction == 1 else 'buy'
                    self.exchange.create_order(
                        symbol=trade.symbol,
                        type='market',
                        side=side,
                        amount=trade.quantity
                    )
            except Exception as e:
                logger.error(f"Error in emergency close: {e}")
        
        self._save_system_state()
        logger.critical("🛑 Emergency shutdown complete")
    
    def _load_history_from_db(self):
        """Restore trade history from DB so Kelly Criterion works from the first signal."""
        if not self._trade_repo:
            return
        try:
            pnl_pcts = self._trade_repo.get_recent_pnl_pcts(limit=200)
            if not pnl_pcts:
                return
            for bot_id, risk_mgr in self.risk_managers.items():
                for pnl_pct in pnl_pcts:
                    risk_mgr.record_trade(pnl_pct)
            self.total_pnl = sum(self._trade_repo.get_recent_pnl_pcts(limit=10_000))  # rough equity
            logger.info(
                f"Loaded {len(pnl_pcts)} historical trades from DB "
                f"(Kelly Criterion seeded)"
            )
        except Exception as exc:
            logger.warning(f"Could not load history from DB: {exc}")

    async def _fetch_historical_data(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Fetch up to `days` of OHLCV history using CCXT pagination."""
        tf_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '4h': 240, '1d': 1440,
        }
        tf_min = tf_minutes.get(timeframe, 5)
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        step_ms = tf_min * 60 * 1000 * 1000  # 1000 candles per batch

        all_ohlcv = []
        limit = 1000

        while True:
            try:
                # Offload blocking CCXT call so the API event loop stays responsive
                batch = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv, symbol, timeframe, since, limit
                )
            except Exception as e:
                logger.error(f"Error fetching history for {symbol} {timeframe}: {e}")
                break

            if not batch:
                break

            all_ohlcv.extend(batch)
            since = batch[-1][0] + tf_min * 60 * 1000  # advance by one candle

            if len(batch) < limit:
                break

            await asyncio.sleep(0.3)  # respect Binance rate limit

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} ({days}d history)")
        return df

    # History window per timeframe — targets ~35-45k candles so memory and
    # training time stay bounded regardless of the container limit.
    # 365d of 1m = ~525k rows → memory balloon + OOM on a no-swap VM.
    _HISTORY_DAYS = {
        '1m': 30,    # ~43k candles
        '3m': 90,    # ~43k
        '5m': 120,   # ~35k
        '15m': 365,  # ~35k
        '30m': 365,  # ~17k
        '1h': 365,   # ~9k
        '4h': 365,
        '1d': 365,
    }

    async def _initialize_models(self):
        """Pre-train ML models with historical data before the live loop starts."""
        logger.info("Pre-training ML models with historical data...")
        initialized = 0

        for bot_id, config in self.bot_configs.items():
            try:
                t0 = time.time()
                days = self._HISTORY_DAYS.get(config.timeframe, 180)
                df = await self._fetch_historical_data(config.symbol, config.timeframe, days=days)
                if df.empty:
                    logger.warning(f"No historical data for {bot_id} — model will warm up on live data")
                    continue
                # Training (sklearn .fit) is CPU-bound — run in a thread so it
                # does not block the shared API event loop for minutes.
                await asyncio.to_thread(self.signal_generators[bot_id].initialize_from_history, df)
                initialized += 1
                logger.info(f"[{bot_id}] Pre-trained in {time.time() - t0:.1f}s ({len(df)} candles)")
            except Exception as e:
                logger.warning(f"Could not pre-train model for {bot_id}: {e}")

        logger.info(f"ML pre-training complete: {initialized}/{len(self.bot_configs)} bots initialized")

    def _save_system_state(self):
        """Save current system state to file."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'active_trades': [asdict(trade) for trade in self.active_trades.values()],
            'trade_history': [asdict(trade) for trade in self.trade_history[-100:]],  # Last 100 trades
            'daily_stats': self.daily_stats[-30:],  # Last 30 days
            'equity_curve': self.equity_curve[-100:]  # Last 100 points
        }
        
        filename = f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 System state saved to {filename}")


class OptimizedSignalGenerator:
    """Optimized signal generator for production use."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        self._retrain_counter = 0
        self._retrain_interval = 500  # retrain every 500 calls (~4h at 30s cadence)

        # Get optimized parameters
        self.params = self._get_optimized_params(symbol, timeframe)
    
    def _get_optimized_params(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get optimized parameters based on backtest results."""
        # Best performing configurations
        if symbol == 'LINK/USDT' and timeframe == '5m':
            return {
                'momentum_threshold': 0.003,   # 0.3% in 5 bars (was 0.8% — too rare)
                'volume_threshold': 1.5,        # 1.5x avg volume (was 1.8)
                'rsi_oversold': 38,
                'rsi_overbought': 62,
                'confidence_multiplier': 1.2,
                'ml_threshold': 0.55
            }
        elif symbol == 'LINK/USDT' and timeframe == '1m':
            return {
                'momentum_threshold': 0.002,   # 0.2% in 5 bars for 1m
                'volume_threshold': 1.5,
                'rsi_oversold': 38,
                'rsi_overbought': 62,
                'confidence_multiplier': 1.2,
                'ml_threshold': 0.55
            }
        elif symbol == 'ADA/USDT' and timeframe == '1m':
            return {
                'momentum_threshold': 0.002,
                'volume_threshold': 1.5,
                'rsi_oversold': 38,
                'rsi_overbought': 62,
                'confidence_multiplier': 1.2,
                'ml_threshold': 0.55
            }
        else:
            return {
                'momentum_threshold': 0.004,
                'volume_threshold': 1.8,
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'confidence_multiplier': 1.0,
                'ml_threshold': 0.58
            }
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals."""
        try:
            # Add technical indicators
            df = self._add_indicators(df)

            # Retrain periodically — never every call (too slow, causes scaler drift)
            self._retrain_counter += 1
            if not self.is_fitted or self._retrain_counter % self._retrain_interval == 0:
                self._update_model(df)
            
            # Generate signals
            signals = []
            
            if len(df) < 50:
                return signals
            
            latest_row = df.iloc[-1]
            current_price = latest_row['close']
            
            # Generate different types of signals
            momentum_signal = self._check_momentum_signal(latest_row)
            mean_reversion_signal = self._check_mean_reversion_signal(latest_row)
            volume_signal = self._check_volume_signal(df.iloc[-2:])
            ml_signal = self._check_ml_signal(df.iloc[-1:]) if self.is_fitted else None
            
            # Combine signals
            combined_signal = self._combine_signals([
                momentum_signal, mean_reversion_signal, volume_signal, ml_signal
            ])
            
            if combined_signal:
                # Calculate stop loss and take profit
                atr = latest_row.get('atr', current_price * 0.02)
                
                if combined_signal['direction'] == 1:
                    stop_loss = current_price - (atr * 1.5)
                    take_profit = current_price + (atr * 2.5)
                else:
                    stop_loss = current_price + (atr * 1.5)
                    take_profit = current_price - (atr * 2.5)
                
                signal = TradeSignal(
                    symbol=self.symbol,
                    direction=combined_signal['direction'],
                    strength=combined_signal['strength'],
                    confidence=combined_signal['confidence'],
                    timestamp=int(latest_row['timestamp']),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata=combined_signal['metadata']
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {self.symbol}: {e}")
            return []
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        return df
    
    def _check_momentum_signal(self, row) -> Optional[Dict]:
        """Check for momentum signals."""
        params = self.params
        
        if (row['momentum_5'] > params['momentum_threshold'] and
            row['volume_ratio'] > params['volume_threshold'] and
            row['rsi'] < params['rsi_overbought']):
            
            return {
                'type': 'momentum',
                'direction': 1,
                'strength': min(abs(row['momentum_5']) * 50, 1.0),
                'confidence': 0.7 * params['confidence_multiplier']
            }
        
        elif (row['momentum_5'] < -params['momentum_threshold'] and
              row['volume_ratio'] > params['volume_threshold'] and
              row['rsi'] > params['rsi_oversold']):
            
            return {
                'type': 'momentum',
                'direction': -1,
                'strength': min(abs(row['momentum_5']) * 50, 1.0),
                'confidence': 0.7 * params['confidence_multiplier']
            }
        
        return None
    
    def _check_mean_reversion_signal(self, row) -> Optional[Dict]:
        """Check for mean reversion signals."""
        params = self.params
        
        if (row['bb_position'] < 0.15 and row['rsi'] < params['rsi_oversold']):
            return {
                'type': 'mean_reversion',
                'direction': 1,
                'strength': min((params['rsi_oversold'] - row['rsi']) / params['rsi_oversold'], 1.0),
                'confidence': 0.8 * params['confidence_multiplier']
            }
        
        elif (row['bb_position'] > 0.85 and row['rsi'] > params['rsi_overbought']):
            return {
                'type': 'mean_reversion',
                'direction': -1,
                'strength': min((row['rsi'] - params['rsi_overbought']) / (100 - params['rsi_overbought']), 1.0),
                'confidence': 0.8 * params['confidence_multiplier']
            }
        
        return None
    
    def _check_volume_signal(self, df_slice) -> Optional[Dict]:
        """Check for volume breakout signals."""
        if len(df_slice) < 2:
            return None
        
        current = df_slice.iloc[-1]
        previous = df_slice.iloc[-2]
        
        price_change = (current['close'] - previous['close']) / previous['close']
        
        if (current['volume_ratio'] > self.params['volume_threshold'] and
            abs(price_change) > 0.005):
            
            direction = 1 if price_change > 0 else -1
            
            return {
                'type': 'volume',
                'direction': direction,
                'strength': min(current['volume_ratio'] / 4, 1.0),
                'confidence': min(abs(price_change) * 100, 0.9)
            }
        
        return None
    
    def _check_ml_signal(self, df_slice) -> Optional[Dict]:
        """Check for ML-based signals."""
        if not self.is_fitted or len(df_slice) == 0:
            return None
        
        try:
            # Prepare features
            features = self._prepare_features(df_slice)
            if features.empty:
                return None
            
            # Get prediction
            X_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Convert to signal — ML model is binary ("will price go up?")
            # It can only generate LONG signals; short signals come from rule-based only.
            if len(proba) >= 2:
                buy_prob = proba[1]

                if buy_prob > self.params['ml_threshold']:
                    return {
                        'type': 'ml',
                        'direction': 1,
                        'strength': min((buy_prob - 0.5) * 2, 1.0),
                        'confidence': buy_prob
                    }
            
        except Exception as e:
            logger.error(f"Error in ML signal generation: {e}")
        
        return None
    
    def _combine_signals(self, signals: List[Optional[Dict]]) -> Optional[Dict]:
        """Combine multiple signals into one."""
        valid_signals = [s for s in signals if s is not None]
        
        if len(valid_signals) < 2:
            return None
        
        # Weighted voting
        weights = {'momentum': 0.3, 'mean_reversion': 0.25, 'volume': 0.25, 'ml': 0.2}
        
        long_vote = 0.0
        short_vote = 0.0
        total_confidence = 0.0
        metadata = {}
        
        for signal in valid_signals:
            weight = weights.get(signal['type'], 0.1)
            weighted_strength = signal['strength'] * signal['confidence'] * weight
            
            if signal['direction'] == 1:
                long_vote += weighted_strength
            else:
                short_vote += weighted_strength
            
            total_confidence += signal['confidence'] * weight
            metadata[signal['type']] = signal
        
        # Decision logic
        if long_vote > short_vote and long_vote > 0.3:
            return {
                'direction': 1,
                'strength': min(long_vote, 1.0),
                'confidence': min(total_confidence, 0.95),
                'metadata': metadata
            }
        elif short_vote > long_vote and short_vote > 0.3:
            return {
                'direction': -1,
                'strength': min(short_vote, 1.0),
                'confidence': min(total_confidence, 0.95),
                'metadata': metadata
            }
        
        return None
    
    def _update_model(self, df: pd.DataFrame):
        """Update ML model with latest data."""
        try:
            if len(df) < 100:
                return
            
            # Prepare features and labels
            features = self._prepare_features(df)
            labels = self._create_labels(df)
            
            if features.empty or labels.empty or len(features) != len(labels):
                return
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            X = features[valid_idx]
            y = labels[valid_idx]
            
            if len(X) < 50 or y.sum() < 5:
                return
            
            if not self.is_fitted:
                self.model = RandomForestClassifier(
                    n_estimators=100, max_depth=6, min_samples_leaf=5,
                    class_weight='balanced', n_jobs=-1, random_state=42
                )

            # Use only recent data; last row already excluded via NaN label (shift(-1))
            recent_data = min(200, len(X))
            X_recent = X.iloc[-recent_data:]
            y_recent = y.iloc[-recent_data:]

            X_scaled = self.scaler.fit_transform(X_recent)
            self.model.fit(X_scaled, y_recent)
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        feature_cols = [
            'sma_20', 'ema_8', 'ema_21', 'rsi', 'bb_position',
            'atr', 'volume_ratio', 'momentum_5', 'momentum_10'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        return df[available_features].ffill().fillna(0)
    
    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create labels for ML training.
        Label at row T = 1 if close[T+1] > close[T] + round-trip costs (0.3%).
        Threshold of 0.003 gives ~25-35% positive rate — balanced enough to train on.
        The last row always gets NaN (no T+1 yet) and is excluded by callers.
        """
        future_returns = df['close'].shift(-1) / df['close'] - 1
        # 0.003 = 0.3% covers round-trip commission+slippage and leaves a real edge
        labels = np.where(future_returns > 0.003, 1, np.where(future_returns.isna(), np.nan, 0))
        return pd.Series(labels, index=df.index, dtype=float)


    def initialize_from_history(self, df: pd.DataFrame):
        """Train the ML model on a large historical dataset at startup.

        Should be called once before live trading begins. Uses walk-forward
        validation to report whether the signal has any predictive lift, then
        trains the final model on the full history.
        """
        try:
            if len(df) < 200:
                logger.warning(f"[{self.symbol}] Too few historical rows ({len(df)}) for initialization")
                return

            df = self._add_indicators(df.copy())

            wf = self.walk_forward_validate(df)
            logger.info(
                f"[{self.symbol}/{self.timeframe}] Walk-forward: "
                f"accuracy={wf.get('avg_accuracy', 0):.3f}, "
                f"baseline={wf.get('avg_baseline', 0):.3f}, "
                f"lift={wf.get('avg_lift', 0):+.3f}, "
                f"valid={wf.get('valid', False)}"
            )

            features = self._prepare_features(df)
            labels = self._create_labels(df)
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            X = features[valid_idx]
            y = labels[valid_idx]

            if len(X) < 100 or y.sum() < 10:
                logger.warning(f"[{self.symbol}] Not enough valid samples after cleaning: {len(X)}")
                return

            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=10,
                class_weight='balanced', n_jobs=-1, random_state=42
            )
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True

            logger.info(
                f"[{self.symbol}/{self.timeframe}] Model trained on {len(X)} samples, "
                f"positive_rate={y.mean():.2%}"
            )

        except Exception as e:
            logger.error(f"Error in initialize_from_history for {self.symbol}: {e}")

    def walk_forward_validate(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """Walk-forward cross-validation across n_splits time folds.

        Each fold trains on the past and tests on the future — never the reverse.
        Returns accuracy metrics and a 'valid' flag (True if lift > 2% above
        majority-class baseline).
        """
        try:
            features = self._prepare_features(df)
            labels = self._create_labels(df)
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            X = features[valid_idx].values
            y = labels[valid_idx].values

            min_test = max(100, len(X) // (n_splits + 2))
            if len(X) < min_test * 3:
                return {'valid': False, 'reason': f'too_few_samples:{len(X)}'}

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=min_test)
            fold_metrics = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                if len(train_idx) < 100 or y[train_idx].sum() < 5:
                    continue

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                fold_scaler = StandardScaler()
                X_tr_s = fold_scaler.fit_transform(X_train)
                X_te_s = fold_scaler.transform(X_test)

                fold_model = RandomForestClassifier(
                    n_estimators=50, max_depth=6, min_samples_leaf=5,
                    class_weight='balanced', random_state=42
                )
                fold_model.fit(X_tr_s, y_train)

                y_pred = fold_model.predict(X_te_s)
                acc = accuracy_score(y_test, y_pred)
                baseline = float(max(y_test.mean(), 1 - y_test.mean()))

                fold_metrics.append({
                    'fold': fold,
                    'accuracy': float(acc),
                    'baseline': baseline,
                    'lift': float(acc - baseline),
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                })

            if not fold_metrics:
                return {'valid': False, 'reason': 'no_valid_folds'}

            avg_acc = float(np.mean([m['accuracy'] for m in fold_metrics]))
            avg_baseline = float(np.mean([m['baseline'] for m in fold_metrics]))
            avg_lift = avg_acc - avg_baseline

            return {
                'valid': avg_lift > 0.02,
                'avg_accuracy': avg_acc,
                'avg_baseline': avg_baseline,
                'avg_lift': avg_lift,
                'folds': fold_metrics,
            }

        except Exception as e:
            logger.error(f"walk_forward_validate error for {self.symbol}: {e}")
            return {'valid': False, 'reason': str(e)}


def create_production_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create production configuration based on backtest results."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,  # Capital mínimo otimizado para Brasil (R$ 6,000)
        max_concurrent_trades=2,  # Reduzido para menor capital
        daily_loss_limit=0.04,  # 4% perda máxima diária
        daily_profit_target=0.025,  # 2.5% meta diária
        emergency_stop_drawdown=0.08,
        paper_trading=True  # Start with paper trading
    )
    
    # Configuração otimizada para capital mínimo - apenas 2 bots mais lucrativos
    bot_configs = [
        BotConfig(
            symbol='LINK/USDT',
            timeframe='5m',
            capital_allocation=0.70,  # 70% para o melhor performer (18.57% retorno)
            max_risk_per_trade=0.025,  # 2.5% risco por trade
            confidence_threshold=0.65,
            stop_loss_pct=0.018,  # 1.8% stop loss
            take_profit_pct=0.035  # 3.5% take profit
        ),
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',
            capital_allocation=0.30,  # 30% para alta frequência (18.10% retorno)
            max_risk_per_trade=0.020,  # 2.0% risco por trade
            confidence_threshold=0.65,
            stop_loss_pct=0.015,  # 1.5% stop loss
            take_profit_pct=0.030  # 3.0% take profit
        )
    ]
    
    return global_config, bot_configs


async def main():
    """Main function to run the production trading system."""
    print("🚀 Production Trading System Starting...")
    print("=" * 60)
    
    # Create configuration
    global_config, bot_configs = create_production_config()
    
    # Display configuration
    print(f"💼 Total Capital: ${global_config.total_capital:,.2f}")
    print(f"📝 Paper Trading: {global_config.paper_trading}")
    print(f"🤖 Number of Bots: {len(bot_configs)}")
    print("\n📊 Bot Configurations:")
    
    for i, config in enumerate(bot_configs, 1):
        print(f"  {i}. {config.symbol} {config.timeframe} - "
              f"{config.capital_allocation:.0%} allocation, "
              f"{config.max_risk_per_trade:.1%} risk per trade")
    
    print("\n" + "=" * 60)
    
    # Initialize and start system
    system = ProductionTradingSystem(global_config, bot_configs)
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())