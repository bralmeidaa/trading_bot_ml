"""
Advanced backtesting engine with comprehensive metrics and analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..core.advanced_risk import AdvancedRiskManager, AdvancedRiskParams
from ..utils.backtest_metrics import calc_max_drawdown, calc_sharpe


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    hold_time: int
    exit_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents an open position."""
    entry_time: int
    entry_price: float
    quantity: float
    side: str
    stop_loss: float
    take_profit: float
    risk_amount: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    commission_pct: float = 0.1  # 0.1% commission
    slippage_pct: float = 0.05   # 0.05% slippage
    
    # Risk management
    use_advanced_risk: bool = True
    risk_params: Optional[AdvancedRiskParams] = None
    
    # Position management
    allow_multiple_positions: bool = False
    max_positions: int = 1
    
    # Execution settings
    execution_delay: int = 0  # Bars delay for execution
    partial_fills: bool = False
    
    # Analysis settings
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    # Basic results
    trades: List[Trade]
    equity_curve: List[Dict[str, Any]]
    positions_history: List[Dict[str, Any]]
    
    # Performance metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    
    # Risk metrics
    var_95: float  # Value at Risk
    cvar_95: float  # Conditional Value at Risk
    calmar_ratio: float
    sterling_ratio: float
    
    # Additional metrics
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Metadata
    config: BacktestConfig
    strategy_params: Dict[str, Any]
    execution_stats: Dict[str, Any]


class AdvancedBacktester:
    """Advanced backtesting engine with comprehensive analysis."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_manager = None
        if config.use_advanced_risk:
            risk_params = config.risk_params or AdvancedRiskParams()
            self.risk_manager = AdvancedRiskManager(risk_params)
        
        # State variables
        self.current_time = 0
        self.current_equity = config.initial_capital
        self.cash = config.initial_capital
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_history: List[Dict] = []
        self.positions_history: List[Dict] = []
        
        # Execution tracking
        self.pending_orders: List[Dict] = []
        self.execution_stats = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "slippage_total": 0.0,
            "commission_total": 0.0
        }
    
    def reset(self) -> None:
        """Reset backtester state."""
        self.current_time = 0
        self.current_equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.positions_history.clear()
        self.pending_orders.clear()
        
        # Reset execution stats
        for key in self.execution_stats:
            self.execution_stats[key] = 0 if isinstance(self.execution_stats[key], int) else 0.0
        
        # Reset risk manager
        if self.risk_manager:
            self.risk_manager = AdvancedRiskManager(self.risk_manager.params)
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        return quantity * price * (self.config.commission_pct / 100)
    
    def _calculate_slippage(self, quantity: float, price: float, side: str) -> Tuple[float, float]:
        """Calculate slippage for a trade."""
        slippage_amount = quantity * price * (self.config.slippage_pct / 100)
        
        # Slippage works against the trader
        if side == 'buy':
            slipped_price = price * (1 + self.config.slippage_pct / 100)
        else:  # sell
            slipped_price = price * (1 - self.config.slippage_pct / 100)
        
        return slipped_price, slippage_amount
    
    def _update_equity(self, current_prices: Dict[str, float]) -> None:
        """Update current equity based on positions and cash."""
        position_value = 0.0
        
        for pos in self.positions:
            # For simplicity, assume single symbol trading
            current_price = list(current_prices.values())[0] if current_prices else pos.entry_price
            if pos.side == 'long':
                position_value += pos.quantity * current_price
            else:  # short
                position_value += pos.quantity * (2 * pos.entry_price - current_price)
        
        self.current_equity = self.cash + position_value
        
        # Update risk manager
        if self.risk_manager:
            self.risk_manager.update_equity(self.current_equity)
    
    def _check_stop_loss_take_profit(self, current_price: float) -> List[Position]:
        """Check if any positions hit stop loss or take profit."""
        positions_to_close = []
        
        for pos in self.positions:
            if pos.side == 'long':
                if current_price <= pos.stop_loss:
                    pos.metadata['exit_reason'] = 'stop_loss'
                    positions_to_close.append(pos)
                elif current_price >= pos.take_profit:
                    pos.metadata['exit_reason'] = 'take_profit'
                    positions_to_close.append(pos)
            else:  # short
                if current_price >= pos.stop_loss:
                    pos.metadata['exit_reason'] = 'stop_loss'
                    positions_to_close.append(pos)
                elif current_price <= pos.take_profit:
                    pos.metadata['exit_reason'] = 'take_profit'
                    positions_to_close.append(pos)
        
        return positions_to_close
    
    def _execute_order(self, order_type: str, quantity: float, price: float, 
                      timestamp: int, metadata: Dict = None) -> Optional[Position]:
        """Execute a buy or sell order."""
        if metadata is None:
            metadata = {}
        
        self.execution_stats["orders_placed"] += 1
        
        # Calculate costs
        commission = self._calculate_commission(quantity, price)
        slipped_price, slippage_cost = self._calculate_slippage(quantity, price, order_type)
        
        total_cost = quantity * slipped_price + commission
        
        if order_type == 'buy':
            if total_cost > self.cash:
                # Insufficient funds
                self.execution_stats["orders_cancelled"] += 1
                return None
            
            self.cash -= total_cost
            self.execution_stats["orders_filled"] += 1
            self.execution_stats["commission_total"] += commission
            self.execution_stats["slippage_total"] += slippage_cost
            
            # Create position
            position = Position(
                entry_time=timestamp,
                entry_price=slipped_price,
                quantity=quantity,
                side='long',
                stop_loss=metadata.get('stop_loss', slipped_price * 0.95),
                take_profit=metadata.get('take_profit', slipped_price * 1.05),
                risk_amount=metadata.get('risk_amount', total_cost),
                confidence=metadata.get('confidence', 0.5),
                metadata=metadata
            )
            
            return position
        
        else:  # sell
            # Find position to close (simplified - assumes FIFO)
            if not self.positions:
                return None
            
            position_to_close = self.positions[0]  # FIFO
            
            # Calculate PnL
            if position_to_close.side == 'long':
                pnl = (slipped_price - position_to_close.entry_price) * position_to_close.quantity - commission
            else:  # short
                pnl = (position_to_close.entry_price - slipped_price) * position_to_close.quantity - commission
            
            pnl_pct = pnl / (position_to_close.entry_price * position_to_close.quantity) * 100
            
            # Update cash
            if position_to_close.side == 'long':
                self.cash += quantity * slipped_price - commission
            else:
                self.cash += position_to_close.quantity * position_to_close.entry_price + pnl
            
            # Record trade
            trade = Trade(
                entry_time=position_to_close.entry_time,
                exit_time=timestamp,
                entry_price=position_to_close.entry_price,
                exit_price=slipped_price,
                quantity=position_to_close.quantity,
                side=position_to_close.side,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees=commission,
                slippage=slippage_cost,
                hold_time=timestamp - position_to_close.entry_time,
                exit_reason=metadata.get('exit_reason', 'signal'),
                metadata=position_to_close.metadata
            )
            
            self.trades.append(trade)
            
            # Update risk manager
            if self.risk_manager:
                trade_info = {
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_time': trade.hold_time,
                    'exit_reason': trade.exit_reason
                }
                self.risk_manager.record_trade(trade_info)
            
            # Remove position
            self.positions.remove(position_to_close)
            
            self.execution_stats["orders_filled"] += 1
            self.execution_stats["commission_total"] += commission
            self.execution_stats["slippage_total"] += slippage_cost
            
            return None
    
    def place_order(self, signal: str, quantity: float, price: float, 
                   timestamp: int, confidence: float = 0.5, 
                   metadata: Dict = None) -> bool:
        """Place a trading order based on signal."""
        if metadata is None:
            metadata = {}
        
        metadata['confidence'] = confidence
        
        # Risk management checks
        if self.risk_manager and signal == 'buy':
            current_positions = [
                {
                    'risk_amount': pos.risk_amount,
                    'entry_time': pos.entry_time,
                    'side': pos.side
                } for pos in self.positions
            ]
            
            should_enter, reason = self.risk_manager.should_enter_position(
                confidence, current_positions
            )
            
            if not should_enter:
                metadata['rejection_reason'] = reason
                return False
            
            # Calculate position size
            volatility = metadata.get('volatility')
            position_size, sizing_info = self.risk_manager.calculate_position_size(
                self.current_equity, price, volatility, confidence
            )
            
            # Override quantity with risk-managed size
            quantity = position_size
            metadata.update(sizing_info)
            
            # Calculate stops
            atr = metadata.get('atr')
            if atr:
                sl, tp = self.risk_manager.calculate_stop_loss_take_profit(
                    price, atr, confidence
                )
                metadata['stop_loss'] = sl
                metadata['take_profit'] = tp
        
        # Execute order
        if signal == 'buy' and len(self.positions) < self.config.max_positions:
            position = self._execute_order('buy', quantity, price, timestamp, metadata)
            if position:
                self.positions.append(position)
                return True
        
        elif signal == 'sell' and self.positions:
            self._execute_order('sell', quantity, price, timestamp, metadata)
            return True
        
        return False
    
    def update(self, timestamp: int, price: float, volume: float = 0, 
              metadata: Dict = None) -> None:
        """Update backtester state with new market data."""
        self.current_time = timestamp
        
        if metadata is None:
            metadata = {}
        
        # Check stop losses and take profits
        positions_to_close = self._check_stop_loss_take_profit(price)
        for pos in positions_to_close:
            self._execute_order('sell', pos.quantity, price, timestamp, pos.metadata)
        
        # Update equity
        self._update_equity({'symbol': price})
        
        # Record equity point
        equity_point = {
            'timestamp': timestamp,
            'equity': self.current_equity,
            'cash': self.cash,
            'positions_value': self.current_equity - self.cash,
            'num_positions': len(self.positions),
            'price': price
        }
        self.equity_history.append(equity_point)
        
        # Record positions snapshot
        positions_snapshot = {
            'timestamp': timestamp,
            'positions': [
                {
                    'entry_time': pos.entry_time,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'side': pos.side,
                    'unrealized_pnl': (price - pos.entry_price) * pos.quantity if pos.side == 'long' 
                                    else (pos.entry_price - price) * pos.quantity,
                    'confidence': pos.confidence
                } for pos in self.positions
            ]
        }
        self.positions_history.append(positions_snapshot)
    
    def finalize(self, final_price: float, final_timestamp: int) -> None:
        """Close all open positions and finalize backtest."""
        # Close all remaining positions
        for pos in self.positions.copy():
            pos.metadata['exit_reason'] = 'end_of_data'
            self._execute_order('sell', pos.quantity, final_price, final_timestamp, pos.metadata)
        
        # Final equity update
        self._update_equity({'symbol': final_price})
    
    def get_results(self, start_date: datetime, end_date: datetime, 
                   strategy_params: Dict = None) -> BacktestResults:
        """Generate comprehensive backtest results."""
        if strategy_params is None:
            strategy_params = {}
        
        # Basic calculations
        initial_capital = self.config.initial_capital
        final_equity = self.current_equity
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Time calculations
        duration_days = (end_date - start_date).days
        annual_return = ((final_equity / initial_capital) ** (365 / max(duration_days, 1)) - 1) * 100
        
        # Equity curve analysis
        equity_values = [point['equity'] for point in self.equity_history]
        returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else []
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0  # Annualized
        sharpe_ratio = (annual_return - self.config.risk_free_rate * 100) / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) * 100 if negative_returns else 0
        sortino_ratio = (annual_return - self.config.risk_free_rate * 100) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        peak_equity = np.maximum.accumulate(equity_values)
        drawdowns = (peak_equity - equity_values) / peak_equity * 100
        max_drawdown = np.max(drawdowns) if drawdowns.size > 0 else 0
        
        # Drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        for dd in drawdowns:
            if dd > 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        cvar_95 = np.mean([r for r in returns if r <= np.percentile(returns, 5)]) * 100 if len(returns) > 0 else 0
        
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        sterling_ratio = annual_return / (max_drawdown + 10) if max_drawdown >= 0 else 0  # +10% penalty
        
        return BacktestResults(
            trades=self.trades,
            equity_curve=self.equity_history,
            positions_history=self.positions_history,
            
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            largest_win=largest_win,
            largest_loss=largest_loss,
            
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            
            config=self.config,
            strategy_params=strategy_params,
            execution_stats=self.execution_stats
        )