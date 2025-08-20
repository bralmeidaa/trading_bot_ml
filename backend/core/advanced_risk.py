"""
Advanced risk management for trading strategies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AdvancedRiskParams:
    """Advanced risk management parameters."""
    # Position sizing
    base_risk_per_trade_pct: float = 1.0  # Base risk per trade
    max_risk_per_trade_pct: float = 3.0   # Maximum risk per trade
    min_risk_per_trade_pct: float = 0.1   # Minimum risk per trade
    
    # Kelly criterion
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (conservative)
    kelly_lookback: int = 100     # Lookback period for Kelly calculation
    
    # Volatility-based sizing
    use_volatility_sizing: bool = True
    target_volatility: float = 0.02  # Target daily volatility (2%)
    volatility_lookback: int = 20    # Lookback for volatility calculation
    
    # Drawdown protection
    max_drawdown_pct: float = 10.0   # Maximum allowed drawdown
    drawdown_reduction_factor: float = 0.5  # Reduce size when approaching max DD
    
    # Correlation limits
    max_correlation: float = 0.7     # Maximum correlation between positions
    correlation_lookback: int = 50   # Lookback for correlation calculation
    
    # Stop loss and take profit
    use_dynamic_stops: bool = True
    atr_stop_multiplier: float = 2.0
    atr_profit_multiplier: float = 3.0
    trailing_stop_pct: float = 0.5   # Trailing stop as % of profit
    
    # Time-based exits
    max_hold_periods: int = 100      # Maximum holding period
    time_decay_factor: float = 0.95  # Reduce position size over time
    
    # Portfolio limits
    max_positions: int = 5           # Maximum concurrent positions
    max_portfolio_risk_pct: float = 5.0  # Maximum total portfolio risk


class AdvancedRiskManager:
    """Advanced risk management system."""
    
    def __init__(self, params: AdvancedRiskParams):
        self.params = params
        self.equity_history: List[float] = []
        self.trade_history: List[Dict] = []
        self.position_history: List[Dict] = []
        self.returns_history: List[float] = []
    
    def update_equity(self, equity: float) -> None:
        """Update equity history."""
        self.equity_history.append(equity)
        
        # Calculate returns
        if len(self.equity_history) > 1:
            ret = (equity / self.equity_history[-2]) - 1
            self.returns_history.append(ret)
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly fraction for position sizing.
        Kelly = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = avg_win / abs(avg_loss)  # Payoff ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply conservative fraction
        return max(0, min(kelly * self.params.kelly_fraction, 0.5))
    
    def calculate_volatility_sizing(self, current_volatility: float) -> float:
        """Calculate position size based on volatility targeting."""
        if current_volatility <= 0:
            return 1.0
        
        # Scale position size inversely with volatility
        vol_multiplier = self.params.target_volatility / current_volatility
        return np.clip(vol_multiplier, 0.1, 3.0)
    
    def calculate_drawdown_adjustment(self, current_equity: float, peak_equity: float) -> float:
        """Adjust position size based on current drawdown."""
        if peak_equity <= 0:
            return 1.0
        
        current_dd = (peak_equity - current_equity) / peak_equity * 100
        
        if current_dd >= self.params.max_drawdown_pct * self.params.drawdown_reduction_factor:
            # Reduce position size as we approach max drawdown
            reduction = 1 - (current_dd / self.params.max_drawdown_pct)
            return max(0.1, reduction)
        
        return 1.0
    
    def calculate_position_size(self, 
                              equity: float,
                              price: float,
                              volatility: Optional[float] = None,
                              confidence: float = 0.5) -> Tuple[float, Dict]:
        """
        Calculate optimal position size using multiple methods.
        
        Returns:
            Tuple of (position_size, sizing_info)
        """
        if price <= 0 or equity <= 0:
            return 0.0, {"error": "Invalid price or equity"}
        
        sizing_info = {
            "base_risk_pct": self.params.base_risk_per_trade_pct,
            "adjustments": {}
        }
        
        # Start with base risk
        risk_pct = self.params.base_risk_per_trade_pct
        
        # Kelly criterion adjustment
        if self.params.use_kelly_criterion and len(self.trade_history) >= 10:
            wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            losses = [t for t in self.trade_history if t.get('pnl', 0) < 0]
            
            if wins and losses:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = np.mean([t['pnl'] for t in wins])
                avg_loss = np.mean([abs(t['pnl']) for t in losses])
                
                kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
                kelly_risk_pct = kelly_fraction * 100
                
                # Use Kelly if it's reasonable
                if 0.1 <= kelly_risk_pct <= 5.0:
                    risk_pct = kelly_risk_pct
                    sizing_info["adjustments"]["kelly"] = kelly_risk_pct
        
        # Volatility adjustment
        if self.params.use_volatility_sizing and volatility is not None:
            vol_multiplier = self.calculate_volatility_sizing(volatility)
            risk_pct *= vol_multiplier
            sizing_info["adjustments"]["volatility"] = vol_multiplier
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + confidence  # Scale from 0.5 to 1.5
        risk_pct *= confidence_multiplier
        sizing_info["adjustments"]["confidence"] = confidence_multiplier
        
        # Drawdown adjustment
        if self.equity_history:
            peak_equity = max(self.equity_history)
            dd_multiplier = self.calculate_drawdown_adjustment(equity, peak_equity)
            risk_pct *= dd_multiplier
            sizing_info["adjustments"]["drawdown"] = dd_multiplier
        
        # Apply limits
        risk_pct = np.clip(risk_pct, 
                          self.params.min_risk_per_trade_pct, 
                          self.params.max_risk_per_trade_pct)
        
        # Calculate position size
        risk_amount = equity * (risk_pct / 100)
        position_size = risk_amount / price
        
        sizing_info["final_risk_pct"] = risk_pct
        sizing_info["risk_amount"] = risk_amount
        sizing_info["position_size"] = position_size
        
        return position_size, sizing_info
    
    def calculate_stop_loss_take_profit(self, 
                                      entry_price: float,
                                      atr: Optional[float] = None,
                                      confidence: float = 0.5) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        if not self.params.use_dynamic_stops or atr is None or atr <= 0:
            # Fallback to percentage-based stops
            sl = entry_price * 0.98  # 2% stop loss
            tp = entry_price * 1.04  # 4% take profit
            return sl, tp
        
        # ATR-based stops
        stop_distance = atr * self.params.atr_stop_multiplier
        profit_distance = atr * self.params.atr_profit_multiplier
        
        # Adjust based on confidence
        confidence_adj = 0.7 + (confidence * 0.6)  # Scale from 0.7 to 1.3
        stop_distance /= confidence_adj
        profit_distance *= confidence_adj
        
        sl = entry_price - stop_distance
        tp = entry_price + profit_distance
        
        return sl, tp
    
    def should_reduce_position(self, 
                             current_price: float,
                             entry_price: float,
                             entry_time: int,
                             current_time: int) -> Tuple[bool, float, str]:
        """
        Determine if position should be reduced and by how much.
        
        Returns:
            Tuple of (should_reduce, reduction_factor, reason)
        """
        # Time-based reduction
        holding_period = current_time - entry_time
        if holding_period > self.params.max_hold_periods:
            time_factor = self.params.time_decay_factor ** (holding_period - self.params.max_hold_periods)
            return True, 1 - time_factor, "time_decay"
        
        # Drawdown-based reduction
        if self.equity_history:
            peak_equity = max(self.equity_history)
            current_equity = self.equity_history[-1]
            current_dd = (peak_equity - current_equity) / peak_equity * 100
            
            if current_dd > self.params.max_drawdown_pct * 0.8:  # 80% of max DD
                reduction = min(0.5, current_dd / self.params.max_drawdown_pct)
                return True, reduction, "drawdown_protection"
        
        return False, 0.0, "none"
    
    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Calculate total portfolio risk metrics."""
        if not positions:
            return {"total_risk_pct": 0.0, "position_count": 0}
        
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        total_equity = self.equity_history[-1] if self.equity_history else 1000.0
        
        portfolio_risk_pct = (total_risk / total_equity) * 100
        
        # Calculate correlation risk (simplified)
        correlation_penalty = 0.0
        if len(positions) > 1:
            # Assume some correlation between positions
            correlation_penalty = len(positions) * 0.1  # 10% penalty per additional position
        
        adjusted_risk_pct = portfolio_risk_pct * (1 + correlation_penalty)
        
        return {
            "total_risk_pct": portfolio_risk_pct,
            "adjusted_risk_pct": adjusted_risk_pct,
            "position_count": len(positions),
            "correlation_penalty": correlation_penalty,
            "within_limits": adjusted_risk_pct <= self.params.max_portfolio_risk_pct
        }
    
    def should_enter_position(self, 
                            signal_strength: float,
                            current_positions: List[Dict]) -> Tuple[bool, str]:
        """Determine if a new position should be entered."""
        # Check position limits
        if len(current_positions) >= self.params.max_positions:
            return False, "max_positions_reached"
        
        # Check portfolio risk
        portfolio_risk = self.calculate_portfolio_risk(current_positions)
        if not portfolio_risk["within_limits"]:
            return False, "portfolio_risk_exceeded"
        
        # Check drawdown limits
        if self.equity_history:
            peak_equity = max(self.equity_history)
            current_equity = self.equity_history[-1]
            current_dd = (peak_equity - current_equity) / peak_equity * 100
            
            if current_dd > self.params.max_drawdown_pct * 0.9:  # 90% of max DD
                return False, "drawdown_limit_approached"
        
        # Signal strength threshold (dynamic based on market conditions)
        min_signal_strength = 0.6
        if len(self.returns_history) > 20:
            recent_volatility = np.std(self.returns_history[-20:])
            # Require higher signal strength in volatile markets
            min_signal_strength += recent_volatility * 10
        
        if signal_strength < min_signal_strength:
            return False, f"signal_too_weak_{signal_strength:.3f}<{min_signal_strength:.3f}"
        
        return True, "approved"
    
    def record_trade(self, trade_info: Dict) -> None:
        """Record a completed trade for analysis."""
        self.trade_history.append(trade_info)
        
        # Keep only recent trades for performance
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
    
    def get_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics."""
        if not self.equity_history:
            return {}
        
        equity_series = np.array(self.equity_history)
        returns = np.array(self.returns_history) if self.returns_history else np.array([])
        
        metrics = {
            "current_equity": equity_series[-1],
            "peak_equity": np.max(equity_series),
            "total_return_pct": ((equity_series[-1] / equity_series[0]) - 1) * 100 if len(equity_series) > 1 else 0,
        }
        
        # Drawdown metrics
        peak_equity = np.maximum.accumulate(equity_series)
        drawdowns = (peak_equity - equity_series) / peak_equity * 100
        metrics["current_drawdown_pct"] = drawdowns[-1]
        metrics["max_drawdown_pct"] = np.max(drawdowns)
        
        if len(returns) > 0:
            # Return metrics
            metrics["avg_return"] = np.mean(returns)
            metrics["return_volatility"] = np.std(returns)
            metrics["sharpe_ratio"] = metrics["avg_return"] / metrics["return_volatility"] if metrics["return_volatility"] > 0 else 0
            
            # Risk metrics
            metrics["var_95"] = np.percentile(returns, 5)  # Value at Risk (95%)
            metrics["cvar_95"] = np.mean(returns[returns <= metrics["var_95"]])  # Conditional VaR
            
            # Skewness and kurtosis
            metrics["skewness"] = stats.skew(returns)
            metrics["kurtosis"] = stats.kurtosis(returns)
        
        # Trade metrics
        if self.trade_history:
            wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
            losses = [t for t in self.trade_history if t.get('pnl', 0) < 0]
            
            metrics["total_trades"] = len(self.trade_history)
            metrics["win_rate"] = len(wins) / len(self.trade_history) if self.trade_history else 0
            metrics["avg_win"] = np.mean([t['pnl'] for t in wins]) if wins else 0
            metrics["avg_loss"] = np.mean([t['pnl'] for t in losses]) if losses else 0
            metrics["profit_factor"] = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else 0
        
        return metrics


def create_conservative_risk_params() -> AdvancedRiskParams:
    """Create conservative risk parameters."""
    return AdvancedRiskParams(
        base_risk_per_trade_pct=0.5,
        max_risk_per_trade_pct=1.5,
        kelly_fraction=0.1,
        max_drawdown_pct=5.0,
        max_positions=3
    )


def create_aggressive_risk_params() -> AdvancedRiskParams:
    """Create aggressive risk parameters."""
    return AdvancedRiskParams(
        base_risk_per_trade_pct=2.0,
        max_risk_per_trade_pct=5.0,
        kelly_fraction=0.5,
        max_drawdown_pct=15.0,
        max_positions=8
    )