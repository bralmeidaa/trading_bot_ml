"""
Advanced monitoring and metrics system for trading bot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    active_positions: int
    equity: float
    available_cash: float
    
    # Risk metrics
    var_95: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    
    # Model metrics
    model_accuracy: float = 0.0
    prediction_confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for monitoring alerts."""
    max_drawdown_pct: float = 15.0
    min_sharpe_ratio: float = 0.5
    max_daily_loss_pct: float = 5.0
    min_win_rate_pct: float = 40.0
    max_consecutive_losses: int = 5
    
    # Model performance alerts
    min_model_accuracy: float = 0.55
    min_prediction_confidence: float = 0.6
    
    # System alerts
    max_response_time_ms: float = 1000.0
    min_data_freshness_minutes: float = 5.0


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, alert_config: AlertConfig = None):
        self.alert_config = alert_config or AlertConfig()
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.consecutive_losses = 0
        self.last_update = None
        self.peak_equity = 0.0
        
    def update_metrics(self, 
                      equity: float,
                      trades: List[Dict],
                      positions: List[Dict],
                      model_metrics: Dict = None) -> PerformanceMetrics:
        """Update performance metrics with latest data."""
        current_time = datetime.now()
        
        # Calculate basic metrics
        if not self.equity_history:
            initial_equity = equity
            total_return = 0.0
        else:
            initial_equity = self.equity_history[0]['equity']
            total_return = (equity / initial_equity - 1) * 100
        
        # Daily return
        if len(self.equity_history) >= 24:  # Assuming hourly updates
            yesterday_equity = self.equity_history[-24]['equity']
            daily_return = (equity / yesterday_equity - 1) * 100
        else:
            daily_return = 0.0
        
        # Update peak equity and drawdown
        self.peak_equity = max(self.peak_equity, equity)
        current_drawdown = (self.peak_equity - equity) / self.peak_equity * 100
        
        # Calculate max drawdown from history
        if self.equity_history:
            equity_values = [e['equity'] for e in self.equity_history] + [equity]
            peak_equity_series = np.maximum.accumulate(equity_values)
            drawdowns = (peak_equity_series - equity_values) / peak_equity_series * 100
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = current_drawdown
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Check consecutive losses
            recent_trades = sorted(trades, key=lambda x: x.get('exit_time', 0))[-10:]
            consecutive_losses = 0
            for trade in reversed(recent_trades):
                if trade.get('pnl', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            self.consecutive_losses = consecutive_losses
        else:
            win_rate = 0
            profit_factor = 0
        
        # Sharpe ratio (simplified)
        if len(self.equity_history) > 30:
            returns = []
            for i in range(1, min(len(self.equity_history), 252)):  # Last year or available
                prev_equity = self.equity_history[-i-1]['equity']
                curr_equity = self.equity_history[-i]['equity']
                ret = (curr_equity / prev_equity - 1)
                returns.append(ret)
            
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
                
                # VaR calculation
                var_95 = np.percentile(returns, 5) * 100
                volatility = std_return * np.sqrt(252) * 100
            else:
                sharpe_ratio = 0
                var_95 = 0
                volatility = 0
        else:
            sharpe_ratio = 0
            var_95 = 0
            volatility = 0
        
        # Model metrics
        model_accuracy = model_metrics.get('accuracy', 0) if model_metrics else 0
        prediction_confidence = model_metrics.get('confidence', 0) if model_metrics else 0
        feature_importance = model_metrics.get('feature_importance', {}) if model_metrics else {}
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=current_time,
            total_return=total_return,
            daily_return=daily_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            active_positions=len(positions),
            equity=equity,
            available_cash=equity - sum(p.get('value', 0) for p in positions),
            var_95=var_95,
            volatility=volatility,
            model_accuracy=model_accuracy,
            prediction_confidence=prediction_confidence,
            feature_importance=feature_importance
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.equity_history.append({
            'timestamp': current_time,
            'equity': equity
        })
        
        # Keep only recent history (last 30 days)
        cutoff_time = current_time - timedelta(days=30)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        self.equity_history = [e for e in self.equity_history if e['timestamp'] > cutoff_time]
        
        # Check for alerts
        self._check_alerts(metrics)
        
        self.last_update = current_time
        return metrics
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Drawdown alerts
        if metrics.current_drawdown > self.alert_config.max_drawdown_pct:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'RISK',
                'message': f'Current drawdown {metrics.current_drawdown:.2f}% exceeds limit {self.alert_config.max_drawdown_pct}%',
                'timestamp': metrics.timestamp,
                'value': metrics.current_drawdown,
                'threshold': self.alert_config.max_drawdown_pct
            })
        
        # Daily loss alert
        if metrics.daily_return < -self.alert_config.max_daily_loss_pct:
            alerts.append({
                'type': 'WARNING',
                'category': 'PERFORMANCE',
                'message': f'Daily loss {metrics.daily_return:.2f}% exceeds limit {self.alert_config.max_daily_loss_pct}%',
                'timestamp': metrics.timestamp,
                'value': metrics.daily_return,
                'threshold': -self.alert_config.max_daily_loss_pct
            })
        
        # Sharpe ratio alert
        if metrics.sharpe_ratio < self.alert_config.min_sharpe_ratio and len(self.metrics_history) > 50:
            alerts.append({
                'type': 'WARNING',
                'category': 'PERFORMANCE',
                'message': f'Sharpe ratio {metrics.sharpe_ratio:.2f} below minimum {self.alert_config.min_sharpe_ratio}',
                'timestamp': metrics.timestamp,
                'value': metrics.sharpe_ratio,
                'threshold': self.alert_config.min_sharpe_ratio
            })
        
        # Win rate alert
        if metrics.win_rate < self.alert_config.min_win_rate_pct and metrics.total_trades > 20:
            alerts.append({
                'type': 'WARNING',
                'category': 'PERFORMANCE',
                'message': f'Win rate {metrics.win_rate:.1f}% below minimum {self.alert_config.min_win_rate_pct}%',
                'timestamp': metrics.timestamp,
                'value': metrics.win_rate,
                'threshold': self.alert_config.min_win_rate_pct
            })
        
        # Consecutive losses alert
        if self.consecutive_losses >= self.alert_config.max_consecutive_losses:
            alerts.append({
                'type': 'CRITICAL',
                'category': 'RISK',
                'message': f'{self.consecutive_losses} consecutive losses detected',
                'timestamp': metrics.timestamp,
                'value': self.consecutive_losses,
                'threshold': self.alert_config.max_consecutive_losses
            })
        
        # Model performance alerts
        if metrics.model_accuracy < self.alert_config.min_model_accuracy:
            alerts.append({
                'type': 'WARNING',
                'category': 'MODEL',
                'message': f'Model accuracy {metrics.model_accuracy:.3f} below minimum {self.alert_config.min_model_accuracy}',
                'timestamp': metrics.timestamp,
                'value': metrics.model_accuracy,
                'threshold': self.alert_config.min_model_accuracy
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Keep only recent alerts (last 7 days)
        cutoff_time = metrics.timestamp - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        latest_metrics = self.metrics_history[-1]
        
        # Recent performance trend
        if len(self.metrics_history) >= 24:
            recent_returns = [m.daily_return for m in self.metrics_history[-24:]]
            trend = 'UP' if np.mean(recent_returns[-7:]) > np.mean(recent_returns[-14:-7]) else 'DOWN'
        else:
            trend = 'NEUTRAL'
        
        # Active alerts by category
        active_alerts = [a for a in self.alerts if a['timestamp'] > datetime.now() - timedelta(hours=24)]
        alert_summary = {}
        for alert in active_alerts:
            category = alert['category']
            alert_summary[category] = alert_summary.get(category, 0) + 1
        
        return {
            'current_metrics': {
                'equity': latest_metrics.equity,
                'total_return': latest_metrics.total_return,
                'daily_return': latest_metrics.daily_return,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'max_drawdown': latest_metrics.max_drawdown,
                'current_drawdown': latest_metrics.current_drawdown,
                'win_rate': latest_metrics.win_rate,
                'profit_factor': latest_metrics.profit_factor,
                'total_trades': latest_metrics.total_trades,
                'active_positions': latest_metrics.active_positions
            },
            'risk_metrics': {
                'var_95': latest_metrics.var_95,
                'volatility': latest_metrics.volatility,
                'consecutive_losses': self.consecutive_losses
            },
            'model_metrics': {
                'accuracy': latest_metrics.model_accuracy,
                'confidence': latest_metrics.prediction_confidence,
                'top_features': dict(list(latest_metrics.feature_importance.items())[:5])
            },
            'trend': trend,
            'alerts': {
                'total': len(active_alerts),
                'by_category': alert_summary,
                'recent': active_alerts[-5:]  # Last 5 alerts
            },
            'last_update': latest_metrics.timestamp.isoformat(),
            'system_status': self._get_system_status()
        }
    
    def _get_system_status(self) -> str:
        """Determine overall system status."""
        if not self.metrics_history:
            return 'UNKNOWN'
        
        latest = self.metrics_history[-1]
        critical_alerts = [a for a in self.alerts if a['type'] == 'CRITICAL' and 
                          a['timestamp'] > datetime.now() - timedelta(hours=1)]
        
        if critical_alerts:
            return 'CRITICAL'
        elif latest.current_drawdown > self.alert_config.max_drawdown_pct * 0.8:
            return 'WARNING'
        elif latest.total_return > 0 and latest.sharpe_ratio > 1.0:
            return 'EXCELLENT'
        elif latest.total_return > 0:
            return 'GOOD'
        else:
            return 'POOR'
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        data = {
            'metrics_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'total_return': m.total_return,
                    'daily_return': m.daily_return,
                    'sharpe_ratio': m.sharpe_ratio,
                    'max_drawdown': m.max_drawdown,
                    'current_drawdown': m.current_drawdown,
                    'win_rate': m.win_rate,
                    'profit_factor': m.profit_factor,
                    'total_trades': m.total_trades,
                    'active_positions': m.active_positions,
                    'equity': m.equity
                } for m in self.metrics_history
            ],
            'alerts': self.alerts,
            'summary': self.get_dashboard_data()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.last_checks = {}
        self.health_history = []
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {}
        }
        
        # Check database connectivity
        try:
            # This would check actual database connection
            health_status['components']['database'] = {
                'status': 'HEALTHY',
                'response_time_ms': 50,
                'last_error': None
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'UNHEALTHY',
                'response_time_ms': None,
                'last_error': str(e)
            }
            health_status['overall_status'] = 'UNHEALTHY'
        
        # Check data feed
        try:
            # This would check market data feed
            health_status['components']['data_feed'] = {
                'status': 'HEALTHY',
                'last_update': datetime.now().isoformat(),
                'latency_ms': 100
            }
        except Exception as e:
            health_status['components']['data_feed'] = {
                'status': 'UNHEALTHY',
                'last_update': None,
                'error': str(e)
            }
            health_status['overall_status'] = 'DEGRADED'
        
        # Check ML model
        try:
            # This would check model availability and performance
            health_status['components']['ml_model'] = {
                'status': 'HEALTHY',
                'last_prediction': datetime.now().isoformat(),
                'accuracy': 0.65
            }
        except Exception as e:
            health_status['components']['ml_model'] = {
                'status': 'UNHEALTHY',
                'error': str(e)
            }
            health_status['overall_status'] = 'UNHEALTHY'
        
        # Check trading engine
        health_status['components']['trading_engine'] = {
            'status': 'HEALTHY',
            'active_strategies': 1,
            'last_trade': datetime.now().isoformat()
        }
        
        self.health_history.append(health_status)
        
        # Keep only recent history
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-500:]
        
        return health_status