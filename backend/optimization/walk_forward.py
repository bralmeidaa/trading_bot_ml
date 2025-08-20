"""
Walk-forward optimization and Monte Carlo analysis for trading strategies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..backtesting.advanced_engine import AdvancedBacktester, BacktestConfig, BacktestResults
from ..core.advanced_risk import AdvancedRiskParams


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""
    # Time windows
    training_window_days: int = 180  # 6 months training
    testing_window_days: int = 30    # 1 month testing
    step_size_days: int = 30         # Move forward by 1 month
    min_training_samples: int = 100  # Minimum samples for training
    
    # Optimization
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, calmar_ratio, total_return
    max_iterations: int = 50
    
    # Validation
    min_trades_per_period: int = 5
    max_drawdown_threshold: float = 20.0  # Stop if DD > 20%
    
    # Monte Carlo
    monte_carlo_runs: int = 1000
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]


@dataclass
class WalkForwardPeriod:
    """Results for a single walk-forward period."""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Training results
    train_samples: int
    optimization_iterations: int
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    
    # Testing results
    test_results: BacktestResults
    out_of_sample_metrics: Dict[str, float]
    
    # Validation flags
    is_valid: bool
    validation_issues: List[str]


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo analysis."""
    n_runs: int
    confidence_intervals: Dict[str, Dict[float, float]]  # metric -> confidence_level -> value
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    worst_case_scenarios: List[Dict[str, Any]]
    best_case_scenarios: List[Dict[str, Any]]
    probability_of_loss: float
    expected_shortfall: float


@dataclass
class WalkForwardResults:
    """Complete walk-forward optimization results."""
    config: WalkForwardConfig
    periods: List[WalkForwardPeriod]
    
    # Aggregate metrics
    total_periods: int
    valid_periods: int
    avg_out_of_sample_return: float
    avg_out_of_sample_sharpe: float
    consistency_score: float  # How consistent results are across periods
    
    # Combined equity curve
    combined_equity_curve: List[Dict[str, Any]]
    combined_trades: List[Any]
    
    # Monte Carlo results
    monte_carlo: Optional[MonteCarloResults] = None
    
    # Stability analysis
    parameter_stability: Dict[str, float]  # How stable each parameter is
    performance_degradation: float  # How much performance degrades over time


class WalkForwardOptimizer:
    """Walk-forward optimization engine."""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
    
    def optimize_strategy(self,
                         data: pd.DataFrame,
                         strategy_class: Any,
                         parameter_ranges: Dict[str, List],
                         backtest_config: BacktestConfig) -> WalkForwardResults:
        """
        Perform walk-forward optimization on a strategy.
        
        Args:
            data: OHLCV data with datetime index
            strategy_class: Strategy class to optimize
            parameter_ranges: Dict of parameter names to lists of values to test
            backtest_config: Backtesting configuration
        """
        # Ensure data is sorted by time
        data = data.sort_values('ts').reset_index(drop=True)
        data['dt'] = pd.to_datetime(data['ts'], unit='ms')
        
        periods = self._create_periods(data)
        walk_forward_periods = []
        
        for i, (train_data, test_data, period_info) in enumerate(periods):
            print(f"Processing period {i+1}/{len(periods)}: {period_info['test_start']} to {period_info['test_end']}")
            
            # Optimize on training data
            best_params, train_metrics, iterations = self._optimize_parameters(
                train_data, strategy_class, parameter_ranges, backtest_config
            )
            
            # Test on out-of-sample data
            test_results = self._backtest_strategy(
                test_data, strategy_class, best_params, backtest_config
            )
            
            # Validate results
            is_valid, issues = self._validate_period_results(test_results, train_data, test_data)
            
            period = WalkForwardPeriod(
                period_id=i,
                train_start=period_info['train_start'],
                train_end=period_info['train_end'],
                test_start=period_info['test_start'],
                test_end=period_info['test_end'],
                train_samples=len(train_data),
                optimization_iterations=iterations,
                best_params=best_params,
                train_metrics=train_metrics,
                test_results=test_results,
                out_of_sample_metrics=self._extract_key_metrics(test_results),
                is_valid=is_valid,
                validation_issues=issues
            )
            
            walk_forward_periods.append(period)
        
        # Aggregate results
        return self._aggregate_results(walk_forward_periods, data)
    
    def _create_periods(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
        """Create training/testing periods for walk-forward analysis."""
        periods = []
        
        start_date = data['dt'].min()
        end_date = data['dt'].max()
        
        current_date = start_date + timedelta(days=self.config.training_window_days)
        
        while current_date + timedelta(days=self.config.testing_window_days) <= end_date:
            # Training period
            train_start = current_date - timedelta(days=self.config.training_window_days)
            train_end = current_date
            
            # Testing period
            test_start = current_date
            test_end = current_date + timedelta(days=self.config.testing_window_days)
            
            # Extract data
            train_mask = (data['dt'] >= train_start) & (data['dt'] < train_end)
            test_mask = (data['dt'] >= test_start) & (data['dt'] < test_end)
            
            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()
            
            # Check minimum samples
            if len(train_data) >= self.config.min_training_samples and len(test_data) > 0:
                period_info = {
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                }
                periods.append((train_data, test_data, period_info))
            
            # Move forward
            current_date += timedelta(days=self.config.step_size_days)
        
        return periods
    
    def _optimize_parameters(self,
                           train_data: pd.DataFrame,
                           strategy_class: Any,
                           parameter_ranges: Dict[str, List],
                           backtest_config: BacktestConfig) -> Tuple[Dict, Dict, int]:
        """Optimize strategy parameters on training data."""
        best_params = {}
        best_score = -np.inf
        best_metrics = {}
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        # Limit iterations
        if len(param_combinations) > self.config.max_iterations:
            # Random sampling if too many combinations
            indices = np.random.choice(len(param_combinations), self.config.max_iterations, replace=False)
            param_combinations = [param_combinations[i] for i in indices]
        
        iterations = 0
        for params in param_combinations:
            try:
                # Backtest with these parameters
                results = self._backtest_strategy(train_data, strategy_class, params, backtest_config)
                
                # Calculate optimization score
                score = self._calculate_optimization_score(results)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = self._extract_key_metrics(results)
                
                iterations += 1
                
            except Exception as e:
                print(f"Error testing parameters {params}: {e}")
                continue
        
        return best_params, best_metrics, iterations
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters."""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _backtest_strategy(self,
                          data: pd.DataFrame,
                          strategy_class: Any,
                          params: Dict,
                          backtest_config: BacktestConfig) -> BacktestResults:
        """Run backtest with given strategy and parameters."""
        # This is a simplified version - in practice, you'd integrate with your strategy classes
        backtester = AdvancedBacktester(backtest_config)
        backtester.reset()
        
        # Initialize strategy with parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        for _, row in data.iterrows():
            timestamp = int(row['ts'])
            price = float(row['close'])
            
            # Get strategy signal (this would be strategy-specific)
            signal = strategy.get_signal(row)  # This method would need to be implemented
            
            if signal == 'buy':
                backtester.place_order('buy', 100, price, timestamp, confidence=0.6)
            elif signal == 'sell':
                backtester.place_order('sell', 100, price, timestamp)
            
            # Update backtester
            backtester.update(timestamp, price, row.get('volume', 0))
        
        # Finalize
        if not data.empty:
            final_row = data.iloc[-1]
            backtester.finalize(float(final_row['close']), int(final_row['ts']))
        
        # Get results
        start_date = pd.to_datetime(data['ts'].min(), unit='ms')
        end_date = pd.to_datetime(data['ts'].max(), unit='ms')
        
        return backtester.get_results(start_date, end_date, params)
    
    def _calculate_optimization_score(self, results: BacktestResults) -> float:
        """Calculate optimization score based on configured metric."""
        if self.config.optimization_metric == "sharpe_ratio":
            return results.sharpe_ratio
        elif self.config.optimization_metric == "calmar_ratio":
            return results.calmar_ratio
        elif self.config.optimization_metric == "total_return":
            return results.total_return
        else:
            # Default to Sharpe ratio
            return results.sharpe_ratio
    
    def _extract_key_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """Extract key metrics from backtest results."""
        return {
            'total_return': results.total_return,
            'annual_return': results.annual_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'total_trades': results.total_trades
        }
    
    def _validate_period_results(self,
                                results: BacktestResults,
                                train_data: pd.DataFrame,
                                test_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate results for a period."""
        issues = []
        
        # Check minimum trades
        if results.total_trades < self.config.min_trades_per_period:
            issues.append(f"Too few trades: {results.total_trades}")
        
        # Check maximum drawdown
        if results.max_drawdown > self.config.max_drawdown_threshold:
            issues.append(f"Excessive drawdown: {results.max_drawdown:.2f}%")
        
        # Check for data issues
        if len(test_data) == 0:
            issues.append("No test data")
        
        # Check for extreme results (possible overfitting)
        if results.sharpe_ratio > 5.0:
            issues.append(f"Suspiciously high Sharpe ratio: {results.sharpe_ratio:.2f}")
        
        return len(issues) == 0, issues
    
    def _aggregate_results(self, periods: List[WalkForwardPeriod], data: pd.DataFrame) -> WalkForwardResults:
        """Aggregate results from all periods."""
        valid_periods = [p for p in periods if p.is_valid]
        
        if not valid_periods:
            raise ValueError("No valid periods found")
        
        # Calculate aggregate metrics
        returns = [p.out_of_sample_metrics['total_return'] for p in valid_periods]
        sharpes = [p.out_of_sample_metrics['sharpe_ratio'] for p in valid_periods]
        
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        consistency_score = 1.0 - (np.std(returns) / (abs(avg_return) + 1e-8))
        
        # Parameter stability analysis
        parameter_stability = self._analyze_parameter_stability(valid_periods)
        
        # Performance degradation analysis
        performance_degradation = self._analyze_performance_degradation(valid_periods)
        
        # Combine equity curves
        combined_equity_curve = self._combine_equity_curves(valid_periods)
        combined_trades = self._combine_trades(valid_periods)
        
        return WalkForwardResults(
            config=self.config,
            periods=periods,
            total_periods=len(periods),
            valid_periods=len(valid_periods),
            avg_out_of_sample_return=avg_return,
            avg_out_of_sample_sharpe=avg_sharpe,
            consistency_score=consistency_score,
            combined_equity_curve=combined_equity_curve,
            combined_trades=combined_trades,
            parameter_stability=parameter_stability,
            performance_degradation=performance_degradation
        )
    
    def _analyze_parameter_stability(self, periods: List[WalkForwardPeriod]) -> Dict[str, float]:
        """Analyze how stable parameters are across periods."""
        if not periods:
            return {}
        
        # Get all parameter names
        all_params = set()
        for period in periods:
            all_params.update(period.best_params.keys())
        
        stability_scores = {}
        
        for param_name in all_params:
            values = []
            for period in periods:
                if param_name in period.best_params:
                    values.append(period.best_params[param_name])
            
            if len(values) > 1:
                # Calculate coefficient of variation as stability measure
                if isinstance(values[0], (int, float)):
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    stability_scores[param_name] = 1.0 - (std_val / (abs(mean_val) + 1e-8))
                else:
                    # For non-numeric parameters, calculate consistency
                    unique_values = len(set(values))
                    stability_scores[param_name] = 1.0 - (unique_values - 1) / len(values)
        
        return stability_scores
    
    def _analyze_performance_degradation(self, periods: List[WalkForwardPeriod]) -> float:
        """Analyze if performance degrades over time."""
        if len(periods) < 3:
            return 0.0
        
        returns = [p.out_of_sample_metrics['total_return'] for p in periods]
        
        # Calculate trend using linear regression
        x = np.arange(len(returns))
        slope, _ = np.polyfit(x, returns, 1)
        
        # Normalize by average return
        avg_return = np.mean(returns)
        degradation = -slope / (abs(avg_return) + 1e-8) if avg_return != 0 else 0
        
        return max(0, degradation)  # Only positive degradation
    
    def _combine_equity_curves(self, periods: List[WalkForwardPeriod]) -> List[Dict[str, Any]]:
        """Combine equity curves from all periods."""
        combined_curve = []
        
        for period in periods:
            for point in period.test_results.equity_curve:
                combined_curve.append({
                    'timestamp': point['timestamp'],
                    'equity': point['equity'],
                    'period_id': period.period_id
                })
        
        return sorted(combined_curve, key=lambda x: x['timestamp'])
    
    def _combine_trades(self, periods: List[WalkForwardPeriod]) -> List[Any]:
        """Combine trades from all periods."""
        combined_trades = []
        
        for period in periods:
            for trade in period.test_results.trades:
                trade_dict = {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'period_id': period.period_id
                }
                combined_trades.append(trade_dict)
        
        return combined_trades


class MonteCarloAnalyzer:
    """Monte Carlo analysis for trading strategies."""
    
    def __init__(self, n_runs: int = 1000):
        self.n_runs = n_runs
    
    def analyze_strategy_robustness(self,
                                  walk_forward_results: WalkForwardResults,
                                  confidence_levels: List[float] = None) -> MonteCarloResults:
        """
        Perform Monte Carlo analysis on walk-forward results.
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Extract trade sequences from valid periods
        valid_periods = [p for p in walk_forward_results.periods if p.is_valid]
        all_trades = []
        
        for period in valid_periods:
            for trade in period.test_results.trades:
                all_trades.append({
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'hold_time': trade.hold_time
                })
        
        if not all_trades:
            raise ValueError("No trades found for Monte Carlo analysis")
        
        # Run Monte Carlo simulations
        simulation_results = []
        
        for run in range(self.n_runs):
            # Bootstrap sample trades
            n_trades = len(all_trades)
            sampled_trades = np.random.choice(all_trades, size=n_trades, replace=True)
            
            # Calculate metrics for this run
            total_pnl = sum(trade['pnl'] for trade in sampled_trades)
            returns = [trade['pnl_pct'] / 100 for trade in sampled_trades]
            
            # Calculate equity curve
            equity = 10000  # Starting equity
            equity_curve = [equity]
            
            for trade in sampled_trades:
                equity += trade['pnl']
                equity_curve.append(equity)
            
            # Calculate metrics
            total_return = (equity / 10000 - 1) * 100
            
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe = 0
            
            # Drawdown
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown = np.max(drawdowns)
            
            simulation_results.append({
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'final_equity': equity,
                'total_trades': len(sampled_trades)
            })
        
        # Analyze results
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'final_equity']
        confidence_intervals = {}
        mean_metrics = {}
        std_metrics = {}
        
        for metric in metrics:
            values = [result[metric] for result in simulation_results]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
            
            confidence_intervals[metric] = {}
            for level in confidence_levels:
                confidence_intervals[metric][level] = np.percentile(values, level * 100)
        
        # Find worst and best case scenarios
        returns = [result['total_return'] for result in simulation_results]
        worst_indices = np.argsort(returns)[:10]  # 10 worst cases
        best_indices = np.argsort(returns)[-10:]  # 10 best cases
        
        worst_cases = [simulation_results[i] for i in worst_indices]
        best_cases = [simulation_results[i] for i in best_indices]
        
        # Probability of loss
        prob_loss = sum(1 for result in simulation_results if result['total_return'] < 0) / len(simulation_results)
        
        # Expected shortfall (average of worst 5% cases)
        worst_5_percent = int(0.05 * len(simulation_results))
        sorted_returns = sorted(returns)
        expected_shortfall = np.mean(sorted_returns[:worst_5_percent]) if worst_5_percent > 0 else 0
        
        return MonteCarloResults(
            n_runs=self.n_runs,
            confidence_intervals=confidence_intervals,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            worst_case_scenarios=worst_cases,
            best_case_scenarios=best_cases,
            probability_of_loss=prob_loss,
            expected_shortfall=expected_shortfall
        )