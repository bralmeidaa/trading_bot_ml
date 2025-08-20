#!/usr/bin/env python3
"""
Massive parameter optimization for maximum profitability.
Focus on short-term trading (1-5min candles) with extensive testing.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from data.binance_client import BinancePublicClient
from ml.features import build_features, build_labels
from ml.models import create_model
from core.advanced_risk import AdvancedRiskParams


class MassiveOptimizer:
    """Massive parameter optimization system."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        self.results = []
        self.best_configs = []
        
        # Optimization search space
        self.timeframes = ["1m", "3m", "5m", "15m"]  # Focus on short-term
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"]
        self.horizons = [1, 2, 3, 5, 8]  # Prediction horizons
        self.costs = [5.0, 10.0, 15.0, 20.0]  # Transaction costs (basis points)
        self.models = ["xgboost", "lightgbm", "ensemble"]
        self.thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        
        # Feature engineering parameters
        self.feature_configs = [
            {"advanced": False, "max_features": 20},
            {"advanced": False, "max_features": 30},
            {"advanced": True, "max_features": 30},
            {"advanced": True, "max_features": 50},
        ]
        
        # Risk management parameters
        self.risk_configs = [
            {"max_position": 0.05, "kelly_fraction": 0.1, "max_drawdown": 0.05},  # Ultra conservative
            {"max_position": 0.10, "kelly_fraction": 0.25, "max_drawdown": 0.10},  # Conservative
            {"max_position": 0.15, "kelly_fraction": 0.5, "max_drawdown": 0.15},   # Moderate
            {"max_position": 0.20, "kelly_fraction": 0.75, "max_drawdown": 0.20},  # Aggressive
        ]
    
    def fetch_data_for_optimization(self, symbol: str, timeframe: str, limit: int = 5000) -> pd.DataFrame:
        """Fetch data for optimization."""
        try:
            print(f"Fetching {symbol} {timeframe} data...")
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                print(f"✓ {symbol} {timeframe}: {len(df)} candles")
                return df
            else:
                print(f"✗ {symbol} {timeframe}: No data")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ {symbol} {timeframe}: Error - {e}")
            return pd.DataFrame()
    
    def optimize_single_config(self, config: Dict) -> Dict:
        """Optimize a single configuration."""
        try:
            symbol = config['symbol']
            timeframe = config['timeframe']
            horizon = config['horizon']
            cost = config['cost']
            model_type = config['model']
            threshold = config['threshold']
            feature_config = config['feature_config']
            risk_config = config['risk_config']
            
            # Fetch data
            df = self.fetch_data_for_optimization(symbol, timeframe, limit=3000)
            if df.empty:
                return {"config": config, "error": "No data"}
            
            # Build features
            fdf = build_features(df, advanced=feature_config['advanced'])
            if fdf.empty:
                return {"config": config, "error": "No features"}
            
            # Build labels
            y = build_labels(fdf, horizon=horizon, cost_bp=cost)
            if len(y) == 0:
                return {"config": config, "error": "No labels"}
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            if len(X) < 500:  # Need minimum data
                return {"config": config, "error": "Insufficient data"}
            
            # Split data (70% train, 30% test)
            split = int(len(X) * 0.7)
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            y_test = y.iloc[split:]
            
            # Train model
            model = create_model(model_type, max_features=feature_config['max_features'])
            train_metrics = model.fit(X_train, y_train.values)
            
            # Backtest strategy
            result = self.backtest_optimized_strategy(
                model, X_test, fdf.iloc[split:], feature_cols, 
                threshold, risk_config
            )
            
            if result:
                result.update({
                    "config": config,
                    "train_metrics": train_metrics,
                    "data_points": len(X_test),
                    "feature_count": len(feature_cols)
                })
                
                # Calculate score (weighted combination of metrics)
                score = self.calculate_optimization_score(result)
                result["optimization_score"] = score
                
                return result
            else:
                return {"config": config, "error": "Backtest failed"}
                
        except Exception as e:
            return {"config": config, "error": str(e)}
    
    def backtest_optimized_strategy(self, model, X_test, price_data, feature_cols, threshold, risk_config):
        """Enhanced backtesting with risk management."""
        try:
            initial_capital = 10000.0
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            # Risk parameters
            max_position_size = risk_config['max_position']
            kelly_fraction = risk_config['kelly_fraction']
            max_drawdown_limit = risk_config['max_drawdown']
            
            peak_equity = initial_capital
            
            for i, (_, row) in enumerate(X_test.iterrows()):
                current_equity = capital + (position * price_data.iloc[i]['close'] if position > 0 else 0)
                equity_curve.append(current_equity)
                
                # Update peak and check drawdown
                peak_equity = max(peak_equity, current_equity)
                current_drawdown = (peak_equity - current_equity) / peak_equity
                
                # Stop trading if max drawdown exceeded
                if current_drawdown > max_drawdown_limit:
                    if position > 0:
                        # Close position
                        capital = position * price_data.iloc[i]['close'] * 0.999  # Account for slippage
                        position = 0
                    continue
                
                # Get prediction
                features = row[feature_cols].values.reshape(1, -1)
                prob = model.predict_proba(pd.DataFrame(features, columns=feature_cols))[0]
                
                current_price = price_data.iloc[i]['close']
                
                # Enhanced trading logic with risk management
                if prob > threshold and position == 0 and current_drawdown < max_drawdown_limit * 0.5:
                    # Calculate position size using Kelly criterion approximation
                    win_rate = 0.6  # Assume based on threshold
                    avg_win_loss_ratio = 1.2  # Conservative estimate
                    
                    kelly_f = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
                    kelly_f = max(0, min(kelly_f * kelly_fraction, max_position_size))
                    
                    # Position size based on current equity
                    position_value = current_equity * kelly_f
                    position = position_value / current_price * 0.999  # Account for fees
                    capital = current_equity - position_value
                    
                    entry_price = current_price
                    entry_time = i
                    
                elif position > 0:
                    # Exit conditions
                    should_exit = (
                        prob < (1 - threshold) or  # Probability reversal
                        i == len(X_test) - 1 or    # End of data
                        current_drawdown > max_drawdown_limit * 0.8  # Approaching limit
                    )
                    
                    if should_exit:
                        # Sell
                        exit_value = position * current_price * 0.999  # Account for fees
                        pnl = exit_value - (position * entry_price)
                        pnl_pct = (exit_value / (position * entry_price) - 1) * 100
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_time': i - entry_time,
                            'position_size': position * entry_price / initial_capital
                        })
                        
                        capital += exit_value
                        position = 0
            
            # Close final position if any
            if position > 0:
                final_value = position * price_data.iloc[-1]['close'] * 0.999
                pnl = final_value - (position * entry_price)
                pnl_pct = (final_value / (position * entry_price) - 1) * 100
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': price_data.iloc[-1]['close'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_time': len(X_test) - entry_time,
                    'position_size': position * entry_price / initial_capital
                })
                
                capital += final_value
            
            # Calculate comprehensive metrics
            final_capital = capital
            total_return = (final_capital / initial_capital - 1) * 100
            
            if not trades:
                return {
                    'total_return': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'calmar_ratio': 0
                }
            
            # Trade statistics
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Risk metrics
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60 / 5)  # Adjust for 5min
            else:
                sharpe_ratio = 0
            
            # Drawdown calculation
            peak_equity_series = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity_series - equity_curve) / peak_equity_series * 100
            max_drawdown = np.max(drawdowns)
            
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Additional metrics
            avg_trade_duration = np.mean([t['hold_time'] for t in trades])
            avg_position_size = np.mean([t['position_size'] for t in trades])
            
            return {
                'total_return': total_return,
                'final_capital': final_capital,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio,
                'avg_trade_duration': avg_trade_duration,
                'avg_position_size': avg_position_size,
                'trades': trades
            }
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return None
    
    def calculate_optimization_score(self, result: Dict) -> float:
        """Calculate optimization score (higher is better)."""
        # Weighted combination of metrics
        weights = {
            'total_return': 0.3,
            'sharpe_ratio': 0.25,
            'win_rate': 0.15,
            'profit_factor': 0.15,
            'calmar_ratio': 0.15
        }
        
        # Normalize metrics
        total_return_norm = min(result['total_return'] / 100, 1.0)  # Cap at 100%
        sharpe_norm = min(result['sharpe_ratio'] / 3.0, 1.0)  # Cap at 3.0
        win_rate_norm = result['win_rate'] / 100
        profit_factor_norm = min(result['profit_factor'] / 3.0, 1.0)  # Cap at 3.0
        calmar_norm = min(result['calmar_ratio'] / 5.0, 1.0)  # Cap at 5.0
        
        # Penalties
        penalty = 0
        if result['total_trades'] < 10:
            penalty += 0.2  # Too few trades
        if result['max_drawdown'] > 20:
            penalty += 0.3  # Too much drawdown
        if result['total_return'] < 0:
            penalty += 0.5  # Negative returns
        
        score = (
            weights['total_return'] * total_return_norm +
            weights['sharpe_ratio'] * sharpe_norm +
            weights['win_rate'] * win_rate_norm +
            weights['profit_factor'] * profit_factor_norm +
            weights['calmar_ratio'] * calmar_norm
        ) - penalty
        
        return max(0, score)  # Ensure non-negative
    
    def run_massive_optimization(self, max_workers: int = None):
        """Run massive parameter optimization."""
        print("🚀 STARTING MASSIVE PARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Generate all configurations
        configs = []
        config_id = 0
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                for horizon in self.horizons:
                    for cost in self.costs:
                        for model in self.models:
                            for threshold in self.thresholds:
                                for feature_config in self.feature_configs:
                                    for risk_config in self.risk_configs:
                                        configs.append({
                                            'id': config_id,
                                            'symbol': symbol,
                                            'timeframe': timeframe,
                                            'horizon': horizon,
                                            'cost': cost,
                                            'model': model,
                                            'threshold': threshold,
                                            'feature_config': feature_config,
                                            'risk_config': risk_config
                                        })
                                        config_id += 1
        
        print(f"Total configurations to test: {len(configs)}")
        print(f"Estimated time: {len(configs) * 30 / 3600:.1f} hours")
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 8)  # Leave one core free, max 8
        
        print(f"Using {max_workers} parallel workers")
        print()
        
        # Run optimization in batches to avoid memory issues
        batch_size = 100
        all_results = []
        
        for batch_start in range(0, len(configs), batch_size):
            batch_end = min(batch_start + batch_size, len(configs))
            batch_configs = configs[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(configs)-1)//batch_size + 1}")
            print(f"Configs {batch_start} to {batch_end-1}")
            
            # Process batch
            batch_results = []
            completed = 0
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs in batch
                future_to_config = {
                    executor.submit(self.optimize_single_config, config): config 
                    for config in batch_configs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    result = future.result()
                    batch_results.append(result)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"  Completed: {completed}/{len(batch_configs)}")
            
            all_results.extend(batch_results)
            
            # Save intermediate results
            self.save_intermediate_results(all_results, batch_start // batch_size + 1)
            
            print(f"Batch completed. Total results so far: {len(all_results)}")
            print()
        
        self.results = all_results
        return all_results
    
    def save_intermediate_results(self, results: List[Dict], batch_num: int):
        """Save intermediate results."""
        filename = f"optimization_results_batch_{batch_num}.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = result.copy()
            if 'trades' in json_result:
                del json_result['trades']  # Remove trades to reduce size
            json_results.append(json_result)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Intermediate results saved to {filename}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results."""
        print("\n🔍 ANALYZING OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Filter successful results
        successful_results = [r for r in results if 'error' not in r and r.get('total_return', 0) > 0]
        
        print(f"Total configurations tested: {len(results)}")
        print(f"Successful configurations: {len(successful_results)}")
        print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
        
        if not successful_results:
            print("❌ No successful configurations found!")
            return {}
        
        # Sort by optimization score
        successful_results.sort(key=lambda x: x.get('optimization_score', 0), reverse=True)
        
        # Top 10 configurations
        top_configs = successful_results[:10]
        
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        print("-" * 80)
        
        for i, result in enumerate(top_configs, 1):
            config = result['config']
            print(f"{i}. Score: {result['optimization_score']:.3f}")
            print(f"   {config['symbol']} {config['timeframe']} | Model: {config['model']}")
            print(f"   Return: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
            print(f"   Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")
            print(f"   Max DD: {result['max_drawdown']:.2f}% | Profit Factor: {result['profit_factor']:.2f}")
            print()
        
        # Analysis by dimensions
        analysis = {
            'top_configs': top_configs,
            'symbol_analysis': self.analyze_by_dimension(successful_results, 'symbol'),
            'timeframe_analysis': self.analyze_by_dimension(successful_results, 'timeframe'),
            'model_analysis': self.analyze_by_dimension(successful_results, 'model'),
            'threshold_analysis': self.analyze_by_dimension(successful_results, 'threshold'),
            'horizon_analysis': self.analyze_by_dimension(successful_results, 'horizon'),
        }
        
        return analysis
    
    def analyze_by_dimension(self, results: List[Dict], dimension: str) -> Dict:
        """Analyze results by a specific dimension."""
        dimension_results = {}
        
        for result in results:
            key = result['config'][dimension]
            if key not in dimension_results:
                dimension_results[key] = []
            dimension_results[key].append(result)
        
        # Calculate statistics for each dimension value
        dimension_stats = {}
        for key, group_results in dimension_results.items():
            scores = [r['optimization_score'] for r in group_results]
            returns = [r['total_return'] for r in group_results]
            
            dimension_stats[key] = {
                'count': len(group_results),
                'avg_score': np.mean(scores),
                'avg_return': np.mean(returns),
                'best_score': np.max(scores),
                'best_return': np.max(returns),
                'success_rate': len([r for r in group_results if r['total_return'] > 0]) / len(group_results)
            }
        
        return dimension_stats
    
    def generate_final_report(self, analysis: Dict):
        """Generate comprehensive final report."""
        report = []
        report.append("=" * 100)
        report.append("🎯 MASSIVE OPTIMIZATION - FINAL REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not analysis:
            report.append("❌ NO SUCCESSFUL CONFIGURATIONS FOUND")
            return "\n".join(report)
        
        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        
        top_config = analysis['top_configs'][0]
        report.append(f"🏆 BEST CONFIGURATION:")
        report.append(f"   Symbol: {top_config['config']['symbol']}")
        report.append(f"   Timeframe: {top_config['config']['timeframe']}")
        report.append(f"   Model: {top_config['config']['model']}")
        report.append(f"   Threshold: {top_config['config']['threshold']}")
        report.append(f"   Horizon: {top_config['config']['horizon']}")
        report.append("")
        report.append(f"📈 PERFORMANCE METRICS:")
        report.append(f"   Total Return: {top_config['total_return']:.2f}%")
        report.append(f"   Sharpe Ratio: {top_config['sharpe_ratio']:.2f}")
        report.append(f"   Win Rate: {top_config['win_rate']:.1f}%")
        report.append(f"   Profit Factor: {top_config['profit_factor']:.2f}")
        report.append(f"   Max Drawdown: {top_config['max_drawdown']:.2f}%")
        report.append(f"   Total Trades: {top_config['total_trades']}")
        report.append("")
        
        # Dimension Analysis
        report.append("🔍 DIMENSION ANALYSIS")
        report.append("-" * 50)
        
        # Best timeframes
        timeframe_stats = analysis['timeframe_analysis']
        best_timeframes = sorted(timeframe_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        report.append("⏰ BEST TIMEFRAMES:")
        for tf, stats in best_timeframes[:3]:
            report.append(f"   {tf}: Avg Score {stats['avg_score']:.3f}, Avg Return {stats['avg_return']:.2f}%")
        report.append("")
        
        # Best symbols
        symbol_stats = analysis['symbol_analysis']
        best_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        report.append("💰 BEST SYMBOLS:")
        for symbol, stats in best_symbols[:3]:
            report.append(f"   {symbol}: Avg Score {stats['avg_score']:.3f}, Avg Return {stats['avg_return']:.2f}%")
        report.append("")
        
        # Best models
        model_stats = analysis['model_analysis']
        best_models = sorted(model_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        report.append("🤖 BEST MODELS:")
        for model, stats in best_models:
            report.append(f"   {model}: Avg Score {stats['avg_score']:.3f}, Avg Return {stats['avg_return']:.2f}%")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        # Timeframe recommendation
        best_tf = best_timeframes[0][0]
        if best_tf in ["1m", "3m", "5m"]:
            report.append("✅ SHORT-TERM TRADING CONFIRMED: Best performance in 1-5 minute timeframes")
        else:
            report.append(f"⚠️  CONSIDER LONGER TIMEFRAMES: Best performance in {best_tf}")
        
        # Model recommendation
        best_model = best_models[0][0]
        report.append(f"✅ RECOMMENDED MODEL: {best_model}")
        
        # Risk assessment
        avg_max_dd = np.mean([c['max_drawdown'] for c in analysis['top_configs'][:5]])
        if avg_max_dd < 10:
            report.append("✅ LOW RISK: Average max drawdown < 10%")
        elif avg_max_dd < 20:
            report.append("⚠️  MODERATE RISK: Average max drawdown 10-20%")
        else:
            report.append("❌ HIGH RISK: Average max drawdown > 20%")
        
        report.append("")
        
        # Implementation Guide
        report.append("🚀 IMPLEMENTATION GUIDE")
        report.append("-" * 50)
        report.append("1. Start with the top configuration for live testing")
        report.append("2. Use paper trading first to validate results")
        report.append("3. Monitor performance closely and adjust if needed")
        report.append("4. Consider ensemble of top 3-5 configurations")
        report.append("5. Implement strict risk management and stop-loss")
        report.append("")
        
        # Top 5 configurations for implementation
        report.append("🎯 TOP 5 CONFIGURATIONS FOR IMPLEMENTATION")
        report.append("-" * 50)
        
        for i, config_result in enumerate(analysis['top_configs'][:5], 1):
            config = config_result['config']
            report.append(f"{i}. {config['symbol']} {config['timeframe']} | {config['model']}")
            report.append(f"   Threshold: {config['threshold']}, Horizon: {config['horizon']}")
            report.append(f"   Return: {config_result['total_return']:.2f}%, Sharpe: {config_result['sharpe_ratio']:.2f}")
            report.append(f"   Trades: {config_result['total_trades']}, Win Rate: {config_result['win_rate']:.1f}%")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main optimization function."""
    print("🎯 MASSIVE PARAMETER OPTIMIZATION FOR MAXIMUM PROFITABILITY")
    print("Focus: Short-term trading (1-5min candles)")
    print("=" * 80)
    
    optimizer = MassiveOptimizer()
    
    # Run massive optimization
    results = optimizer.run_massive_optimization(max_workers=4)  # Adjust based on your system
    
    # Analyze results
    analysis = optimizer.analyze_results(results)
    
    # Generate final report
    final_report = optimizer.generate_final_report(analysis)
    
    # Display and save report
    print(final_report)
    
    # Save detailed results
    with open("massive_optimization_results.json", "w") as f:
        json.dump({
            'results': [r for r in results if 'trades' not in r],  # Exclude trades for size
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Save final report
    with open("OPTIMIZATION_FINAL_REPORT.md", "w") as f:
        f.write(final_report)
    
    print(f"\n📁 Results saved to:")
    print(f"   - massive_optimization_results.json")
    print(f"   - OPTIMIZATION_FINAL_REPORT.md")
    
    return analysis


if __name__ == "__main__":
    analysis = main()