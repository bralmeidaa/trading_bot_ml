#!/usr/bin/env python3
"""
Focused optimization for short-term trading profitability.
Optimized approach with smart parameter selection.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Any
import time
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from data.binance_client import BinancePublicClient
from ml.features import build_features, build_labels
from ml.models import create_model


class FocusedOptimizer:
    """Focused optimization for maximum profitability."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        self.results = []
        
    def fetch_high_quality_data(self, symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
        """Fetch high-quality data for optimization."""
        try:
            print(f"Fetching {symbol} {timeframe}...")
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv and len(ohlcv) > 500:  # Minimum data requirement
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                
                # Data quality checks
                if df['volume'].sum() > 0 and not df['close'].isna().any():
                    print(f"✓ {symbol} {timeframe}: {len(df)} candles")
                    return df
            
            print(f"✗ {symbol} {timeframe}: Insufficient quality data")
            return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ {symbol} {timeframe}: Error - {e}")
            return pd.DataFrame()
    
    def enhanced_backtest(self, model, X_test, price_data, feature_cols, config):
        """Enhanced backtesting with realistic conditions."""
        try:
            # Configuration
            threshold = config['threshold']
            stop_loss = config.get('stop_loss', 0.02)  # 2% stop loss
            take_profit = config.get('take_profit', 0.04)  # 4% take profit
            max_hold_time = config.get('max_hold_time', 50)  # Max 50 candles
            
            initial_capital = 10000.0
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            entry_price = 0
            entry_time = 0
            
            for i, (_, row) in enumerate(X_test.iterrows()):
                current_price = price_data.iloc[i]['close']
                
                # Update equity
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                
                # Get prediction
                features = row[feature_cols].values.reshape(1, -1)
                prob = model.predict_proba(pd.DataFrame(features, columns=feature_cols))[0]
                
                if position == 0:
                    # Entry logic
                    if prob > threshold:
                        # Calculate position size (fixed 10% of capital for consistency)
                        position_value = capital * 0.1
                        position = position_value / current_price * 0.998  # Account for fees
                        capital -= position_value
                        entry_price = current_price
                        entry_time = i
                
                else:
                    # Exit logic
                    hold_time = i - entry_time
                    price_change = (current_price - entry_price) / entry_price
                    
                    should_exit = (
                        prob < (1 - threshold) or  # Signal reversal
                        price_change <= -stop_loss or  # Stop loss
                        price_change >= take_profit or  # Take profit
                        hold_time >= max_hold_time or  # Max hold time
                        i == len(X_test) - 1  # End of data
                    )
                    
                    if should_exit:
                        # Exit position
                        exit_value = position * current_price * 0.998  # Account for fees
                        pnl = exit_value - (position * entry_price)
                        pnl_pct = (exit_value / (position * entry_price) - 1) * 100
                        
                        # Determine exit reason
                        exit_reason = "signal"
                        if price_change <= -stop_loss:
                            exit_reason = "stop_loss"
                        elif price_change >= take_profit:
                            exit_reason = "take_profit"
                        elif hold_time >= max_hold_time:
                            exit_reason = "time_limit"
                        elif i == len(X_test) - 1:
                            exit_reason = "end_of_data"
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_time': hold_time,
                            'exit_reason': exit_reason,
                            'entry_prob': prob  # Store entry probability
                        })
                        
                        capital += exit_value
                        position = 0
            
            # Calculate metrics
            if not trades:
                return None
            
            final_capital = capital
            total_return = (final_capital / initial_capital - 1) * 100
            
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
                # Adjust Sharpe for timeframe
                timeframe_multiplier = {
                    '1m': 252 * 24 * 60,
                    '3m': 252 * 24 * 20,
                    '5m': 252 * 24 * 12,
                    '15m': 252 * 24 * 4,
                    '1h': 252 * 24
                }
                multiplier = timeframe_multiplier.get(config.get('timeframe', '5m'), 252 * 24 * 12)
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(multiplier)
            else:
                sharpe_ratio = 0
            
            # Drawdown
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown = np.max(drawdowns)
            
            # Additional metrics
            avg_hold_time = np.mean([t['hold_time'] for t in trades])
            
            # Exit reason analysis
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
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
                'avg_hold_time': avg_hold_time,
                'exit_reasons': exit_reasons,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return None
    
    def calculate_fitness_score(self, result: Dict) -> float:
        """Calculate fitness score for optimization."""
        if not result:
            return 0
        
        # Base score from return
        return_score = min(result['total_return'] / 50, 1.0)  # Normalize to 50% max
        
        # Sharpe bonus
        sharpe_bonus = min(result['sharpe_ratio'] / 2.0, 0.5)  # Max 0.5 bonus
        
        # Win rate bonus
        win_rate_bonus = (result['win_rate'] - 50) / 100 if result['win_rate'] > 50 else 0
        
        # Profit factor bonus
        pf_bonus = min((result['profit_factor'] - 1) / 2, 0.3) if result['profit_factor'] > 1 else 0
        
        # Penalties
        penalty = 0
        if result['total_trades'] < 10:
            penalty += 0.3
        if result['max_drawdown'] > 15:
            penalty += 0.2
        if result['total_return'] < 0:
            penalty += 1.0
        
        score = return_score + sharpe_bonus + win_rate_bonus + pf_bonus - penalty
        return max(0, score)
    
    def optimize_configuration(self, symbol: str, timeframe: str):
        """Optimize a single symbol-timeframe combination."""
        print(f"\n🎯 Optimizing {symbol} {timeframe}")
        print("-" * 50)
        
        # Fetch data
        df = self.fetch_high_quality_data(symbol, timeframe, limit=2000)
        if df.empty:
            return []
        
        # Parameter ranges for this symbol/timeframe
        horizons = [1, 2, 3, 5] if timeframe in ['1m', '3m'] else [2, 3, 5, 8]
        costs = [8.0, 12.0, 16.0, 20.0]  # Transaction costs
        models = ["xgboost", "lightgbm", "ensemble"]
        thresholds = [0.55, 0.6, 0.65, 0.7, 0.75]
        
        # Risk management parameters
        stop_losses = [0.015, 0.02, 0.025]  # 1.5%, 2%, 2.5%
        take_profits = [0.03, 0.04, 0.05]   # 3%, 4%, 5%
        max_hold_times = [20, 30, 50] if timeframe in ['1m', '3m'] else [10, 15, 25]
        
        results = []
        config_count = 0
        total_configs = len(horizons) * len(costs) * len(models) * len(thresholds) * len(stop_losses) * len(take_profits) * len(max_hold_times)
        
        print(f"Testing {total_configs} configurations...")
        
        for horizon in horizons:
            for cost in costs:
                for model_type in models:
                    for threshold in thresholds:
                        for stop_loss in stop_losses:
                            for take_profit in take_profits:
                                for max_hold_time in max_hold_times:
                                    config_count += 1
                                    
                                    if config_count % 50 == 0:
                                        print(f"  Progress: {config_count}/{total_configs} ({config_count/total_configs*100:.1f}%)")
                                    
                                    try:
                                        # Build features
                                        fdf = build_features(df, advanced=False)  # Start with basic features
                                        if fdf.empty:
                                            continue
                                        
                                        # Build labels
                                        y = build_labels(fdf, horizon=horizon, cost_bp=cost)
                                        if len(y) == 0:
                                            continue
                                        
                                        # Prepare data
                                        feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
                                        X = fdf[feature_cols]
                                        
                                        if len(X) < 300:  # Minimum data
                                            continue
                                        
                                        # Split data
                                        split = int(len(X) * 0.7)
                                        X_train = X.iloc[:split]
                                        X_test = X.iloc[split:]
                                        y_train = y.iloc[:split]
                                        
                                        # Train model
                                        model = create_model(model_type, max_features=25)
                                        train_metrics = model.fit(X_train, y_train.values)
                                        
                                        # Backtest
                                        config = {
                                            'symbol': symbol,
                                            'timeframe': timeframe,
                                            'horizon': horizon,
                                            'cost': cost,
                                            'model': model_type,
                                            'threshold': threshold,
                                            'stop_loss': stop_loss,
                                            'take_profit': take_profit,
                                            'max_hold_time': max_hold_time
                                        }
                                        
                                        result = self.enhanced_backtest(
                                            model, X_test, fdf.iloc[split:], feature_cols, config
                                        )
                                        
                                        if result:
                                            result['config'] = config
                                            result['train_metrics'] = train_metrics
                                            result['fitness_score'] = self.calculate_fitness_score(result)
                                            results.append(result)
                                    
                                    except Exception as e:
                                        continue
        
        # Sort by fitness score
        results.sort(key=lambda x: x['fitness_score'], reverse=True)
        
        print(f"✓ Completed {symbol} {timeframe}: {len(results)} successful configurations")
        if results:
            best = results[0]
            print(f"  Best: Return {best['total_return']:.2f}%, Sharpe {best['sharpe_ratio']:.2f}, Score {best['fitness_score']:.3f}")
        
        return results[:20]  # Return top 20 configurations
    
    def run_focused_optimization(self):
        """Run focused optimization on selected pairs."""
        print("🚀 FOCUSED OPTIMIZATION FOR SHORT-TERM TRADING")
        print("=" * 80)
        
        # Focus on most liquid pairs and short timeframes
        test_pairs = [
            ("BTCUSDT", "1m"),
            ("BTCUSDT", "3m"),
            ("BTCUSDT", "5m"),
            ("ETHUSDT", "1m"),
            ("ETHUSDT", "3m"),
            ("ETHUSDT", "5m"),
            ("ADAUSDT", "3m"),
            ("ADAUSDT", "5m"),
        ]
        
        all_results = []
        
        for symbol, timeframe in test_pairs:
            results = self.optimize_configuration(symbol, timeframe)
            all_results.extend(results)
        
        # Global ranking
        all_results.sort(key=lambda x: x['fitness_score'], reverse=True)
        
        return all_results
    
    def generate_optimization_report(self, results: List[Dict]) -> str:
        """Generate comprehensive optimization report."""
        if not results:
            return "❌ No successful configurations found!"
        
        report = []
        report.append("=" * 100)
        report.append("🎯 FOCUSED OPTIMIZATION REPORT - SHORT-TERM TRADING")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total successful configurations: {len(results)}")
        report.append("")
        
        # Top 10 configurations
        report.append("🏆 TOP 10 CONFIGURATIONS")
        report.append("-" * 80)
        
        for i, result in enumerate(results[:10], 1):
            config = result['config']
            report.append(f"{i}. {config['symbol']} {config['timeframe']} | {config['model']}")
            report.append(f"   Threshold: {config['threshold']}, Horizon: {config['horizon']}, Cost: {config['cost']}bp")
            report.append(f"   Stop Loss: {config['stop_loss']*100:.1f}%, Take Profit: {config['take_profit']*100:.1f}%")
            report.append(f"   📈 Return: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
            report.append(f"   📊 Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")
            report.append(f"   📉 Max DD: {result['max_drawdown']:.2f}% | Profit Factor: {result['profit_factor']:.2f}")
            report.append(f"   🎯 Fitness Score: {result['fitness_score']:.3f}")
            report.append("")
        
        # Analysis by timeframe
        timeframe_analysis = {}
        for result in results:
            tf = result['config']['timeframe']
            if tf not in timeframe_analysis:
                timeframe_analysis[tf] = []
            timeframe_analysis[tf].append(result)
        
        report.append("⏰ TIMEFRAME ANALYSIS")
        report.append("-" * 50)
        
        for tf in sorted(timeframe_analysis.keys()):
            tf_results = timeframe_analysis[tf]
            avg_return = np.mean([r['total_return'] for r in tf_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in tf_results])
            best_return = max([r['total_return'] for r in tf_results])
            
            report.append(f"{tf}: {len(tf_results)} configs | Avg Return: {avg_return:.2f}% | Best: {best_return:.2f}%")
        
        report.append("")
        
        # Symbol analysis
        symbol_analysis = {}
        for result in results:
            symbol = result['config']['symbol']
            if symbol not in symbol_analysis:
                symbol_analysis[symbol] = []
            symbol_analysis[symbol].append(result)
        
        report.append("💰 SYMBOL ANALYSIS")
        report.append("-" * 50)
        
        for symbol in sorted(symbol_analysis.keys()):
            symbol_results = symbol_analysis[symbol]
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            best_return = max([r['total_return'] for r in symbol_results])
            
            report.append(f"{symbol}: {len(symbol_results)} configs | Avg Return: {avg_return:.2f}% | Best: {best_return:.2f}%")
        
        report.append("")
        
        # Key insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        # Best timeframe
        best_tf_results = max(timeframe_analysis.items(), key=lambda x: np.mean([r['total_return'] for r in x[1]]))
        report.append(f"✅ BEST TIMEFRAME: {best_tf_results[0]} (avg return: {np.mean([r['total_return'] for r in best_tf_results[1]]):.2f}%)")
        
        # Best symbol
        best_symbol_results = max(symbol_analysis.items(), key=lambda x: np.mean([r['total_return'] for r in x[1]]))
        report.append(f"✅ BEST SYMBOL: {best_symbol_results[0]} (avg return: {np.mean([r['total_return'] for r in best_symbol_results[1]]):.2f}%)")
        
        # Model performance
        model_performance = {}
        for result in results[:50]:  # Top 50
            model = result['config']['model']
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(result['total_return'])
        
        best_model = max(model_performance.items(), key=lambda x: np.mean(x[1]))
        report.append(f"✅ BEST MODEL: {best_model[0]} (avg return in top 50: {np.mean(best_model[1]):.2f}%)")
        
        # Risk assessment
        top_10_dd = [r['max_drawdown'] for r in results[:10]]
        avg_dd = np.mean(top_10_dd)
        if avg_dd < 8:
            risk_level = "LOW"
        elif avg_dd < 15:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        report.append(f"⚠️  RISK LEVEL: {risk_level} (avg max drawdown in top 10: {avg_dd:.2f}%)")
        
        report.append("")
        
        # Implementation recommendations
        report.append("🚀 IMPLEMENTATION RECOMMENDATIONS")
        report.append("-" * 50)
        
        best_config = results[0]['config']
        report.append("1. RECOMMENDED CONFIGURATION:")
        report.append(f"   - Symbol: {best_config['symbol']}")
        report.append(f"   - Timeframe: {best_config['timeframe']}")
        report.append(f"   - Model: {best_config['model']}")
        report.append(f"   - Entry Threshold: {best_config['threshold']}")
        report.append(f"   - Stop Loss: {best_config['stop_loss']*100:.1f}%")
        report.append(f"   - Take Profit: {best_config['take_profit']*100:.1f}%")
        report.append(f"   - Max Hold Time: {best_config['max_hold_time']} candles")
        report.append("")
        
        report.append("2. RISK MANAGEMENT:")
        report.append("   - Start with small position sizes (1-2% of capital)")
        report.append("   - Use strict stop losses as configured")
        report.append("   - Monitor performance daily")
        report.append("   - Have a maximum daily loss limit (5% of capital)")
        report.append("")
        
        report.append("3. MONITORING:")
        report.append("   - Track win rate (should stay above 50%)")
        report.append("   - Monitor drawdown (stop if exceeds 10%)")
        report.append("   - Review performance weekly")
        report.append("   - Re-optimize monthly")
        
        return "\n".join(report)


def main():
    """Main optimization function."""
    print("🎯 FOCUSED OPTIMIZATION FOR SHORT-TERM TRADING PROFITABILITY")
    print("=" * 80)
    
    start_time = time.time()
    
    optimizer = FocusedOptimizer()
    results = optimizer.run_focused_optimization()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Optimization completed in {duration/60:.1f} minutes")
    
    # Generate report
    report = optimizer.generate_optimization_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    with open(f"focused_optimization_results_{timestamp}.json", "w") as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration / 60
        }, f, indent=2, default=str)
    
    # Save report
    with open(f"FOCUSED_OPTIMIZATION_REPORT_{timestamp}.md", "w") as f:
        f.write(report)
    
    print(f"\n📁 Results saved:")
    print(f"   - focused_optimization_results_{timestamp}.json")
    print(f"   - FOCUSED_OPTIMIZATION_REPORT_{timestamp}.md")
    
    if results:
        best = results[0]
        print(f"\n🎉 BEST CONFIGURATION FOUND:")
        print(f"   {best['config']['symbol']} {best['config']['timeframe']} | {best['config']['model']}")
        print(f"   Return: {best['total_return']:.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.2f}%")
    
    return results


if __name__ == "__main__":
    results = main()