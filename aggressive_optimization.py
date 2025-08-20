#!/usr/bin/env python3
"""
Aggressive optimization for higher profitability.
Focus on realistic trading scenarios with better parameters.
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


class AggressiveOptimizer:
    """Aggressive optimization for maximum profitability."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        
    def fetch_extended_data(self, symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
        """Fetch extended data for better optimization."""
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 500:
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def aggressive_backtest(self, model, X_test, price_data, feature_cols, config):
        """Aggressive backtesting with realistic position sizing."""
        try:
            threshold = config['threshold']
            stop_loss = config.get('stop_loss', 0.03)
            take_profit = config.get('take_profit', 0.06)
            max_hold = config.get('max_hold', 20)
            position_size = config.get('position_size', 0.2)  # 20% of capital per trade
            
            initial_capital = 10000.0
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            entry_price = 0
            entry_time = 0
            
            for i, (_, row) in enumerate(X_test.iterrows()):
                current_price = price_data.iloc[i]['close']
                
                # Current equity
                current_equity = capital + (position * current_price if position > 0 else 0)
                equity_curve.append(current_equity)
                
                # Get prediction
                features = row[feature_cols].values.reshape(1, -1)
                features_df = pd.DataFrame(features, columns=feature_cols)
                prob = model.predict_proba(features_df)[0]
                
                if position == 0 and prob > threshold:
                    # Enter position with aggressive sizing
                    position_value = capital * position_size
                    position = position_value / current_price * 0.997  # Account for fees/slippage
                    capital -= position_value
                    entry_price = current_price
                    entry_time = i
                    
                elif position > 0:
                    # Check exit conditions
                    price_change = (current_price - entry_price) / entry_price
                    hold_time = i - entry_time
                    
                    # More aggressive exit conditions
                    should_exit = (
                        prob < (threshold - 0.1) or  # Stronger reversal signal
                        price_change <= -stop_loss or
                        price_change >= take_profit or
                        hold_time >= max_hold or
                        i == len(X_test) - 1
                    )
                    
                    if should_exit:
                        # Exit position
                        exit_value = position * current_price * 0.997  # Account for fees/slippage
                        pnl = exit_value - (position * entry_price)
                        pnl_pct = (exit_value / (position * entry_price) - 1) * 100
                        
                        # Determine exit reason
                        exit_reason = "signal"
                        if price_change <= -stop_loss:
                            exit_reason = "stop_loss"
                        elif price_change >= take_profit:
                            exit_reason = "take_profit"
                        elif hold_time >= max_hold:
                            exit_reason = "time_limit"
                        
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_time': hold_time,
                            'exit_reason': exit_reason,
                            'position_size_pct': position_size * 100
                        })
                        
                        capital += exit_value
                        position = 0
            
            if not trades:
                return None
            
            # Calculate comprehensive metrics
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
                # Adjust for timeframe
                periods_per_year = {
                    '1m': 365 * 24 * 60,
                    '3m': 365 * 24 * 20,
                    '5m': 365 * 24 * 12,
                    '15m': 365 * 24 * 4
                }
                periods = periods_per_year.get(config.get('timeframe', '5m'), 365 * 24 * 12)
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(periods)
            else:
                sharpe_ratio = 0
            
            # Drawdown calculation
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown = np.max(drawdowns)
            
            # Additional metrics
            avg_hold_time = np.mean([t['hold_time'] for t in trades])
            
            # Exit reason breakdown
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Calculate expectancy
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            
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
                'expectancy': expectancy,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }
            
        except Exception as e:
            return None
    
    def calculate_aggressive_score(self, result: Dict) -> float:
        """Calculate aggressive optimization score."""
        if not result:
            return 0
        
        # Base score from return (more weight on high returns)
        return_score = min(result['total_return'] / 30, 2.0)  # Up to 2.0 for 30%+ returns
        
        # Sharpe bonus (higher weight)
        sharpe_bonus = min(result['sharpe_ratio'] / 1.5, 1.0)
        
        # Win rate bonus
        win_rate_bonus = max(0, (result['win_rate'] - 45) / 50)  # Bonus above 45%
        
        # Profit factor bonus
        pf_bonus = min((result['profit_factor'] - 1) / 1.5, 0.8) if result['profit_factor'] > 1 else 0
        
        # Expectancy bonus
        expectancy_bonus = min(result['expectancy'] / 100, 0.5) if result['expectancy'] > 0 else 0
        
        # Trade frequency bonus (more trades = more opportunities)
        trade_bonus = min(result['total_trades'] / 50, 0.3)
        
        # Penalties
        penalty = 0
        if result['total_trades'] < 5:
            penalty += 0.5  # Too few trades
        if result['max_drawdown'] > 25:
            penalty += 0.3  # Too much drawdown
        if result['total_return'] < 0:
            penalty += 2.0  # Negative returns
        if result['win_rate'] < 35:
            penalty += 0.4  # Very low win rate
        
        score = (return_score + sharpe_bonus + win_rate_bonus + 
                pf_bonus + expectancy_bonus + trade_bonus - penalty)
        
        return max(0, score)
    
    def test_aggressive_config(self, symbol: str, timeframe: str, config: Dict) -> Dict:
        """Test aggressive configuration."""
        try:
            # Fetch more data for better training
            df = self.fetch_extended_data(symbol, timeframe, limit=2000)
            if df.empty or len(df) < 800:
                return None
            
            # Build features with advanced indicators
            fdf = build_features(df, advanced=True)
            if fdf.empty or len(fdf) < 400:
                return None
            
            # Build labels with lower cost (more aggressive)
            y = build_labels(fdf, horizon=config['horizon'], cost_bp=config['cost'])
            if len(y) < 100:
                return None
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            # Use more training data
            split = int(len(X) * 0.8)
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            
            # Train model with more features
            model = create_model(config['model'], max_features=min(40, len(feature_cols)))
            train_metrics = model.fit(X_train, y_train.values)
            
            # Aggressive backtest
            config_with_tf = config.copy()
            config_with_tf['timeframe'] = timeframe
            
            result = self.aggressive_backtest(model, X_test, fdf.iloc[split:], feature_cols, config_with_tf)
            
            if result:
                result['config'] = config
                result['symbol'] = symbol
                result['timeframe'] = timeframe
                result['train_metrics'] = train_metrics
                result['score'] = self.calculate_aggressive_score(result)
                
            return result
            
        except Exception as e:
            return None
    
    def run_aggressive_optimization(self):
        """Run aggressive optimization."""
        print("🔥 AGGRESSIVE OPTIMIZATION FOR MAXIMUM PROFITABILITY")
        print("=" * 70)
        
        # Aggressive configurations focused on profitability
        aggressive_configs = [
            # Ultra short-term high frequency
            {"horizon": 1, "cost": 5, "model": "xgboost", "threshold": 0.55, "stop_loss": 0.02, "take_profit": 0.05, "max_hold": 15, "position_size": 0.25},
            {"horizon": 1, "cost": 5, "model": "lightgbm", "threshold": 0.6, "stop_loss": 0.025, "take_profit": 0.06, "max_hold": 20, "position_size": 0.3},
            {"horizon": 1, "cost": 8, "model": "ensemble", "threshold": 0.65, "stop_loss": 0.03, "take_profit": 0.07, "max_hold": 25, "position_size": 0.2},
            
            # Short-term momentum
            {"horizon": 2, "cost": 8, "model": "xgboost", "threshold": 0.6, "stop_loss": 0.025, "take_profit": 0.06, "max_hold": 20, "position_size": 0.25},
            {"horizon": 2, "cost": 10, "model": "lightgbm", "threshold": 0.65, "stop_loss": 0.03, "take_profit": 0.08, "max_hold": 30, "position_size": 0.3},
            {"horizon": 2, "cost": 12, "model": "ensemble", "threshold": 0.7, "stop_loss": 0.035, "take_profit": 0.09, "max_hold": 35, "position_size": 0.25},
            
            # Medium-term swing
            {"horizon": 3, "cost": 10, "model": "xgboost", "threshold": 0.65, "stop_loss": 0.03, "take_profit": 0.08, "max_hold": 40, "position_size": 0.3},
            {"horizon": 3, "cost": 12, "model": "lightgbm", "threshold": 0.7, "stop_loss": 0.035, "take_profit": 0.1, "max_hold": 50, "position_size": 0.35},
            {"horizon": 3, "cost": 15, "model": "ensemble", "threshold": 0.75, "stop_loss": 0.04, "take_profit": 0.12, "max_hold": 60, "position_size": 0.3},
            
            # Longer-term trend following
            {"horizon": 5, "cost": 15, "model": "xgboost", "threshold": 0.7, "stop_loss": 0.04, "take_profit": 0.12, "max_hold": 80, "position_size": 0.4},
            {"horizon": 5, "cost": 18, "model": "lightgbm", "threshold": 0.75, "stop_loss": 0.045, "take_profit": 0.15, "max_hold": 100, "position_size": 0.35},
            {"horizon": 5, "cost": 20, "model": "ensemble", "threshold": 0.8, "stop_loss": 0.05, "take_profit": 0.18, "max_hold": 120, "position_size": 0.3},
        ]
        
        # Test pairs with focus on high volatility
        test_pairs = [
            ("BTCUSDT", "1m"),
            ("BTCUSDT", "3m"),
            ("BTCUSDT", "5m"),
            ("ETHUSDT", "1m"),
            ("ETHUSDT", "3m"),
            ("ETHUSDT", "5m"),
            ("ADAUSDT", "3m"),
            ("ADAUSDT", "5m"),
            ("BNBUSDT", "5m"),
            ("SOLUSDT", "5m"),
        ]
        
        print(f"Testing {len(aggressive_configs)} configs on {len(test_pairs)} pairs...")
        print(f"Total combinations: {len(aggressive_configs) * len(test_pairs)}")
        
        all_results = []
        total_tests = len(aggressive_configs) * len(test_pairs)
        completed = 0
        
        for symbol, timeframe in test_pairs:
            print(f"\n🎯 Testing {symbol} {timeframe}...")
            
            for config in aggressive_configs:
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total_tests} ({completed/total_tests*100:.1f}%)")
                
                result = self.test_aggressive_config(symbol, timeframe, config)
                if result and result['total_return'] > 0:  # Only keep profitable results
                    all_results.append(result)
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n✅ Completed! Found {len(all_results)} profitable configurations")
        
        return all_results
    
    def generate_aggressive_report(self, results: List[Dict]) -> str:
        """Generate aggressive optimization report."""
        if not results:
            return "❌ No profitable configurations found!"
        
        report = []
        report.append("=" * 100)
        report.append("🔥 AGGRESSIVE OPTIMIZATION REPORT - MAXIMUM PROFITABILITY")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Profitable configurations found: {len(results)}")
        report.append("")
        
        # Top 15 results for better analysis
        report.append("🏆 TOP 15 PROFITABLE CONFIGURATIONS")
        report.append("-" * 80)
        
        for i, result in enumerate(results[:15], 1):
            config = result['config']
            report.append(f"{i}. {result['symbol']} {result['timeframe']} | {config['model']}")
            report.append(f"   Horizon: {config['horizon']}, Cost: {config['cost']}bp, Threshold: {config['threshold']}")
            report.append(f"   Position Size: {config['position_size']*100:.0f}%, Stop: {config['stop_loss']*100:.1f}%, Take: {config['take_profit']*100:.1f}%")
            report.append(f"   📈 Return: {result['total_return']:.2f}% | Sharpe: {result['sharpe_ratio']:.2f}")
            report.append(f"   📊 Trades: {result['total_trades']} | Win Rate: {result['win_rate']:.1f}%")
            report.append(f"   💰 Profit Factor: {result['profit_factor']:.2f} | Expectancy: ${result['expectancy']:.2f}")
            report.append(f"   📉 Max DD: {result['max_drawdown']:.2f}% | Avg Hold: {result['avg_hold_time']:.1f}")
            report.append(f"   🎯 Score: {result['score']:.3f}")
            report.append("")
        
        # Performance analysis
        if results:
            returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            win_rates = [r['win_rate'] for r in results]
            
            report.append("📊 PERFORMANCE STATISTICS")
            report.append("-" * 50)
            report.append(f"Average Return: {np.mean(returns):.2f}%")
            report.append(f"Best Return: {np.max(returns):.2f}%")
            report.append(f"Average Sharpe: {np.mean(sharpe_ratios):.2f}")
            report.append(f"Average Win Rate: {np.mean(win_rates):.1f}%")
            report.append(f"Configs with >10% return: {len([r for r in results if r['total_return'] > 10])}")
            report.append(f"Configs with >20% return: {len([r for r in results if r['total_return'] > 20])}")
            report.append("")
        
        # Timeframe analysis
        tf_analysis = {}
        for result in results:
            tf = result['timeframe']
            if tf not in tf_analysis:
                tf_analysis[tf] = []
            tf_analysis[tf].append(result)
        
        report.append("⏰ TIMEFRAME ANALYSIS")
        report.append("-" * 50)
        for tf in sorted(tf_analysis.keys()):
            tf_results = tf_analysis[tf]
            avg_return = np.mean([r['total_return'] for r in tf_results])
            best_return = np.max([r['total_return'] for r in tf_results])
            avg_trades = np.mean([r['total_trades'] for r in tf_results])
            
            report.append(f"{tf}: {len(tf_results)} configs | Avg: {avg_return:.2f}% | Best: {best_return:.2f}% | Avg Trades: {avg_trades:.1f}")
        
        report.append("")
        
        # Symbol analysis
        symbol_analysis = {}
        for result in results:
            symbol = result['symbol']
            if symbol not in symbol_analysis:
                symbol_analysis[symbol] = []
            symbol_analysis[symbol].append(result)
        
        report.append("💰 SYMBOL ANALYSIS")
        report.append("-" * 50)
        for symbol in sorted(symbol_analysis.keys()):
            symbol_results = symbol_analysis[symbol]
            avg_return = np.mean([r['total_return'] for r in symbol_results])
            best_return = np.max([r['total_return'] for r in symbol_results])
            
            report.append(f"{symbol}: {len(symbol_results)} configs | Avg: {avg_return:.2f}% | Best: {best_return:.2f}%")
        
        report.append("")
        
        # Model analysis
        model_analysis = {}
        for result in results:
            model = result['config']['model']
            if model not in model_analysis:
                model_analysis[model] = []
            model_analysis[model].append(result)
        
        report.append("🤖 MODEL ANALYSIS")
        report.append("-" * 50)
        for model in sorted(model_analysis.keys()):
            model_results = model_analysis[model]
            avg_return = np.mean([r['total_return'] for r in model_results])
            best_return = np.max([r['total_return'] for r in model_results])
            
            report.append(f"{model}: {len(model_results)} configs | Avg: {avg_return:.2f}% | Best: {best_return:.2f}%")
        
        report.append("")
        
        # Key insights
        report.append("💡 KEY INSIGHTS")
        report.append("-" * 50)
        
        if results:
            # Best performing combinations
            best_tf = max(tf_analysis.items(), key=lambda x: np.mean([r['total_return'] for r in x[1]]))
            best_symbol = max(symbol_analysis.items(), key=lambda x: np.mean([r['total_return'] for r in x[1]]))
            best_model = max(model_analysis.items(), key=lambda x: np.mean([r['total_return'] for r in x[1]]))
            
            report.append(f"🎯 BEST TIMEFRAME: {best_tf[0]} (avg: {np.mean([r['total_return'] for r in best_tf[1]]):.2f}%)")
            report.append(f"🎯 BEST SYMBOL: {best_symbol[0]} (avg: {np.mean([r['total_return'] for r in best_symbol[1]]):.2f}%)")
            report.append(f"🎯 BEST MODEL: {best_model[0]} (avg: {np.mean([r['total_return'] for r in best_model[1]]):.2f}%)")
            
            # Risk assessment
            high_return_configs = [r for r in results if r['total_return'] > 15]
            if high_return_configs:
                avg_dd_high_return = np.mean([r['max_drawdown'] for r in high_return_configs])
                report.append(f"⚠️  HIGH RETURN RISK: Avg drawdown for >15% return configs: {avg_dd_high_return:.2f}%")
            
            # Trading frequency
            high_freq_configs = [r for r in results if r['total_trades'] > 20]
            if high_freq_configs:
                avg_return_high_freq = np.mean([r['total_return'] for r in high_freq_configs])
                report.append(f"📈 HIGH FREQUENCY: Avg return for >20 trades configs: {avg_return_high_freq:.2f}%")
        
        report.append("")
        
        # Implementation recommendations
        if results:
            best = results[0]
            report.append("🚀 IMPLEMENTATION RECOMMENDATIONS")
            report.append("-" * 50)
            report.append("TOP CONFIGURATION FOR LIVE TRADING:")
            report.append(f"Symbol: {best['symbol']}")
            report.append(f"Timeframe: {best['timeframe']}")
            report.append(f"Model: {best['config']['model']}")
            report.append(f"Entry Threshold: {best['config']['threshold']}")
            report.append(f"Position Size: {best['config']['position_size']*100:.0f}% of capital")
            report.append(f"Stop Loss: {best['config']['stop_loss']*100:.1f}%")
            report.append(f"Take Profit: {best['config']['take_profit']*100:.1f}%")
            report.append(f"Max Hold Time: {best['config']['max_hold']} candles")
            report.append("")
            report.append(f"EXPECTED PERFORMANCE:")
            report.append(f"Return: {best['total_return']:.2f}%")
            report.append(f"Win Rate: {best['win_rate']:.1f}%")
            report.append(f"Profit Factor: {best['profit_factor']:.2f}")
            report.append(f"Max Drawdown: {best['max_drawdown']:.2f}%")
            report.append("")
            
            report.append("RISK MANAGEMENT RULES:")
            report.append("1. Start with 50% of recommended position size")
            report.append("2. Increase position size gradually as confidence builds")
            report.append("3. Set daily loss limit at 5% of capital")
            report.append("4. Stop trading if drawdown exceeds 15%")
            report.append("5. Review and re-optimize weekly")
            report.append("")
            
            # Alternative configurations
            report.append("ALTERNATIVE CONFIGURATIONS (Top 3):")
            for i, alt in enumerate(results[1:4], 2):
                report.append(f"{i}. {alt['symbol']} {alt['timeframe']} | Return: {alt['total_return']:.2f}% | DD: {alt['max_drawdown']:.2f}%")
        
        return "\n".join(report)


def main():
    """Main function."""
    print("🔥 AGGRESSIVE OPTIMIZATION FOR MAXIMUM PROFITABILITY")
    print("=" * 70)
    
    start_time = time.time()
    
    optimizer = AggressiveOptimizer()
    results = optimizer.run_aggressive_optimization()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Optimization completed in {duration/60:.1f} minutes")
    
    # Generate report
    report = optimizer.generate_aggressive_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"aggressive_optimization_results_{timestamp}.json", "w") as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration / 60,
            'total_configs_tested': len(results)
        }, f, indent=2, default=str)
    
    with open(f"AGGRESSIVE_OPTIMIZATION_REPORT_{timestamp}.md", "w") as f:
        f.write(report)
    
    print(f"\n📁 Results saved:")
    print(f"   - aggressive_optimization_results_{timestamp}.json")
    print(f"   - AGGRESSIVE_OPTIMIZATION_REPORT_{timestamp}.md")
    
    if results:
        best = results[0]
        print(f"\n🎉 BEST CONFIGURATION FOUND:")
        print(f"   {best['symbol']} {best['timeframe']} | {best['config']['model']}")
        print(f"   Return: {best['total_return']:.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1f}% | Max DD: {best['max_drawdown']:.2f}%")
        print(f"   Trades: {best['total_trades']} | Score: {best['score']:.3f}")
    else:
        print("\n❌ No profitable configurations found. Consider:")
        print("   - Adjusting parameters")
        print("   - Using different timeframes")
        print("   - Trying different symbols")
    
    return results


if __name__ == "__main__":
    results = main()