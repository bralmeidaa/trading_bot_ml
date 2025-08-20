#!/usr/bin/env python3
"""
Quick and efficient optimization for short-term trading.
Smart parameter selection with reduced search space.
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


class QuickOptimizer:
    """Quick optimization with smart parameter selection."""
    
    def __init__(self):
        self.client = BinancePublicClient()
        
    def fetch_data(self, symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
        """Fetch data efficiently."""
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 300:
                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def quick_backtest(self, model, X_test, price_data, feature_cols, config):
        """Quick backtesting with essential metrics."""
        try:
            threshold = config['threshold']
            stop_loss = config.get('stop_loss', 0.02)
            take_profit = config.get('take_profit', 0.04)
            
            capital = 10000.0
            position = 0
            trades = []
            
            for i, (_, row) in enumerate(X_test.iterrows()):
                current_price = price_data.iloc[i]['close']
                
                # Get prediction (simplified)
                features = row[feature_cols].values.reshape(1, -1)
                features_df = pd.DataFrame(features, columns=feature_cols)
                prob = model.predict_proba(features_df)[0]
                
                if position == 0 and prob > threshold:
                    # Enter position
                    position = (capital * 0.1) / current_price * 0.998
                    entry_price = current_price
                    entry_idx = i
                    
                elif position > 0:
                    # Check exit conditions
                    price_change = (current_price - entry_price) / entry_price
                    hold_time = i - entry_idx
                    
                    if (prob < (1 - threshold) or 
                        price_change <= -stop_loss or 
                        price_change >= take_profit or 
                        hold_time >= 30 or 
                        i == len(X_test) - 1):
                        
                        # Exit position
                        pnl = position * (current_price - entry_price) * 0.998
                        trades.append({
                            'pnl': pnl,
                            'pnl_pct': (current_price / entry_price - 1) * 100,
                            'hold_time': hold_time
                        })
                        position = 0
            
            if not trades:
                return None
            
            # Calculate metrics
            total_pnl = sum(t['pnl'] for t in trades)
            total_return = (total_pnl / 10000) * 100
            
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] < 0])
            avg_loss = avg_loss if not np.isnan(avg_loss) else 0
            
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            return {
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            return None
    
    def test_configuration(self, symbol: str, timeframe: str, config: Dict) -> Dict:
        """Test a single configuration quickly."""
        try:
            # Fetch data
            df = self.fetch_data(symbol, timeframe)
            if df.empty:
                return None
            
            # Build features (basic only for speed)
            fdf = build_features(df, advanced=False)
            if fdf.empty:
                return None
            
            # Build labels
            y = build_labels(fdf, horizon=config['horizon'], cost_bp=config['cost'])
            if len(y) == 0:
                return None
            
            # Prepare data
            feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
            X = fdf[feature_cols]
            
            if len(X) < 200:
                return None
            
            # Split data
            split = int(len(X) * 0.75)  # More training data
            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            
            # Train model
            model = create_model(config['model'], max_features=20)  # Fewer features for speed
            model.fit(X_train, y_train.values)
            
            # Backtest
            result = self.quick_backtest(model, X_test, fdf.iloc[split:], feature_cols, config)
            
            if result:
                result['config'] = config
                result['symbol'] = symbol
                result['timeframe'] = timeframe
                
                # Calculate simple score
                score = 0
                if result['total_return'] > 0:
                    score += result['total_return'] / 20  # Normalize
                if result['win_rate'] > 50:
                    score += (result['win_rate'] - 50) / 50
                if result['profit_factor'] > 1:
                    score += min((result['profit_factor'] - 1), 1)
                if result['total_trades'] >= 10:
                    score += 0.2
                
                result['score'] = max(0, score)
                
            return result
            
        except Exception as e:
            return None
    
    def run_quick_optimization(self):
        """Run quick optimization on key configurations."""
        print("⚡ QUICK OPTIMIZATION FOR SHORT-TERM TRADING")
        print("=" * 60)
        
        # Focus on most promising combinations
        test_configs = [
            # High-frequency configs
            {"symbol": "BTCUSDT", "timeframe": "1m", "horizon": 1, "cost": 10, "model": "xgboost", "threshold": 0.65, "stop_loss": 0.015, "take_profit": 0.03},
            {"symbol": "BTCUSDT", "timeframe": "1m", "horizon": 2, "cost": 10, "model": "lightgbm", "threshold": 0.6, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "BTCUSDT", "timeframe": "3m", "horizon": 1, "cost": 8, "model": "xgboost", "threshold": 0.7, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "BTCUSDT", "timeframe": "3m", "horizon": 2, "cost": 12, "model": "ensemble", "threshold": 0.65, "stop_loss": 0.025, "take_profit": 0.05},
            {"symbol": "BTCUSDT", "timeframe": "5m", "horizon": 2, "cost": 10, "model": "xgboost", "threshold": 0.6, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "BTCUSDT", "timeframe": "5m", "horizon": 3, "cost": 15, "model": "lightgbm", "threshold": 0.65, "stop_loss": 0.025, "take_profit": 0.05},
            
            {"symbol": "ETHUSDT", "timeframe": "1m", "horizon": 1, "cost": 8, "model": "xgboost", "threshold": 0.7, "stop_loss": 0.015, "take_profit": 0.03},
            {"symbol": "ETHUSDT", "timeframe": "3m", "horizon": 2, "cost": 10, "model": "lightgbm", "threshold": 0.65, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "ETHUSDT", "timeframe": "5m", "horizon": 2, "cost": 12, "model": "ensemble", "threshold": 0.6, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "ETHUSDT", "timeframe": "5m", "horizon": 3, "cost": 15, "model": "xgboost", "threshold": 0.65, "stop_loss": 0.025, "take_profit": 0.05},
            
            {"symbol": "ADAUSDT", "timeframe": "3m", "horizon": 2, "cost": 12, "model": "lightgbm", "threshold": 0.65, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "ADAUSDT", "timeframe": "5m", "horizon": 3, "cost": 15, "model": "xgboost", "threshold": 0.6, "stop_loss": 0.025, "take_profit": 0.05},
            
            {"symbol": "BNBUSDT", "timeframe": "5m", "horizon": 2, "cost": 10, "model": "ensemble", "threshold": 0.65, "stop_loss": 0.02, "take_profit": 0.04},
            {"symbol": "SOLUSDT", "timeframe": "5m", "horizon": 3, "cost": 15, "model": "lightgbm", "threshold": 0.6, "stop_loss": 0.025, "take_profit": 0.05},
        ]
        
        # Add variations of best performing configs
        variations = []
        base_configs = test_configs[:6]  # Take first 6 as base
        
        for base in base_configs:
            # Threshold variations
            for threshold in [0.55, 0.6, 0.65, 0.7, 0.75]:
                if threshold != base['threshold']:
                    var = base.copy()
                    var['threshold'] = threshold
                    variations.append(var)
            
            # Stop loss variations
            for sl in [0.015, 0.02, 0.025, 0.03]:
                if sl != base['stop_loss']:
                    var = base.copy()
                    var['stop_loss'] = sl
                    variations.append(var)
        
        all_configs = test_configs + variations[:30]  # Limit variations
        
        print(f"Testing {len(all_configs)} configurations...")
        
        results = []
        for i, config in enumerate(all_configs):
            if i % 5 == 0:
                print(f"Progress: {i}/{len(all_configs)} ({i/len(all_configs)*100:.1f}%)")
            
            result = self.test_configuration(config['symbol'], config['timeframe'], config)
            if result:
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n✅ Completed! Found {len(results)} successful configurations")
        
        return results
    
    def generate_quick_report(self, results: List[Dict]) -> str:
        """Generate quick optimization report."""
        if not results:
            return "❌ No successful configurations found!"
        
        report = []
        report.append("=" * 80)
        report.append("⚡ QUICK OPTIMIZATION REPORT - SHORT-TERM TRADING")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Successful configurations: {len(results)}")
        report.append("")
        
        # Top 10 results
        report.append("🏆 TOP 10 CONFIGURATIONS")
        report.append("-" * 60)
        
        for i, result in enumerate(results[:10], 1):
            config = result['config']
            report.append(f"{i}. {result['symbol']} {result['timeframe']} | {config['model']}")
            report.append(f"   Threshold: {config['threshold']}, Horizon: {config['horizon']}")
            report.append(f"   Stop: {config['stop_loss']*100:.1f}%, Take: {config['take_profit']*100:.1f}%")
            report.append(f"   📈 Return: {result['total_return']:.2f}% | Trades: {result['total_trades']}")
            report.append(f"   📊 Win Rate: {result['win_rate']:.1f}% | PF: {result['profit_factor']:.2f}")
            report.append(f"   🎯 Score: {result['score']:.3f}")
            report.append("")
        
        # Quick analysis
        profitable_configs = [r for r in results if r['total_return'] > 0]
        high_win_rate = [r for r in results if r['win_rate'] > 60]
        
        report.append("📊 QUICK ANALYSIS")
        report.append("-" * 40)
        report.append(f"Profitable configs: {len(profitable_configs)}/{len(results)} ({len(profitable_configs)/len(results)*100:.1f}%)")
        report.append(f"High win rate (>60%): {len(high_win_rate)}")
        
        if profitable_configs:
            avg_return = np.mean([r['total_return'] for r in profitable_configs])
            best_return = max([r['total_return'] for r in profitable_configs])
            report.append(f"Average return (profitable): {avg_return:.2f}%")
            report.append(f"Best return: {best_return:.2f}%")
        
        report.append("")
        
        # Timeframe analysis
        tf_analysis = {}
        for result in results:
            tf = result['timeframe']
            if tf not in tf_analysis:
                tf_analysis[tf] = []
            tf_analysis[tf].append(result)
        
        report.append("⏰ TIMEFRAME PERFORMANCE")
        report.append("-" * 40)
        for tf in sorted(tf_analysis.keys()):
            tf_results = tf_analysis[tf]
            profitable = [r for r in tf_results if r['total_return'] > 0]
            if tf_results:
                avg_return = np.mean([r['total_return'] for r in profitable]) if profitable else 0
                report.append(f"{tf}: {len(profitable)}/{len(tf_results)} profitable (avg: {avg_return:.2f}%)")
        
        report.append("")
        
        # Symbol analysis
        symbol_analysis = {}
        for result in results:
            symbol = result['symbol']
            if symbol not in symbol_analysis:
                symbol_analysis[symbol] = []
            symbol_analysis[symbol].append(result)
        
        report.append("💰 SYMBOL PERFORMANCE")
        report.append("-" * 40)
        for symbol in sorted(symbol_analysis.keys()):
            symbol_results = symbol_analysis[symbol]
            profitable = [r for r in symbol_results if r['total_return'] > 0]
            if symbol_results:
                avg_return = np.mean([r['total_return'] for r in profitable]) if profitable else 0
                report.append(f"{symbol}: {len(profitable)}/{len(symbol_results)} profitable (avg: {avg_return:.2f}%)")
        
        report.append("")
        
        # Recommendations
        if results:
            best = results[0]
            report.append("🚀 IMPLEMENTATION RECOMMENDATION")
            report.append("-" * 40)
            report.append("BEST CONFIGURATION:")
            report.append(f"Symbol: {best['symbol']}")
            report.append(f"Timeframe: {best['timeframe']}")
            report.append(f"Model: {best['config']['model']}")
            report.append(f"Entry Threshold: {best['config']['threshold']}")
            report.append(f"Stop Loss: {best['config']['stop_loss']*100:.1f}%")
            report.append(f"Take Profit: {best['config']['take_profit']*100:.1f}%")
            report.append(f"Expected Return: {best['total_return']:.2f}%")
            report.append(f"Win Rate: {best['win_rate']:.1f}%")
            report.append("")
            
            report.append("RISK MANAGEMENT:")
            report.append("- Start with 1-2% position sizes")
            report.append("- Use the configured stop loss strictly")
            report.append("- Monitor performance daily")
            report.append("- Set maximum daily loss limit (3-5%)")
            report.append("- Re-evaluate weekly")
        
        return "\n".join(report)


def main():
    """Main function."""
    print("⚡ QUICK OPTIMIZATION FOR SHORT-TERM TRADING")
    print("=" * 60)
    
    start_time = time.time()
    
    optimizer = QuickOptimizer()
    results = optimizer.run_quick_optimization()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Optimization completed in {duration/60:.1f} minutes")
    
    # Generate report
    report = optimizer.generate_quick_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"quick_optimization_results_{timestamp}.json", "w") as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration / 60
        }, f, indent=2, default=str)
    
    with open(f"QUICK_OPTIMIZATION_REPORT_{timestamp}.md", "w") as f:
        f.write(report)
    
    print(f"\n📁 Results saved:")
    print(f"   - quick_optimization_results_{timestamp}.json")
    print(f"   - QUICK_OPTIMIZATION_REPORT_{timestamp}.md")
    
    return results


if __name__ == "__main__":
    results = main()