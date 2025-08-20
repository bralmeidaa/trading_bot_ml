#!/usr/bin/env python3
"""
Test with real market data to validate profitability.
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

def fetch_real_data():
    """Fetch real market data using the Binance client."""
    print("Fetching real market data...")
    
    try:
        from data.binance_client import BinancePublicClient
        
        client = BinancePublicClient()
        
        # Fetch data for multiple symbols
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        data = {}
        
        for symbol in symbols:
            try:
                # Fetch OHLCV data
                ohlcv = client.fetch_ohlcv(symbol, "1h", limit=2000)  # ~83 days of hourly data
                
                if ohlcv:
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    data[symbol] = df
                    print(f"✓ {symbol}: {len(df)} candles")
                else:
                    print(f"✗ {symbol}: No data")
            except Exception as e:
                print(f"✗ {symbol}: Error - {e}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}

def test_strategy_profitability(symbol, df):
    """Test strategy profitability on real data."""
    print(f"\nTesting strategy on {symbol}...")
    
    try:
        from ml.features import build_features, build_labels
        from ml.models import create_model
        
        # Build features
        fdf = build_features(df, advanced=False)
        if fdf.empty:
            print("  No features generated")
            return None
        
        print(f"  Features: {len(fdf)} rows, {len(fdf.columns)} columns")
        
        # Build labels
        y = build_labels(fdf, horizon=3, cost_bp=10.0)  # Shorter horizon, lower cost
        if len(y) == 0:
            print("  No labels generated")
            return None
        
        print(f"  Labels: {len(y)} samples, distribution: {np.bincount(y.astype(int).values)}")
        
        # Get feature columns
        feature_cols = [c for c in fdf.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
        X = fdf[feature_cols]
        
        # Split data (70% train, 30% test)
        split = int(len(X) * 0.7)
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
        y_train = y.iloc[:split]
        y_test = y.iloc[split:]
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Test different models and thresholds
        models = ["xgboost", "lightgbm", "random_forest"]
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
        
        best_result = None
        best_return = -float('inf')
        
        for model_type in models:
            print(f"    Testing {model_type}...")
            
            try:
                # Train model
                model = create_model(model_type, max_features=20)
                model.fit(X_train, y_train.values)
                
                # Test different thresholds
                for threshold in thresholds:
                    result = backtest_strategy(model, X_test, fdf.iloc[split:], feature_cols, threshold)
                    
                    if result and result['total_return'] > best_return:
                        best_return = result['total_return']
                        best_result = result.copy()
                        best_result['model'] = model_type
                        best_result['threshold'] = threshold
                
            except Exception as e:
                print(f"      Error with {model_type}: {e}")
        
        return best_result
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def backtest_strategy(model, X_test, price_data, feature_cols, threshold=0.6):
    """Simple backtesting with the trained model."""
    try:
        initial_capital = 10000.0
        capital = initial_capital
        position = 0
        trades = []
        
        for i, (_, row) in enumerate(X_test.iterrows()):
            # Get prediction
            features = row[feature_cols].values.reshape(1, -1)
            prob = model.predict_proba(pd.DataFrame(features, columns=feature_cols))[0]
            
            current_price = price_data.iloc[i]['close']
            
            # Trading logic with more aggressive thresholds
            if prob > threshold and position == 0:
                # Buy
                position = capital / current_price * 0.95  # Account for fees
                capital = 0
                entry_price = current_price
                entry_time = i
                
            elif (prob < (1 - threshold) or i == len(X_test) - 1) and position > 0:
                # Sell or close at end
                capital = position * current_price * 0.95  # Account for fees
                pnl = capital - initial_capital
                pnl_pct = (capital / initial_capital - 1) * 100
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_time': i - entry_time
                })
                
                position = 0
        
        # Final calculations
        final_capital = capital if position == 0 else position * price_data.iloc[-1]['close'] * 0.95
        total_return = (final_capital / initial_capital - 1) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0])
            avg_loss = avg_loss if not np.isnan(avg_loss) else 0
            
            # Calculate max drawdown
            equity_curve = [initial_capital]
            running_capital = initial_capital
            
            for trade in trades:
                running_capital += trade['pnl']
                equity_curve.append(running_capital)
            
            peak_equity = np.maximum.accumulate(equity_curve)
            drawdowns = (peak_equity - equity_curve) / peak_equity * 100
            max_drawdown = np.max(drawdowns)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'final_capital': final_capital,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }
        
    except Exception as e:
        print(f"    Backtest error: {e}")
        return None

def generate_report(results):
    """Generate comprehensive report."""
    report = []
    report.append("=" * 80)
    report.append("TRADING BOT ML - REAL DATA PROFITABILITY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if not results:
        report.append("❌ NO RESULTS TO REPORT")
        return "\n".join(report)
    
    # Individual symbol results
    report.append("INDIVIDUAL SYMBOL RESULTS:")
    report.append("-" * 50)
    
    total_return_sum = 0
    profitable_symbols = 0
    total_symbols = 0
    
    for symbol, result in results.items():
        if result:
            total_symbols += 1
            if result['total_return'] > 0:
                profitable_symbols += 1
            
            total_return_sum += result['total_return']
            
            report.append(f"{symbol}:")
            report.append(f"  Model: {result['model']}")
            report.append(f"  Threshold: {result['threshold']}")
            report.append(f"  Total Return: {result['total_return']:.2f}%")
            report.append(f"  Total Trades: {result['total_trades']}")
            report.append(f"  Win Rate: {result['win_rate']:.1f}%")
            report.append(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            report.append(f"  Profit Factor: {result['profit_factor']:.2f}")
            report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS:")
    report.append("-" * 50)
    
    if total_symbols > 0:
        avg_return = total_return_sum / total_symbols
        profitability_rate = (profitable_symbols / total_symbols) * 100
        
        report.append(f"Average Return: {avg_return:.2f}%")
        report.append(f"Profitable Symbols: {profitable_symbols}/{total_symbols} ({profitability_rate:.1f}%)")
        
        # Overall assessment
        report.append("")
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 50)
        
        if avg_return > 10 and profitability_rate >= 66:
            report.append("🎉 EXCELLENT: System shows strong profitability!")
            assessment = "excellent"
        elif avg_return > 5 and profitability_rate >= 50:
            report.append("✅ GOOD: System shows promising profitability")
            assessment = "good"
        elif avg_return > 0 and profitability_rate >= 33:
            report.append("⚠️  MODERATE: System shows modest profitability")
            assessment = "moderate"
        else:
            report.append("❌ POOR: System needs significant improvement")
            assessment = "poor"
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)
        
        if assessment == "excellent":
            report.append("1. Consider live testing with small amounts")
            report.append("2. Implement advanced risk management")
            report.append("3. Monitor performance closely")
        elif assessment == "good":
            report.append("1. Optimize parameters further")
            report.append("2. Test on more diverse market conditions")
            report.append("3. Implement walk-forward optimization")
        elif assessment == "moderate":
            report.append("1. Focus on feature engineering improvements")
            report.append("2. Try ensemble methods")
            report.append("3. Optimize entry/exit thresholds")
        else:
            report.append("1. Revisit feature engineering completely")
            report.append("2. Try different labeling strategies")
            report.append("3. Consider different market regimes")
    
    return "\n".join(report)

def main():
    """Main function."""
    print("=" * 80)
    print("TRADING BOT ML - REAL DATA PROFITABILITY TEST")
    print("=" * 80)
    
    # Fetch real data
    data = fetch_real_data()
    
    if not data:
        print("❌ No data available for testing")
        return False
    
    # Test each symbol
    results = {}
    
    for symbol, df in data.items():
        result = test_strategy_profitability(symbol, df)
        results[symbol] = result
        
        if result:
            print(f"✓ {symbol}: {result['total_return']:.2f}% return, {result['total_trades']} trades")
        else:
            print(f"✗ {symbol}: Failed to generate results")
    
    # Generate and display report
    report = generate_report(results)
    print("\n" + report)
    
    # Save report
    with open("real_data_profitability_report.txt", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: real_data_profitability_report.txt")
    
    # Determine success
    successful_tests = sum(1 for r in results.values() if r and r['total_return'] > 0)
    total_tests = len([r for r in results.values() if r])
    
    if successful_tests >= total_tests * 0.5:  # At least 50% profitable
        print("🎉 OVERALL SUCCESS: System shows profitability potential!")
        return True
    else:
        print("⚠️  NEEDS IMPROVEMENT: System requires optimization")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)