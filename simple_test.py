#!/usr/bin/env python3
"""
Simple test to validate basic ML functionality and profitability.
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

def test_basic_imports():
    """Test if we can import basic modules."""
    print("Testing basic imports...")
    
    try:
        from utils.indicators import compute_indicators
        print("✓ indicators module imported")
    except Exception as e:
        print(f"✗ indicators import failed: {e}")
        return False
    
    try:
        from ml.features import build_features, build_labels
        print("✓ features module imported")
    except Exception as e:
        print(f"✗ features import failed: {e}")
        return False
    
    try:
        from ml.models import create_model
        print("✓ models module imported")
    except Exception as e:
        print(f"✗ models import failed: {e}")
        return False
    
    return True

def generate_sample_data(n_samples=1000):
    """Generate sample OHLCV data for testing."""
    print(f"Generating {n_samples} samples of test data...")
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Start with a base price
    base_price = 50000.0
    
    # Generate price movements (random walk with trend)
    returns = np.random.normal(0.0001, 0.02, n_samples)  # Small positive drift, 2% daily vol
    prices = [base_price]
    
    for ret in returns:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    prices = prices[1:]  # Remove the initial price
    
    # Generate OHLCV data
    data = []
    base_ts = int(datetime(2023, 1, 1).timestamp() * 1000)
    
    for i, close in enumerate(prices):
        # Generate realistic OHLC
        volatility = abs(np.random.normal(0, 0.01))  # Daily volatility
        
        high = close * (1 + volatility * np.random.uniform(0, 1))
        low = close * (1 - volatility * np.random.uniform(0, 1))
        
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.005))
        else:
            open_price = prices[i-1]  # Previous close
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'ts': base_ts + i * 3600000,  # Hourly data
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated data: {len(df)} rows")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"  Date range: {pd.to_datetime(df['ts'].min(), unit='ms')} to {pd.to_datetime(df['ts'].max(), unit='ms')}")
    
    return df

def test_feature_engineering(df):
    """Test feature engineering."""
    print("\nTesting feature engineering...")
    
    try:
        from ml.features import build_features, build_labels
        
        # Build features
        print(f"Input data: {len(df)} rows")
        fdf = build_features(df, advanced=False)  # Start with basic features first
        print(f"✓ Features built: {len(fdf)} rows, {len(fdf.columns)} columns")
        
        if len(fdf) == 0:
            print("  Trying with basic features only...")
            fdf = build_features(df, advanced=False)
            print(f"  Basic features: {len(fdf)} rows, {len(fdf.columns)} columns")
        
        # Check for key features
        expected_features = ['rsi14', 'bb_pos', 'atr_pct', 'ret1']
        found_features = [f for f in expected_features if f in fdf.columns]
        print(f"✓ Key features found: {found_features}")
        
        # Build labels
        y = build_labels(fdf, horizon=5, cost_bp=15.0)
        print(f"✓ Labels built: {len(y)} samples")
        if len(y) > 0:
            y_int = y.astype(int)
            print(f"  Label distribution: {np.bincount(y_int.values)}")
        else:
            print("  No labels generated")
        
        return fdf, y
        
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_ml_models(X, y):
    """Test ML models."""
    print("\nTesting ML models...")
    
    try:
        from ml.models import create_model
        
        # Get feature columns
        feature_cols = [c for c in X.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
        X_features = X[feature_cols]
        
        print(f"Using {len(feature_cols)} features for training")
        
        # Test different model types
        model_types = ["ensemble", "xgboost", "lightgbm", "random_forest"]
        
        results = {}
        
        for model_type in model_types:
            print(f"  Testing {model_type}...")
            
            try:
                model = create_model(model_type, max_features=30)
                
                # Split data
                split = int(len(X_features) * 0.7)
                X_train = X_features.iloc[:split]
                X_test = X_features.iloc[split:]
                y_train = y.iloc[:split]
                y_test = y.iloc[split:]
                
                # Train model
                train_metrics = model.fit(X_train, y_train.values)
                print(f"    Training completed: {train_metrics}")
                
                # Test predictions
                y_pred_proba = model.predict_proba(X_test)
                y_pred = model.predict(X_test, threshold=0.6)
                
                # Calculate metrics
                accuracy = np.mean(y_pred == y_test.values)
                
                results[model_type] = {
                    "accuracy": accuracy,
                    "train_metrics": train_metrics,
                    "n_predictions": len(y_pred),
                    "positive_predictions": np.sum(y_pred)
                }
                
                print(f"    Accuracy: {accuracy:.3f}, Positive predictions: {np.sum(y_pred)}/{len(y_pred)}")
                
            except Exception as e:
                print(f"    ✗ {model_type} failed: {e}")
                results[model_type] = {"error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"✗ ML model testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_simple_strategy(df, X, y):
    """Test a simple trading strategy."""
    print("\nTesting simple trading strategy...")
    
    try:
        from ml.models import create_model
        
        # Get feature columns
        feature_cols = [c for c in X.columns if c not in ("ts", "open", "high", "low", "close", "volume", "dt")]
        X_features = X[feature_cols]
        
        # Split data
        split = int(len(X_features) * 0.7)
        X_train = X_features.iloc[:split]
        X_test = X_features.iloc[split:]
        y_train = y.iloc[:split]
        
        # Train model
        model = create_model("xgboost", max_features=20)
        model.fit(X_train, y_train.values)
        
        # Get test data with prices
        test_data = X.iloc[split:].copy()
        
        # Simple backtesting
        initial_capital = 10000.0
        capital = initial_capital
        position = 0
        trades = []
        
        for i, (_, row) in enumerate(test_data.iterrows()):
            # Get prediction
            features = row[feature_cols].values.reshape(1, -1)
            prob = model.predict_proba(pd.DataFrame(features, columns=feature_cols))[0]
            
            current_price = row['close']
            
            # Simple strategy: buy if prob > 0.6, sell if prob < 0.4
            if prob > 0.6 and position == 0:
                # Buy
                position = capital / current_price
                capital = 0
                entry_price = current_price
                entry_time = i
                
            elif prob < 0.4 and position > 0:
                # Sell
                capital = position * current_price
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
        
        # Close final position if any
        if position > 0:
            capital = position * test_data.iloc[-1]['close']
            pnl = capital - initial_capital
            pnl_pct = (capital / initial_capital - 1) * 100
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': test_data.iloc[-1]['close'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'hold_time': len(test_data) - entry_time
            })
        
        # Calculate results
        final_capital = capital if position == 0 else position * test_data.iloc[-1]['close']
        total_return = (final_capital / initial_capital - 1) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0])
            avg_loss = avg_loss if not np.isnan(avg_loss) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        results = {
            'total_return': total_return,
            'final_capital': final_capital,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        print(f"✓ Strategy backtest completed:")
        print(f"  Total return: {total_return:.2f}%")
        print(f"  Total trades: {len(trades)}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Final capital: ${final_capital:.2f}")
        
        return results
        
    except Exception as e:
        print(f"✗ Strategy testing failed: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Main test function."""
    print("=" * 60)
    print("TRADING BOT ML - SIMPLE VALIDATION TEST")
    print("=" * 60)
    
    # Test 1: Basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed. Cannot continue.")
        return False
    
    # Test 2: Generate sample data
    df = generate_sample_data(2000)  # 2000 hours ≈ 83 days
    
    # Test 3: Feature engineering
    X, y = test_feature_engineering(df)
    if X is None or y is None:
        print("❌ Feature engineering failed. Cannot continue.")
        return False
    
    # Test 4: ML models
    ml_results = test_ml_models(X, y)
    if not ml_results:
        print("❌ ML model testing failed. Cannot continue.")
        return False
    
    # Test 5: Simple strategy
    strategy_results = test_simple_strategy(df, X, y)
    if not strategy_results:
        print("❌ Strategy testing failed.")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    print("\nML Model Performance:")
    for model_type, results in ml_results.items():
        if "error" not in results:
            print(f"  {model_type}: Accuracy {results['accuracy']:.3f}, Positive predictions: {results['positive_predictions']}")
        else:
            print(f"  {model_type}: ERROR - {results['error']}")
    
    print(f"\nStrategy Performance:")
    print(f"  Total Return: {strategy_results['total_return']:.2f}%")
    print(f"  Total Trades: {strategy_results['total_trades']}")
    print(f"  Win Rate: {strategy_results['win_rate']:.1f}%")
    
    # Assessment
    print(f"\nASSESSMENT:")
    if strategy_results['total_return'] > 5:
        print("✅ STRATEGY SHOWS GOOD PROFITABILITY")
    elif strategy_results['total_return'] > 0:
        print("⚠️  STRATEGY SHOWS MODEST PROFITABILITY")
    else:
        print("❌ STRATEGY NEEDS IMPROVEMENT")
    
    # Check if any models worked
    working_models = [m for m, r in ml_results.items() if "error" not in r]
    if len(working_models) >= 2:
        print("✅ MULTIPLE ML MODELS WORKING")
    elif len(working_models) >= 1:
        print("⚠️  SOME ML MODELS WORKING")
    else:
        print("❌ NO ML MODELS WORKING")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)