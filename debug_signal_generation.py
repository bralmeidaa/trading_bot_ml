#!/usr/bin/env python3
"""
Debug script to understand why signals are not being generated.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import OptimizedSignalGenerator, create_production_config
import ccxt

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_signal_generation():
    """Debug signal generation step by step."""
    
    print("🔍 DEBUGGING SIGNAL GENERATION")
    print("=" * 60)
    
    # Test with real market data
    try:
        exchange = ccxt.binance({
            'sandbox': True,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        exchange.load_markets()
        print("✅ Connected to Binance")
    except Exception as e:
        print(f"⚠️ Cannot connect to Binance: {e}")
        print("Using mock data instead...")
        exchange = None
    
    # Test configurations
    test_configs = [
        ('LINK/USDT', '5m'),
        ('LINK/USDT', '1m'),
        ('BTC/USDT', '5m')
    ]
    
    for symbol, timeframe in test_configs:
        print(f"\n🧪 Testing {symbol} {timeframe}")
        print("-" * 40)
        
        # Create signal generator
        generator = OptimizedSignalGenerator(symbol, timeframe)
        
        # Print parameters
        print("📊 Parameters:")
        for key, value in generator.params.items():
            print(f"  {key}: {value}")
        
        # Get market data
        try:
            if exchange:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            else:
                # Generate mock data
                ohlcv = generate_mock_data(200)
            
            if not ohlcv or len(ohlcv) < 100:
                print(f"❌ Insufficient data: {len(ohlcv) if ohlcv else 0} candles")
                continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"✅ Data loaded: {len(df)} candles")
            
            # Add indicators
            df_with_indicators = generator._add_indicators(df)
            
            # Check indicators
            latest_row = df_with_indicators.iloc[-1]
            required_indicators = ['sma_20', 'ema_8', 'ema_21', 'rsi', 'bb_upper', 'bb_lower', 'atr']
            
            print("📈 Latest indicators:")
            for indicator in required_indicators:
                value = latest_row.get(indicator, 'N/A')
                is_nan = pd.isna(value) if value != 'N/A' else True
                status = "❌ NaN" if is_nan else f"✅ {value:.4f}"
                print(f"  {indicator}: {status}")
            
            # Check if indicators are ready
            indicators_ready = not any(pd.isna(latest_row[indicator]) for indicator in required_indicators)
            print(f"🎯 Indicators ready: {'✅ Yes' if indicators_ready else '❌ No'}")
            
            if not indicators_ready:
                print("⚠️ Skipping signal generation - indicators not ready")
                continue
            
            # Test individual signal types
            print("\n🔍 Testing individual signals:")
            
            # Momentum signal
            momentum_signal = generator._check_momentum_signal(latest_row)
            print(f"  Momentum: {'✅ Found' if momentum_signal else '❌ None'}")
            if momentum_signal:
                print(f"    Direction: {momentum_signal['direction']}, Confidence: {momentum_signal['confidence']:.3f}")
            else:
                # Debug momentum conditions
                print(f"    momentum_5: {latest_row.get('momentum_5', 'N/A'):.6f} (threshold: ±{generator.params['momentum_threshold']})")
                print(f"    volume_ratio: {latest_row.get('volume_ratio', 'N/A'):.2f} (threshold: {generator.params['volume_threshold']})")
                print(f"    rsi: {latest_row.get('rsi', 'N/A'):.1f} (range: {generator.params['rsi_oversold']}-{generator.params['rsi_overbought']})")
            
            # Mean reversion signal
            mean_reversion_signal = generator._check_mean_reversion_signal(latest_row)
            print(f"  Mean Reversion: {'✅ Found' if mean_reversion_signal else '❌ None'}")
            if mean_reversion_signal:
                print(f"    Direction: {mean_reversion_signal['direction']}, Confidence: {mean_reversion_signal['confidence']:.3f}")
            else:
                print(f"    bb_position: {latest_row.get('bb_position', 'N/A'):.3f} (extremes: <0.15 or >0.85)")
                print(f"    rsi: {latest_row.get('rsi', 'N/A'):.1f} (oversold: <{generator.params['rsi_oversold']}, overbought: >{generator.params['rsi_overbought']})")
            
            # Volume signal
            volume_signal = generator._check_volume_signal(df_with_indicators.iloc[-2:])
            print(f"  Volume: {'✅ Found' if volume_signal else '❌ None'}")
            if volume_signal:
                print(f"    Direction: {volume_signal['direction']}, Confidence: {volume_signal['confidence']:.3f}")
            else:
                if len(df_with_indicators) >= 2:
                    current = df_with_indicators.iloc[-1]
                    previous = df_with_indicators.iloc[-2]
                    price_change = (current['close'] - previous['close']) / previous['close']
                    print(f"    volume_ratio: {current.get('volume_ratio', 'N/A'):.2f} (threshold: {generator.params['volume_threshold']})")
                    print(f"    price_change: {price_change:.6f} (threshold: ±0.005)")
            
            # ML signal
            ml_signal = generator._check_ml_signal(df_with_indicators.iloc[-1:])
            print(f"  ML: {'✅ Found' if ml_signal else '❌ None'}")
            print(f"    Model fitted: {'✅ Yes' if generator.is_fitted else '❌ No'}")
            
            # Try to generate signals
            signals = generator.generate_signals(df)
            print(f"\n🎯 Final signals generated: {len(signals)}")
            
            if signals:
                for i, signal in enumerate(signals):
                    print(f"  Signal {i+1}: {signal.direction} direction, confidence {signal.confidence:.3f}")
            
            # Test with more relaxed parameters
            print("\n🔧 Testing with relaxed parameters:")
            original_params = generator.params.copy()
            
            # Make parameters more lenient
            generator.params.update({
                'momentum_threshold': 0.003,  # Much lower
                'volume_threshold': 1.2,      # Much lower
                'rsi_oversold': 40,           # Less extreme
                'rsi_overbought': 60,         # Less extreme
                'ml_threshold': 0.45          # Lower
            })
            
            relaxed_signals = generator.generate_signals(df)
            print(f"  Relaxed signals: {len(relaxed_signals)}")
            
            # Restore original parameters
            generator.params = original_params
            
        except Exception as e:
            print(f"❌ Error testing {symbol} {timeframe}: {e}")
            import traceback
            traceback.print_exc()

def generate_mock_data(periods: int):
    """Generate mock OHLCV data for testing."""
    np.random.seed(42)
    
    # Start with a base price
    base_price = 100.0
    timestamps = []
    ohlcv = []
    
    current_time = int(datetime.now().timestamp() * 1000)
    
    for i in range(periods):
        # Generate realistic price movement
        change = np.random.normal(0, 0.02)  # 2% volatility
        base_price *= (1 + change)
        
        # Generate OHLC from close price
        high = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low = base_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = base_price * (1 + np.random.normal(0, 0.005))
        
        # Generate volume
        volume = np.random.uniform(1000, 10000)
        
        # Timestamp (5 minutes apart)
        timestamp = current_time - (periods - i) * 5 * 60 * 1000
        
        ohlcv.append([timestamp, open_price, high, low, base_price, volume])
    
    return ohlcv

def test_parameter_sensitivity():
    """Test how sensitive the system is to parameter changes."""
    print("\n🎛️ PARAMETER SENSITIVITY TEST")
    print("=" * 60)
    
    # Generate test data
    ohlcv = generate_mock_data(200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Test different parameter sets
    parameter_sets = [
        ("Very Conservative", {
            'momentum_threshold': 0.015,
            'volume_threshold': 2.5,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'ml_threshold': 0.65
        }),
        ("Conservative", {
            'momentum_threshold': 0.010,
            'volume_threshold': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ml_threshold': 0.60
        }),
        ("Moderate", {
            'momentum_threshold': 0.007,
            'volume_threshold': 1.6,
            'rsi_oversold': 33,
            'rsi_overbought': 67,
            'ml_threshold': 0.55
        }),
        ("Aggressive", {
            'momentum_threshold': 0.005,
            'volume_threshold': 1.3,
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'ml_threshold': 0.50
        }),
        ("Very Aggressive", {
            'momentum_threshold': 0.003,
            'volume_threshold': 1.1,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'ml_threshold': 0.45
        })
    ]
    
    for name, params in parameter_sets:
        generator = OptimizedSignalGenerator('BTC/USDT', '5m')
        generator.params.update(params)
        
        signals = generator.generate_signals(df)
        signal_frequency = len(signals) / len(df) * 100
        
        print(f"{name:15}: {len(signals):2d} signals ({signal_frequency:4.1f}%)")

if __name__ == "__main__":
    debug_signal_generation()
    test_parameter_sensitivity()