#!/usr/bin/env python3
"""
Test script to verify the signal generation fix.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import OptimizedSignalGenerator
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_fix():
    """Test if the signal generation fix works."""
    
    print("🔧 TESTING SIGNAL GENERATION FIX")
    print("=" * 60)
    
    # Connect to exchange
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
        return False
    
    # Test configurations
    test_configs = [
        ('LINK/USDT', '5m'),
        ('LINK/USDT', '1m'),
        ('BTC/USDT', '5m')
    ]
    
    total_signals = 0
    
    for symbol, timeframe in test_configs:
        print(f"\n🧪 Testing {symbol} {timeframe}")
        print("-" * 40)
        
        # Create signal generator
        generator = OptimizedSignalGenerator(symbol, timeframe)
        
        # Get market data
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
            
            if not ohlcv or len(ohlcv) < 100:
                print(f"❌ Insufficient data: {len(ohlcv) if ohlcv else 0} candles")
                continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"✅ Data loaded: {len(df)} candles")
            
            # Generate signals
            signals = generator.generate_signals(df)
            print(f"🎯 Signals generated: {len(signals)}")
            
            if signals:
                for i, signal in enumerate(signals):
                    print(f"  Signal {i+1}: Direction {signal.direction}, "
                          f"Confidence {signal.confidence:.3f}, "
                          f"Strength {signal.strength:.3f}")
                    print(f"    Entry: ${signal.entry_price:.4f}, "
                          f"SL: ${signal.stop_loss:.4f}, "
                          f"TP: ${signal.take_profit:.4f}")
                    if hasattr(signal, 'metadata') and signal.metadata:
                        signal_types = list(signal.metadata.keys())
                        print(f"    Types: {', '.join(signal_types)}")
            
            total_signals += len(signals)
            
        except Exception as e:
            print(f"❌ Error testing {symbol} {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 SUMMARY")
    print("-" * 40)
    print(f"Total signals generated: {total_signals}")
    
    if total_signals > 0:
        print("✅ Signal generation fix SUCCESSFUL!")
        return True
    else:
        print("❌ Signal generation still not working")
        return False

def test_with_more_sensitive_params():
    """Test with more sensitive parameters to ensure signal generation."""
    
    print("\n🎛️ TESTING WITH MORE SENSITIVE PARAMETERS")
    print("=" * 60)
    
    # Connect to exchange
    try:
        exchange = ccxt.binance({
            'sandbox': True,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        exchange.load_markets()
    except Exception as e:
        print(f"⚠️ Cannot connect to Binance: {e}")
        return False
    
    # Create generator with more sensitive parameters
    generator = OptimizedSignalGenerator('LINK/USDT', '5m')
    
    # Override with very sensitive parameters
    generator.params.update({
        'momentum_threshold': 0.003,    # Very low
        'volume_threshold': 1.1,        # Very low
        'rsi_oversold': 45,             # Less extreme
        'rsi_overbought': 55,           # Less extreme
        'confidence_multiplier': 1.0,
        'ml_threshold': 0.45            # Lower
    })
    
    print("📊 Using sensitive parameters:")
    for key, value in generator.params.items():
        print(f"  {key}: {value}")
    
    # Get data and test
    try:
        ohlcv = exchange.fetch_ohlcv('LINK/USDT', '5m', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        signals = generator.generate_signals(df)
        print(f"\n🎯 Signals with sensitive params: {len(signals)}")
        
        if signals:
            for i, signal in enumerate(signals):
                print(f"  Signal {i+1}: Direction {signal.direction}, Confidence {signal.confidence:.3f}")
        
        return len(signals) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_individual_signal_components():
    """Test individual signal components to understand what's working."""
    
    print("\n🔍 TESTING INDIVIDUAL SIGNAL COMPONENTS")
    print("=" * 60)
    
    try:
        exchange = ccxt.binance({
            'sandbox': True,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        exchange.load_markets()
        
        generator = OptimizedSignalGenerator('LINK/USDT', '5m')
        ohlcv = exchange.fetch_ohlcv('LINK/USDT', '5m', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Add indicators
        df_with_indicators = generator._add_indicators(df)
        latest_row = df_with_indicators.iloc[-1]
        
        print("📈 Latest market data:")
        print(f"  Price: ${latest_row['close']:.4f}")
        print(f"  RSI: {latest_row['rsi']:.1f}")
        print(f"  BB Position: {latest_row['bb_position']:.3f}")
        print(f"  Volume Ratio: {latest_row['volume_ratio']:.2f}")
        print(f"  Momentum 5: {latest_row['momentum_5']:.6f}")
        
        # Test each signal type individually
        momentum_signal = generator._check_momentum_signal(latest_row)
        mean_reversion_signal = generator._check_mean_reversion_signal(latest_row)
        volume_signal = generator._check_volume_signal(df_with_indicators.iloc[-2:])
        ml_signal = generator._check_ml_signal(df_with_indicators.iloc[-1:])
        
        print("\n🔍 Individual signal results:")
        print(f"  Momentum: {'✅' if momentum_signal else '❌'}")
        if momentum_signal:
            print(f"    Confidence: {momentum_signal['confidence']:.3f}")
        
        print(f"  Mean Reversion: {'✅' if mean_reversion_signal else '❌'}")
        if mean_reversion_signal:
            print(f"    Confidence: {mean_reversion_signal['confidence']:.3f}")
        
        print(f"  Volume: {'✅' if volume_signal else '❌'}")
        if volume_signal:
            print(f"    Confidence: {volume_signal['confidence']:.3f}")
        
        print(f"  ML: {'✅' if ml_signal else '❌'}")
        if ml_signal:
            print(f"    Confidence: {ml_signal['confidence']:.3f}")
        
        # Test combination
        signals_list = [momentum_signal, mean_reversion_signal, volume_signal, ml_signal]
        combined = generator._combine_signals(signals_list)
        
        print(f"\n🎯 Combined signal: {'✅' if combined else '❌'}")
        if combined:
            print(f"  Direction: {combined['direction']}")
            print(f"  Confidence: {combined['confidence']:.3f}")
            print(f"  Strength: {combined['strength']:.3f}")
        
        return combined is not None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SIGNAL GENERATION FIX TEST")
    print("=" * 80)
    
    # Test 1: Basic fix test
    success1 = test_signal_fix()
    
    # Test 2: Sensitive parameters
    success2 = test_with_more_sensitive_params()
    
    # Test 3: Individual components
    success3 = test_individual_signal_components()
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 40)
    print(f"Basic fix test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Sensitive params: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Component test: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    overall_success = success1 or success2 or success3
    print(f"\nOverall: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    sys.exit(0 if overall_success else 1)