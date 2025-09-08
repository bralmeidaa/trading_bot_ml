#!/usr/bin/env python3
"""
Diagnostic Tool for Trading Bot Analysis
This script helps analyze why the bot has low trading activity.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enhanced_signal_generator import EnhancedSignalGenerator
from production_trading_system import OptimizedSignalGenerator
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBotDiagnostic:
    """Diagnostic tool to analyze trading bot performance and signal generation."""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'sandbox': True,  # Use testnet
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
    
    async def run_full_diagnostic(self):
        """Run comprehensive diagnostic analysis."""
        logger.info("🔍 Starting Trading Bot Diagnostic Analysis...")
        
        # Test symbols and timeframes
        test_configs = [
            ('LINK/USDT', '5m'),
            ('LINK/USDT', '1m'),
            ('ADA/USDT', '1m'),
            ('BNB/USDT', '1m')
        ]
        
        results = {}
        
        for symbol, timeframe in test_configs:
            logger.info(f"\n📊 Analyzing {symbol} {timeframe}...")
            result = await self.analyze_symbol_timeframe(symbol, timeframe)
            results[f"{symbol}_{timeframe}"] = result
        
        # Generate summary report
        self.generate_diagnostic_report(results)
        
        return results
    
    async def analyze_symbol_timeframe(self, symbol: str, timeframe: str):
        """Analyze signal generation for a specific symbol/timeframe."""
        try:
            # Fetch recent data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            if len(df) < 100:
                return {'error': 'Insufficient data'}
            
            # Test both signal generators
            original_generator = OptimizedSignalGenerator(symbol, timeframe)
            enhanced_generator = EnhancedSignalGenerator(symbol, timeframe)
            
            # Analyze last 100 periods
            analysis_periods = 100
            original_signals = []
            enhanced_signals = []
            
            logger.info(f"   Testing signal generation over last {analysis_periods} periods...")
            
            for i in range(analysis_periods):
                if i + 50 >= len(df):
                    break
                
                # Get data slice
                data_slice = df.iloc[:len(df)-analysis_periods+i+1].copy()
                
                # Test original generator
                try:
                    orig_signals = original_generator.generate_signals(data_slice)
                    original_signals.extend(orig_signals)
                except Exception as e:
                    logger.debug(f"Original generator error: {e}")
                
                # Test enhanced generator
                try:
                    enh_signals = enhanced_generator.generate_signals(data_slice)
                    enhanced_signals.extend(enh_signals)
                except Exception as e:
                    logger.debug(f"Enhanced generator error: {e}")
            
            # Calculate statistics
            current_price = df.iloc[-1]['close']
            price_volatility = df['close'].pct_change().std() * 100
            volume_avg = df['volume'].mean()
            volume_current = df.iloc[-1]['volume']
            
            # Get signal stats from enhanced generator
            signal_stats = enhanced_generator.get_signal_stats()
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_points': len(df),
                'current_price': current_price,
                'price_volatility_pct': price_volatility,
                'volume_ratio': volume_current / volume_avg if volume_avg > 0 else 0,
                'original_signals_count': len(original_signals),
                'enhanced_signals_count': len(enhanced_signals),
                'original_signals_per_day': len(original_signals) * (1440 / self._timeframe_to_minutes(timeframe)) / analysis_periods,
                'enhanced_signals_per_day': len(enhanced_signals) * (1440 / self._timeframe_to_minutes(timeframe)) / analysis_periods,
                'signal_breakdown': signal_stats,
                'last_signal_time': datetime.now().isoformat() if enhanced_signals else None,
                'market_conditions': self._assess_market_conditions(df)
            }
            
            logger.info(f"   ✅ Original: {len(original_signals)} signals, Enhanced: {len(enhanced_signals)} signals")
            logger.info(f"   📈 Volatility: {price_volatility:.2f}%, Volume Ratio: {result['volume_ratio']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
            return {'error': str(e)}
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        if timeframe == '1m':
            return 1
        elif timeframe == '3m':
            return 3
        elif timeframe == '5m':
            return 5
        elif timeframe == '15m':
            return 15
        elif timeframe == '30m':
            return 30
        elif timeframe == '1h':
            return 60
        else:
            return 5  # default
    
    def _assess_market_conditions(self, df: pd.DataFrame) -> dict:
        """Assess current market conditions."""
        recent_data = df.tail(50)
        
        # Calculate trend
        sma_20 = recent_data['close'].rolling(20).mean()
        current_price = recent_data['close'].iloc[-1]
        trend = "bullish" if current_price > sma_20.iloc[-1] else "bearish"
        
        # Calculate RSI
        delta = recent_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Volume analysis
        volume_trend = "high" if recent_data['volume'].iloc[-1] > recent_data['volume'].mean() * 1.5 else "normal"
        
        return {
            'trend': trend,
            'rsi': current_rsi,
            'rsi_condition': 'oversold' if current_rsi < 30 else 'overbought' if current_rsi > 70 else 'neutral',
            'volume_trend': volume_trend,
            'volatility': recent_data['close'].pct_change().std() * 100
        }
    
    def generate_diagnostic_report(self, results: dict):
        """Generate comprehensive diagnostic report."""
        logger.info("\n" + "="*80)
        logger.info("📋 TRADING BOT DIAGNOSTIC REPORT")
        logger.info("="*80)
        
        total_original_signals = sum(r.get('original_signals_count', 0) for r in results.values() if isinstance(r, dict))
        total_enhanced_signals = sum(r.get('enhanced_signals_count', 0) for r in results.values() if isinstance(r, dict))
        
        logger.info(f"\n🎯 SIGNAL GENERATION SUMMARY:")
        logger.info(f"   Original System: {total_original_signals} total signals")
        logger.info(f"   Enhanced System: {total_enhanced_signals} total signals")
        logger.info(f"   Improvement: {((total_enhanced_signals - total_original_signals) / max(total_original_signals, 1) * 100):+.1f}%")
        
        logger.info(f"\n📊 DETAILED ANALYSIS BY SYMBOL/TIMEFRAME:")
        
        for key, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                logger.info(f"\n   {result['symbol']} {result['timeframe']}:")
                logger.info(f"      Original signals/day: {result['original_signals_per_day']:.1f}")
                logger.info(f"      Enhanced signals/day: {result['enhanced_signals_per_day']:.1f}")
                logger.info(f"      Market conditions: {result['market_conditions']['trend']}, RSI: {result['market_conditions']['rsi']:.1f}")
                logger.info(f"      Volatility: {result['price_volatility_pct']:.2f}%, Volume: {result['volume_ratio']:.2f}x")
        
        logger.info(f"\n💡 RECOMMENDATIONS:")
        
        if total_original_signals < 5:
            logger.info("   🔴 CRITICAL: Very low signal generation detected!")
            logger.info("   ✅ Switch to Enhanced Signal Generator")
            logger.info("   ✅ Lower confidence thresholds to 0.50-0.55")
            logger.info("   ✅ Add more trading pairs (ADA/USDT, BNB/USDT)")
            logger.info("   ✅ Focus on 1m timeframe for higher frequency")
        elif total_original_signals < 20:
            logger.info("   🟡 MODERATE: Signal generation could be improved")
            logger.info("   ✅ Consider using Enhanced Signal Generator")
            logger.info("   ✅ Slightly lower confidence thresholds")
        else:
            logger.info("   🟢 GOOD: Signal generation appears healthy")
        
        logger.info(f"\n🔧 IMMEDIATE ACTIONS:")
        logger.info("   1. Replace OptimizedSignalGenerator with EnhancedSignalGenerator")
        logger.info("   2. Use optimized_config.py for more active trading")
        logger.info("   3. Monitor for 24-48 hours and adjust if needed")
        logger.info("   4. Consider adding debug logging for real-time monitoring")
        
        logger.info("\n" + "="*80)

async def main():
    """Run diagnostic analysis."""
    diagnostic = TradingBotDiagnostic()
    await diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    asyncio.run(main())