#!/usr/bin/env python3
"""
BTC Trend Following Bot
Especializado em seguir tendências de longo prazo do Bitcoin
"""

import asyncio
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .base_specialized_bot import BaseSpecializedBot, BotSpecialization

class BTCTrendBot(BaseSpecializedBot):
    """Bot especializado em trend following para Bitcoin."""
    
    def __init__(self):
        specialization = BotSpecialization(
            name="BTC_Trend_Master",
            symbols=["BTC/USDT"],
            strategy_type="trend_following",
            timeframes=["1h", "4h"],
            max_positions=2,
            risk_per_trade=0.012,  # Mais agressivo para BTC
            specialized_indicators=[
                "EMA_21", "EMA_55", "EMA_200", 
                "MACD", "ADX", "Parabolic_SAR",
                "Volume_Profile", "Trend_Strength"
            ],
            entry_conditions={
                "trend_alignment": True,
                "momentum_confirmation": True,
                "volume_support": True,
                "pullback_entry": True,
                "adx_threshold": 25,
                "ema_separation": 0.02
            },
            exit_conditions={
                "trailing_stop": True,
                "trend_reversal": True,
                "momentum_divergence": True,
                "time_based": False
            },
            performance_targets={
                "win_rate": 0.65,
                "avg_pnl": 0.015,
                "max_drawdown": 0.08,
                "sharpe_ratio": 1.5
            },
            notes="Especializado em capturar grandes movimentos de tendência do BTC"
        )
        
        super().__init__(specialization)
        
        # BTC-specific parameters
        self.trend_confirmation_periods = [21, 55, 200]
        self.momentum_lookback = 14
        self.volume_threshold = 1.5
        self.pullback_depth = 0.618  # Fibonacci retracement
        
    async def analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict:
        """Analyze BTC-specific market conditions."""
        
        # Simulate getting price data (in production, use real data)
        price_data = await self._get_price_data(symbol, timeframe)
        
        if price_data is None:
            return {"error": "No price data"}
        
        # Calculate BTC-specific indicators
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        volume = price_data['volume'].values
        
        # Trend Analysis
        ema_21 = talib.EMA(close, timeperiod=21)
        ema_55 = talib.EMA(close, timeperiod=55)
        ema_200 = talib.EMA(close, timeperiod=200)
        
        # Momentum
        macd, macd_signal, macd_hist = talib.MACD(close)
        adx = talib.ADX(high, low, close, timeperiod=14)
        rsi = talib.RSI(close, timeperiod=14)
        
        # Parabolic SAR
        sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # Volume analysis
        volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
        volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
        
        # Trend strength calculation
        trend_strength = self._calculate_trend_strength(ema_21, ema_55, ema_200, adx)
        
        # Market structure
        higher_highs = self._count_higher_highs(high[-20:])
        higher_lows = self._count_higher_lows(low[-20:])
        
        return {
            "trend_direction": self._determine_trend_direction(ema_21, ema_55, ema_200),
            "trend_strength": trend_strength,
            "momentum_score": self._calculate_momentum_score(macd, macd_hist, rsi, adx),
            "volume_support": volume_ratio > self.volume_threshold,
            "sar_signal": "bullish" if close[-1] > sar[-1] else "bearish",
            "market_structure": {
                "higher_highs": higher_highs,
                "higher_lows": higher_lows,
                "structure_score": (higher_highs + higher_lows) / 2
            },
            "pullback_opportunity": self._detect_pullback_opportunity(close, ema_21),
            "timeframe": timeframe,
            "last_price": close[-1]
        }
    
    async def generate_specialized_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Generate BTC trend following signal."""
        
        market_conditions = await self.analyze_market_conditions(symbol, timeframe)
        
        if "error" in market_conditions:
            return None
        
        # Get price data for signal generation
        price_data = await self._get_price_data(symbol, timeframe)
        features = await self.calculate_specialized_features(price_data)
        
        # Signal generation logic
        signal_strength = 0.0
        direction = None
        entry_price = market_conditions["last_price"]
        
        # Bullish signal conditions
        if (market_conditions["trend_direction"] == "bullish" and
            market_conditions["trend_strength"] > 0.6 and
            market_conditions["momentum_score"] > 0.5 and
            market_conditions["volume_support"] and
            market_conditions["pullback_opportunity"]):
            
            direction = "long"
            signal_strength = (
                market_conditions["trend_strength"] * 0.3 +
                market_conditions["momentum_score"] * 0.3 +
                market_conditions["market_structure"]["structure_score"] * 0.2 +
                (1.0 if market_conditions["volume_support"] else 0.0) * 0.2
            )
        
        # Bearish signal conditions (for trend reversal or short opportunities)
        elif (market_conditions["trend_direction"] == "bearish" and
              market_conditions["trend_strength"] > 0.6 and
              market_conditions["momentum_score"] < -0.5):
            
            direction = "short"
            signal_strength = (
                market_conditions["trend_strength"] * 0.3 +
                abs(market_conditions["momentum_score"]) * 0.3 +
                (1.0 - market_conditions["market_structure"]["structure_score"]) * 0.2 +
                (1.0 if market_conditions["volume_support"] else 0.0) * 0.2
            )
        
        if direction and signal_strength > 0.65:  # High threshold for BTC trend bot
            
            # Calculate stop loss and take profit
            atr = features.get("atr_14", entry_price * 0.02)
            
            if direction == "long":
                stop_loss = entry_price - (atr * 2.5)
                take_profit = entry_price + (atr * 5.0)  # 2:1 R/R
            else:
                stop_loss = entry_price + (atr * 2.5)
                take_profit = entry_price - (atr * 5.0)
            
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": signal_strength,
                "timeframe": timeframe,
                "strategy": "btc_trend_following",
                "features": features,
                "market_conditions": market_conditions,
                "risk_reward_ratio": 2.0,
                "timestamp": datetime.now()
            }
        
        return None
    
    async def calculate_specialized_features(self, price_data: pd.DataFrame) -> Dict:
        """Calculate BTC-specific features."""
        
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        volume = price_data['volume'].values
        
        features = {}
        
        # Trend features
        features["ema_21"] = talib.EMA(close, timeperiod=21)[-1]
        features["ema_55"] = talib.EMA(close, timeperiod=55)[-1]
        features["ema_200"] = talib.EMA(close, timeperiod=200)[-1]
        
        # Momentum features
        macd, macd_signal, macd_hist = talib.MACD(close)
        features["macd"] = macd[-1]
        features["macd_signal"] = macd_signal[-1]
        features["macd_histogram"] = macd_hist[-1]
        
        features["rsi_14"] = talib.RSI(close, timeperiod=14)[-1]
        features["adx"] = talib.ADX(high, low, close, timeperiod=14)[-1]
        
        # Volatility features
        features["atr_14"] = talib.ATR(high, low, close, timeperiod=14)[-1]
        features["bb_upper"], features["bb_middle"], features["bb_lower"] = talib.BBANDS(close)
        features["bb_width"] = (features["bb_upper"][-1] - features["bb_lower"][-1]) / features["bb_middle"][-1]
        
        # Volume features
        features["volume_sma_20"] = talib.SMA(volume.astype(float), timeperiod=20)[-1]
        features["volume_ratio"] = volume[-1] / features["volume_sma_20"] if features["volume_sma_20"] > 0 else 1
        
        # Price action features
        features["price_vs_ema21"] = (close[-1] - features["ema_21"]) / features["ema_21"]
        features["ema21_vs_ema55"] = (features["ema_21"] - features["ema_55"]) / features["ema_55"]
        features["ema55_vs_ema200"] = (features["ema_55"] - features["ema_200"]) / features["ema_200"]
        
        # Specialized BTC features
        features["trend_alignment_score"] = self._calculate_trend_alignment(
            features["ema_21"], features["ema_55"], features["ema_200"]
        )
        
        return features
    
    def validate_specialized_conditions(self, signal: Dict, market_conditions: Dict) -> Tuple[bool, str]:
        """Validate BTC trend following conditions."""
        
        # Check trend strength
        if market_conditions["trend_strength"] < 0.6:
            return False, "Trend strength too weak"
        
        # Check momentum alignment
        if signal["direction"] == "long" and market_conditions["momentum_score"] < 0.4:
            return False, "Momentum not aligned for long position"
        
        if signal["direction"] == "short" and market_conditions["momentum_score"] > -0.4:
            return False, "Momentum not aligned for short position"
        
        # Check volume support
        if not market_conditions["volume_support"]:
            return False, "Insufficient volume support"
        
        # Check risk/reward ratio
        if signal["risk_reward_ratio"] < 1.5:
            return False, "Risk/reward ratio too low"
        
        # Check confidence threshold
        if signal["confidence"] < 0.65:
            return False, "Signal confidence below threshold"
        
        return True, "All BTC trend conditions met"
    
    def _determine_trend_direction(self, ema_21, ema_55, ema_200) -> str:
        """Determine overall trend direction."""
        if ema_21[-1] > ema_55[-1] > ema_200[-1]:
            return "bullish"
        elif ema_21[-1] < ema_55[-1] < ema_200[-1]:
            return "bearish"
        else:
            return "sideways"
    
    def _calculate_trend_strength(self, ema_21, ema_55, ema_200, adx) -> float:
        """Calculate trend strength score."""
        
        # EMA separation
        ema_separation = abs(ema_21[-1] - ema_55[-1]) / ema_55[-1]
        ema_separation_score = min(ema_separation / 0.05, 1.0)  # Normalize
        
        # ADX strength
        adx_score = min(adx[-1] / 50.0, 1.0)  # Normalize ADX
        
        # EMA slope
        ema21_slope = (ema_21[-1] - ema_21[-5]) / ema_21[-5]
        slope_score = min(abs(ema21_slope) / 0.02, 1.0)
        
        return (ema_separation_score * 0.4 + adx_score * 0.4 + slope_score * 0.2)
    
    def _calculate_momentum_score(self, macd, macd_hist, rsi, adx) -> float:
        """Calculate momentum score."""
        
        # MACD momentum
        macd_score = 1.0 if macd[-1] > 0 else -1.0
        macd_hist_score = 1.0 if macd_hist[-1] > macd_hist[-2] else -1.0
        
        # RSI momentum
        if rsi[-1] > 70:
            rsi_score = 1.0
        elif rsi[-1] < 30:
            rsi_score = -1.0
        else:
            rsi_score = (rsi[-1] - 50) / 50.0
        
        # ADX trend strength
        adx_strength = min(adx[-1] / 50.0, 1.0)
        
        momentum = (macd_score * 0.3 + macd_hist_score * 0.3 + rsi_score * 0.4) * adx_strength
        
        return momentum
    
    def _count_higher_highs(self, highs) -> int:
        """Count higher highs in recent price action."""
        count = 0
        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                count += 1
        return count / len(highs)
    
    def _count_higher_lows(self, lows) -> int:
        """Count higher lows in recent price action."""
        count = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1]:
                count += 1
        return count / len(lows)
    
    def _detect_pullback_opportunity(self, close, ema_21) -> bool:
        """Detect if current price is a good pullback entry."""
        
        # Check if price pulled back to EMA21
        distance_to_ema = abs(close[-1] - ema_21[-1]) / ema_21[-1]
        
        # Good pullback if within 2% of EMA21
        return distance_to_ema < 0.02
    
    def _calculate_trend_alignment(self, ema_21, ema_55, ema_200) -> float:
        """Calculate how well EMAs are aligned."""
        
        if ema_21 > ema_55 > ema_200:
            return 1.0  # Perfect bullish alignment
        elif ema_21 < ema_55 < ema_200:
            return -1.0  # Perfect bearish alignment
        else:
            return 0.0  # No clear alignment
    
    async def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get price data (simulated for now)."""
        
        # Simulate price data
        periods = 200
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate realistic BTC price data
        base_price = 50000
        returns = np.random.normal(0.0005, 0.02, periods)  # BTC-like volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.normal(1000, 200, periods)
        })
        
        return df

# Test the BTC Trend Bot
async def test_btc_trend_bot():
    """Test the BTC Trend Bot."""
    
    bot = BTCTrendBot()
    
    print("🚀 TESTE DO BTC TREND BOT")
    print("=" * 50)
    
    # Run specialized cycle
    signals = await bot.run_specialized_cycle()
    
    print(f"Sinais gerados: {len(signals)}")
    
    for signal in signals:
        print(f"\n📊 SINAL BTC TREND:")
        print(f"Direção: {signal['direction']}")
        print(f"Preço de entrada: ${signal['entry_price']:.2f}")
        print(f"Stop loss: ${signal['stop_loss']:.2f}")
        print(f"Take profit: ${signal['take_profit']:.2f}")
        print(f"Confiança: {signal['confidence']:.2f}")
        print(f"R/R: {signal['risk_reward_ratio']:.1f}")
        print(f"Estratégia: {signal['strategy']}")
    
    # Performance summary
    summary = bot.get_performance_summary()
    print(f"\n📈 RESUMO DE PERFORMANCE:")
    print(f"Bot: {summary['bot_name']}")
    print(f"Especialização: {summary['specialization']}")
    print(f"Símbolos: {summary['symbols']}")
    print(f"Score de especialização: {summary['stats']['specialization_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_btc_trend_bot())