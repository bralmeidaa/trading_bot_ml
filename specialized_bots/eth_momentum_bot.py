#!/usr/bin/env python3
"""
ETH Momentum Bot
Especializado em capturar movimentos de momentum do Ethereum
"""

import asyncio
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .base_specialized_bot import BaseSpecializedBot, BotSpecialization

class ETHMomentumBot(BaseSpecializedBot):
    """Bot especializado em momentum trading para Ethereum."""
    
    def __init__(self):
        specialization = BotSpecialization(
            name="ETH_Momentum_Hunter",
            symbols=["ETH/USDT"],
            strategy_type="momentum_trading",
            timeframes=["15m", "1h"],
            max_positions=3,
            risk_per_trade=0.010,
            specialized_indicators=[
                "RSI", "MACD", "Stochastic", "Williams_R",
                "Volume_Oscillator", "Price_Rate_of_Change",
                "Momentum_Divergence", "Breakout_Strength"
            ],
            entry_conditions={
                "momentum_acceleration": True,
                "volume_confirmation": True,
                "breakout_validation": True,
                "rsi_momentum": True,
                "macd_crossover": True,
                "stoch_alignment": True
            },
            exit_conditions={
                "momentum_exhaustion": True,
                "divergence_signal": True,
                "overbought_exit": True,
                "trailing_stop": True
            },
            performance_targets={
                "win_rate": 0.60,
                "avg_pnl": 0.012,
                "max_drawdown": 0.06,
                "sharpe_ratio": 1.8
            },
            notes="Especializado em capturar movimentos rápidos de momentum do ETH"
        )
        
        super().__init__(specialization)
        
        # ETH-specific momentum parameters
        self.momentum_lookback = 10
        self.volume_surge_threshold = 2.0
        self.rsi_momentum_threshold = 60
        self.breakout_confirmation_periods = 3
        
    async def analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict:
        """Analyze ETH momentum conditions."""
        
        price_data = await self._get_price_data(symbol, timeframe)
        
        if price_data is None:
            return {"error": "No price data"}
        
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        volume = price_data['volume'].values
        
        # Momentum indicators
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        williams_r = talib.WILLR(high, low, close, timeperiod=14)
        
        # Volume analysis
        volume_sma = talib.SMA(volume.astype(float), timeperiod=20)
        volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
        
        # Price momentum
        roc = talib.ROC(close, timeperiod=10)  # Rate of Change
        momentum = talib.MOM(close, timeperiod=10)
        
        # Momentum acceleration
        momentum_acceleration = self._calculate_momentum_acceleration(close)
        
        # Breakout analysis
        breakout_strength = self._analyze_breakout_strength(high, low, close)
        
        # Divergence detection
        price_momentum_divergence = self._detect_momentum_divergence(close, rsi)
        
        return {
            "momentum_score": self._calculate_momentum_score(rsi, macd_hist, stoch_k, williams_r),
            "momentum_acceleration": momentum_acceleration,
            "volume_surge": volume_ratio > self.volume_surge_threshold,
            "volume_ratio": volume_ratio,
            "rsi_momentum": rsi[-1] > self.rsi_momentum_threshold or rsi[-1] < (100 - self.rsi_momentum_threshold),
            "macd_signal": "bullish" if macd[-1] > macd_signal[-1] else "bearish",
            "stoch_alignment": self._check_stochastic_alignment(stoch_k, stoch_d),
            "breakout_strength": breakout_strength,
            "momentum_divergence": price_momentum_divergence,
            "price_roc": roc[-1],
            "momentum_raw": momentum[-1],
            "timeframe": timeframe,
            "last_price": close[-1]
        }
    
    async def generate_specialized_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Generate ETH momentum signal."""
        
        market_conditions = await self.analyze_market_conditions(symbol, timeframe)
        
        if "error" in market_conditions:
            return None
        
        price_data = await self._get_price_data(symbol, timeframe)
        features = await self.calculate_specialized_features(price_data)
        
        signal_strength = 0.0
        direction = None
        entry_price = market_conditions["last_price"]
        
        # Bullish momentum signal
        if (market_conditions["momentum_score"] > 0.6 and
            market_conditions["momentum_acceleration"] > 0.5 and
            market_conditions["volume_surge"] and
            market_conditions["rsi_momentum"] and
            market_conditions["macd_signal"] == "bullish" and
            market_conditions["breakout_strength"] > 0.6):
            
            direction = "long"
            signal_strength = (
                market_conditions["momentum_score"] * 0.25 +
                market_conditions["momentum_acceleration"] * 0.25 +
                market_conditions["breakout_strength"] * 0.20 +
                (1.0 if market_conditions["volume_surge"] else 0.0) * 0.15 +
                (1.0 if market_conditions["stoch_alignment"] == "bullish" else 0.0) * 0.15
            )
        
        # Bearish momentum signal
        elif (market_conditions["momentum_score"] < -0.6 and
              market_conditions["momentum_acceleration"] < -0.5 and
              market_conditions["volume_surge"] and
              market_conditions["macd_signal"] == "bearish"):
            
            direction = "short"
            signal_strength = (
                abs(market_conditions["momentum_score"]) * 0.25 +
                abs(market_conditions["momentum_acceleration"]) * 0.25 +
                market_conditions["breakout_strength"] * 0.20 +
                (1.0 if market_conditions["volume_surge"] else 0.0) * 0.15 +
                (1.0 if market_conditions["stoch_alignment"] == "bearish" else 0.0) * 0.15
            )
        
        if direction and signal_strength > 0.70:  # High threshold for momentum
            
            # Calculate dynamic stop loss and take profit based on volatility
            atr = features.get("atr_14", entry_price * 0.015)
            
            # Momentum trades have tighter stops and quicker profits
            if direction == "long":
                stop_loss = entry_price - (atr * 1.5)  # Tighter stop
                take_profit = entry_price + (atr * 3.0)  # 2:1 R/R
            else:
                stop_loss = entry_price + (atr * 1.5)
                take_profit = entry_price - (atr * 3.0)
            
            return {
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": signal_strength,
                "timeframe": timeframe,
                "strategy": "eth_momentum",
                "features": features,
                "market_conditions": market_conditions,
                "risk_reward_ratio": 2.0,
                "momentum_type": "acceleration" if market_conditions["momentum_acceleration"] > 0.7 else "standard",
                "timestamp": datetime.now()
            }
        
        return None
    
    async def calculate_specialized_features(self, price_data: pd.DataFrame) -> Dict:
        """Calculate ETH momentum-specific features."""
        
        close = price_data['close'].values
        high = price_data['high'].values
        low = price_data['low'].values
        volume = price_data['volume'].values
        
        features = {}
        
        # Momentum oscillators
        features["rsi_14"] = talib.RSI(close, timeperiod=14)[-1]
        features["rsi_7"] = talib.RSI(close, timeperiod=7)[-1]  # Faster RSI
        
        macd, macd_signal, macd_hist = talib.MACD(close)
        features["macd"] = macd[-1]
        features["macd_signal"] = macd_signal[-1]
        features["macd_histogram"] = macd_hist[-1]
        
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        features["stoch_k"] = stoch_k[-1]
        features["stoch_d"] = stoch_d[-1]
        
        features["williams_r"] = talib.WILLR(high, low, close, timeperiod=14)[-1]
        
        # Price momentum
        features["roc_10"] = talib.ROC(close, timeperiod=10)[-1]
        features["roc_5"] = talib.ROC(close, timeperiod=5)[-1]
        features["momentum_10"] = talib.MOM(close, timeperiod=10)[-1]
        
        # Volatility
        features["atr_14"] = talib.ATR(high, low, close, timeperiod=14)[-1]
        features["atr_7"] = talib.ATR(high, low, close, timeperiod=7)[-1]
        
        # Volume momentum
        features["volume_sma_10"] = talib.SMA(volume.astype(float), timeperiod=10)[-1]
        features["volume_sma_20"] = talib.SMA(volume.astype(float), timeperiod=20)[-1]
        features["volume_ratio"] = volume[-1] / features["volume_sma_20"] if features["volume_sma_20"] > 0 else 1
        
        # Specialized momentum features
        features["momentum_acceleration"] = self._calculate_momentum_acceleration(close)
        features["momentum_consistency"] = self._calculate_momentum_consistency(close)
        features["breakout_momentum"] = self._calculate_breakout_momentum(high, low, close)
        
        # Multi-timeframe momentum
        features["momentum_alignment"] = self._calculate_momentum_alignment(
            features["rsi_7"], features["rsi_14"], features["macd_histogram"]
        )
        
        return features
    
    def validate_specialized_conditions(self, signal: Dict, market_conditions: Dict) -> Tuple[bool, str]:
        """Validate ETH momentum conditions."""
        
        # Check momentum strength
        if abs(market_conditions["momentum_score"]) < 0.6:
            return False, "Momentum score too weak"
        
        # Check momentum acceleration
        if signal["direction"] == "long" and market_conditions["momentum_acceleration"] < 0.4:
            return False, "Insufficient bullish momentum acceleration"
        
        if signal["direction"] == "short" and market_conditions["momentum_acceleration"] > -0.4:
            return False, "Insufficient bearish momentum acceleration"
        
        # Check volume confirmation
        if not market_conditions["volume_surge"]:
            return False, "No volume surge confirmation"
        
        # Check breakout strength
        if market_conditions["breakout_strength"] < 0.5:
            return False, "Breakout strength insufficient"
        
        # Check for momentum divergence (negative signal)
        if market_conditions["momentum_divergence"]:
            return False, "Momentum divergence detected"
        
        # Check confidence threshold
        if signal["confidence"] < 0.70:
            return False, "Signal confidence below momentum threshold"
        
        return True, "All ETH momentum conditions met"
    
    def _calculate_momentum_score(self, rsi, macd_hist, stoch_k, williams_r) -> float:
        """Calculate comprehensive momentum score."""
        
        # RSI momentum
        rsi_score = (rsi[-1] - 50) / 50.0  # Normalize around 50
        
        # MACD histogram momentum
        macd_score = 1.0 if macd_hist[-1] > 0 else -1.0
        macd_acceleration = 1.0 if macd_hist[-1] > macd_hist[-2] else -1.0
        
        # Stochastic momentum
        stoch_score = (stoch_k[-1] - 50) / 50.0
        
        # Williams %R momentum
        williams_score = (williams_r[-1] + 50) / 50.0  # Normalize
        
        # Combine scores
        momentum = (
            rsi_score * 0.3 +
            (macd_score + macd_acceleration) * 0.25 +
            stoch_score * 0.25 +
            williams_score * 0.2
        )
        
        return np.clip(momentum, -1.0, 1.0)
    
    def _calculate_momentum_acceleration(self, close) -> float:
        """Calculate momentum acceleration."""
        
        # Calculate rate of change over different periods
        roc_5 = (close[-1] - close[-6]) / close[-6]
        roc_10 = (close[-1] - close[-11]) / close[-11]
        roc_20 = (close[-1] - close[-21]) / close[-21]
        
        # Acceleration is when shorter periods show stronger momentum
        if roc_5 > roc_10 > roc_20 and roc_5 > 0:
            return min(roc_5 / 0.05, 1.0)  # Normalize positive acceleration
        elif roc_5 < roc_10 < roc_20 and roc_5 < 0:
            return max(roc_5 / 0.05, -1.0)  # Normalize negative acceleration
        else:
            return 0.0
    
    def _calculate_momentum_consistency(self, close) -> float:
        """Calculate how consistent the momentum is."""
        
        # Calculate returns over last 10 periods
        returns = []
        for i in range(1, 11):
            if len(close) > i:
                returns.append((close[-i] - close[-i-1]) / close[-i-1])
        
        if not returns:
            return 0.0
        
        # Consistency is measured by how many returns have the same sign
        positive_returns = sum(1 for r in returns if r > 0)
        negative_returns = sum(1 for r in returns if r < 0)
        
        consistency = max(positive_returns, negative_returns) / len(returns)
        return consistency
    
    def _analyze_breakout_strength(self, high, low, close) -> float:
        """Analyze strength of price breakout."""
        
        # Calculate recent high/low levels
        recent_high = np.max(high[-20:-1])  # Exclude current candle
        recent_low = np.min(low[-20:-1])
        
        current_price = close[-1]
        
        # Check for breakout
        if current_price > recent_high:
            # Bullish breakout strength
            breakout_distance = (current_price - recent_high) / recent_high
            return min(breakout_distance / 0.02, 1.0)  # Normalize
        elif current_price < recent_low:
            # Bearish breakout strength
            breakout_distance = (recent_low - current_price) / recent_low
            return min(breakout_distance / 0.02, 1.0)  # Normalize
        else:
            return 0.0
    
    def _detect_momentum_divergence(self, close, rsi) -> bool:
        """Detect momentum divergence."""
        
        # Simple divergence detection
        # Price making higher highs but RSI making lower highs (bearish divergence)
        # Price making lower lows but RSI making higher lows (bullish divergence)
        
        if len(close) < 20 or len(rsi) < 20:
            return False
        
        # Recent price and RSI peaks/troughs
        price_recent_high = np.max(close[-10:])
        price_previous_high = np.max(close[-20:-10])
        
        rsi_recent_high = np.max(rsi[-10:])
        rsi_previous_high = np.max(rsi[-20:-10])
        
        # Bearish divergence
        if price_recent_high > price_previous_high and rsi_recent_high < rsi_previous_high:
            return True
        
        # Bullish divergence
        price_recent_low = np.min(close[-10:])
        price_previous_low = np.min(close[-20:-10])
        
        rsi_recent_low = np.min(rsi[-10:])
        rsi_previous_low = np.min(rsi[-20:-10])
        
        if price_recent_low < price_previous_low and rsi_recent_low > rsi_previous_low:
            return True
        
        return False
    
    def _check_stochastic_alignment(self, stoch_k, stoch_d) -> str:
        """Check stochastic alignment."""
        
        if stoch_k[-1] > stoch_d[-1] and stoch_k[-1] > 50:
            return "bullish"
        elif stoch_k[-1] < stoch_d[-1] and stoch_k[-1] < 50:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_breakout_momentum(self, high, low, close) -> float:
        """Calculate momentum of breakout."""
        
        # Volume-weighted momentum calculation would be ideal
        # For now, use price momentum at breakout levels
        
        range_20 = np.max(high[-20:]) - np.min(low[-20:])
        current_momentum = abs(close[-1] - close[-2]) / close[-2]
        
        # Normalize momentum relative to recent range
        normalized_momentum = current_momentum / (range_20 / close[-1])
        
        return min(normalized_momentum * 10, 1.0)
    
    def _calculate_momentum_alignment(self, rsi_7, rsi_14, macd_hist) -> float:
        """Calculate alignment between different momentum indicators."""
        
        # All indicators should point in same direction
        rsi_7_signal = 1 if rsi_7 > 50 else -1
        rsi_14_signal = 1 if rsi_14 > 50 else -1
        macd_signal = 1 if macd_hist > 0 else -1
        
        alignment = (rsi_7_signal + rsi_14_signal + macd_signal) / 3.0
        
        return abs(alignment)  # Return strength of alignment
    
    async def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get price data (simulated for ETH)."""
        
        periods = 200
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='15T')
        
        # Generate realistic ETH price data with higher volatility
        base_price = 3000
        returns = np.random.normal(0.0003, 0.025, periods)  # ETH-like volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.015, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.015, periods))),
            'close': prices,
            'volume': np.random.normal(800, 150, periods)
        })
        
        return df

# Test the ETH Momentum Bot
async def test_eth_momentum_bot():
    """Test the ETH Momentum Bot."""
    
    bot = ETHMomentumBot()
    
    print("⚡ TESTE DO ETH MOMENTUM BOT")
    print("=" * 50)
    
    # Run specialized cycle
    signals = await bot.run_specialized_cycle()
    
    print(f"Sinais gerados: {len(signals)}")
    
    for signal in signals:
        print(f"\n📊 SINAL ETH MOMENTUM:")
        print(f"Direção: {signal['direction']}")
        print(f"Preço de entrada: ${signal['entry_price']:.2f}")
        print(f"Stop loss: ${signal['stop_loss']:.2f}")
        print(f"Take profit: ${signal['take_profit']:.2f}")
        print(f"Confiança: {signal['confidence']:.2f}")
        print(f"Tipo de momentum: {signal['momentum_type']}")
        print(f"R/R: {signal['risk_reward_ratio']:.1f}")
    
    # Performance summary
    summary = bot.get_performance_summary()
    print(f"\n📈 RESUMO DE PERFORMANCE:")
    print(f"Bot: {summary['bot_name']}")
    print(f"Especialização: {summary['specialization']}")
    print(f"Score de especialização: {summary['stats']['specialization_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_eth_momentum_bot())