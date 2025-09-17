#!/usr/bin/env python3
"""
Market Regime Detection System
Detecta automaticamente o regime de mercado e adapta estratégias
Regimes: Trending Bull, Trending Bear, Ranging, High Volatility
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

class MarketRegime(Enum):
    """Tipos de regime de mercado."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    TRANSITIONAL = "transitional"

@dataclass
class RegimeAnalysis:
    """Resultado da análise de regime."""
    regime: MarketRegime
    confidence: float
    trend_strength: float
    volatility_level: float
    volume_profile: str
    breakout_frequency: float
    support_resistance_strength: float
    regime_duration: int  # períodos no regime atual
    factors: List[str]
    timestamp: datetime

class MarketRegimeDetector:
    """Detector avançado de regime de mercado."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.regime_history = {}  # Histórico por símbolo
        self.regime_cache = {}    # Cache de análises recentes
        
    def _default_config(self) -> Dict:
        """Configuração padrão do detector."""
        return {
            "trend_strength_periods": [20, 50, 100],
            "volatility_periods": [14, 30],
            "volume_periods": [20, 50],
            "breakout_lookback": 20,
            "regime_confirmation_periods": 3,
            "thresholds": {
                "strong_trend": 0.7,
                "weak_trend": 0.3,
                "high_volatility": 0.8,
                "normal_volatility": 0.5,
                "high_volume": 1.5,
                "low_volume": 0.7
            }
        }
    
    async def detect_regime(self, symbol: str, price_data: pd.DataFrame) -> RegimeAnalysis:
        """Detecta o regime de mercado atual."""
        
        # Calcular indicadores base
        indicators = self._calculate_regime_indicators(price_data)
        
        # Analisar cada componente
        trend_analysis = self._analyze_trend(indicators)
        volatility_analysis = self._analyze_volatility(indicators)
        volume_analysis = self._analyze_volume(indicators)
        structure_analysis = self._analyze_market_structure(price_data, indicators)
        
        # Determinar regime
        regime, confidence = self._determine_regime(
            trend_analysis, volatility_analysis, volume_analysis, structure_analysis
        )
        
        # Calcular duração do regime
        regime_duration = self._calculate_regime_duration(symbol, regime)
        
        # Compilar fatores
        factors = self._compile_regime_factors(
            trend_analysis, volatility_analysis, volume_analysis, structure_analysis
        )
        
        analysis = RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_analysis['strength'],
            volatility_level=volatility_analysis['level'],
            volume_profile=volume_analysis['profile'],
            breakout_frequency=structure_analysis['breakout_frequency'],
            support_resistance_strength=structure_analysis['sr_strength'],
            regime_duration=regime_duration,
            factors=factors,
            timestamp=datetime.now()
        )
        
        # Atualizar histórico
        self._update_regime_history(symbol, analysis)
        
        return analysis
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcula indicadores para análise de regime."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # Trend Indicators
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)
        indicators['sma_100'] = talib.SMA(close, timeperiod=100)
        indicators['ema_20'] = talib.EMA(close, timeperiod=20)
        indicators['ema_50'] = talib.EMA(close, timeperiod=50)
        
        # Volatility Indicators
        indicators['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        indicators['atr_30'] = talib.ATR(high, low, close, timeperiod=30)
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
        
        # Momentum Indicators
        indicators['rsi'] = talib.RSI(close, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Volume Indicators
        indicators['volume_sma_20'] = talib.SMA(volume.astype(float), timeperiod=20)
        indicators['volume_sma_50'] = talib.SMA(volume.astype(float), timeperiod=50)
        indicators['obv'] = talib.OBV(close, volume.astype(float))
        
        # Price Action
        indicators['high_20'] = pd.Series(high).rolling(20).max().values
        indicators['low_20'] = pd.Series(low).rolling(20).min().values
        indicators['close_series'] = close
        
        return indicators
    
    def _analyze_trend(self, indicators: Dict) -> Dict:
        """Analisa força e direção da tendência."""
        
        # Análise de médias móveis
        sma_20 = indicators['sma_20'][-1]
        sma_50 = indicators['sma_50'][-1]
        sma_100 = indicators['sma_100'][-1]
        current_price = indicators['close_series'][-1]
        
        # Direção da tendência
        trend_direction = 0
        if sma_20 > sma_50 > sma_100 and current_price > sma_20:
            trend_direction = 1  # Bullish
        elif sma_20 < sma_50 < sma_100 and current_price < sma_20:
            trend_direction = -1  # Bearish
        
        # Força da tendência usando ADX
        adx = indicators['adx'][-1] if not np.isnan(indicators['adx'][-1]) else 25
        trend_strength = min(adx / 50.0, 1.0)  # Normalizar ADX
        
        # Consistência da tendência
        ma_alignment = 0
        if sma_20 > sma_50 > sma_100:
            ma_alignment = 1
        elif sma_20 < sma_50 < sma_100:
            ma_alignment = 1
        
        # Momentum da tendência
        price_above_ma20 = current_price > sma_20
        ma20_slope = (sma_20 - indicators['sma_20'][-5]) / indicators['sma_20'][-5]
        
        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'consistency': ma_alignment,
            'momentum': abs(ma20_slope),
            'price_vs_ma': price_above_ma20
        }
    
    def _analyze_volatility(self, indicators: Dict) -> Dict:
        """Analisa nível de volatilidade."""
        
        # ATR normalizado
        atr_14 = indicators['atr_14'][-1]
        atr_30 = indicators['atr_30'][-1]
        current_price = indicators['close_series'][-1]
        
        atr_pct_14 = atr_14 / current_price
        atr_pct_30 = atr_30 / current_price
        
        # Bollinger Bands width
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        bb_middle = indicators['bb_middle'][-1]
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Volatility level
        volatility_score = (atr_pct_14 * 2 + atr_pct_30 + bb_width) / 4
        
        # Volatility trend
        atr_trend = (atr_14 - indicators['atr_14'][-5]) / indicators['atr_14'][-5]
        
        return {
            'level': min(volatility_score * 10, 1.0),  # Normalizar
            'trend': atr_trend,
            'atr_pct': atr_pct_14,
            'bb_width': bb_width
        }
    
    def _analyze_volume(self, indicators: Dict) -> Dict:
        """Analisa perfil de volume."""
        
        current_volume = indicators['volume_sma_20'][-1]
        avg_volume_20 = np.mean(indicators['volume_sma_20'][-20:])
        avg_volume_50 = indicators['volume_sma_50'][-1]
        
        # Volume ratio
        volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
        volume_ratio_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
        
        # Volume trend
        volume_trend = (current_volume - indicators['volume_sma_20'][-5]) / indicators['volume_sma_20'][-5]
        
        # Volume profile
        if volume_ratio_20 > 1.5:
            profile = "high"
        elif volume_ratio_20 < 0.7:
            profile = "low"
        else:
            profile = "normal"
        
        return {
            'profile': profile,
            'ratio_20': volume_ratio_20,
            'ratio_50': volume_ratio_50,
            'trend': volume_trend
        }
    
    def _analyze_market_structure(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analisa estrutura de mercado."""
        
        # Support/Resistance strength
        high_20 = indicators['high_20'][-1]
        low_20 = indicators['low_20'][-1]
        current_price = indicators['close_series'][-1]
        
        # Distance from S/R levels
        distance_from_high = (high_20 - current_price) / current_price
        distance_from_low = (current_price - low_20) / current_price
        
        # S/R strength (quanto mais próximo, mais forte)
        sr_strength = 1 - min(distance_from_high, distance_from_low)
        
        # Breakout frequency (últimos 20 períodos)
        breakouts = 0
        for i in range(-20, -1):
            if i < -len(df):
                continue
            if df['high'].iloc[i] > indicators['high_20'][i-1]:
                breakouts += 1
            if df['low'].iloc[i] < indicators['low_20'][i-1]:
                breakouts += 1
        
        breakout_frequency = breakouts / 20.0
        
        # Higher highs and lower lows
        recent_highs = df['high'].tail(10).values
        recent_lows = df['low'].tail(10).values
        
        higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
        
        return {
            'sr_strength': sr_strength,
            'breakout_frequency': breakout_frequency,
            'higher_highs': higher_highs,
            'lower_lows': lower_lows,
            'distance_from_high': distance_from_high,
            'distance_from_low': distance_from_low
        }
    
    def _determine_regime(self, trend: Dict, volatility: Dict, 
                         volume: Dict, structure: Dict) -> Tuple[MarketRegime, float]:
        """Determina o regime de mercado baseado nas análises."""
        
        thresholds = self.config['thresholds']
        
        # Scores para cada regime
        regime_scores = {
            MarketRegime.TRENDING_BULL: 0,
            MarketRegime.TRENDING_BEAR: 0,
            MarketRegime.RANGING: 0,
            MarketRegime.HIGH_VOLATILITY: 0,
            MarketRegime.TRANSITIONAL: 0
        }
        
        # Trending Bull
        if (trend['direction'] == 1 and 
            trend['strength'] > thresholds['strong_trend'] and
            volatility['level'] < thresholds['high_volatility']):
            regime_scores[MarketRegime.TRENDING_BULL] += 0.8
            
        if volume['profile'] == "high" and trend['direction'] == 1:
            regime_scores[MarketRegime.TRENDING_BULL] += 0.2
            
        # Trending Bear
        if (trend['direction'] == -1 and 
            trend['strength'] > thresholds['strong_trend'] and
            volatility['level'] < thresholds['high_volatility']):
            regime_scores[MarketRegime.TRENDING_BEAR] += 0.8
            
        if volume['profile'] == "high" and trend['direction'] == -1:
            regime_scores[MarketRegime.TRENDING_BEAR] += 0.2
            
        # Ranging
        if (trend['strength'] < thresholds['weak_trend'] and
            volatility['level'] < thresholds['normal_volatility'] and
            structure['breakout_frequency'] < 0.3):
            regime_scores[MarketRegime.RANGING] += 0.9
            
        if structure['sr_strength'] > 0.7:
            regime_scores[MarketRegime.RANGING] += 0.1
            
        # High Volatility
        if volatility['level'] > thresholds['high_volatility']:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.7
            
        if structure['breakout_frequency'] > 0.6:
            regime_scores[MarketRegime.HIGH_VOLATILITY] += 0.3
            
        # Transitional (fallback)
        if max(regime_scores.values()) < 0.6:
            regime_scores[MarketRegime.TRANSITIONAL] = 0.5
        
        # Determinar regime com maior score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        return best_regime, confidence
    
    def _calculate_regime_duration(self, symbol: str, current_regime: MarketRegime) -> int:
        """Calcula há quantos períodos estamos no regime atual."""
        if symbol not in self.regime_history:
            return 1
        
        history = self.regime_history[symbol]
        if not history:
            return 1
        
        # Contar períodos consecutivos no mesmo regime
        duration = 1
        for analysis in reversed(history[-10:]):  # Últimos 10 períodos
            if analysis.regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _compile_regime_factors(self, trend: Dict, volatility: Dict, 
                               volume: Dict, structure: Dict) -> List[str]:
        """Compila fatores que influenciaram a determinação do regime."""
        factors = []
        
        # Trend factors
        if trend['strength'] > 0.7:
            direction = "alta" if trend['direction'] == 1 else "baixa"
            factors.append(f"Tendência forte de {direction}")
        elif trend['strength'] < 0.3:
            factors.append("Tendência fraca/lateral")
        
        # Volatility factors
        if volatility['level'] > 0.8:
            factors.append("Volatilidade extrema")
        elif volatility['level'] < 0.3:
            factors.append("Volatilidade baixa")
        
        # Volume factors
        if volume['profile'] == "high":
            factors.append("Volume alto")
        elif volume['profile'] == "low":
            factors.append("Volume baixo")
        
        # Structure factors
        if structure['breakout_frequency'] > 0.5:
            factors.append("Muitos breakouts")
        elif structure['breakout_frequency'] < 0.2:
            factors.append("Poucos breakouts")
        
        if structure['sr_strength'] > 0.7:
            factors.append("Suporte/resistência forte")
        
        return factors
    
    def _update_regime_history(self, symbol: str, analysis: RegimeAnalysis):
        """Atualiza histórico de regimes."""
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        self.regime_history[symbol].append(analysis)
        
        # Manter apenas últimos 100 registros
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]
    
    def get_regime_statistics(self, symbol: str) -> Dict:
        """Retorna estatísticas do regime para um símbolo."""
        if symbol not in self.regime_history:
            return {}
        
        history = self.regime_history[symbol]
        if not history:
            return {}
        
        # Contar regimes
        regime_counts = {}
        for analysis in history[-50:]:  # Últimos 50 períodos
            regime = analysis.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Calcular médias
        avg_confidence = np.mean([a.confidence for a in history[-20:]])
        avg_volatility = np.mean([a.volatility_level for a in history[-20:]])
        avg_trend_strength = np.mean([a.trend_strength for a in history[-20:]])
        
        return {
            'regime_distribution': regime_counts,
            'avg_confidence': avg_confidence,
            'avg_volatility': avg_volatility,
            'avg_trend_strength': avg_trend_strength,
            'current_regime': history[-1].regime.value if history else None,
            'regime_duration': history[-1].regime_duration if history else 0
        }

# Exemplo de uso
async def test_regime_detection():
    """Testa o detector de regime."""
    
    # Simular dados de preço
    dates = pd.date_range(end=datetime.now(), periods=200, freq='15T')
    
    # Simular diferentes regimes
    base_price = 50000
    prices = []
    
    # Trending bull (primeiros 50 períodos)
    trend_prices = base_price * np.exp(np.cumsum(np.random.normal(0.001, 0.01, 50)))
    prices.extend(trend_prices)
    
    # Ranging (próximos 50 períodos)
    range_prices = trend_prices[-1] + np.random.normal(0, trend_prices[-1] * 0.02, 50)
    prices.extend(range_prices)
    
    # High volatility (próximos 50 períodos)
    vol_prices = range_prices[-1] * np.exp(np.cumsum(np.random.normal(0, 0.05, 50)))
    prices.extend(vol_prices)
    
    # Trending bear (últimos 50 períodos)
    bear_prices = vol_prices[-1] * np.exp(np.cumsum(np.random.normal(-0.001, 0.01, 50)))
    prices.extend(bear_prices)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.normal(10000, 2000, 200)
    })
    
    # Testar detector
    detector = MarketRegimeDetector()
    
    print("🔍 TESTE DO DETECTOR DE REGIME")
    print("=" * 50)
    
    # Testar diferentes períodos
    test_periods = [50, 100, 150, 200]  # Diferentes regimes simulados
    
    for period in test_periods:
        test_data = df.iloc[:period]
        analysis = await detector.detect_regime("BTC/USDT", test_data)
        
        print(f"\n📊 Período {period} (Regime esperado: {['Trending Bull', 'Ranging', 'High Vol', 'Trending Bear'][min(period//50, 3)]})")
        print(f"Regime detectado: {analysis.regime.value}")
        print(f"Confiança: {analysis.confidence:.2f}")
        print(f"Força da tendência: {analysis.trend_strength:.2f}")
        print(f"Volatilidade: {analysis.volatility_level:.2f}")
        print(f"Perfil de volume: {analysis.volume_profile}")
        print(f"Duração do regime: {analysis.regime_duration} períodos")
        print(f"Fatores: {', '.join(analysis.factors)}")
    
    # Estatísticas finais
    stats = detector.get_regime_statistics("BTC/USDT")
    print(f"\n📈 ESTATÍSTICAS FINAIS:")
    print(f"Distribuição de regimes: {stats.get('regime_distribution', {})}")
    print(f"Confiança média: {stats.get('avg_confidence', 0):.2f}")
    print(f"Volatilidade média: {stats.get('avg_volatility', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(test_regime_detection())