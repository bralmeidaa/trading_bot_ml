#!/usr/bin/env python3
"""
Advanced Signal Quality Filters
Implementa filtros multi-camada para melhorar assertividade dos sinais
Meta: Reduzir trades para ~5/dia com 55%+ win rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json

class AdvancedSignalFilter:
    """Filtro avançado de qualidade de sinais."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_weights = {
            'technical': 0.25,
            'market_structure': 0.25,
            'binance_sentiment': 0.25,
            'ml_confidence': 0.25
        }
        self.quality_threshold = 0.65  # Ajustado para permitir mais sinais de qualidade
        
    async def evaluate_signal_quality(self, signal_data: Dict) -> Dict:
        """Avalia qualidade do sinal usando filtros multi-camada."""
        
        # Layer 1: Filtro Técnico Base
        technical_score = await self._evaluate_technical_layer(signal_data)
        
        # Layer 2: Estrutura de Mercado
        structure_score = await self._evaluate_market_structure(signal_data)
        
        # Layer 3: Sentiment Binance
        sentiment_score = await self._evaluate_binance_sentiment(signal_data)
        
        # Layer 4: Confiança ML Rigorosa
        ml_score = await self._evaluate_ml_confidence(signal_data)
        
        # Calcular score final
        quality_score = (
            technical_score * self.quality_weights['technical'] +
            structure_score * self.quality_weights['market_structure'] +
            sentiment_score * self.quality_weights['binance_sentiment'] +
            ml_score * self.quality_weights['ml_confidence']
        )
        
        return {
            'quality_score': quality_score,
            'pass_threshold': quality_score >= self.quality_threshold,
            'layer_scores': {
                'technical': technical_score,
                'market_structure': structure_score,
                'binance_sentiment': sentiment_score,
                'ml_confidence': ml_score
            },
            'rejection_reasons': self._get_rejection_reasons(
                technical_score, structure_score, sentiment_score, ml_score
            )
        }
    
    async def _evaluate_technical_layer(self, signal_data: Dict) -> float:
        """Layer 1: Avalia condições técnicas básicas."""
        df = signal_data['price_data']
        score = 0.0
        max_score = 4.0
        
        # RSI entre 30-70 (evitar extremos)
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        current_rsi = rsi[-1]
        if 30 <= current_rsi <= 70:
            score += 1.0
        elif 25 <= current_rsi <= 75:
            score += 0.5
            
        # MACD com divergência confirmada
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        if len(macd_hist) >= 2:
            if signal_data['direction'] == 'long' and macd_hist[-1] > macd_hist[-2] > 0:
                score += 1.0
            elif signal_data['direction'] == 'short' and macd_hist[-1] < macd_hist[-2] < 0:
                score += 1.0
            elif abs(macd_hist[-1]) > abs(macd_hist[-2]):
                score += 0.5
                
        # Volume acima da média 20 períodos
        volume_ma = df['volume'].rolling(20).mean()
        current_volume = df['volume'].iloc[-1]
        if current_volume > volume_ma.iloc[-1] * 1.2:
            score += 1.0
        elif current_volume > volume_ma.iloc[-1]:
            score += 0.5
            
        # Bollinger Bands não em squeeze
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values)
        bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        bb_width_ma = np.mean([(bb_upper[i] - bb_lower[i]) / bb_middle[i] 
                              for i in range(-20, -1)])
        if bb_width > bb_width_ma * 0.8:  # Não em squeeze extremo
            score += 1.0
        elif bb_width > bb_width_ma * 0.6:
            score += 0.5
            
        return score / max_score
    
    async def _evaluate_market_structure(self, signal_data: Dict) -> float:
        """Layer 2: Avalia estrutura de mercado."""
        df = signal_data['price_data']
        score = 0.0
        max_score = 4.0
        
        # Suporte/Resistência próximos identificados
        support_resistance = self._find_support_resistance(df)
        current_price = df['close'].iloc[-1]
        
        # Verificar se não está muito próximo de S/R (pode causar rejeição)
        min_distance = current_price * 0.005  # 0.5% mínimo
        near_sr = any(abs(current_price - level) < min_distance 
                     for level in support_resistance)
        if not near_sr:
            score += 1.0
        elif len(support_resistance) > 0:
            score += 0.5
            
        # Trend de curto prazo alinhado com médio prazo
        ma_short = talib.SMA(df['close'].values, timeperiod=10)
        ma_medium = talib.SMA(df['close'].values, timeperiod=50)
        
        short_trend = 1 if ma_short[-1] > ma_short[-5] else -1
        medium_trend = 1 if ma_medium[-1] > ma_medium[-10] else -1
        
        if signal_data['direction'] == 'long' and short_trend == medium_trend == 1:
            score += 1.0
        elif signal_data['direction'] == 'short' and short_trend == medium_trend == -1:
            score += 1.0
        elif short_trend == medium_trend:
            score += 0.5
            
        # Não há eventos de alta volatilidade programados
        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
        current_atr = atr[-1]
        avg_atr = np.mean(atr[-20:])
        
        if current_atr <= avg_atr * 1.5:  # Volatilidade normal
            score += 1.0
        elif current_atr <= avg_atr * 2.0:
            score += 0.5
            
        # Correlação com BTC dentro de limites normais (para altcoins)
        if signal_data['symbol'] != 'BTC/USDT':
            # Simulação de correlação (em implementação real, usar dados BTC)
            correlation_score = 0.7  # Placeholder
            if 0.3 <= correlation_score <= 0.8:  # Correlação moderada
                score += 1.0
            else:
                score += 0.3
        else:
            score += 1.0  # BTC sempre passa neste critério
            
        return score / max_score
    
    async def _evaluate_binance_sentiment(self, signal_data: Dict) -> float:
        """Layer 3: Avalia sentiment usando dados da Binance."""
        score = 0.0
        max_score = 4.0
        
        try:
            # Funding rate não extremo (±0.1%)
            funding_rate = await self._get_funding_rate(signal_data['symbol'])
            if funding_rate is not None:
                if abs(funding_rate) <= 0.001:  # ±0.1%
                    score += 1.0
                elif abs(funding_rate) <= 0.002:  # ±0.2%
                    score += 0.5
            else:
                score += 0.5  # Neutro se não disponível
                
            # Long/Short ratio balanceado (0.8-1.2)
            ls_ratio = await self._get_long_short_ratio(signal_data['symbol'])
            if ls_ratio is not None:
                if 0.8 <= ls_ratio <= 1.2:
                    score += 1.0
                elif 0.6 <= ls_ratio <= 1.5:
                    score += 0.5
            else:
                score += 0.5
                
            # Open Interest crescente (momentum)
            oi_trend = await self._get_oi_trend(signal_data['symbol'])
            if oi_trend == 'increasing':
                score += 1.0
            elif oi_trend == 'stable':
                score += 0.5
            else:
                score += 0.5  # Neutro se não disponível
                
            # Taker buy/sell ratio favorável
            taker_ratio = await self._get_taker_ratio(signal_data['symbol'])
            if taker_ratio is not None:
                if signal_data['direction'] == 'long' and taker_ratio > 1.1:
                    score += 1.0
                elif signal_data['direction'] == 'short' and taker_ratio < 0.9:
                    score += 1.0
                elif 0.9 <= taker_ratio <= 1.1:
                    score += 0.7  # Neutro é ok
                else:
                    score += 0.3
            else:
                score += 0.5
                
        except Exception as e:
            print(f"Erro ao avaliar sentiment Binance: {e}")
            score = max_score * 0.5  # Score neutro em caso de erro
            
        return score / max_score
    
    async def _evaluate_ml_confidence(self, signal_data: Dict) -> float:
        """Layer 4: Avalia confiança ML de forma rigorosa."""
        score = 0.0
        max_score = 4.0
        
        # Confidence score > 0.75 (vs atual 0.55-0.65)
        ml_confidence = signal_data.get('ml_confidence', 0.0)
        if ml_confidence >= 0.80:
            score += 1.0
        elif ml_confidence >= 0.75:
            score += 0.7
        elif ml_confidence >= 0.70:
            score += 0.4
            
        # Múltiplos modelos concordando (ensemble)
        ensemble_agreement = signal_data.get('ensemble_agreement', 0.0)
        if ensemble_agreement >= 0.8:
            score += 1.0
        elif ensemble_agreement >= 0.7:
            score += 0.6
        elif ensemble_agreement >= 0.6:
            score += 0.3
            
        # Feature importance alta nos indicadores chave
        feature_importance = signal_data.get('feature_importance', {})
        key_features = ['rsi', 'macd', 'volume', 'price_momentum']
        avg_importance = np.mean([feature_importance.get(f, 0) for f in key_features])
        
        if avg_importance >= 0.7:
            score += 1.0
        elif avg_importance >= 0.5:
            score += 0.6
        elif avg_importance >= 0.3:
            score += 0.3
            
        # Backtesting recente positivo
        recent_performance = signal_data.get('recent_backtest_winrate', 0.0)
        if recent_performance >= 0.6:
            score += 1.0
        elif recent_performance >= 0.5:
            score += 0.6
        elif recent_performance >= 0.4:
            score += 0.3
            
        return score / max_score
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> List[float]:
        """Encontra níveis de suporte e resistência."""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
                
        # Retornar níveis únicos mais recentes
        all_levels = resistance_levels + support_levels
        return list(set(all_levels))[-10:]  # Últimos 10 níveis únicos
    
    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Obtém funding rate da Binance."""
        try:
            # Placeholder - implementar chamada real à API
            # Em produção: usar endpoint /fapi/v1/fundingRate
            return 0.0001  # Simulação de funding rate neutro
        except:
            return None
    
    async def _get_long_short_ratio(self, symbol: str) -> Optional[float]:
        """Obtém ratio long/short da Binance."""
        try:
            # Placeholder - implementar chamada real à API
            # Em produção: usar endpoint /fapi/v1/globalLongShortAccountRatio
            return 1.0  # Simulação de ratio balanceado
        except:
            return None
    
    async def _get_oi_trend(self, symbol: str) -> str:
        """Obtém tendência do Open Interest."""
        try:
            # Placeholder - implementar chamada real à API
            # Em produção: usar endpoint /fapi/v1/openInterest
            return 'stable'  # Simulação
        except:
            return 'unknown'
    
    async def _get_taker_ratio(self, symbol: str) -> Optional[float]:
        """Obtém ratio taker buy/sell."""
        try:
            # Placeholder - implementar chamada real à API
            # Em produção: usar endpoint /fapi/v1/takerlongshortRatio
            return 1.0  # Simulação de ratio neutro
        except:
            return None
    
    def _get_rejection_reasons(self, tech: float, struct: float, 
                             sent: float, ml: float) -> List[str]:
        """Retorna razões para rejeição do sinal."""
        reasons = []
        
        if tech < 0.6:
            reasons.append("Condições técnicas desfavoráveis")
        if struct < 0.6:
            reasons.append("Estrutura de mercado inadequada")
        if sent < 0.6:
            reasons.append("Sentiment Binance negativo")
        if ml < 0.7:
            reasons.append("Confiança ML insuficiente")
            
        return reasons

class QualityBasedTradingSystem:
    """Sistema de trading baseado em qualidade de sinais."""
    
    def __init__(self, config_path: str = 'trading_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.signal_filter = AdvancedSignalFilter(self.config)
        self.daily_trade_count = 0
        self.max_daily_trades = 5
        self.quality_stats = {
            'signals_evaluated': 0,
            'signals_passed': 0,
            'signals_rejected': 0,
            'avg_quality_score': 0.0
        }
    
    async def evaluate_trading_signal(self, signal_data: Dict) -> Dict:
        """Avalia sinal de trading com filtros de qualidade."""
        
        # Verificar limite diário de trades
        if self.daily_trade_count >= self.max_daily_trades:
            return {
                'approved': False,
                'reason': 'Daily trade limit reached',
                'quality_score': 0.0
            }
        
        # Avaliar qualidade do sinal
        quality_result = await self.signal_filter.evaluate_signal_quality(signal_data)
        
        # Atualizar estatísticas
        self.quality_stats['signals_evaluated'] += 1
        if quality_result['pass_threshold']:
            self.quality_stats['signals_passed'] += 1
            self.daily_trade_count += 1
        else:
            self.quality_stats['signals_rejected'] += 1
        
        # Atualizar média de qualidade
        current_avg = self.quality_stats['avg_quality_score']
        n = self.quality_stats['signals_evaluated']
        new_score = quality_result['quality_score']
        self.quality_stats['avg_quality_score'] = (current_avg * (n-1) + new_score) / n
        
        return {
            'approved': quality_result['pass_threshold'],
            'quality_score': quality_result['quality_score'],
            'layer_scores': quality_result['layer_scores'],
            'rejection_reasons': quality_result['rejection_reasons'],
            'daily_trades_remaining': self.max_daily_trades - self.daily_trade_count
        }
    
    def get_quality_statistics(self) -> Dict:
        """Retorna estatísticas de qualidade."""
        total = self.quality_stats['signals_evaluated']
        if total == 0:
            return self.quality_stats
        
        return {
            **self.quality_stats,
            'pass_rate': self.quality_stats['signals_passed'] / total,
            'rejection_rate': self.quality_stats['signals_rejected'] / total
        }
    
    def reset_daily_counters(self):
        """Reseta contadores diários (chamar a cada novo dia)."""
        self.daily_trade_count = 0

# Exemplo de uso
async def test_signal_quality():
    """Testa o sistema de qualidade de sinais."""
    
    # Dados de exemplo de um sinal
    sample_signal = {
        'symbol': 'BTC/USDT',
        'direction': 'long',
        'ml_confidence': 0.78,
        'ensemble_agreement': 0.82,
        'feature_importance': {
            'rsi': 0.8,
            'macd': 0.7,
            'volume': 0.6,
            'price_momentum': 0.75
        },
        'recent_backtest_winrate': 0.65,
        'price_data': pd.DataFrame({
            'open': np.random.randn(100) + 50000,
            'high': np.random.randn(100) + 50100,
            'low': np.random.randn(100) + 49900,
            'close': np.random.randn(100) + 50000,
            'volume': np.random.randn(100) * 1000 + 10000
        })
    }
    
    # Testar sistema
    trading_system = QualityBasedTradingSystem()
    result = await trading_system.evaluate_trading_signal(sample_signal)
    
    print("🔍 TESTE DO SISTEMA DE QUALIDADE")
    print("=" * 50)
    print(f"Sinal aprovado: {result['approved']}")
    print(f"Score de qualidade: {result['quality_score']:.3f}")
    print(f"Trades restantes hoje: {result['daily_trades_remaining']}")
    
    if not result['approved']:
        print(f"Razões de rejeição: {result['rejection_reasons']}")
    
    print(f"\nScores por camada:")
    for layer, score in result['layer_scores'].items():
        print(f"  {layer}: {score:.3f}")

if __name__ == "__main__":
    asyncio.run(test_signal_quality())