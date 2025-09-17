#!/usr/bin/env python3
"""
Sistema de Trading Integrado com Detecção de Regime
Combina o sistema otimizado V2 com detecção de regime de mercado
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from optimized_trading_system_v2 import OptimizedTradingSystemV2, TradingSignal
from market_regime_detector import MarketRegimeDetector, MarketRegime
from regime_based_strategy import RegimeBasedStrategyAdapter
from advanced_signal_filters import QualityBasedTradingSystem
from binance_advanced_data import BinanceAdvancedData, MarketSentimentAnalyzer

class RegimeIntegratedTradingSystem:
    """Sistema de trading com detecção de regime integrada."""
    
    def __init__(self, config_path: str = 'trading_config.json'):
        # Sistemas base
        self.base_system = OptimizedTradingSystemV2(config_path)
        self.regime_adapter = RegimeBasedStrategyAdapter()
        self.regime_detector = MarketRegimeDetector()
        
        # Estado do sistema
        self.current_regimes = {}
        self.regime_stats = {}
        self.enhanced_stats = {
            'regime_changes_today': 0,
            'trades_by_regime': {},
            'regime_performance': {},
            'adaptive_adjustments': 0
        }
        
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Inicializa todos os componentes."""
        await self.base_system.initialize()
        self.logger.info("Sistema de Trading com Regime Integrado inicializado")
    
    async def cleanup(self):
        """Limpa recursos."""
        await self.base_system.cleanup()
    
    async def enhanced_signal_generation(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Gera sinal com análise de regime integrada."""
        
        try:
            # 1. Obter dados de preço
            price_data = await self.base_system._get_price_data(symbol, timeframe)
            if price_data is None:
                return None
            
            # 2. Detectar regime atual
            regime_result = await self.regime_adapter.analyze_and_adapt(symbol, price_data)
            regime_analysis = regime_result['regime_analysis']
            regime_config = regime_result['regime_config']
            
            # 3. Atualizar regime atual
            self.current_regimes[symbol] = regime_analysis.regime
            
            # 4. Verificar se deve fazer trade neste regime
            should_trade, trade_reason = self.regime_adapter.should_trade(
                symbol, 0.75, 0.80  # Valores temporários
            )
            
            if not should_trade:
                self.logger.info(f"Trade pausado para {symbol}: {trade_reason}")
                return None
            
            # 5. Gerar sinal base
            base_signal = await self.base_system.generate_enhanced_signal(symbol, timeframe)
            if base_signal is None:
                return None
            
            # 6. Adaptar sinal baseado no regime
            enhanced_signal = await self._adapt_signal_to_regime(
                base_signal, regime_analysis, regime_config
            )
            
            # 7. Validação final com regime
            final_validation = await self._final_regime_validation(
                enhanced_signal, regime_analysis, regime_config
            )
            
            if not final_validation['approved']:
                self.logger.info(f"Sinal rejeitado na validação final: {final_validation['reason']}")
                return None
            
            # 8. Adicionar metadados de regime
            enhanced_signal.regime_data = {
                'regime': regime_analysis.regime.value,
                'confidence': regime_analysis.confidence,
                'regime_duration': regime_analysis.regime_duration,
                'strategy_type': regime_config.strategy_type,
                'adapted_params': {
                    'risk_per_trade': regime_config.risk_per_trade,
                    'stop_loss_multiplier': regime_config.stop_loss_multiplier,
                    'take_profit_multiplier': regime_config.take_profit_multiplier
                }
            }
            
            self.logger.info(f"Sinal gerado com regime {regime_analysis.regime.value} para {symbol}")
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Erro na geração de sinal com regime para {symbol}: {e}")
            return None
    
    async def _adapt_signal_to_regime(self, signal: TradingSignal, 
                                    regime_analysis, regime_config) -> TradingSignal:
        """Adapta sinal baseado no regime detectado."""
        
        # Ajustar stop loss e take profit baseado no regime
        atr_estimate = abs(signal.entry_price - signal.stop_loss) / 2.0  # Estimativa do ATR
        
        # Aplicar multiplicadores do regime
        if signal.direction == 'long':
            signal.stop_loss = signal.entry_price - (atr_estimate * regime_config.stop_loss_multiplier)
            signal.take_profit = signal.entry_price + (atr_estimate * regime_config.take_profit_multiplier)
        else:
            signal.stop_loss = signal.entry_price + (atr_estimate * regime_config.stop_loss_multiplier)
            signal.take_profit = signal.entry_price - (atr_estimate * regime_config.take_profit_multiplier)
        
        # Ajustar confiança baseado no regime
        regime_confidence_bonus = 0.0
        if regime_analysis.regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            if regime_analysis.confidence > 0.8:
                regime_confidence_bonus = 0.1
        elif regime_analysis.regime == MarketRegime.RANGING:
            if regime_analysis.confidence > 0.8:
                regime_confidence_bonus = 0.05
        
        signal.confidence = min(1.0, signal.confidence + regime_confidence_bonus)
        
        return signal
    
    async def _final_regime_validation(self, signal: TradingSignal, 
                                     regime_analysis, regime_config) -> Dict:
        """Validação final considerando o regime."""
        
        # Validações específicas por regime
        if regime_analysis.regime == MarketRegime.HIGH_VOLATILITY:
            if regime_analysis.confidence < 0.9:
                return {
                    'approved': False,
                    'reason': 'High volatility regime requires >90% confidence'
                }
        
        if regime_analysis.regime == MarketRegime.TRANSITIONAL:
            if signal.confidence < 0.8:
                return {
                    'approved': False,
                    'reason': 'Transitional regime requires >80% signal confidence'
                }
        
        # Validar alinhamento de direção com regime
        if regime_analysis.regime == MarketRegime.TRENDING_BULL and signal.direction == 'short':
            if regime_analysis.confidence > 0.7:  # Regime bem estabelecido
                return {
                    'approved': False,
                    'reason': 'Short signal conflicts with established bull trend'
                }
        
        if regime_analysis.regime == MarketRegime.TRENDING_BEAR and signal.direction == 'long':
            if regime_analysis.confidence > 0.7:  # Regime bem estabelecido
                return {
                    'approved': False,
                    'reason': 'Long signal conflicts with established bear trend'
                }
        
        return {'approved': True, 'reason': 'Passed regime validation'}
    
    async def execute_regime_aware_trade(self, signal: TradingSignal) -> Dict:
        """Executa trade com consciência de regime."""
        
        try:
            # Obter configuração do regime
            regime_data = getattr(signal, 'regime_data', {})
            regime = regime_data.get('regime', 'unknown')
            
            # Calcular tamanho da posição baseado no regime
            base_result = await self.base_system.execute_trade(signal)
            if not base_result['success']:
                return base_result
            
            # Ajustar tamanho da posição baseado no regime
            trade_data = base_result['trade_data']
            regime_config = self.regime_adapter.regime_configs.get(
                MarketRegime(regime), None
            )
            
            if regime_config:
                # Aplicar fator de sizing do regime
                original_size = trade_data['position_size']
                adjusted_size = original_size * regime_config.position_sizing_factor
                trade_data['position_size'] = adjusted_size
                trade_data['regime_adjusted'] = True
                trade_data['size_adjustment_factor'] = regime_config.position_sizing_factor
            
            # Adicionar dados de regime ao trade
            trade_data['regime_data'] = regime_data
            
            # Atualizar estatísticas
            self._update_regime_trade_stats(regime, trade_data)
            
            self.logger.info(f"Trade executado com regime {regime}: {trade_data['trade_id']}")
            
            return {
                'success': True,
                'trade_id': trade_data['trade_id'],
                'trade_data': trade_data,
                'regime_applied': regime
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao executar trade com regime: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_regime_trade_stats(self, regime: str, trade_data: Dict):
        """Atualiza estatísticas de trades por regime."""
        if regime not in self.enhanced_stats['trades_by_regime']:
            self.enhanced_stats['trades_by_regime'][regime] = {
                'count': 0,
                'total_risk': 0.0,
                'avg_confidence': 0.0
            }
        
        regime_stats = self.enhanced_stats['trades_by_regime'][regime]
        regime_stats['count'] += 1
        regime_stats['total_risk'] += trade_data.get('position_size', 0) * trade_data.get('entry_price', 0)
        
        # Atualizar média de confiança
        regime_data = trade_data.get('regime_data', {})
        confidence = regime_data.get('confidence', 0)
        current_avg = regime_stats['avg_confidence']
        count = regime_stats['count']
        regime_stats['avg_confidence'] = (current_avg * (count - 1) + confidence) / count
    
    async def run_enhanced_trading_cycle(self):
        """Executa ciclo de trading com regime integrado."""
        
        # Obter símbolos configurados
        for bot_config in self.base_system.config.get('bot_configs', []):
            if not bot_config.get('enabled', True):
                continue
            
            symbol = bot_config['symbol']
            timeframe = bot_config['timeframe']
            
            # Gerar sinal com regime
            signal = await self.enhanced_signal_generation(symbol, timeframe)
            
            if signal:
                # Executar trade com regime
                result = await self.execute_regime_aware_trade(signal)
                if result['success']:
                    self.logger.info(f"Novo trade com regime {result['regime_applied']} para {symbol}")
        
        # Monitorar trades ativos (usar sistema base)
        await self.base_system.monitor_trades()
    
    def get_enhanced_system_status(self) -> Dict:
        """Retorna status do sistema com dados de regime."""
        
        base_status = self.base_system.get_system_status()
        
        # Adicionar dados de regime
        regime_status = {
            'current_regimes': {
                symbol: regime.value for symbol, regime in self.current_regimes.items()
            },
            'regime_statistics': self.regime_adapter.get_regime_statistics(),
            'enhanced_stats': self.enhanced_stats,
            'regime_distribution': self._get_regime_distribution()
        }
        
        return {
            **base_status,
            'regime_data': regime_status,
            'system_version': 'Regime-Integrated v1.0'
        }
    
    def _get_regime_distribution(self) -> Dict:
        """Calcula distribuição atual de regimes."""
        if not self.current_regimes:
            return {}
        
        distribution = {}
        for regime in self.current_regimes.values():
            regime_name = regime.value
            distribution[regime_name] = distribution.get(regime_name, 0) + 1
        
        total = len(self.current_regimes)
        return {
            regime: count / total for regime, count in distribution.items()
        }
    
    async def get_regime_analysis_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Retorna análise de regime para um símbolo específico."""
        try:
            price_data = await self.base_system._get_price_data(symbol, '15m')
            if price_data is None:
                return None
            
            regime_analysis = await self.regime_detector.detect_regime(symbol, price_data)
            regime_config = self.regime_adapter.regime_configs.get(regime_analysis.regime)
            
            return {
                'regime': regime_analysis.regime.value,
                'confidence': regime_analysis.confidence,
                'trend_strength': regime_analysis.trend_strength,
                'volatility_level': regime_analysis.volatility_level,
                'volume_profile': regime_analysis.volume_profile,
                'breakout_frequency': regime_analysis.breakout_frequency,
                'regime_duration': regime_analysis.regime_duration,
                'factors': regime_analysis.factors,
                'strategy_config': {
                    'strategy_type': regime_config.strategy_type if regime_config else None,
                    'max_trades_per_day': regime_config.max_trades_per_day if regime_config else None,
                    'quality_threshold': regime_config.quality_threshold if regime_config else None,
                    'risk_per_trade': regime_config.risk_per_trade if regime_config else None
                } if regime_config else None
            }
        except Exception as e:
            self.logger.error(f"Erro ao obter análise de regime para {symbol}: {e}")
            return None

# Exemplo de uso
async def test_regime_integrated_system():
    """Testa o sistema integrado com regime."""
    
    system = RegimeIntegratedTradingSystem()
    await system.initialize()
    
    try:
        print("🚀 SISTEMA DE TRADING COM REGIME INTEGRADO")
        print("=" * 60)
        
        # Executar alguns ciclos
        for i in range(3):
            print(f"\n📊 Ciclo {i+1}")
            await system.run_enhanced_trading_cycle()
            
            # Status do sistema
            status = system.get_enhanced_system_status()
            print(f"Trades ativos: {status['active_trades']}")
            print(f"Regimes atuais: {status['regime_data']['current_regimes']}")
            print(f"Trades por regime: {status['regime_data']['enhanced_stats']['trades_by_regime']}")
            
            await asyncio.sleep(1)
        
        # Análise de regime por símbolo
        print(f"\n🔍 ANÁLISE DE REGIME POR SÍMBOLO:")
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        for symbol in symbols:
            analysis = await system.get_regime_analysis_for_symbol(symbol)
            if analysis:
                print(f"\n{symbol}:")
                print(f"  Regime: {analysis['regime']}")
                print(f"  Confiança: {analysis['confidence']:.2f}")
                print(f"  Estratégia: {analysis['strategy_config']['strategy_type'] if analysis['strategy_config'] else 'N/A'}")
                print(f"  Max trades/dia: {analysis['strategy_config']['max_trades_per_day'] if analysis['strategy_config'] else 'N/A'}")
        
        print(f"\n📈 STATUS FINAL:")
        final_status = system.get_enhanced_system_status()
        print(json.dumps(final_status, indent=2, default=str))
        
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(test_regime_integrated_system())