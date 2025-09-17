#!/usr/bin/env python3
"""
Regime-Based Strategy Adapter
Adapta parâmetros de trading baseado no regime de mercado detectado
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeAnalysis

@dataclass
class RegimeStrategyConfig:
    """Configuração de estratégia para um regime específico."""
    regime: MarketRegime
    max_trades_per_day: int
    quality_threshold: float
    confidence_threshold: float
    risk_per_trade: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    position_sizing_factor: float
    enabled_indicators: List[str]
    strategy_type: str
    entry_conditions: Dict
    exit_conditions: Dict
    notes: str

class RegimeBasedStrategyAdapter:
    """Adaptador de estratégia baseado em regime de mercado."""
    
    def __init__(self, config_path: str = 'regime_strategy_config.json'):
        self.config_path = config_path
        self.regime_configs = self._load_regime_configs()
        self.regime_detector = MarketRegimeDetector()
        self.current_regimes = {}  # Por símbolo
        self.strategy_history = {}  # Histórico de mudanças
        
    def _load_regime_configs(self) -> Dict[MarketRegime, RegimeStrategyConfig]:
        """Carrega configurações de estratégia por regime."""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            configs = {}
            for regime_name, config_data in data.items():
                regime = MarketRegime(regime_name)
                configs[regime] = RegimeStrategyConfig(**config_data)
            
            return configs
        except FileNotFoundError:
            return self._create_default_regime_configs()
    
    def _create_default_regime_configs(self) -> Dict[MarketRegime, RegimeStrategyConfig]:
        """Cria configurações padrão para cada regime."""
        
        configs = {
            MarketRegime.TRENDING_BULL: RegimeStrategyConfig(
                regime=MarketRegime.TRENDING_BULL,
                max_trades_per_day=3,
                quality_threshold=0.65,
                confidence_threshold=0.70,
                risk_per_trade=0.008,  # 0.8% por trade
                stop_loss_multiplier=2.0,
                take_profit_multiplier=4.0,
                position_sizing_factor=1.2,  # Posições maiores em bull trend
                enabled_indicators=["EMA_20", "EMA_50", "MACD", "Volume", "RSI"],
                strategy_type="trend_following",
                entry_conditions={
                    "trend_alignment": True,
                    "volume_confirmation": True,
                    "pullback_entry": True,
                    "rsi_range": [30, 70]
                },
                exit_conditions={
                    "trailing_stop": True,
                    "profit_target": True,
                    "trend_reversal": True
                },
                notes="Estratégia agressiva para trends de alta - foco em momentum"
            ),
            
            MarketRegime.TRENDING_BEAR: RegimeStrategyConfig(
                regime=MarketRegime.TRENDING_BEAR,
                max_trades_per_day=2,
                quality_threshold=0.70,
                confidence_threshold=0.75,
                risk_per_trade=0.006,  # Mais conservador
                stop_loss_multiplier=1.5,
                take_profit_multiplier=3.0,
                position_sizing_factor=0.8,  # Posições menores
                enabled_indicators=["EMA_20", "EMA_50", "MACD", "Volume", "RSI"],
                strategy_type="trend_following_short",
                entry_conditions={
                    "trend_alignment": True,
                    "volume_confirmation": True,
                    "rally_entry": True,  # Vender nos rallies
                    "rsi_range": [30, 70]
                },
                exit_conditions={
                    "trailing_stop": True,
                    "profit_target": True,
                    "trend_reversal": True
                },
                notes="Estratégia conservadora para trends de baixa - shorts em rallies"
            ),
            
            MarketRegime.RANGING: RegimeStrategyConfig(
                regime=MarketRegime.RANGING,
                max_trades_per_day=4,
                quality_threshold=0.75,
                confidence_threshold=0.80,
                risk_per_trade=0.005,  # Menor risco
                stop_loss_multiplier=1.0,  # Stops mais apertados
                take_profit_multiplier=2.0,  # Targets menores
                position_sizing_factor=1.0,
                enabled_indicators=["RSI", "Bollinger_Bands", "Support_Resistance", "Volume"],
                strategy_type="mean_reversion",
                entry_conditions={
                    "oversold_overbought": True,
                    "support_resistance_bounce": True,
                    "bollinger_touch": True,
                    "rsi_range": [20, 80]  # Mais extremo
                },
                exit_conditions={
                    "mean_reversion": True,
                    "opposite_extreme": True,
                    "time_based": True  # Sair se não se mover
                },
                notes="Mean reversion em mercados laterais - comprar suporte/vender resistência"
            ),
            
            MarketRegime.HIGH_VOLATILITY: RegimeStrategyConfig(
                regime=MarketRegime.HIGH_VOLATILITY,
                max_trades_per_day=1,
                quality_threshold=0.85,  # Muito rigoroso
                confidence_threshold=0.85,
                risk_per_trade=0.003,  # Risco mínimo
                stop_loss_multiplier=3.0,  # Stops mais largos
                take_profit_multiplier=6.0,  # Targets maiores
                position_sizing_factor=0.5,  # Posições muito pequenas
                enabled_indicators=["ATR", "Bollinger_Bands", "Volume", "MACD"],
                strategy_type="volatility_breakout",
                entry_conditions={
                    "volatility_expansion": True,
                    "volume_spike": True,
                    "clear_direction": True,
                    "atr_filter": True
                },
                exit_conditions={
                    "volatility_contraction": True,
                    "profit_target": True,
                    "time_based": True
                },
                notes="Estratégia ultra-conservadora para alta volatilidade - poucos trades, alta qualidade"
            ),
            
            MarketRegime.TRANSITIONAL: RegimeStrategyConfig(
                regime=MarketRegime.TRANSITIONAL,
                max_trades_per_day=1,
                quality_threshold=0.80,
                confidence_threshold=0.80,
                risk_per_trade=0.004,
                stop_loss_multiplier=2.0,
                take_profit_multiplier=3.0,
                position_sizing_factor=0.7,
                enabled_indicators=["EMA_20", "RSI", "Volume", "MACD"],
                strategy_type="conservative",
                entry_conditions={
                    "high_confidence_only": True,
                    "multiple_confirmations": True,
                    "clear_setup": True
                },
                exit_conditions={
                    "quick_exit": True,
                    "profit_target": True,
                    "regime_change": True
                },
                notes="Estratégia conservadora para períodos de transição - aguardar clareza"
            )
        }
        
        # Salvar configurações padrão
        self._save_regime_configs(configs)
        return configs
    
    def _save_regime_configs(self, configs: Dict[MarketRegime, RegimeStrategyConfig]):
        """Salva configurações de regime."""
        data = {}
        for regime, config in configs.items():
            config_dict = asdict(config)
            config_dict['regime'] = regime.value  # Converter enum para string
            data[regime.value] = config_dict
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def analyze_and_adapt(self, symbol: str, price_data) -> Dict:
        """Analisa regime e adapta estratégia."""
        
        # Detectar regime atual
        regime_analysis = await self.regime_detector.detect_regime(symbol, price_data)
        
        # Obter configuração para o regime
        regime_config = self.regime_configs.get(regime_analysis.regime)
        if not regime_config:
            regime_config = self.regime_configs[MarketRegime.TRANSITIONAL]
        
        # Verificar se houve mudança de regime
        previous_regime = self.current_regimes.get(symbol)
        regime_changed = previous_regime != regime_analysis.regime
        
        # Atualizar regime atual
        self.current_regimes[symbol] = regime_analysis.regime
        
        # Registrar mudança
        if regime_changed:
            self._log_regime_change(symbol, previous_regime, regime_analysis.regime)
        
        # Adaptar parâmetros baseado na confiança
        adapted_config = self._adapt_config_by_confidence(regime_config, regime_analysis.confidence)
        
        return {
            'regime_analysis': regime_analysis,
            'regime_config': adapted_config,
            'regime_changed': regime_changed,
            'previous_regime': previous_regime.value if previous_regime else None,
            'adaptation_factors': self._get_adaptation_factors(regime_analysis, adapted_config)
        }
    
    def _adapt_config_by_confidence(self, base_config: RegimeStrategyConfig, 
                                   confidence: float) -> RegimeStrategyConfig:
        """Adapta configuração baseada na confiança da detecção."""
        
        # Criar cópia da configuração
        adapted = RegimeStrategyConfig(**asdict(base_config))
        
        # Ajustar parâmetros baseado na confiança
        confidence_factor = confidence  # 0.0 a 1.0
        
        # Quanto menor a confiança, mais conservador
        if confidence < 0.7:
            adapted.max_trades_per_day = max(1, int(adapted.max_trades_per_day * 0.5))
            adapted.quality_threshold = min(0.9, adapted.quality_threshold + 0.1)
            adapted.confidence_threshold = min(0.9, adapted.confidence_threshold + 0.1)
            adapted.risk_per_trade *= 0.7
            adapted.position_sizing_factor *= 0.8
        elif confidence > 0.8:
            # Alta confiança - pode ser um pouco mais agressivo
            adapted.max_trades_per_day = int(adapted.max_trades_per_day * 1.2)
            adapted.quality_threshold = max(0.5, adapted.quality_threshold - 0.05)
            adapted.position_sizing_factor *= 1.1
        
        return adapted
    
    def _log_regime_change(self, symbol: str, old_regime: Optional[MarketRegime], 
                          new_regime: MarketRegime):
        """Registra mudança de regime."""
        if symbol not in self.strategy_history:
            self.strategy_history[symbol] = []
        
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'old_regime': old_regime.value if old_regime else None,
            'new_regime': new_regime.value,
            'change_type': 'regime_change'
        }
        
        self.strategy_history[symbol].append(change_record)
        
        # Manter apenas últimos 50 registros
        if len(self.strategy_history[symbol]) > 50:
            self.strategy_history[symbol] = self.strategy_history[symbol][-50:]
        
        print(f"🔄 MUDANÇA DE REGIME - {symbol}: {old_regime.value if old_regime else 'None'} → {new_regime.value}")
    
    def _get_adaptation_factors(self, regime_analysis: RegimeAnalysis, 
                               config: RegimeStrategyConfig) -> List[str]:
        """Retorna fatores que influenciaram a adaptação."""
        factors = []
        
        # Fatores do regime
        factors.extend(regime_analysis.factors)
        
        # Fatores de confiança
        if regime_analysis.confidence < 0.7:
            factors.append("Baixa confiança - parâmetros mais conservadores")
        elif regime_analysis.confidence > 0.8:
            factors.append("Alta confiança - parâmetros otimizados")
        
        # Fatores de duração
        if regime_analysis.regime_duration > 10:
            factors.append("Regime estabelecido - estratégia estável")
        elif regime_analysis.regime_duration < 3:
            factors.append("Regime recente - cautela extra")
        
        return factors
    
    def get_current_strategy_summary(self, symbol: str) -> Dict:
        """Retorna resumo da estratégia atual para um símbolo."""
        if symbol not in self.current_regimes:
            return {"error": "No regime data for symbol"}
        
        current_regime = self.current_regimes[symbol]
        config = self.regime_configs[current_regime]
        
        return {
            'symbol': symbol,
            'current_regime': current_regime.value,
            'strategy_type': config.strategy_type,
            'max_trades_per_day': config.max_trades_per_day,
            'quality_threshold': config.quality_threshold,
            'risk_per_trade': config.risk_per_trade,
            'notes': config.notes,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_regime_statistics(self) -> Dict:
        """Retorna estatísticas dos regimes."""
        stats = {}
        
        for symbol in self.current_regimes:
            regime_stats = self.regime_detector.get_regime_statistics(symbol)
            stats[symbol] = {
                'current_regime': self.current_regimes[symbol].value,
                'regime_stats': regime_stats,
                'strategy_summary': self.get_current_strategy_summary(symbol)
            }
        
        return stats
    
    def should_trade(self, symbol: str, signal_quality: float, 
                    ml_confidence: float) -> Tuple[bool, str]:
        """Determina se deve fazer trade baseado no regime atual."""
        
        if symbol not in self.current_regimes:
            return False, "No regime data available"
        
        current_regime = self.current_regimes[symbol]
        config = self.regime_configs[current_regime]
        
        # Verificar thresholds
        if signal_quality < config.quality_threshold:
            return False, f"Signal quality {signal_quality:.2f} below threshold {config.quality_threshold:.2f}"
        
        if ml_confidence < config.confidence_threshold:
            return False, f"ML confidence {ml_confidence:.2f} below threshold {config.confidence_threshold:.2f}"
        
        # Regimes específicos que podem pausar trading
        if current_regime == MarketRegime.HIGH_VOLATILITY:
            return True, "High volatility regime - proceed with extreme caution"
        
        if current_regime == MarketRegime.TRANSITIONAL:
            return True, "Transitional regime - conservative approach"
        
        return True, f"Approved for {current_regime.value} regime"

# Exemplo de uso
async def test_regime_strategy_adapter():
    """Testa o adaptador de estratégia baseado em regime."""
    
    # Simular dados de preço
    import pandas as pd
    
    dates = pd.date_range(end=datetime.now(), periods=100, freq='15T')
    base_price = 50000
    prices = base_price * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
        'close': prices,
        'volume': np.random.normal(10000, 2000, 100)
    })
    
    # Testar adaptador
    adapter = RegimeBasedStrategyAdapter()
    
    print("🎯 TESTE DO ADAPTADOR DE ESTRATÉGIA")
    print("=" * 50)
    
    # Analisar e adaptar
    result = await adapter.analyze_and_adapt("BTC/USDT", df)
    
    print(f"Regime detectado: {result['regime_analysis'].regime.value}")
    print(f"Confiança: {result['regime_analysis'].confidence:.2f}")
    print(f"Mudança de regime: {result['regime_changed']}")
    
    config = result['regime_config']
    print(f"\n📋 CONFIGURAÇÃO ADAPTADA:")
    print(f"Tipo de estratégia: {config.strategy_type}")
    print(f"Max trades/dia: {config.max_trades_per_day}")
    print(f"Quality threshold: {config.quality_threshold:.2f}")
    print(f"Confidence threshold: {config.confidence_threshold:.2f}")
    print(f"Risco por trade: {config.risk_per_trade:.3f}")
    print(f"Stop loss multiplier: {config.stop_loss_multiplier}x")
    print(f"Take profit multiplier: {config.take_profit_multiplier}x")
    
    print(f"\n🔍 FATORES DE ADAPTAÇÃO:")
    for factor in result['adaptation_factors']:
        print(f"  • {factor}")
    
    # Testar decisão de trade
    should_trade, reason = adapter.should_trade("BTC/USDT", 0.75, 0.80)
    print(f"\n💹 DECISÃO DE TRADE:")
    print(f"Deve fazer trade: {should_trade}")
    print(f"Razão: {reason}")
    
    # Resumo da estratégia
    summary = adapter.get_current_strategy_summary("BTC/USDT")
    print(f"\n📊 RESUMO DA ESTRATÉGIA:")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(test_regime_strategy_adapter())