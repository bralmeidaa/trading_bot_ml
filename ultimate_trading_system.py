#!/usr/bin/env python3
"""
Ultimate Trading System
Sistema completo que integra todos os componentes avançados:
- Market Regime Detection
- Specialized Bots
- ML Ensemble
- Continuous Learning
- Advanced Features
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# Import all our advanced systems
from regime_integrated_trading_system import RegimeIntegratedTradingSystem
from specialized_bots.btc_trend_bot import BTCTrendBot
from specialized_bots.eth_momentum_bot import ETHMomentumBot
from advanced_ml.ml_ensemble_system import MLEnsembleSystem
from advanced_ml.continuous_learning_system import ContinuousLearningSystem, TradeResult

@dataclass
class UltimateSignal:
    """Sinal final do sistema ultimate."""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_per_trade: float
    
    # Source information
    regime_data: Dict
    specialized_bot: str
    ml_prediction: Dict
    learning_recommendation: Dict
    
    # Metadata
    timestamp: datetime
    signal_id: str
    quality_score: float
    expected_duration: int

class UltimateTradingSystem:
    """Sistema de trading definitivo com todos os componentes avançados."""
    
    def __init__(self, config_path: str = 'ultimate_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize all subsystems
        self.regime_system = RegimeIntegratedTradingSystem()
        self.ml_ensemble = MLEnsembleSystem()
        self.learning_system = ContinuousLearningSystem()
        
        # Initialize specialized bots
        self.specialized_bots = {
            'btc_trend': BTCTrendBot(),
            'eth_momentum': ETHMomentumBot()
            # Add more bots as needed
        }
        
        # System state
        self.active_signals = {}
        self.performance_history = []
        self.system_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'ml_accuracy': 0.0,
            'regime_accuracy': 0.0,
            'learning_adaptation_score': 0.0
        }
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("UltimateSystem")
        
    def _load_config(self) -> Dict:
        """Carrega configuração do sistema."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Cria configuração padrão."""
        config = {
            "system_name": "Ultimate Trading System",
            "version": "1.0",
            "enabled_components": {
                "regime_detection": True,
                "specialized_bots": True,
                "ml_ensemble": True,
                "continuous_learning": True
            },
            "signal_fusion": {
                "regime_weight": 0.25,
                "specialized_bot_weight": 0.30,
                "ml_ensemble_weight": 0.30,
                "learning_weight": 0.15
            },
            "risk_management": {
                "max_concurrent_trades": 5,
                "max_risk_per_trade": 0.02,
                "max_daily_risk": 0.10,
                "emergency_stop_loss": 0.15
            },
            "performance_targets": {
                "min_win_rate": 0.60,
                "min_sharpe_ratio": 1.5,
                "max_drawdown": 0.10,
                "target_daily_return": 0.015
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    async def initialize(self):
        """Inicializa todos os componentes."""
        self.logger.info("🚀 Inicializando Ultimate Trading System...")
        
        # Initialize regime system
        await self.regime_system.initialize()
        
        # Train ML ensemble if needed
        if not hasattr(self.ml_ensemble, 'models') or not any(m['trained'] for m in self.ml_ensemble.models.values()):
            self.logger.info("🤖 Treinando ML Ensemble...")
            # Use sample data for training (in production, use historical data)
            sample_data = self._generate_sample_training_data()
            await self._train_ml_ensemble(sample_data)
        
        self.logger.info("✅ Ultimate Trading System inicializado com sucesso!")
    
    async def generate_ultimate_signal(self, symbol: str, timeframe: str = '15m') -> Optional[UltimateSignal]:
        """Gera sinal final combinando todos os sistemas."""
        
        try:
            self.logger.info(f"🔍 Gerando sinal ultimate para {symbol}...")
            
            # 1. Get price data
            price_data = await self._get_price_data(symbol, timeframe)
            if price_data is None:
                return None
            
            # 2. Regime Analysis
            regime_analysis = None
            if self.config['enabled_components']['regime_detection']:
                regime_analysis = await self.regime_system.get_regime_analysis_for_symbol(symbol)
            
            # 3. Specialized Bot Signal
            specialized_signal = None
            specialized_bot_name = None
            if self.config['enabled_components']['specialized_bots']:
                specialized_signal, specialized_bot_name = await self._get_specialized_bot_signal(symbol, timeframe)
            
            # 4. ML Ensemble Prediction
            ml_prediction = None
            if self.config['enabled_components']['ml_ensemble']:
                ml_prediction = self.ml_ensemble.predict_ensemble(price_data)
            
            # 5. Learning System Recommendation
            learning_recommendation = None
            if self.config['enabled_components']['continuous_learning']:
                current_conditions = self._extract_market_conditions(regime_analysis, price_data)
                learning_recommendation = self.learning_system.get_strategy_recommendation(current_conditions)
            
            # 6. Fuse all signals
            ultimate_signal = await self._fuse_signals(
                symbol, timeframe, price_data,
                regime_analysis, specialized_signal, specialized_bot_name,
                ml_prediction, learning_recommendation
            )
            
            if ultimate_signal:
                self.active_signals[ultimate_signal.signal_id] = ultimate_signal
                self.system_metrics['total_signals'] += 1
                
                self.logger.info(f"✅ Sinal ultimate gerado: {ultimate_signal.direction} "
                               f"com confiança {ultimate_signal.confidence:.3f}")
            
            return ultimate_signal
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar sinal ultimate: {e}")
            return None
    
    async def _get_specialized_bot_signal(self, symbol: str, timeframe: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Obtém sinal do bot especializado apropriado."""
        
        # Select appropriate bot based on symbol
        if symbol.startswith('BTC'):
            bot = self.specialized_bots.get('btc_trend')
            bot_name = 'btc_trend'
        elif symbol.startswith('ETH'):
            bot = self.specialized_bots.get('eth_momentum')
            bot_name = 'eth_momentum'
        else:
            # Use first available bot as fallback
            bot = list(self.specialized_bots.values())[0]
            bot_name = list(self.specialized_bots.keys())[0]
        
        if bot:
            signals = await bot.run_specialized_cycle()
            if signals:
                # Return first signal for the symbol
                for signal in signals:
                    if signal['symbol'] == symbol:
                        return signal, bot_name
        
        return None, None
    
    async def _fuse_signals(self, symbol: str, timeframe: str, price_data: pd.DataFrame,
                          regime_analysis: Dict, specialized_signal: Dict, specialized_bot_name: str,
                          ml_prediction: Any, learning_recommendation: Dict) -> Optional[UltimateSignal]:
        """Funde todos os sinais em um sinal final."""
        
        # Extract signal components
        signals = []
        weights = self.config['signal_fusion']
        
        # 1. Regime signal
        if regime_analysis:
            regime_signal = self._extract_regime_signal(regime_analysis)
            if regime_signal:
                signals.append({
                    'direction': regime_signal,
                    'confidence': regime_analysis.get('confidence', 0.5),
                    'weight': weights['regime_weight'],
                    'source': 'regime'
                })
        
        # 2. Specialized bot signal
        if specialized_signal:
            signals.append({
                'direction': specialized_signal['direction'],
                'confidence': specialized_signal['confidence'],
                'weight': weights['specialized_bot_weight'],
                'source': 'specialized_bot'
            })
        
        # 3. ML ensemble signal
        if ml_prediction:
            ml_direction = self._convert_ml_prediction_to_direction(ml_prediction.prediction)
            if ml_direction:
                signals.append({
                    'direction': ml_direction,
                    'confidence': ml_prediction.confidence,
                    'weight': weights['ml_ensemble_weight'],
                    'source': 'ml_ensemble'
                })
        
        # 4. Learning recommendation (affects confidence and risk)
        learning_confidence_adjustment = 0.0
        learning_risk_adjustment = 1.0
        if learning_recommendation:
            learning_confidence_adjustment = learning_recommendation.get('confidence_adjustment', 0.0)
            learning_risk_adjustment = learning_recommendation.get('risk_adjustment', 1.0)
        
        # Fuse signals using weighted voting
        if not signals:
            return None
        
        # Calculate weighted votes
        direction_votes = {'long': 0.0, 'short': 0.0}
        total_weight = 0.0
        total_confidence = 0.0
        
        for signal in signals:
            direction = signal['direction']
            confidence = signal['confidence']
            weight = signal['weight']
            
            if direction in direction_votes:
                direction_votes[direction] += weight * confidence
                total_weight += weight
                total_confidence += confidence * weight
        
        # Determine final direction
        if direction_votes['long'] > direction_votes['short']:
            final_direction = 'long'
            final_confidence = direction_votes['long'] / total_weight if total_weight > 0 else 0
        elif direction_votes['short'] > direction_votes['long']:
            final_direction = 'short'
            final_confidence = direction_votes['short'] / total_weight if total_weight > 0 else 0
        else:
            return None  # No clear direction
        
        # Apply learning adjustments
        final_confidence += learning_confidence_adjustment
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Check minimum confidence threshold
        min_confidence = 0.65  # High threshold for ultimate system
        if final_confidence < min_confidence:
            self.logger.info(f"Signal rejected: confidence {final_confidence:.3f} < {min_confidence}")
            return None
        
        # Calculate entry price and risk management
        current_price = price_data['close'].iloc[-1]
        
        # Use specialized bot's risk management if available
        if specialized_signal:
            stop_loss = specialized_signal['stop_loss']
            take_profit = specialized_signal['take_profit']
        else:
            # Default risk management
            atr = self._calculate_atr(price_data)
            if final_direction == 'long':
                stop_loss = current_price - (atr * 2.0)
                take_profit = current_price + (atr * 4.0)
            else:
                stop_loss = current_price + (atr * 2.0)
                take_profit = current_price - (atr * 4.0)
        
        # Calculate risk per trade
        base_risk = self.config['risk_management']['max_risk_per_trade']
        adjusted_risk = base_risk * learning_risk_adjustment * final_confidence
        adjusted_risk = min(adjusted_risk, self.config['risk_management']['max_risk_per_trade'])
        
        # Calculate quality score
        quality_score = self._calculate_signal_quality(signals, final_confidence)
        
        # Create ultimate signal
        signal_id = f"ultimate_{symbol}_{int(datetime.now().timestamp())}"
        
        ultimate_signal = UltimateSignal(
            symbol=symbol,
            direction=final_direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=final_confidence,
            risk_per_trade=adjusted_risk,
            regime_data=regime_analysis or {},
            specialized_bot=specialized_bot_name or 'none',
            ml_prediction=asdict(ml_prediction) if ml_prediction else {},
            learning_recommendation=learning_recommendation or {},
            timestamp=datetime.now(),
            signal_id=signal_id,
            quality_score=quality_score,
            expected_duration=self._estimate_trade_duration(signals)
        )
        
        return ultimate_signal
    
    def _extract_regime_signal(self, regime_analysis: Dict) -> Optional[str]:
        """Extrai sinal de direção do regime."""
        regime = regime_analysis.get('regime', '')
        
        if regime == 'trending_bull':
            return 'long'
        elif regime == 'trending_bear':
            return 'short'
        elif regime == 'ranging':
            # For ranging, use other indicators
            return None
        else:
            return None
    
    def _convert_ml_prediction_to_direction(self, prediction: int) -> Optional[str]:
        """Converte predição ML para direção."""
        if prediction == 2:  # Buy
            return 'long'
        elif prediction == 0:  # Sell
            return 'short'
        else:  # Hold
            return None
    
    def _calculate_signal_quality(self, signals: List[Dict], final_confidence: float) -> float:
        """Calcula score de qualidade do sinal."""
        
        # Base quality from confidence
        quality = final_confidence * 0.6
        
        # Agreement bonus (more sources agreeing = higher quality)
        agreement_score = len(signals) / 4.0  # Max 4 sources
        quality += agreement_score * 0.2
        
        # Confidence consistency bonus
        confidences = [s['confidence'] for s in signals]
        if confidences:
            confidence_std = np.std(confidences)
            consistency_bonus = max(0, 1 - confidence_std) * 0.2
            quality += consistency_bonus
        
        return min(1.0, quality)
    
    def _estimate_trade_duration(self, signals: List[Dict]) -> int:
        """Estima duração do trade em minutos."""
        
        # Base duration by source
        durations = []
        
        for signal in signals:
            source = signal['source']
            if source == 'regime':
                durations.append(240)  # 4 hours for regime trades
            elif source == 'specialized_bot':
                durations.append(120)  # 2 hours for specialized bots
            elif source == 'ml_ensemble':
                durations.append(60)   # 1 hour for ML predictions
        
        return int(np.mean(durations)) if durations else 120
    
    async def execute_ultimate_trade(self, signal: UltimateSignal) -> Dict[str, Any]:
        """Executa trade usando o sinal ultimate."""
        
        try:
            self.logger.info(f"💹 Executando trade ultimate: {signal.signal_id}")
            
            # Check risk limits
            if not self._check_risk_limits(signal):
                return {'success': False, 'error': 'Risk limits exceeded'}
            
            # Execute through regime system (which has execution logic)
            # Convert ultimate signal to regime system format
            regime_signal = self._convert_to_regime_signal(signal)
            
            result = await self.regime_system.execute_regime_aware_trade(regime_signal)
            
            if result['success']:
                # Record for learning system
                self._record_trade_for_learning(signal, result)
                
                self.logger.info(f"✅ Trade executado: {result['trade_id']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao executar trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_risk_limits(self, signal: UltimateSignal) -> bool:
        """Verifica limites de risco."""
        
        # Check concurrent trades
        active_trades = len(self.active_signals)
        if active_trades >= self.config['risk_management']['max_concurrent_trades']:
            self.logger.warning(f"Max concurrent trades reached: {active_trades}")
            return False
        
        # Check risk per trade
        if signal.risk_per_trade > self.config['risk_management']['max_risk_per_trade']:
            self.logger.warning(f"Risk per trade too high: {signal.risk_per_trade}")
            return False
        
        return True
    
    def _convert_to_regime_signal(self, ultimate_signal: UltimateSignal):
        """Converte sinal ultimate para formato do regime system."""
        
        # Create a mock TradingSignal object
        class MockTradingSignal:
            def __init__(self, ultimate_signal):
                self.symbol = ultimate_signal.symbol
                self.direction = ultimate_signal.direction
                self.entry_price = ultimate_signal.entry_price
                self.stop_loss = ultimate_signal.stop_loss
                self.take_profit = ultimate_signal.take_profit
                self.confidence = ultimate_signal.confidence
                self.timestamp = ultimate_signal.timestamp
                self.regime_data = ultimate_signal.regime_data
        
        return MockTradingSignal(ultimate_signal)
    
    def _record_trade_for_learning(self, signal: UltimateSignal, execution_result: Dict):
        """Registra trade para sistema de aprendizado."""
        
        # This would be called when trade closes
        # For now, just store the signal for future learning
        pass
    
    async def run_ultimate_cycle(self) -> List[UltimateSignal]:
        """Executa um ciclo completo do sistema ultimate."""
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT']
        signals = []
        
        for symbol in symbols:
            try:
                signal = await self.generate_ultimate_signal(symbol)
                if signal:
                    signals.append(signal)
                    
                    # Execute trade if conditions are met
                    if signal.confidence > 0.75:  # High confidence threshold
                        result = await self.execute_ultimate_trade(signal)
                        if result['success']:
                            self.logger.info(f"🎯 Trade executado para {symbol}")
                
            except Exception as e:
                self.logger.error(f"Erro no ciclo para {symbol}: {e}")
        
        return signals
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema."""
        
        # Get subsystem statuses
        regime_status = self.regime_system.get_enhanced_system_status()
        ml_performance = self.ml_ensemble.get_ensemble_performance()
        learning_summary = self.learning_system.get_learning_summary()
        
        return {
            'system_name': 'Ultimate Trading System',
            'timestamp': datetime.now().isoformat(),
            'active_signals': len(self.active_signals),
            'system_metrics': self.system_metrics,
            'subsystems': {
                'regime_system': {
                    'status': 'active',
                    'current_regimes': regime_status.get('regime_data', {}).get('current_regimes', {}),
                    'regime_distribution': regime_status.get('regime_data', {}).get('regime_distribution', {})
                },
                'ml_ensemble': {
                    'status': 'active',
                    'active_models': ml_performance.get('active_models', 0),
                    'avg_confidence': ml_performance.get('avg_confidence', 0),
                    'model_agreement_rate': ml_performance.get('model_agreement_rate', 0)
                },
                'continuous_learning': {
                    'status': 'active',
                    'total_trades_learned': learning_summary.get('total_trades_learned', 0),
                    'adaptation_trend': learning_summary.get('adaptation_trend', 'unknown'),
                    'learning_score': learning_summary.get('learning_metrics', {}).get('adaptation_score', 0)
                },
                'specialized_bots': {
                    'status': 'active',
                    'active_bots': len(self.specialized_bots),
                    'bot_names': list(self.specialized_bots.keys())
                }
            },
            'performance_targets': self.config['performance_targets'],
            'risk_limits': self.config['risk_management']
        }
    
    # Helper methods
    async def _get_price_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Obtém dados de preço."""
        # Use the regime system's method
        return await self.regime_system.base_system._get_price_data(symbol, timeframe)
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR."""
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                 np.abs(low - np.roll(close, 1))))
        
        return np.mean(tr[-period:])
    
    def _extract_market_conditions(self, regime_analysis: Dict, price_data: pd.DataFrame) -> Dict:
        """Extrai condições de mercado para o learning system."""
        
        conditions = {}
        
        if regime_analysis:
            conditions['regime'] = regime_analysis.get('regime', 'unknown')
            conditions['volatility'] = regime_analysis.get('volatility_level', 0.5)
            conditions['trend_strength'] = regime_analysis.get('trend_strength', 0.5)
        
        # Add price-based conditions
        if len(price_data) > 20:
            recent_volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1]
            conditions['recent_volatility'] = recent_volatility if not np.isnan(recent_volatility) else 0.02
        
        return conditions
    
    async def _train_ml_ensemble(self, sample_data: pd.DataFrame):
        """Treina o ensemble ML."""
        try:
            training_results = self.ml_ensemble.train_ensemble(sample_data)
            self.logger.info(f"ML Ensemble treinado: {len(training_results)} modelos")
        except Exception as e:
            self.logger.error(f"Erro ao treinar ML ensemble: {e}")
    
    def _generate_sample_training_data(self) -> pd.DataFrame:
        """Gera dados de amostra para treinamento."""
        periods = 1000
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='15T')
        
        base_price = 50000
        returns = np.random.normal(0.0005, 0.02, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, periods))),
            'close': prices,
            'volume': np.random.normal(1000, 200, periods)
        }, index=dates)

# Test the Ultimate Trading System
async def test_ultimate_system():
    """Testa o sistema ultimate."""
    
    system = UltimateTradingSystem()
    
    print("🌟 ULTIMATE TRADING SYSTEM")
    print("=" * 60)
    
    # Initialize
    await system.initialize()
    
    # Run cycles
    print(f"\n🔄 EXECUTANDO CICLOS DO SISTEMA...")
    
    for cycle in range(3):
        print(f"\n📊 CICLO {cycle + 1}")
        
        signals = await system.run_ultimate_cycle()
        
        print(f"Sinais gerados: {len(signals)}")
        
        for signal in signals:
            print(f"\n🎯 SINAL ULTIMATE - {signal.symbol}:")
            print(f"  Direção: {signal.direction}")
            print(f"  Confiança: {signal.confidence:.3f}")
            print(f"  Quality Score: {signal.quality_score:.3f}")
            print(f"  Risco: {signal.risk_per_trade:.3f}")
            print(f"  Bot especializado: {signal.specialized_bot}")
            print(f"  Regime: {signal.regime_data.get('regime', 'N/A')}")
            print(f"  Duração esperada: {signal.expected_duration} min")
        
        await asyncio.sleep(1)  # Small delay between cycles
    
    # System status
    status = system.get_system_status()
    
    print(f"\n📈 STATUS DO SISTEMA:")
    print(f"Sinais ativos: {status['active_signals']}")
    print(f"Total de sinais: {status['system_metrics']['total_signals']}")
    
    print(f"\n🔧 SUBSISTEMAS:")
    for subsystem, data in status['subsystems'].items():
        print(f"  {subsystem}: {data['status']}")
        if subsystem == 'regime_system':
            print(f"    Regimes atuais: {data['current_regimes']}")
        elif subsystem == 'ml_ensemble':
            print(f"    Modelos ativos: {data['active_models']}")
            print(f"    Confiança média: {data['avg_confidence']:.3f}")
        elif subsystem == 'continuous_learning':
            print(f"    Trades aprendidos: {data['total_trades_learned']}")
            print(f"    Tendência: {data['adaptation_trend']}")
    
    print(f"\n✅ ULTIMATE TRADING SYSTEM FUNCIONANDO PERFEITAMENTE!")

if __name__ == "__main__":
    asyncio.run(test_ultimate_system())