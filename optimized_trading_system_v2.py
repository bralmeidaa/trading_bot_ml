#!/usr/bin/env python3
"""
Sistema de Trading Otimizado V2
Integra filtros avançados de qualidade e dados da Binance
Meta: 5 trades/dia com 55%+ win rate
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from advanced_signal_filters import AdvancedSignalFilter, QualityBasedTradingSystem
from binance_advanced_data import BinanceAdvancedData, MarketSentimentAnalyzer

@dataclass
class TradingSignal:
    """Estrutura de um sinal de trading."""
    symbol: str
    direction: str  # 'long' or 'short'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    quality_score: float = 0.0
    sentiment_data: Dict = None
    rejection_reasons: List[str] = None

class OptimizedTradingSystemV2:
    """Sistema de trading otimizado com filtros avançados."""
    
    def __init__(self, config_path: str = 'trading_config.json'):
        # Carregar configuração
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Inicializar componentes
        self.quality_system = QualityBasedTradingSystem(config_path)
        self.binance_data = None
        self.sentiment_analyzer = None
        
        # Estado do sistema
        self.active_trades = {}
        self.daily_stats = {
            'trades_executed': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'total_pnl': 0.0,
            'signals_evaluated': 0,
            'signals_approved': 0,
            'avg_quality_score': 0.0
        }
        
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Inicializa conexões e componentes assíncronos."""
        self.binance_data = BinanceAdvancedData()
        await self.binance_data.__aenter__()
        self.sentiment_analyzer = MarketSentimentAnalyzer(self.binance_data)
        
        self.logger.info("Sistema de Trading Otimizado V2 inicializado")
    
    async def cleanup(self):
        """Limpa recursos."""
        if self.binance_data:
            await self.binance_data.__aexit__(None, None, None)
    
    async def generate_enhanced_signal(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Gera sinal com análise avançada."""
        
        try:
            # 1. Obter dados de preço (simulação - em produção usar API real)
            price_data = await self._get_price_data(symbol, timeframe)
            if price_data is None:
                return None
            
            # 2. Análise técnica básica
            technical_analysis = await self._perform_technical_analysis(price_data)
            
            # 3. Análise de sentiment da Binance
            sentiment_data = await self.sentiment_analyzer.analyze_market_sentiment(symbol)
            
            # 4. Gerar sinal base
            base_signal = await self._generate_base_signal(
                symbol, price_data, technical_analysis, sentiment_data
            )
            
            if base_signal is None:
                return None
            
            # 5. Avaliar qualidade do sinal
            signal_data = {
                'symbol': symbol,
                'direction': base_signal.direction,
                'ml_confidence': base_signal.confidence,
                'ensemble_agreement': 0.8,  # Simulação
                'feature_importance': {
                    'rsi': 0.8,
                    'macd': 0.7,
                    'volume': 0.6,
                    'price_momentum': 0.75
                },
                'recent_backtest_winrate': 0.65,
                'price_data': price_data
            }
            
            quality_result = await self.quality_system.evaluate_trading_signal(signal_data)
            
            # 6. Aplicar resultado da qualidade
            base_signal.quality_score = quality_result['quality_score']
            base_signal.sentiment_data = sentiment_data
            base_signal.rejection_reasons = quality_result.get('rejection_reasons', [])
            
            if not quality_result['approved']:
                self.logger.info(f"Sinal rejeitado para {symbol}: {quality_result['rejection_reasons']}")
                return None
            
            self.logger.info(f"Sinal aprovado para {symbol} - Quality: {quality_result['quality_score']:.3f}")
            return base_signal
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar sinal para {symbol}: {e}")
            return None
    
    async def _get_price_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Obtém dados de preço (simulação)."""
        # Em produção, usar API real da Binance
        # Por enquanto, simular dados
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='5T')
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        
        # Simular movimento de preços
        returns = np.random.normal(0, 0.01, limit)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, limit)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, limit))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, limit))),
            'close': prices,
            'volume': np.random.normal(10000, 2000, limit)
        })
        
        return df
    
    async def _perform_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """Realiza análise técnica."""
        try:
            import talib
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            analysis = {
                'rsi': talib.RSI(close, timeperiod=14)[-1],
                'macd': talib.MACD(close)[0][-1],
                'bb_upper': talib.BBANDS(close)[0][-1],
                'bb_lower': talib.BBANDS(close)[2][-1],
                'atr': talib.ATR(high, low, close, timeperiod=14)[-1],
                'volume_sma': talib.SMA(volume, timeperiod=20)[-1],
                'price_sma_20': talib.SMA(close, timeperiod=20)[-1],
                'price_sma_50': talib.SMA(close, timeperiod=50)[-1]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise técnica: {e}")
            return {}
    
    async def _generate_base_signal(self, symbol: str, price_data: pd.DataFrame, 
                                  technical: Dict, sentiment: Dict) -> Optional[TradingSignal]:
        """Gera sinal base usando análise técnica e sentiment."""
        
        current_price = price_data['close'].iloc[-1]
        rsi = technical.get('rsi', 50)
        sentiment_score = sentiment.get('sentiment_score', 0.5)
        
        # Lógica simplificada para geração de sinal
        signal_strength = 0.0
        direction = None
        
        # Análise RSI
        if rsi < 35:
            signal_strength += 0.3
            direction = 'long'
        elif rsi > 65:
            signal_strength += 0.3
            direction = 'short'
        
        # Análise de sentiment
        if sentiment_score > 0.6:
            if direction == 'long' or direction is None:
                signal_strength += 0.4
                direction = 'long'
        elif sentiment_score < 0.4:
            if direction == 'short' or direction is None:
                signal_strength += 0.4
                direction = 'short'
        
        # Verificar se temos sinal válido
        if signal_strength < 0.6 or direction is None:
            return None
        
        # Calcular stop loss e take profit
        atr = technical.get('atr', current_price * 0.02)
        
        if direction == 'long':
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 4)
        else:
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 4)
        
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            confidence=signal_strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=datetime.now()
        )
    
    async def execute_trade(self, signal: TradingSignal) -> Dict:
        """Executa trade baseado no sinal."""
        
        try:
            # Calcular tamanho da posição
            bot_config = self._get_bot_config(signal.symbol)
            if not bot_config:
                return {'success': False, 'error': 'Bot config not found'}
            
            capital_allocation = bot_config.get('capital_allocation', 0.2)
            max_risk = bot_config.get('max_risk_per_trade', 0.01)
            
            total_capital = self.config['global_config']['total_capital']
            allocated_capital = total_capital * capital_allocation
            
            # Calcular tamanho baseado no risco
            risk_amount = allocated_capital * max_risk
            price_diff = abs(signal.entry_price - signal.stop_loss)
            position_size = risk_amount / price_diff
            
            # Simular execução (em produção, usar API real)
            trade_id = f"{signal.symbol}_{int(signal.timestamp.timestamp())}"
            
            trade_data = {
                'trade_id': trade_id,
                'symbol': signal.symbol,
                'direction': signal.direction,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': position_size,
                'quality_score': signal.quality_score,
                'sentiment_data': signal.sentiment_data,
                'timestamp': signal.timestamp,
                'status': 'active'
            }
            
            self.active_trades[trade_id] = trade_data
            self.daily_stats['trades_executed'] += 1
            
            self.logger.info(f"Trade executado: {trade_id} - {signal.direction} {signal.symbol} @ {signal.entry_price}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'trade_data': trade_data
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao executar trade: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_bot_config(self, symbol: str) -> Optional[Dict]:
        """Obtém configuração do bot para o símbolo."""
        for bot_config in self.config.get('bot_configs', []):
            if bot_config.get('symbol') == symbol:
                return bot_config
        return None
    
    async def monitor_trades(self):
        """Monitora trades ativos."""
        for trade_id, trade_data in list(self.active_trades.items()):
            if trade_data['status'] != 'active':
                continue
            
            # Simular verificação de preço atual
            current_price = await self._get_current_price(trade_data['symbol'])
            if current_price is None:
                continue
            
            # Verificar stop loss e take profit
            if trade_data['direction'] == 'long':
                if current_price <= trade_data['stop_loss']:
                    await self._close_trade(trade_id, current_price, 'stop_loss')
                elif current_price >= trade_data['take_profit']:
                    await self._close_trade(trade_id, current_price, 'take_profit')
            else:  # short
                if current_price >= trade_data['stop_loss']:
                    await self._close_trade(trade_id, current_price, 'stop_loss')
                elif current_price <= trade_data['take_profit']:
                    await self._close_trade(trade_id, current_price, 'take_profit')
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual (simulação)."""
        # Em produção, usar API real
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
        return base_price * (1 + np.random.normal(0, 0.01))
    
    async def _close_trade(self, trade_id: str, exit_price: float, reason: str):
        """Fecha trade."""
        trade_data = self.active_trades[trade_id]
        
        # Calcular PnL
        entry_price = trade_data['entry_price']
        position_size = trade_data['position_size']
        
        if trade_data['direction'] == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        
        # Atualizar trade
        trade_data.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': reason,
            'status': 'closed',
            'exit_timestamp': datetime.now()
        })
        
        # Atualizar estatísticas
        self.daily_stats['total_pnl'] += pnl
        if pnl > 0:
            self.daily_stats['trades_won'] += 1
        else:
            self.daily_stats['trades_lost'] += 1
        
        self.logger.info(f"Trade fechado: {trade_id} - PnL: ${pnl:.2f} ({reason})")
    
    async def run_trading_cycle(self):
        """Executa um ciclo completo de trading."""
        
        # Obter símbolos configurados
        symbols = [bot['symbol'] for bot in self.config.get('bot_configs', [])]
        
        for bot_config in self.config.get('bot_configs', []):
            if not bot_config.get('enabled', True):
                continue
            
            symbol = bot_config['symbol']
            timeframe = bot_config['timeframe']
            
            # Gerar sinal
            signal = await self.generate_enhanced_signal(symbol, timeframe)
            
            if signal:
                # Executar trade
                result = await self.execute_trade(signal)
                if result['success']:
                    self.logger.info(f"Novo trade iniciado para {symbol}")
        
        # Monitorar trades ativos
        await self.monitor_trades()
    
    def get_system_status(self) -> Dict:
        """Retorna status do sistema."""
        quality_stats = self.quality_system.get_quality_statistics()
        
        win_rate = 0.0
        if self.daily_stats['trades_won'] + self.daily_stats['trades_lost'] > 0:
            win_rate = self.daily_stats['trades_won'] / (self.daily_stats['trades_won'] + self.daily_stats['trades_lost'])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_trades': len([t for t in self.active_trades.values() if t['status'] == 'active']),
            'daily_stats': {
                **self.daily_stats,
                'win_rate': win_rate,
                'avg_pnl_per_trade': self.daily_stats['total_pnl'] / max(self.daily_stats['trades_executed'], 1)
            },
            'quality_stats': quality_stats,
            'system_health': 'healthy' if len(self.active_trades) < 10 else 'warning'
        }

# Exemplo de uso
async def test_optimized_system():
    """Testa o sistema otimizado."""
    
    system = OptimizedTradingSystemV2()
    await system.initialize()
    
    try:
        print("🚀 SISTEMA DE TRADING OTIMIZADO V2")
        print("=" * 50)
        
        # Executar alguns ciclos
        for i in range(3):
            print(f"\n📊 Ciclo {i+1}")
            await system.run_trading_cycle()
            
            status = system.get_system_status()
            print(f"Trades ativos: {status['active_trades']}")
            print(f"Trades executados: {status['daily_stats']['trades_executed']}")
            print(f"Win rate: {status['daily_stats']['win_rate']:.1%}")
            print(f"PnL total: ${status['daily_stats']['total_pnl']:.2f}")
            
            await asyncio.sleep(1)  # Simular intervalo
        
        print(f"\n📈 STATUS FINAL:")
        final_status = system.get_system_status()
        print(json.dumps(final_status, indent=2, default=str))
        
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(test_optimized_system())