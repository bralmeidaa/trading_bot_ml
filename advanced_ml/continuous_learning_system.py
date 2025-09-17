#!/usr/bin/env python3
"""
Continuous Learning System
Sistema que aprende continuamente com os resultados dos trades
"""

import asyncio
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import logging

@dataclass
class TradeResult:
    """Resultado de um trade para aprendizado."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_percentage: float
    duration_minutes: int
    strategy_used: str
    features_at_entry: Dict
    market_conditions: Dict
    confidence_at_entry: float
    timestamp_entry: datetime
    timestamp_exit: datetime
    success: bool

@dataclass
class LearningMetrics:
    """Métricas de aprendizado."""
    total_trades: int
    successful_trades: int
    win_rate: float
    avg_pnl: float
    avg_duration: int
    strategy_performance: Dict[str, Dict]
    feature_performance: Dict[str, float]
    market_condition_performance: Dict[str, Dict]
    confidence_calibration: Dict[str, float]
    learning_rate: float
    adaptation_score: float

class ContinuousLearningSystem:
    """Sistema de aprendizado contínuo para trading."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.trade_history = deque(maxlen=max_history)
        self.learning_metrics = None
        self.strategy_weights = {}
        self.feature_importance_dynamic = {}
        self.market_condition_adjustments = {}
        self.confidence_calibration = {}
        
        # A/B Testing
        self.ab_tests = {}
        self.current_experiments = {}
        
        # Performance tracking
        self.performance_windows = {
            'short_term': deque(maxlen=50),    # Last 50 trades
            'medium_term': deque(maxlen=200),  # Last 200 trades
            'long_term': deque(maxlen=1000)    # Last 1000 trades
        }
        
        # Adaptation parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05
        self.min_trades_for_learning = 20
        
        # Logging
        self.logger = logging.getLogger("ContinuousLearning")
        
    def add_trade_result(self, trade_result: TradeResult):
        """Adiciona resultado de trade para aprendizado."""
        
        # Add to history
        self.trade_history.append(trade_result)
        
        # Add to performance windows
        for window in self.performance_windows.values():
            window.append(trade_result)
        
        # Trigger learning if we have enough data
        if len(self.trade_history) >= self.min_trades_for_learning:
            self._update_learning_metrics()
            self._adapt_strategies()
            self._calibrate_confidence()
            self._update_feature_importance()
        
        self.logger.info(f"Trade result added: {trade_result.trade_id} - "
                        f"PnL: {trade_result.pnl_percentage:.2f}%")
    
    def _update_learning_metrics(self):
        """Atualiza métricas de aprendizado."""
        
        if not self.trade_history:
            return
        
        trades = list(self.trade_history)
        
        # Basic metrics
        total_trades = len(trades)
        successful_trades = sum(1 for t in trades if t.success)
        win_rate = successful_trades / total_trades if total_trades > 0 else 0
        avg_pnl = np.mean([t.pnl_percentage for t in trades])
        avg_duration = np.mean([t.duration_minutes for t in trades])
        
        # Strategy performance
        strategy_performance = {}
        for trade in trades:
            strategy = trade.strategy_used
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'total_pnl': 0.0,
                    'avg_duration': 0.0
                }
            
            strategy_performance[strategy]['trades'] += 1
            if trade.success:
                strategy_performance[strategy]['wins'] += 1
            strategy_performance[strategy]['total_pnl'] += trade.pnl_percentage
        
        # Calculate strategy metrics
        for strategy, data in strategy_performance.items():
            if data['trades'] > 0:
                data['win_rate'] = data['wins'] / data['trades']
                data['avg_pnl'] = data['total_pnl'] / data['trades']
        
        # Feature performance analysis
        feature_performance = self._analyze_feature_performance(trades)
        
        # Market condition performance
        market_condition_performance = self._analyze_market_condition_performance(trades)
        
        # Confidence calibration
        confidence_calibration = self._analyze_confidence_calibration(trades)
        
        # Adaptation score (how well we're adapting)
        adaptation_score = self._calculate_adaptation_score()
        
        self.learning_metrics = LearningMetrics(
            total_trades=total_trades,
            successful_trades=successful_trades,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_duration=avg_duration,
            strategy_performance=strategy_performance,
            feature_performance=feature_performance,
            market_condition_performance=market_condition_performance,
            confidence_calibration=confidence_calibration,
            learning_rate=self.learning_rate,
            adaptation_score=adaptation_score
        )
    
    def _analyze_feature_performance(self, trades: List[TradeResult]) -> Dict[str, float]:
        """Analisa performance das features."""
        
        feature_performance = {}
        
        # Group trades by feature values
        for trade in trades:
            if not trade.features_at_entry:
                continue
            
            for feature_name, feature_value in trade.features_at_entry.items():
                if feature_name not in feature_performance:
                    feature_performance[feature_name] = []
                
                # Store feature value and trade success
                feature_performance[feature_name].append({
                    'value': feature_value,
                    'success': trade.success,
                    'pnl': trade.pnl_percentage
                })
        
        # Calculate correlation between feature values and success
        feature_correlations = {}
        for feature_name, data in feature_performance.items():
            if len(data) < 10:  # Need minimum data
                continue
            
            values = [d['value'] for d in data if isinstance(d['value'], (int, float))]
            successes = [d['success'] for d in data if isinstance(d['value'], (int, float))]
            
            if len(values) > 5:
                try:
                    correlation = np.corrcoef(values, successes)[0, 1]
                    feature_correlations[feature_name] = correlation if not np.isnan(correlation) else 0
                except:
                    feature_correlations[feature_name] = 0
        
        return feature_correlations
    
    def _analyze_market_condition_performance(self, trades: List[TradeResult]) -> Dict[str, Dict]:
        """Analisa performance por condições de mercado."""
        
        condition_performance = {}
        
        for trade in trades:
            if not trade.market_conditions:
                continue
            
            # Analyze key market conditions
            for condition, value in trade.market_conditions.items():
                if condition not in condition_performance:
                    condition_performance[condition] = {
                        'trades': [],
                        'win_rate': 0.0,
                        'avg_pnl': 0.0
                    }
                
                condition_performance[condition]['trades'].append({
                    'value': value,
                    'success': trade.success,
                    'pnl': trade.pnl_percentage
                })
        
        # Calculate metrics for each condition
        for condition, data in condition_performance.items():
            trades_data = data['trades']
            if len(trades_data) > 5:
                successes = sum(1 for t in trades_data if t['success'])
                data['win_rate'] = successes / len(trades_data)
                data['avg_pnl'] = np.mean([t['pnl'] for t in trades_data])
        
        return condition_performance
    
    def _analyze_confidence_calibration(self, trades: List[TradeResult]) -> Dict[str, float]:
        """Analisa calibração da confiança."""
        
        confidence_buckets = {
            'low': [],      # 0.0 - 0.5
            'medium': [],   # 0.5 - 0.7
            'high': [],     # 0.7 - 0.9
            'very_high': [] # 0.9 - 1.0
        }
        
        for trade in trades:
            confidence = trade.confidence_at_entry
            
            if confidence < 0.5:
                bucket = 'low'
            elif confidence < 0.7:
                bucket = 'medium'
            elif confidence < 0.9:
                bucket = 'high'
            else:
                bucket = 'very_high'
            
            confidence_buckets[bucket].append(trade.success)
        
        # Calculate actual success rate for each confidence bucket
        calibration = {}
        for bucket, successes in confidence_buckets.items():
            if len(successes) > 0:
                calibration[bucket] = sum(successes) / len(successes)
            else:
                calibration[bucket] = 0.0
        
        return calibration
    
    def _calculate_adaptation_score(self) -> float:
        """Calcula score de adaptação do sistema."""
        
        if len(self.trade_history) < 100:
            return 0.5  # Neutral score
        
        # Compare recent performance vs historical
        recent_trades = list(self.performance_windows['short_term'])
        older_trades = list(self.trade_history)[-200:-50]  # Trades 50-200 ago
        
        if len(recent_trades) < 20 or len(older_trades) < 20:
            return 0.5
        
        recent_win_rate = sum(1 for t in recent_trades if t.success) / len(recent_trades)
        older_win_rate = sum(1 for t in older_trades if t.success) / len(older_trades)
        
        recent_avg_pnl = np.mean([t.pnl_percentage for t in recent_trades])
        older_avg_pnl = np.mean([t.pnl_percentage for t in older_trades])
        
        # Adaptation score based on improvement
        win_rate_improvement = (recent_win_rate - older_win_rate) / (older_win_rate + 0.01)
        pnl_improvement = (recent_avg_pnl - older_avg_pnl) / (abs(older_avg_pnl) + 0.01)
        
        adaptation_score = (win_rate_improvement + pnl_improvement) / 2
        
        # Normalize to 0-1 range
        return max(0, min(1, 0.5 + adaptation_score))
    
    def _adapt_strategies(self):
        """Adapta pesos das estratégias baseado na performance."""
        
        if not self.learning_metrics:
            return
        
        strategy_performance = self.learning_metrics.strategy_performance
        
        # Calculate new weights based on performance
        total_performance = 0
        strategy_scores = {}
        
        for strategy, data in strategy_performance.items():
            if data['trades'] >= 10:  # Minimum trades for reliable statistics
                # Combine win rate and average PnL
                score = (data['win_rate'] * 0.6 + 
                        min(data['avg_pnl'] / 2.0, 0.4) * 0.4)  # Cap PnL contribution
                strategy_scores[strategy] = max(score, 0.1)  # Minimum weight
                total_performance += strategy_scores[strategy]
        
        # Normalize weights
        if total_performance > 0:
            for strategy, score in strategy_scores.items():
                new_weight = score / total_performance
                old_weight = self.strategy_weights.get(strategy, 1.0 / len(strategy_scores))
                
                # Smooth adaptation
                adapted_weight = old_weight + self.learning_rate * (new_weight - old_weight)
                self.strategy_weights[strategy] = adapted_weight
                
                self.logger.info(f"Strategy {strategy} weight: {old_weight:.3f} → {adapted_weight:.3f}")
    
    def _calibrate_confidence(self):
        """Calibra sistema de confiança."""
        
        if not self.learning_metrics:
            return
        
        calibration = self.learning_metrics.confidence_calibration
        
        # Adjust confidence thresholds based on actual performance
        for bucket, actual_success_rate in calibration.items():
            expected_rates = {
                'low': 0.25,
                'medium': 0.60,
                'high': 0.80,
                'very_high': 0.90
            }
            
            expected = expected_rates.get(bucket, 0.5)
            adjustment = (actual_success_rate - expected) * self.learning_rate
            
            self.confidence_calibration[bucket] = adjustment
            
            self.logger.info(f"Confidence {bucket}: expected {expected:.2f}, "
                           f"actual {actual_success_rate:.2f}, adjustment {adjustment:.3f}")
    
    def _update_feature_importance(self):
        """Atualiza importância dinâmica das features."""
        
        if not self.learning_metrics:
            return
        
        feature_performance = self.learning_metrics.feature_performance
        
        # Update dynamic feature importance
        for feature, correlation in feature_performance.items():
            old_importance = self.feature_importance_dynamic.get(feature, 0.5)
            
            # Convert correlation to importance (0-1 scale)
            new_importance = 0.5 + (correlation * 0.5)  # Scale correlation to 0-1
            new_importance = max(0.1, min(0.9, new_importance))  # Clamp
            
            # Smooth adaptation
            adapted_importance = old_importance + self.learning_rate * (new_importance - old_importance)
            self.feature_importance_dynamic[feature] = adapted_importance
    
    def get_strategy_recommendation(self, current_conditions: Dict) -> Dict[str, Any]:
        """Retorna recomendação de estratégia baseada no aprendizado."""
        
        if not self.learning_metrics:
            return {
                'recommended_strategy': 'default',
                'confidence_adjustment': 0.0,
                'risk_adjustment': 1.0,
                'reasoning': 'Insufficient learning data'
            }
        
        # Find best performing strategy for current conditions
        best_strategy = None
        best_score = 0
        
        for strategy, data in self.learning_metrics.strategy_performance.items():
            if data['trades'] >= 5:  # Minimum trades
                score = data['win_rate'] * 0.7 + min(data['avg_pnl'] / 2.0, 0.3) * 0.3
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        # Confidence adjustment based on calibration
        confidence_adjustment = 0.0
        if self.confidence_calibration:
            avg_adjustment = np.mean(list(self.confidence_calibration.values()))
            confidence_adjustment = avg_adjustment
        
        # Risk adjustment based on recent performance
        risk_adjustment = 1.0
        if len(self.performance_windows['short_term']) > 10:
            recent_win_rate = sum(1 for t in self.performance_windows['short_term'] if t.success) / len(self.performance_windows['short_term'])
            if recent_win_rate > 0.7:
                risk_adjustment = 1.2  # Increase risk when performing well
            elif recent_win_rate < 0.4:
                risk_adjustment = 0.7  # Decrease risk when performing poorly
        
        return {
            'recommended_strategy': best_strategy or 'default',
            'confidence_adjustment': confidence_adjustment,
            'risk_adjustment': risk_adjustment,
            'strategy_weights': self.strategy_weights.copy(),
            'adaptation_score': self.learning_metrics.adaptation_score,
            'reasoning': f'Based on {len(self.trade_history)} trades, '
                        f'adaptation score: {self.learning_metrics.adaptation_score:.2f}'
        }
    
    def start_ab_test(self, test_name: str, variants: Dict[str, Any], 
                     traffic_split: Dict[str, float]):
        """Inicia teste A/B."""
        
        self.ab_tests[test_name] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'results': {variant: [] for variant in variants.keys()},
            'start_time': datetime.now(),
            'active': True
        }
        
        self.logger.info(f"A/B test started: {test_name} with variants {list(variants.keys())}")
    
    def get_ab_test_variant(self, test_name: str) -> Optional[str]:
        """Retorna variante do teste A/B para usar."""
        
        if test_name not in self.ab_tests or not self.ab_tests[test_name]['active']:
            return None
        
        test = self.ab_tests[test_name]
        traffic_split = test['traffic_split']
        
        # Random selection based on traffic split
        rand = np.random.random()
        cumulative = 0
        
        for variant, split in traffic_split.items():
            cumulative += split
            if rand <= cumulative:
                return variant
        
        return list(traffic_split.keys())[0]  # Fallback
    
    def record_ab_test_result(self, test_name: str, variant: str, trade_result: TradeResult):
        """Registra resultado do teste A/B."""
        
        if test_name in self.ab_tests and self.ab_tests[test_name]['active']:
            self.ab_tests[test_name]['results'][variant].append(trade_result)
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analisa resultados do teste A/B."""
        
        if test_name not in self.ab_tests:
            return {}
        
        test = self.ab_tests[test_name]
        results = test['results']
        
        analysis = {}
        
        for variant, trades in results.items():
            if len(trades) > 0:
                win_rate = sum(1 for t in trades if t.success) / len(trades)
                avg_pnl = np.mean([t.pnl_percentage for t in trades])
                
                analysis[variant] = {
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': sum(t.pnl_percentage for t in trades)
                }
        
        # Determine winner
        if len(analysis) >= 2:
            winner = max(analysis.items(), key=lambda x: x[1]['win_rate'] * 0.7 + x[1]['avg_pnl'] * 0.3)
            analysis['winner'] = winner[0]
            analysis['confidence'] = self._calculate_ab_test_confidence(results)
        
        return analysis
    
    def _calculate_ab_test_confidence(self, results: Dict[str, List[TradeResult]]) -> float:
        """Calcula confiança estatística do teste A/B."""
        
        # Simplified confidence calculation
        # In production, use proper statistical tests
        
        variants = list(results.keys())
        if len(variants) < 2:
            return 0.0
        
        # Compare first two variants
        trades_a = results[variants[0]]
        trades_b = results[variants[1]]
        
        if len(trades_a) < 10 or len(trades_b) < 10:
            return 0.0  # Need more data
        
        win_rate_a = sum(1 for t in trades_a if t.success) / len(trades_a)
        win_rate_b = sum(1 for t in trades_b if t.success) / len(trades_b)
        
        # Simple confidence based on difference and sample size
        difference = abs(win_rate_a - win_rate_b)
        sample_size_factor = min(len(trades_a), len(trades_b)) / 100.0
        
        confidence = min(difference * sample_size_factor * 10, 0.95)
        
        return confidence
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Retorna resumo do aprendizado."""
        
        if not self.learning_metrics:
            return {'status': 'insufficient_data'}
        
        return {
            'learning_metrics': asdict(self.learning_metrics),
            'strategy_weights': self.strategy_weights,
            'feature_importance': dict(sorted(
                self.feature_importance_dynamic.items(),
                key=lambda x: abs(x[1]), reverse=True
            )[:20]),  # Top 20 features
            'confidence_calibration': self.confidence_calibration,
            'active_ab_tests': len([t for t in self.ab_tests.values() if t['active']]),
            'total_trades_learned': len(self.trade_history),
            'adaptation_trend': self._get_adaptation_trend()
        }
    
    def _get_adaptation_trend(self) -> str:
        """Retorna tendência de adaptação."""
        
        if len(self.trade_history) < 100:
            return 'learning'
        
        adaptation_score = self.learning_metrics.adaptation_score
        
        if adaptation_score > 0.7:
            return 'improving'
        elif adaptation_score > 0.4:
            return 'stable'
        else:
            return 'declining'

# Test the Continuous Learning System
def test_continuous_learning():
    """Test the Continuous Learning System."""
    
    learning_system = ContinuousLearningSystem()
    
    print("🧠 TESTE DO SISTEMA DE APRENDIZADO CONTÍNUO")
    print("=" * 60)
    
    # Simulate trade results
    strategies = ['btc_trend', 'eth_momentum', 'sol_breakout', 'mean_reversion']
    
    for i in range(200):  # Simulate 200 trades
        # Random trade result
        strategy = np.random.choice(strategies)
        success = np.random.random() > 0.4  # 60% win rate
        pnl_pct = np.random.normal(1.5 if success else -1.2, 0.8)
        
        trade_result = TradeResult(
            trade_id=f"trade_{i:03d}",
            symbol=np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']),
            direction=np.random.choice(['long', 'short']),
            entry_price=50000 + np.random.normal(0, 1000),
            exit_price=50000 + pnl_pct * 500,
            pnl=pnl_pct * 500,
            pnl_percentage=pnl_pct,
            duration_minutes=np.random.randint(15, 240),
            strategy_used=strategy,
            features_at_entry={
                'rsi_14': np.random.uniform(20, 80),
                'macd_hist': np.random.normal(0, 0.1),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'trend_strength': np.random.uniform(0, 1)
            },
            market_conditions={
                'volatility': np.random.uniform(0.1, 0.8),
                'trend_direction': np.random.choice(['bullish', 'bearish', 'sideways']),
                'volume_surge': np.random.random() > 0.7
            },
            confidence_at_entry=np.random.uniform(0.5, 0.95),
            timestamp_entry=datetime.now() - timedelta(hours=i),
            timestamp_exit=datetime.now() - timedelta(hours=i) + timedelta(minutes=np.random.randint(15, 240)),
            success=success
        )
        
        learning_system.add_trade_result(trade_result)
    
    # Get learning summary
    summary = learning_system.get_learning_summary()
    
    print(f"📊 RESUMO DO APRENDIZADO:")
    print(f"Total de trades: {summary['total_trades_learned']}")
    print(f"Win rate geral: {summary['learning_metrics']['win_rate']:.3f}")
    print(f"PnL médio: {summary['learning_metrics']['avg_pnl']:.2f}%")
    print(f"Score de adaptação: {summary['learning_metrics']['adaptation_score']:.3f}")
    print(f"Tendência: {summary['adaptation_trend']}")
    
    print(f"\n🎯 PERFORMANCE POR ESTRATÉGIA:")
    for strategy, data in summary['learning_metrics']['strategy_performance'].items():
        print(f"  {strategy}: {data['trades']} trades, "
              f"win rate {data['win_rate']:.3f}, "
              f"PnL médio {data['avg_pnl']:.2f}%")
    
    print(f"\n⚖️ PESOS ADAPTADOS DAS ESTRATÉGIAS:")
    for strategy, weight in summary['strategy_weights'].items():
        print(f"  {strategy}: {weight:.3f}")
    
    print(f"\n🔧 TOP 10 FEATURES MAIS IMPORTANTES:")
    for feature, importance in list(summary['feature_importance'].items())[:10]:
        print(f"  {feature}: {importance:.3f}")
    
    # Test strategy recommendation
    current_conditions = {
        'volatility': 0.5,
        'trend_direction': 'bullish',
        'volume_surge': True
    }
    
    recommendation = learning_system.get_strategy_recommendation(current_conditions)
    print(f"\n💡 RECOMENDAÇÃO ATUAL:")
    print(f"Estratégia recomendada: {recommendation['recommended_strategy']}")
    print(f"Ajuste de confiança: {recommendation['confidence_adjustment']:.3f}")
    print(f"Ajuste de risco: {recommendation['risk_adjustment']:.3f}")
    print(f"Razão: {recommendation['reasoning']}")
    
    # Test A/B testing
    learning_system.start_ab_test(
        'risk_levels',
        {'conservative': 0.5, 'aggressive': 1.5},
        {'conservative': 0.5, 'aggressive': 0.5}
    )
    
    print(f"\n🧪 TESTE A/B INICIADO:")
    for i in range(20):
        variant = learning_system.get_ab_test_variant('risk_levels')
        print(f"  Trade {i+1}: usando variante '{variant}'")
    
    print(f"\n✅ SISTEMA DE APRENDIZADO CONTÍNUO FUNCIONANDO!")

if __name__ == "__main__":
    test_continuous_learning()