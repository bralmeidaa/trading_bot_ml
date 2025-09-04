#!/usr/bin/env python3
"""
Optimized Trading Configuration
Analysis and recommendations for improving trading effectiveness while maintaining robustness.
"""

from production_trading_system import GlobalConfig, BotConfig
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

class TradingAnalysis:
    """Analysis of current trading system performance and optimization recommendations."""
    
    def __init__(self):
        self.current_issues = self._identify_current_issues()
        self.optimization_recommendations = self._generate_recommendations()
    
    def _identify_current_issues(self) -> Dict[str, Any]:
        """Identify issues with current configuration."""
        return {
            "low_trade_frequency": {
                "description": "Only 2 trades in 5 days indicates overly conservative parameters",
                "causes": [
                    "confidence_threshold=0.65 is too high (65%)",
                    "ml_threshold=0.55-0.6 is conservative", 
                    "momentum_threshold=0.008-0.012 may be restrictive",
                    "Only 2 trading pairs (limited opportunities)",
                    "Requires multiple signal alignment"
                ],
                "impact": "Missing profitable opportunities"
            },
            "limited_diversification": {
                "description": "Only LINK/USDT on 2 timeframes",
                "causes": [
                    "Single asset concentration risk",
                    "Limited market condition coverage",
                    "No adaptation to different volatility regimes"
                ],
                "impact": "Reduced opportunities and higher risk"
            },
            "conservative_risk_management": {
                "description": "While good for safety, may limit profitability",
                "causes": [
                    "max_concurrent_trades=2 (very low)",
                    "max_risk_per_trade=2-2.5% (conservative for crypto)",
                    "daily_loss_limit=4% (restrictive)"
                ],
                "impact": "Lower potential returns"
            }
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        return {
            "parameter_optimization": {
                "confidence_threshold": {
                    "current": 0.65,
                    "recommended": 0.55,
                    "reasoning": "Lower threshold increases opportunities while maintaining quality"
                },
                "ml_threshold": {
                    "current": "0.55-0.6",
                    "recommended": "0.52-0.58",
                    "reasoning": "Slightly more aggressive for crypto market dynamics"
                },
                "momentum_threshold": {
                    "current": "0.008-0.012",
                    "recommended": "0.006-0.010",
                    "reasoning": "Better suited for crypto volatility patterns"
                }
            },
            "diversification": {
                "add_pairs": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT"],
                "add_timeframes": ["3m", "15m"],
                "reasoning": "Diversification reduces risk and increases opportunities"
            },
            "risk_management": {
                "max_concurrent_trades": {
                    "current": 2,
                    "recommended": 4,
                    "reasoning": "Allow more opportunities while maintaining control"
                },
                "max_risk_per_trade": {
                    "current": "2-2.5%",
                    "recommended": "2.5-3.5%",
                    "reasoning": "Appropriate for crypto volatility"
                }
            }
        }
    
    def print_analysis(self):
        """Print comprehensive analysis."""
        print("🔍 TRADING SYSTEM ANALYSIS")
        print("=" * 60)
        
        print("\n❌ CURRENT ISSUES:")
        for issue, details in self.current_issues.items():
            print(f"\n{issue.upper().replace('_', ' ')}:")
            print(f"  Description: {details['description']}")
            print(f"  Impact: {details['impact']}")
            print("  Causes:")
            for cause in details['causes']:
                print(f"    • {cause}")
        
        print("\n✅ OPTIMIZATION RECOMMENDATIONS:")
        for category, recommendations in self.optimization_recommendations.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            if isinstance(recommendations, dict):
                for param, details in recommendations.items():
                    if isinstance(details, dict) and 'current' in details:
                        print(f"  {param}:")
                        print(f"    Current: {details['current']}")
                        print(f"    Recommended: {details['recommended']}")
                        print(f"    Reasoning: {details['reasoning']}")
                    else:
                        print(f"  {param}: {details}")


def create_optimized_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create optimized configuration for better trading effectiveness."""
    
    # More balanced global configuration
    global_config = GlobalConfig(
        total_capital=1200.0,
        max_concurrent_trades=4,  # Increased from 2
        daily_loss_limit=0.05,  # Slightly increased to 5%
        daily_profit_target=0.03,  # Keep 3% target
        emergency_stop_drawdown=0.08,
        paper_trading=True
    )
    
    # Diversified bot configurations with optimized parameters
    bot_configs = [
        # High-frequency scalping bots
        BotConfig(
            symbol='BTC/USDT',
            timeframe='1m',
            capital_allocation=0.25,  # 25%
            max_risk_per_trade=0.025,
            confidence_threshold=0.55,  # Lowered from 0.65
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            enabled=True
        ),
        BotConfig(
            symbol='ETH/USDT', 
            timeframe='1m',
            capital_allocation=0.20,  # 20%
            max_risk_per_trade=0.025,
            confidence_threshold=0.55,
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            enabled=True
        ),
        
        # Medium-frequency trend following
        BotConfig(
            symbol='LINK/USDT',
            timeframe='5m',
            capital_allocation=0.20,  # 20%
            max_risk_per_trade=0.030,
            confidence_threshold=0.58,
            stop_loss_pct=0.020,
            take_profit_pct=0.035,
            enabled=True
        ),
        BotConfig(
            symbol='ADA/USDT',
            timeframe='5m', 
            capital_allocation=0.15,  # 15%
            max_risk_per_trade=0.030,
            confidence_threshold=0.58,
            stop_loss_pct=0.020,
            take_profit_pct=0.035,
            enabled=True
        ),
        
        # Swing trading bots
        BotConfig(
            symbol='SOL/USDT',
            timeframe='15m',
            capital_allocation=0.20,  # 20%
            max_risk_per_trade=0.035,
            confidence_threshold=0.60,
            stop_loss_pct=0.025,
            take_profit_pct=0.045,
            enabled=True
        )
    ]
    
    return global_config, bot_configs


def create_conservative_optimized_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create a more conservative but still optimized configuration."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,
        max_concurrent_trades=3,  # Moderate increase
        daily_loss_limit=0.045,  # 4.5%
        daily_profit_target=0.025,  # 2.5%
        emergency_stop_drawdown=0.08,
        paper_trading=True
    )
    
    # Focus on proven performers with slight optimization
    bot_configs = [
        BotConfig(
            symbol='LINK/USDT',
            timeframe='5m',
            capital_allocation=0.40,  # 40% - main performer
            max_risk_per_trade=0.025,
            confidence_threshold=0.60,  # Slightly lowered
            stop_loss_pct=0.018,
            take_profit_pct=0.035,
            enabled=True
        ),
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',
            capital_allocation=0.25,  # 25%
            max_risk_per_trade=0.022,
            confidence_threshold=0.58,  # Lowered
            stop_loss_pct=0.015,
            take_profit_pct=0.030,
            enabled=True
        ),
        BotConfig(
            symbol='BTC/USDT',
            timeframe='5m',
            capital_allocation=0.35,  # 35% - add BTC for diversification
            max_risk_per_trade=0.028,
            confidence_threshold=0.58,
            stop_loss_pct=0.020,
            take_profit_pct=0.038,
            enabled=True
        )
    ]
    
    return global_config, bot_configs


def create_aggressive_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create more aggressive configuration for higher activity."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,
        max_concurrent_trades=6,  # Higher activity
        daily_loss_limit=0.06,  # 6%
        daily_profit_target=0.04,  # 4%
        emergency_stop_drawdown=0.10,
        paper_trading=True
    )
    
    bot_configs = [
        # Multiple 1m scalping bots
        BotConfig(
            symbol='BTC/USDT',
            timeframe='1m',
            capital_allocation=0.20,
            max_risk_per_trade=0.030,
            confidence_threshold=0.52,  # More aggressive
            stop_loss_pct=0.012,
            take_profit_pct=0.022,
            enabled=True
        ),
        BotConfig(
            symbol='ETH/USDT',
            timeframe='1m', 
            capital_allocation=0.18,
            max_risk_per_trade=0.030,
            confidence_threshold=0.52,
            stop_loss_pct=0.012,
            take_profit_pct=0.022,
            enabled=True
        ),
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',
            capital_allocation=0.15,
            max_risk_per_trade=0.028,
            confidence_threshold=0.50,
            stop_loss_pct=0.012,
            take_profit_pct=0.020,
            enabled=True
        ),
        
        # 3m and 5m bots
        BotConfig(
            symbol='ADA/USDT',
            timeframe='3m',
            capital_allocation=0.15,
            max_risk_per_trade=0.032,
            confidence_threshold=0.54,
            stop_loss_pct=0.015,
            take_profit_pct=0.028,
            enabled=True
        ),
        BotConfig(
            symbol='SOL/USDT',
            timeframe='5m',
            capital_allocation=0.17,
            max_risk_per_trade=0.035,
            confidence_threshold=0.55,
            stop_loss_pct=0.018,
            take_profit_pct=0.032,
            enabled=True
        ),
        BotConfig(
            symbol='MATIC/USDT',
            timeframe='5m',
            capital_allocation=0.15,
            max_risk_per_trade=0.035,
            confidence_threshold=0.55,
            stop_loss_pct=0.018,
            take_profit_pct=0.032,
            enabled=True
        )
    ]
    
    return global_config, bot_configs


def get_optimized_signal_params() -> Dict[str, Dict[str, Any]]:
    """Get optimized signal generation parameters."""
    return {
        'BTC/USDT_1m': {
            'momentum_threshold': 0.006,  # More sensitive
            'volume_threshold': 1.5,      # Lower threshold
            'rsi_oversold': 32,           # Slightly adjusted
            'rsi_overbought': 68,
            'confidence_multiplier': 1.1,
            'ml_threshold': 0.52
        },
        'ETH/USDT_1m': {
            'momentum_threshold': 0.007,
            'volume_threshold': 1.6,
            'rsi_oversold': 33,
            'rsi_overbought': 67,
            'confidence_multiplier': 1.1,
            'ml_threshold': 0.53
        },
        'LINK/USDT_5m': {
            'momentum_threshold': 0.007,  # Slightly more sensitive
            'volume_threshold': 1.6,      # Lower threshold
            'rsi_oversold': 33,
            'rsi_overbought': 67,
            'confidence_multiplier': 1.15,
            'ml_threshold': 0.53
        },
        'LINK/USDT_1m': {
            'momentum_threshold': 0.006,
            'volume_threshold': 1.5,
            'rsi_oversold': 32,
            'rsi_overbought': 68,
            'confidence_multiplier': 1.1,
            'ml_threshold': 0.52
        },
        'default': {
            'momentum_threshold': 0.008,
            'volume_threshold': 1.8,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'confidence_multiplier': 1.0,
            'ml_threshold': 0.55
        }
    }


if __name__ == "__main__":
    # Run analysis
    analysis = TradingAnalysis()
    analysis.print_analysis()
    
    print("\n" + "="*60)
    print("🚀 CONFIGURATION OPTIONS")
    print("="*60)
    
    print("\n1. CONSERVATIVE OPTIMIZED (Recommended for start):")
    global_config, bot_configs = create_conservative_optimized_config()
    print(f"   • {len(bot_configs)} bots, {global_config.max_concurrent_trades} max concurrent")
    print(f"   • Daily loss limit: {global_config.daily_loss_limit:.1%}")
    for config in bot_configs:
        print(f"   • {config.symbol} {config.timeframe}: {config.capital_allocation:.0%} allocation")
    
    print("\n2. BALANCED OPTIMIZED:")
    global_config, bot_configs = create_optimized_config()
    print(f"   • {len(bot_configs)} bots, {global_config.max_concurrent_trades} max concurrent")
    print(f"   • Daily loss limit: {global_config.daily_loss_limit:.1%}")
    
    print("\n3. AGGRESSIVE (Higher activity):")
    global_config, bot_configs = create_aggressive_config()
    print(f"   • {len(bot_configs)} bots, {global_config.max_concurrent_trades} max concurrent")
    print(f"   • Daily loss limit: {global_config.daily_loss_limit:.1%}")