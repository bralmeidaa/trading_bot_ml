#!/usr/bin/env python3
"""
Optimized Configuration for Increased Trading Activity
This configuration reduces restrictive parameters while maintaining risk management.
"""

from production_trading_system import GlobalConfig, BotConfig
from typing import Tuple, List

def create_optimized_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create optimized configuration for more active trading."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,
        max_concurrent_trades=3,  # Increased from 2 to 3
        daily_loss_limit=0.05,  # Slightly increased from 4% to 5%
        daily_profit_target=0.04,  # Increased from 2.5% to 4%
        emergency_stop_drawdown=0.08,
        paper_trading=True
    )
    
    # More aggressive but still safe configuration
    bot_configs = [
        BotConfig(
            symbol='LINK/USDT',
            timeframe='5m',
            capital_allocation=0.50,  # Reduced from 70% to 50%
            max_risk_per_trade=0.025,
            confidence_threshold=0.55,  # Reduced from 0.65 to 0.55
            stop_loss_pct=0.018,
            take_profit_pct=0.035,
            enabled=True
        ),
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',
            capital_allocation=0.30,
            max_risk_per_trade=0.020,
            confidence_threshold=0.50,  # Reduced from 0.65 to 0.50
            stop_loss_pct=0.015,
            take_profit_pct=0.030,
            enabled=True
        ),
        # Add ADA/USDT for more opportunities
        BotConfig(
            symbol='ADA/USDT',
            timeframe='1m',
            capital_allocation=0.20,  # 20% allocation
            max_risk_per_trade=0.020,
            confidence_threshold=0.50,  # Lower threshold
            stop_loss_pct=0.015,
            take_profit_pct=0.030,
            enabled=True
        )
    ]
    
    return global_config, bot_configs

def create_debug_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create very active configuration for debugging and testing."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,
        max_concurrent_trades=4,
        daily_loss_limit=0.06,  # 6% for testing
        daily_profit_target=0.05,  # 5% target
        emergency_stop_drawdown=0.10,
        paper_trading=True
    )
    
    # Very active configuration for testing
    bot_configs = [
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',  # Focus on 1m for more signals
            capital_allocation=0.40,
            max_risk_per_trade=0.025,
            confidence_threshold=0.45,  # Very low threshold
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            enabled=True
        ),
        BotConfig(
            symbol='ADA/USDT',
            timeframe='1m',
            capital_allocation=0.30,
            max_risk_per_trade=0.020,
            confidence_threshold=0.45,
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            enabled=True
        ),
        BotConfig(
            symbol='BNB/USDT',
            timeframe='1m',
            capital_allocation=0.30,
            max_risk_per_trade=0.020,
            confidence_threshold=0.45,
            stop_loss_pct=0.015,
            take_profit_pct=0.025,
            enabled=True
        )
    ]
    
    return global_config, bot_configs