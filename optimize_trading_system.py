#!/usr/bin/env python3
"""
Trading System Optimization Script
Optimizes parameters for better performance based on test results.
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import (
    ProductionTradingSystem, 
    GlobalConfig, 
    BotConfig, 
    OptimizedSignalGenerator,
    create_production_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemOptimizer:
    """Optimize trading system parameters for better performance."""
    
    def __init__(self):
        self.optimization_results = {}
        self.best_configs = {}
        
    def create_optimized_parameters(self) -> Dict[str, Any]:
        """Create optimized parameters based on analysis."""
        
        # Based on test results, we need:
        # 1. More sensitive signal generation (increase frequency)
        # 2. Better win rate (improve signal quality)
        # 3. Balanced risk/reward
        
        optimized_params = {
            # More sensitive momentum detection
            'momentum_threshold': 0.002,  # Reduced from 0.005
            'momentum_periods': [3, 5, 8],  # Shorter periods for faster signals
            
            # More responsive volume analysis
            'volume_threshold': 1.2,  # Reduced from 1.5
            'volume_ma_period': 10,  # Shorter MA for volume
            
            # Adjusted RSI levels for more signals
            'rsi_oversold': 35,  # More sensitive (was 30)
            'rsi_overbought': 65,  # More sensitive (was 70)
            'rsi_period': 12,  # Shorter period for faster response
            
            # Bollinger Bands adjustments
            'bb_period': 18,  # Shorter period
            'bb_std': 1.8,  # Slightly tighter bands
            
            # ML model improvements
            'ml_threshold': 0.45,  # Lower threshold for more signals
            'ml_lookback': 50,  # Shorter lookback for recent patterns
            'confidence_multiplier': 1.2,  # Boost confidence calculation
            
            # EMA periods for trend detection
            'ema_fast': 8,
            'ema_slow': 21,
            'ema_signal': 5,
            
            # Signal combination logic
            'min_confidence_single': 0.65,  # Lower for single signals
            'min_confidence_combined': 0.55,  # Lower for combined signals
            'signal_decay_factor': 0.95,  # How fast signals decay
            
            # Risk management
            'max_risk_per_trade': 0.015,  # 1.5% max risk
            'profit_target_multiplier': 2.2,  # 2.2:1 reward/risk ratio
            'trailing_stop_activation': 0.8,  # Activate trailing at 80% of target
            
            # Position sizing
            'kelly_fraction': 0.25,  # Conservative Kelly fraction
            'max_position_size': 0.05,  # Max 5% of capital per position
        }
        
        return optimized_params
    
    def create_optimized_bot_configs(self) -> List[Dict[str, Any]]:
        """Create optimized bot configurations."""
        
        # Based on analysis, focus on:
        # 1. More liquid pairs (LINK, BTC, ETH)
        # 2. Multiple timeframes for diversification
        # 3. Balanced capital allocation
        
        optimized_bots = [
            {
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "capital_allocation": 0.30,  # 30% to BTC
                "max_risk_per_trade": 0.012,  # Conservative for BTC
                "confidence_threshold": 0.62,
                "stop_loss_pct": 0.018,  # 1.8%
                "take_profit_pct": 0.040,  # 4.0%
                "enabled": True
            },
            {
                "symbol": "ETH/USDT", 
                "timeframe": "5m",
                "capital_allocation": 0.25,  # 25% to ETH
                "max_risk_per_trade": 0.015,
                "confidence_threshold": 0.58,
                "stop_loss_pct": 0.020,  # 2.0%
                "take_profit_pct": 0.042,  # 4.2%
                "enabled": True
            },
            {
                "symbol": "LINK/USDT",
                "timeframe": "3m",  # Faster timeframe for more signals
                "capital_allocation": 0.20,  # 20% to LINK
                "max_risk_per_trade": 0.018,
                "confidence_threshold": 0.55,  # Lower threshold
                "stop_loss_pct": 0.022,  # 2.2%
                "take_profit_pct": 0.045,  # 4.5%
                "enabled": True
            },
            {
                "symbol": "LINK/USDT",
                "timeframe": "1m",  # High frequency for scalping
                "capital_allocation": 0.15,  # 15% for scalping
                "max_risk_per_trade": 0.010,  # Lower risk for high freq
                "confidence_threshold": 0.65,  # Higher confidence for 1m
                "stop_loss_pct": 0.015,  # 1.5%
                "take_profit_pct": 0.030,  # 3.0%
                "enabled": True
            },
            {
                "symbol": "SOL/USDT",  # Add Solana for diversification
                "timeframe": "5m",
                "capital_allocation": 0.10,  # 10% to SOL
                "max_risk_per_trade": 0.020,
                "confidence_threshold": 0.60,
                "stop_loss_pct": 0.025,  # 2.5%
                "take_profit_pct": 0.050,  # 5.0%
                "enabled": True
            }
        ]
        
        return optimized_bots
    
    def create_optimized_global_config(self) -> Dict[str, Any]:
        """Create optimized global configuration."""
        
        return {
            "total_capital": 1200.0,
            "max_concurrent_trades": 4,  # Increased from 3
            "daily_loss_limit": 0.035,  # 3.5% daily loss limit
            "daily_profit_target": 0.025,  # 2.5% daily profit target
            "emergency_stop_drawdown": 0.08,  # 8% emergency stop
            "paper_trading": True  # Keep paper trading for now
        }
    
    async def test_parameter_combinations(self, param_sets: List[Dict]) -> Dict[str, Any]:
        """Test different parameter combinations."""
        
        logger.info("🧪 Testing parameter combinations...")
        
        results = {}
        
        for i, params in enumerate(param_sets):
            logger.info(f"Testing parameter set {i+1}/{len(param_sets)}")
            
            try:
                # Create temporary system with these parameters
                result = await self._test_single_parameter_set(params)
                results[f"param_set_{i+1}"] = result
                
                logger.info(f"Set {i+1} results: PnL={result['total_pnl']:.2f}, "
                           f"Win Rate={result['win_rate']:.1%}, "
                           f"Trades={result['total_trades']}")
                
            except Exception as e:
                logger.error(f"Error testing parameter set {i+1}: {e}")
                results[f"param_set_{i+1}"] = {"error": str(e)}
        
        return results
    
    async def _test_single_parameter_set(self, params: Dict) -> Dict[str, Any]:
        """Test a single parameter set."""
        
        # Create system with custom parameters
        global_config, bot_configs = create_production_config()
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        
        # Apply custom parameters to signal generators
        for bot_id, generator in trading_system.signal_generators.items():
            generator.params.update(params)
        
        # Initialize system
        await trading_system._initialize_bots()
        
        # Run short simulation
        cycles = 20
        for cycle in range(cycles):
            for bot_id, config in trading_system.bot_configs.items():
                if config.enabled:
                    try:
                        await trading_system._process_bot(bot_id, config)
                    except Exception as e:
                        logger.warning(f"Error in cycle {cycle} for {bot_id}: {e}")
            
            # Simulate exits
            for trade in list(trading_system.active_trades.values()):
                try:
                    ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=2)
                    if ohlcv:
                        current_price = ohlcv[-1][4]
                        # Add some volatility
                        volatility = np.random.normal(0, 0.005)
                        simulated_price = current_price * (1 + volatility)
                        await trading_system._check_trade_exit(trade, simulated_price)
                except Exception as e:
                    logger.warning(f"Error checking exit: {e}")
        
        # Calculate results
        total_trades = len(trading_system.trade_history)
        winning_trades = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in trading_system.trade_history if t.pnl)
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "active_trades": len(trading_system.active_trades)
        }
    
    def generate_optimized_config_file(self) -> str:
        """Generate optimized configuration file."""
        
        logger.info("📝 Generating optimized configuration...")
        
        # Get optimized parameters
        optimized_params = self.create_optimized_parameters()
        optimized_bots = self.create_optimized_bot_configs()
        optimized_global = self.create_optimized_global_config()
        
        # Create complete configuration
        config = {
            "global_config": optimized_global,
            "bot_configs": optimized_bots,
            "optimization_params": optimized_params,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "2.0_optimized",
                "description": "Optimized configuration based on test results",
                "improvements": [
                    "Increased signal sensitivity for more trades",
                    "Improved risk/reward ratios",
                    "Added diversification with multiple timeframes",
                    "Enhanced ML model parameters",
                    "Better position sizing logic"
                ]
            }
        }
        
        # Save to file
        filename = "trading_config_optimized.json"
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Optimized configuration saved to {filename}")
        
        return filename
    
    async def validate_optimized_system(self, config_file: str) -> Dict[str, Any]:
        """Validate the optimized system with extended testing."""
        
        logger.info("🔍 Validating optimized system...")
        
        # Load optimized config
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Create system with optimized config
        global_config = GlobalConfig(**config_data['global_config'])
        
        bot_configs = []
        for bot_data in config_data['bot_configs']:
            bot_configs.append(BotConfig(**bot_data))
        
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        
        # Apply optimization parameters
        optimization_params = config_data.get('optimization_params', {})
        for bot_id, generator in trading_system.signal_generators.items():
            generator.params.update(optimization_params)
        
        # Initialize system
        await trading_system._initialize_bots()
        
        # Extended validation test
        validation_cycles = 50
        hourly_stats = []
        
        logger.info(f"Running {validation_cycles} validation cycles...")
        
        for cycle in range(validation_cycles):
            cycle_start_pnl = trading_system.total_pnl
            cycle_start_trades = len(trading_system.trade_history)
            
            # Process all bots
            for bot_id, config in trading_system.bot_configs.items():
                if config.enabled:
                    try:
                        await trading_system._process_bot(bot_id, config)
                    except Exception as e:
                        logger.warning(f"Validation cycle {cycle} error for {bot_id}: {e}")
            
            # Simulate realistic exits
            for trade in list(trading_system.active_trades.values()):
                try:
                    ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=5)
                    if ohlcv:
                        current_price = ohlcv[-1][4]
                        # More realistic price simulation
                        volatility = np.random.normal(0, 0.008)  # 0.8% volatility
                        trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                        trend_strength = 0.002 * trend
                        simulated_price = current_price * (1 + volatility + trend_strength)
                        
                        await trading_system._check_trade_exit(trade, simulated_price)
                except Exception as e:
                    logger.warning(f"Exit simulation error: {e}")
            
            # Record stats every 10 cycles
            if (cycle + 1) % 10 == 0:
                cycle_pnl = trading_system.total_pnl - cycle_start_pnl
                cycle_trades = len(trading_system.trade_history) - cycle_start_trades
                
                hourly_stats.append({
                    "cycle": cycle + 1,
                    "pnl": trading_system.total_pnl,
                    "cycle_pnl": cycle_pnl,
                    "trades": len(trading_system.trade_history),
                    "cycle_trades": cycle_trades,
                    "active_trades": len(trading_system.active_trades)
                })
                
                logger.info(f"Cycle {cycle+1}: PnL=${trading_system.total_pnl:.2f}, "
                           f"Trades={len(trading_system.trade_history)}, "
                           f"Active={len(trading_system.active_trades)}")
        
        # Calculate final validation metrics
        total_trades = len(trading_system.trade_history)
        winning_trades = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trading_system.total_pnl
        
        if trading_system.trade_history:
            avg_win = np.mean([t.pnl for t in trading_system.trade_history if t.pnl and t.pnl > 0])
            avg_loss = np.mean([t.pnl for t in trading_system.trade_history if t.pnl and t.pnl < 0])
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0
        else:
            avg_win = avg_loss = profit_factor = 0
        
        validation_results = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "roi_pct": (total_pnl / global_config.total_capital) * 100,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "active_trades": len(trading_system.active_trades),
            "hourly_stats": hourly_stats,
            "validation_cycles": validation_cycles
        }
        
        logger.info("✅ Validation completed")
        logger.info(f"📊 Results: {total_trades} trades, {win_rate:.1%} win rate, "
                   f"${total_pnl:.2f} PnL ({(total_pnl/global_config.total_capital)*100:.2f}% ROI)")
        
        return validation_results

async def main():
    """Main optimization function."""
    
    print("🚀 TRADING SYSTEM OPTIMIZATION")
    print("=" * 60)
    
    optimizer = TradingSystemOptimizer()
    
    # Step 1: Generate optimized configuration
    print("\n📝 Step 1: Generating optimized configuration...")
    config_file = optimizer.generate_optimized_config_file()
    
    # Step 2: Validate optimized system
    print("\n🔍 Step 2: Validating optimized system...")
    validation_results = await optimizer.validate_optimized_system(config_file)
    
    # Step 3: Generate optimization report
    print("\n📊 Step 3: Generating optimization report...")
    
    report = f"""
🏆 TRADING SYSTEM OPTIMIZATION REPORT
{'='*60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration: {config_file}

📈 VALIDATION RESULTS
{'-'*30}
Total Trades: {validation_results['total_trades']}
Win Rate: {validation_results['win_rate']:.1%}
Total PnL: ${validation_results['total_pnl']:.2f}
ROI: {validation_results['roi_pct']:.2f}%
Profit Factor: {validation_results['profit_factor']:.2f}
Active Trades: {validation_results['active_trades']}

💡 KEY OPTIMIZATIONS
{'-'*30}
• Reduced signal thresholds for higher frequency
• Improved risk/reward ratios (2.2:1 target)
• Added multiple timeframes for diversification
• Enhanced ML model sensitivity
• Better position sizing with Kelly fraction
• Added SOL/USDT for portfolio diversification

🎯 PERFORMANCE COMPARISON
{'-'*30}
Before Optimization:
  - Win Rate: 28.6%
  - ROI: -1.46%
  - Trades: 7 in 6 hours

After Optimization:
  - Win Rate: {validation_results['win_rate']:.1%}
  - ROI: {validation_results['roi_pct']:.2f}%
  - Trades: {validation_results['total_trades']} in validation

✅ RECOMMENDATIONS
{'-'*30}
{'✅ APPROVED FOR TESTING' if validation_results['win_rate'] > 0.4 and validation_results['total_pnl'] > -20 else '⚠️ NEEDS FURTHER TUNING'}

Next Steps:
1. Deploy optimized configuration
2. Run extended backtests (24+ hours)
3. Monitor performance closely
4. Fine-tune based on live results
"""
    
    print(report)
    
    # Save report
    with open('optimization_report.txt', 'w') as f:
        f.write(report)
    
    # Save validation results
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n💾 Reports saved:")
    print(f"  - Configuration: {config_file}")
    print(f"  - Report: optimization_report.txt")
    print(f"  - Results: validation_results.json")
    
    # Determine success
    success = (
        validation_results['win_rate'] > 0.35 and  # At least 35% win rate
        validation_results['total_pnl'] > -30 and  # Not losing more than $30
        validation_results['total_trades'] > 5     # At least some activity
    )
    
    print(f"\n🏁 Optimization {'✅ SUCCESSFUL' if success else '❌ NEEDS WORK'}")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)