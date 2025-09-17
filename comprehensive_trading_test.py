#!/usr/bin/env python3
"""
Comprehensive Trading System Test
Tests the complete trading system with historical data simulation.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import (
    ProductionTradingSystem, 
    GlobalConfig, 
    BotConfig, 
    create_production_config,
    OptimizedSignalGenerator
)
from mock_exchange import MockExchange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemTester:
    """Comprehensive testing class for the trading system."""
    
    def __init__(self):
        self.results = {
            'initialization': False,
            'signal_generation': {},
            'trading_simulation': {},
            'performance_metrics': {},
            'errors': []
        }
    
    async def test_system_initialization(self) -> bool:
        """Test system initialization with real configuration."""
        logger.info("🧪 Testing system initialization...")
        
        try:
            # Load production configuration
            global_config, bot_configs = create_production_config()
            
            # Create trading system
            trading_system = ProductionTradingSystem(global_config, bot_configs)
            
            # Initialize bots
            await trading_system._initialize_bots()
            
            # Check initialization results
            enabled_bots = sum(1 for config in trading_system.bot_configs.values() if config.enabled)
            total_bots = len(trading_system.bot_configs)
            
            logger.info(f"✅ System initialized: {enabled_bots}/{total_bots} bots enabled")
            
            self.results['initialization'] = True
            self.trading_system = trading_system
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.results['errors'].append(f"Initialization: {e}")
            return False
    
    def test_signal_generation(self) -> bool:
        """Test signal generation for each bot with historical data."""
        logger.info("🔍 Testing signal generation...")
        
        try:
            for bot_id, config in self.trading_system.bot_configs.items():
                if not config.enabled:
                    continue
                
                logger.info(f"Testing signals for {config.symbol} {config.timeframe}")
                
                # Get historical data
                ohlcv = self.trading_system.exchange.fetch_ohlcv(
                    config.symbol, config.timeframe, limit=500
                )
                
                if not ohlcv or len(ohlcv) < 100:
                    logger.warning(f"Insufficient data for {config.symbol}")
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Test signal generator
                signal_generator = self.trading_system.signal_generators[bot_id]
                signals = signal_generator.generate_signals(df)
                
                # Analyze signals
                signal_analysis = {
                    'total_signals': len(signals),
                    'signal_frequency': len(signals) / len(df) * 100,
                    'model_fitted': signal_generator.is_fitted,
                    'avg_confidence': np.mean([s.confidence for s in signals]) if signals else 0,
                    'long_signals': sum(1 for s in signals if s.direction == 1),
                    'short_signals': sum(1 for s in signals if s.direction == -1)
                }
                
                self.results['signal_generation'][bot_id] = signal_analysis
                
                logger.info(f"  📊 {config.symbol}: {signal_analysis['total_signals']} signals "
                           f"({signal_analysis['signal_frequency']:.1f}% frequency)")
                logger.info(f"  🤖 Model fitted: {signal_analysis['model_fitted']}, "
                           f"Avg confidence: {signal_analysis['avg_confidence']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {e}")
            self.results['errors'].append(f"Signal generation: {e}")
            return False
    
    async def test_trading_simulation(self, days: int = 5) -> bool:
        """Test trading simulation over multiple days."""
        logger.info(f"📈 Testing trading simulation for {days} days...")
        
        try:
            # Reset system state
            self.trading_system.active_trades = {}
            self.trading_system.daily_pnl = 0.0
            self.trading_system.total_pnl = 0.0
            self.trading_system.trade_history = []
            
            # Simulate trading for specified days
            simulation_results = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'max_concurrent_trades': 0,
                'daily_results': []
            }
            
            for day in range(days):
                logger.info(f"📅 Simulating day {day + 1}/{days}")
                
                daily_trades_start = len(self.trading_system.trade_history)
                daily_pnl_start = self.trading_system.total_pnl
                
                # Simulate multiple trading cycles per day
                for cycle in range(24):  # 24 cycles per day (hourly)
                    try:
                        # Process each bot
                        for bot_id, config in self.trading_system.bot_configs.items():
                            if not config.enabled:
                                continue
                            
                            await self.trading_system._process_bot(bot_id, config)
                        
                        # Update metrics
                        self.trading_system._update_metrics()
                        
                        # Track max concurrent trades
                        current_trades = len(self.trading_system.active_trades)
                        simulation_results['max_concurrent_trades'] = max(
                            simulation_results['max_concurrent_trades'], 
                            current_trades
                        )
                        
                        # Small delay to simulate real-time
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        logger.warning(f"Error in trading cycle {cycle}: {e}")
                
                # Calculate daily results
                daily_trades_end = len(self.trading_system.trade_history)
                daily_pnl_end = self.trading_system.total_pnl
                
                daily_result = {
                    'day': day + 1,
                    'trades': daily_trades_end - daily_trades_start,
                    'pnl': daily_pnl_end - daily_pnl_start,
                    'active_trades': len(self.trading_system.active_trades)
                }
                
                simulation_results['daily_results'].append(daily_result)
                logger.info(f"  📊 Day {day + 1}: {daily_result['trades']} trades, "
                           f"PnL: ${daily_result['pnl']:.2f}")
            
            # Calculate final results
            simulation_results['total_trades'] = len(self.trading_system.trade_history)
            simulation_results['total_pnl'] = self.trading_system.total_pnl
            simulation_results['profitable_trades'] = sum(
                1 for trade in self.trading_system.trade_history 
                if trade.pnl and trade.pnl > 0
            )
            
            self.results['trading_simulation'] = simulation_results
            
            logger.info(f"✅ Trading simulation completed:")
            logger.info(f"  📊 Total trades: {simulation_results['total_trades']}")
            logger.info(f"  💰 Total PnL: ${simulation_results['total_pnl']:.2f}")
            logger.info(f"  📈 Win rate: {simulation_results['profitable_trades']/max(simulation_results['total_trades'], 1)*100:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading simulation failed: {e}")
            self.results['errors'].append(f"Trading simulation: {e}")
            return False
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        logger.info("📊 Calculating performance metrics...")
        
        try:
            trades = self.trading_system.trade_history
            if not trades:
                logger.warning("No trades to analyze")
                return {}
            
            # Basic metrics
            total_trades = len(trades)
            profitable_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # PnL metrics
            total_pnl = sum(t.pnl for t in trades if t.pnl)
            avg_win = np.mean([t.pnl for t in trades if t.pnl and t.pnl > 0]) if profitable_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl and t.pnl < 0]) if (total_trades - profitable_trades) > 0 else 0
            
            # Risk metrics
            returns = [t.pnl_pct for t in trades if t.pnl_pct]
            if returns:
                volatility = np.std(returns)
                sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
                max_drawdown = self._calculate_max_drawdown([t.pnl for t in trades if t.pnl])
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            metrics = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'roi': total_pnl / self.trading_system.global_config.total_capital
            }
            
            self.results['performance_metrics'] = metrics
            
            logger.info("📈 Performance Metrics:")
            logger.info(f"  Total Trades: {metrics['total_trades']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
            logger.info(f"  Total PnL: ${metrics['total_pnl']:.2f}")
            logger.info(f"  ROI: {metrics['roi']:.1%}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance calculation failed: {e}")
            self.results['errors'].append(f"Performance metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, pnl_series: List[float]) -> float:
        """Calculate maximum drawdown from PnL series."""
        if not pnl_series:
            return 0.0
        
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / self.trading_system.global_config.total_capital
        return abs(np.min(drawdown))
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("🤖 COMPREHENSIVE TRADING SYSTEM TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Initialization results
        report.append("🔧 SYSTEM INITIALIZATION")
        report.append("-" * 40)
        report.append(f"Status: {'✅ PASSED' if self.results['initialization'] else '❌ FAILED'}")
        report.append("")
        
        # Signal generation results
        report.append("🔍 SIGNAL GENERATION ANALYSIS")
        report.append("-" * 40)
        for bot_id, analysis in self.results['signal_generation'].items():
            report.append(f"{bot_id}:")
            report.append(f"  • Total signals: {analysis['total_signals']}")
            report.append(f"  • Signal frequency: {analysis['signal_frequency']:.1f}%")
            report.append(f"  • Model fitted: {analysis['model_fitted']}")
            report.append(f"  • Average confidence: {analysis['avg_confidence']:.3f}")
            report.append(f"  • Long/Short ratio: {analysis['long_signals']}/{analysis['short_signals']}")
            report.append("")
        
        # Trading simulation results
        if 'trading_simulation' in self.results and self.results['trading_simulation']:
            sim = self.results['trading_simulation']
            report.append("📈 TRADING SIMULATION RESULTS")
            report.append("-" * 40)
            report.append(f"Total trades executed: {sim['total_trades']}")
            report.append(f"Profitable trades: {sim['profitable_trades']}")
            report.append(f"Win rate: {sim['profitable_trades']/max(sim['total_trades'], 1)*100:.1f}%")
            report.append(f"Total PnL: ${sim['total_pnl']:.2f}")
            report.append(f"Max concurrent trades: {sim['max_concurrent_trades']}")
            report.append("")
            
            # Daily breakdown
            report.append("📅 DAILY BREAKDOWN:")
            for day_result in sim['daily_results']:
                report.append(f"  Day {day_result['day']}: {day_result['trades']} trades, "
                             f"PnL: ${day_result['pnl']:.2f}")
            report.append("")
        
        # Performance metrics
        if 'performance_metrics' in self.results and self.results['performance_metrics']:
            metrics = self.results['performance_metrics']
            report.append("📊 PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"ROI: {metrics['roi']:.1%}")
            report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            report.append(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
            report.append(f"Average Win: ${metrics['avg_win']:.2f}")
            report.append(f"Average Loss: ${metrics['avg_loss']:.2f}")
            report.append("")
        
        # Errors
        if self.results['errors']:
            report.append("❌ ERRORS ENCOUNTERED")
            report.append("-" * 40)
            for error in self.results['errors']:
                report.append(f"  • {error}")
            report.append("")
        
        # Recommendations
        report.append("✅ RECOMMENDATIONS")
        report.append("-" * 40)
        
        if self.results['initialization']:
            report.append("✅ System initialization: PASSED")
        else:
            report.append("❌ System initialization: FAILED - Check configuration and dependencies")
        
        # Signal generation assessment
        total_signals = sum(
            analysis['total_signals'] 
            for analysis in self.results['signal_generation'].values()
        )
        
        if total_signals > 0:
            report.append("✅ Signal generation: WORKING")
            if total_signals < 10:
                report.append("⚠️  Signal frequency is low - consider adjusting parameters")
        else:
            report.append("❌ Signal generation: NO SIGNALS - Check parameters and data")
        
        # Trading assessment
        if 'trading_simulation' in self.results and self.results['trading_simulation']:
            sim = self.results['trading_simulation']
            if sim['total_trades'] > 0:
                report.append("✅ Trading execution: WORKING")
                win_rate = sim['profitable_trades'] / sim['total_trades']
                if win_rate > 0.6:
                    report.append("✅ Win rate is good (>60%)")
                elif win_rate > 0.4:
                    report.append("⚠️  Win rate is acceptable (40-60%)")
                else:
                    report.append("❌ Win rate is low (<40%) - Review strategy")
            else:
                report.append("❌ Trading execution: NO TRADES - Check signal generation")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def run_comprehensive_test(self) -> bool:
        """Run all tests in sequence."""
        logger.info("🚀 Starting comprehensive trading system test...")
        
        # Test 1: System initialization
        if not await self.test_system_initialization():
            return False
        
        # Test 2: Signal generation
        if not self.test_signal_generation():
            return False
        
        # Test 3: Trading simulation
        if not await self.test_trading_simulation(days=3):
            return False
        
        # Test 4: Performance metrics
        self.calculate_performance_metrics()
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        with open('comprehensive_test_report.txt', 'w') as f:
            f.write(report)
        
        # Print report
        print(report)
        
        logger.info("✅ Comprehensive test completed!")
        return True

async def main():
    """Main function to run comprehensive tests."""
    tester = TradingSystemTester()
    success = await tester.run_comprehensive_test()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)