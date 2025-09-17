#!/usr/bin/env python3
"""
Final comprehensive performance test with realistic trading simulation.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import (
    ProductionTradingSystem, 
    GlobalConfig, 
    BotConfig, 
    create_production_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing class."""
    
    def __init__(self):
        self.results = {
            'test_duration_hours': 0,
            'total_signals_generated': 0,
            'total_trades_entered': 0,
            'total_trades_completed': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_trade_duration_minutes': 0.0,
            'max_concurrent_trades': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'daily_stats': [],
            'trade_details': []
        }
    
    async def run_extended_simulation(self, hours: int = 24) -> dict:
        """Run extended trading simulation."""
        
        print(f"🚀 RUNNING {hours}-HOUR TRADING SIMULATION")
        print("=" * 60)
        
        # Create trading system
        global_config, bot_configs = create_production_config()
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        
        # Initialize system
        await trading_system._initialize_bots()
        
        print(f"✅ System initialized")
        print(f"💰 Starting capital: ${global_config.total_capital:,.2f}")
        print(f"🤖 Active bots: {sum(1 for c in trading_system.bot_configs.values() if c.enabled)}")
        
        # Simulation parameters
        cycles_per_hour = 12  # Every 5 minutes
        total_cycles = hours * cycles_per_hour
        start_time = time.time()
        
        # Track performance metrics
        equity_curve = [global_config.total_capital]
        hourly_stats = []
        
        print(f"\n📊 Starting simulation: {total_cycles} cycles over {hours} hours")
        print("-" * 60)
        
        for cycle in range(total_cycles):
            cycle_start = time.time()
            
            # Process all bots
            signals_this_cycle = 0
            trades_entered_this_cycle = 0
            
            for bot_id, config in trading_system.bot_configs.items():
                if not config.enabled:
                    continue
                
                try:
                    # Count active trades before processing
                    active_before = len([t for t in trading_system.active_trades.values() 
                                       if t.symbol == config.symbol])
                    
                    # Process bot
                    await trading_system._process_bot(bot_id, config)
                    
                    # Count active trades after processing
                    active_after = len([t for t in trading_system.active_trades.values() 
                                      if t.symbol == config.symbol])
                    
                    # If new trades were created
                    if active_after > active_before:
                        trades_entered_this_cycle += (active_after - active_before)
                    
                except Exception as e:
                    logger.error(f"Error processing {bot_id} in cycle {cycle}: {e}")
            
            # Check for trade exits with simulated price movements
            await self._simulate_price_movements_and_exits(trading_system)
            
            # Update metrics
            current_equity = global_config.total_capital + trading_system.total_pnl
            equity_curve.append(current_equity)
            
            # Track max concurrent trades
            current_active = len(trading_system.active_trades)
            self.results['max_concurrent_trades'] = max(
                self.results['max_concurrent_trades'], 
                current_active
            )
            
            # Log progress every hour
            if (cycle + 1) % cycles_per_hour == 0:
                hour = (cycle + 1) // cycles_per_hour
                completed_trades = len(trading_system.trade_history)
                
                hourly_stat = {
                    'hour': hour,
                    'total_pnl': trading_system.total_pnl,
                    'equity': current_equity,
                    'completed_trades': completed_trades,
                    'active_trades': current_active
                }
                hourly_stats.append(hourly_stat)
                
                print(f"Hour {hour:2d}: PnL ${trading_system.total_pnl:8.2f} | "
                      f"Equity ${current_equity:8.2f} | "
                      f"Trades {completed_trades:3d} | "
                      f"Active {current_active}")
            
            # Small delay to simulate real-time
            await asyncio.sleep(0.01)
        
        # Calculate final metrics
        end_time = time.time()
        self.results['test_duration_hours'] = (end_time - start_time) / 3600
        
        await self._calculate_final_metrics(trading_system, equity_curve, hourly_stats)
        
        return self.results
    
    async def _simulate_price_movements_and_exits(self, trading_system):
        """Simulate realistic price movements and check for trade exits."""
        
        for trade in list(trading_system.active_trades.values()):
            try:
                # Get current market data
                ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=5)
                if not ohlcv:
                    continue
                
                current_price = ohlcv[-1][4]  # Close price
                
                # Simulate some price volatility
                volatility = 0.002  # 0.2% volatility
                price_change = np.random.normal(0, volatility)
                simulated_price = current_price * (1 + price_change)
                
                # Check for exit conditions
                await trading_system._check_trade_exit(trade, simulated_price)
                
            except Exception as e:
                logger.error(f"Error simulating price movement for {trade.id}: {e}")
    
    async def _calculate_final_metrics(self, trading_system, equity_curve, hourly_stats):
        """Calculate comprehensive performance metrics."""
        
        # Basic metrics
        self.results['total_trades_entered'] = len(trading_system.trade_history) + len(trading_system.active_trades)
        self.results['total_trades_completed'] = len(trading_system.trade_history)
        self.results['total_pnl'] = trading_system.total_pnl
        
        # Win rate
        if trading_system.trade_history:
            profitable_trades = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
            self.results['win_rate'] = profitable_trades / len(trading_system.trade_history)
            
            # Average trade duration
            durations = []
            for trade in trading_system.trade_history:
                if trade.exit_time and trade.entry_time:
                    duration_ms = trade.exit_time - trade.entry_time
                    duration_minutes = duration_ms / (1000 * 60)
                    durations.append(duration_minutes)
            
            if durations:
                self.results['avg_trade_duration_minutes'] = np.mean(durations)
        
        # Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 0:
                self.results['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized
        
        # Max drawdown
        if len(equity_curve) > 1:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            self.results['max_drawdown'] = abs(np.min(drawdown))
        
        # Store detailed results
        self.results['daily_stats'] = hourly_stats
        self.results['trade_details'] = [
            {
                'symbol': t.symbol,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'reason': t.reason,
                'duration_minutes': (t.exit_time - t.entry_time) / (1000 * 60) if t.exit_time and t.entry_time else None
            }
            for t in trading_system.trade_history
        ]
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        
        report = []
        report.append("=" * 80)
        report.append("🏆 FINAL TRADING SYSTEM PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Duration: {self.results['test_duration_hours']:.2f} hours")
        report.append("")
        
        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total PnL: ${self.results['total_pnl']:,.2f}")
        report.append(f"ROI: {(self.results['total_pnl'] / 1200) * 100:.2f}%")
        report.append(f"Total Trades: {self.results['total_trades_completed']}")
        report.append(f"Win Rate: {self.results['win_rate']:.1%}")
        report.append(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {self.results['max_drawdown']:.1%}")
        report.append("")
        
        # Trading Activity
        report.append("📈 TRADING ACTIVITY")
        report.append("-" * 40)
        report.append(f"Trades Entered: {self.results['total_trades_entered']}")
        report.append(f"Trades Completed: {self.results['total_trades_completed']}")
        report.append(f"Max Concurrent Trades: {self.results['max_concurrent_trades']}")
        report.append(f"Avg Trade Duration: {self.results['avg_trade_duration_minutes']:.1f} minutes")
        report.append("")
        
        # Trade Analysis
        if self.results['trade_details']:
            report.append("💰 TRADE ANALYSIS")
            report.append("-" * 40)
            
            profitable_trades = [t for t in self.results['trade_details'] if t['pnl'] and t['pnl'] > 0]
            losing_trades = [t for t in self.results['trade_details'] if t['pnl'] and t['pnl'] < 0]
            
            if profitable_trades:
                avg_win = np.mean([t['pnl'] for t in profitable_trades])
                report.append(f"Profitable Trades: {len(profitable_trades)}")
                report.append(f"Average Win: ${avg_win:.2f}")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                report.append(f"Losing Trades: {len(losing_trades)}")
                report.append(f"Average Loss: ${avg_loss:.2f}")
            
            if profitable_trades and losing_trades:
                profit_factor = abs(avg_win / avg_loss)
                report.append(f"Profit Factor: {profit_factor:.2f}")
            
            report.append("")
        
        # Hourly Performance
        if self.results['daily_stats']:
            report.append("⏰ HOURLY PERFORMANCE")
            report.append("-" * 40)
            for stat in self.results['daily_stats'][-10:]:  # Last 10 hours
                report.append(f"Hour {stat['hour']:2d}: PnL ${stat['total_pnl']:8.2f} | "
                             f"Equity ${stat['equity']:8.2f} | "
                             f"Trades {stat['completed_trades']:3d}")
            report.append("")
        
        # System Assessment
        report.append("✅ SYSTEM ASSESSMENT")
        report.append("-" * 40)
        
        # Performance rating
        roi_pct = (self.results['total_pnl'] / 1200) * 100
        if roi_pct > 5:
            performance = "🟢 EXCELLENT"
        elif roi_pct > 2:
            performance = "🟡 GOOD"
        elif roi_pct > 0:
            performance = "🟠 ACCEPTABLE"
        else:
            performance = "🔴 NEEDS IMPROVEMENT"
        
        report.append(f"Performance Rating: {performance}")
        
        # Win rate assessment
        if self.results['win_rate'] > 0.6:
            win_assessment = "🟢 HIGH"
        elif self.results['win_rate'] > 0.4:
            win_assessment = "🟡 MODERATE"
        else:
            win_assessment = "🔴 LOW"
        
        report.append(f"Win Rate Assessment: {win_assessment}")
        
        # Risk assessment
        if self.results['max_drawdown'] < 0.05:
            risk_assessment = "🟢 LOW RISK"
        elif self.results['max_drawdown'] < 0.10:
            risk_assessment = "🟡 MODERATE RISK"
        else:
            risk_assessment = "🔴 HIGH RISK"
        
        report.append(f"Risk Assessment: {risk_assessment}")
        report.append("")
        
        # Recommendations
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 40)
        
        if self.results['total_trades_completed'] < 5:
            report.append("• Consider adjusting parameters to increase trade frequency")
        
        if self.results['win_rate'] < 0.5:
            report.append("• Review and optimize signal generation logic")
        
        if self.results['max_drawdown'] > 0.08:
            report.append("• Implement stricter risk management controls")
        
        if roi_pct > 0:
            report.append("• System shows positive performance - consider live testing")
        
        report.append("• Continue monitoring and optimizing based on market conditions")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main test function."""
    
    print("🚀 FINAL PERFORMANCE TEST")
    print("=" * 80)
    
    # Create tester
    tester = PerformanceTester()
    
    # Run extended simulation (6 hours for comprehensive test)
    results = await tester.run_extended_simulation(hours=6)
    
    # Generate and display report
    report = tester.generate_performance_report()
    
    # Save report to file
    with open('final_performance_report.txt', 'w') as f:
        f.write(report)
    
    # Save detailed results as JSON
    with open('final_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Display report
    print(report)
    
    # Determine success
    success = (
        results['total_trades_completed'] > 0 and
        results['total_pnl'] > -50  # Not losing more than $50
    )
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)