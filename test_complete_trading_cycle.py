#!/usr/bin/env python3
"""
Test complete trading cycle: signal generation -> trade entry -> trade exit.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_complete_trading_cycle():
    """Test the complete trading cycle."""
    
    print("🔄 TESTING COMPLETE TRADING CYCLE")
    print("=" * 60)
    
    # Create trading system
    global_config, bot_configs = create_production_config()
    trading_system = ProductionTradingSystem(global_config, bot_configs)
    
    # Initialize system
    await trading_system._initialize_bots()
    
    print(f"✅ System initialized with {len(trading_system.bot_configs)} bots")
    print(f"💰 Total capital: ${global_config.total_capital:,.2f}")
    
    # Test each bot individually
    for bot_id, config in trading_system.bot_configs.items():
        if not config.enabled:
            continue
        
        print(f"\n🤖 Testing {config.symbol} {config.timeframe}")
        print("-" * 40)
        
        try:
            # Process bot to generate signals and potentially enter trades
            await trading_system._process_bot(bot_id, config)
            
            # Check if any trades were created
            bot_trades = [trade for trade in trading_system.active_trades.values() 
                         if trade.symbol == config.symbol]
            
            print(f"📊 Active trades for {config.symbol}: {len(bot_trades)}")
            
            if bot_trades:
                for trade in bot_trades:
                    print(f"  Trade ID: {trade.id}")
                    print(f"  Direction: {trade.direction}")
                    print(f"  Entry Price: ${trade.entry_price:.4f}")
                    print(f"  Stop Loss: ${trade.stop_loss:.4f}")
                    print(f"  Take Profit: ${trade.take_profit:.4f}")
                    print(f"  Quantity: {trade.quantity:.6f}")
                    
                    # Simulate price movement to trigger exit
                    await test_trade_exit(trading_system, trade, config)
            else:
                print("  No trades created")
        
        except Exception as e:
            print(f"❌ Error testing {config.symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY")
    print("-" * 40)
    print(f"Active trades: {len(trading_system.active_trades)}")
    print(f"Completed trades: {len(trading_system.trade_history)}")
    print(f"Total PnL: ${trading_system.total_pnl:.2f}")
    print(f"Daily PnL: ${trading_system.daily_pnl:.2f}")
    
    if trading_system.trade_history:
        print("\n📈 Trade History:")
        for i, trade in enumerate(trading_system.trade_history):
            print(f"  Trade {i+1}: {trade.symbol} {trade.direction} "
                  f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) - {trade.reason}")
    
    return len(trading_system.trade_history) > 0

async def test_trade_exit(trading_system, trade, config):
    """Test trade exit by simulating price movements."""
    
    print(f"  🎯 Testing trade exit scenarios...")
    
    # Get current market data
    ohlcv = trading_system.exchange.fetch_ohlcv(config.symbol, config.timeframe, limit=10)
    if not ohlcv:
        print("  ❌ No market data for exit test")
        return
    
    current_price = ohlcv[-1][4]  # Close price
    
    # Test 1: Normal price movement (no exit)
    print(f"    Current price: ${current_price:.4f}")
    await trading_system._check_trade_exit(trade, current_price)
    
    if trade.status == "closed":
        print(f"    ✅ Trade exited at current price")
        return
    
    # Test 2: Simulate stop loss hit
    if trade.direction == 1:
        # Long trade - simulate price drop to stop loss
        stop_loss_price = trade.stop_loss - 0.01  # Slightly below stop loss
    else:
        # Short trade - simulate price rise to stop loss
        stop_loss_price = trade.stop_loss + 0.01  # Slightly above stop loss
    
    print(f"    Testing stop loss at ${stop_loss_price:.4f}")
    await trading_system._check_trade_exit(trade, stop_loss_price)
    
    if trade.status == "closed":
        print(f"    ✅ Trade exited via stop loss")
        return
    
    # Test 3: Simulate take profit hit
    if trade.direction == 1:
        # Long trade - simulate price rise to take profit
        take_profit_price = trade.take_profit + 0.01  # Slightly above take profit
    else:
        # Short trade - simulate price drop to take profit
        take_profit_price = trade.take_profit - 0.01  # Slightly below take profit
    
    print(f"    Testing take profit at ${take_profit_price:.4f}")
    await trading_system._check_trade_exit(trade, take_profit_price)
    
    if trade.status == "closed":
        print(f"    ✅ Trade exited via take profit")
        return
    
    print(f"    ⚠️ Trade did not exit in any scenario")

async def test_multiple_trading_cycles():
    """Test multiple trading cycles over time."""
    
    print("\n🔄 TESTING MULTIPLE TRADING CYCLES")
    print("=" * 60)
    
    # Create trading system
    global_config, bot_configs = create_production_config()
    trading_system = ProductionTradingSystem(global_config, bot_configs)
    
    # Initialize system
    await trading_system._initialize_bots()
    
    cycles = 10
    trades_created = 0
    trades_completed = 0
    
    for cycle in range(cycles):
        print(f"\n📅 Cycle {cycle + 1}/{cycles}")
        
        # Process all bots
        for bot_id, config in trading_system.bot_configs.items():
            if not config.enabled:
                continue
            
            try:
                await trading_system._process_bot(bot_id, config)
            except Exception as e:
                logger.error(f"Error in cycle {cycle} for {bot_id}: {e}")
        
        # Count active trades
        current_active = len(trading_system.active_trades)
        current_completed = len(trading_system.trade_history)
        
        if current_active > trades_created:
            trades_created = current_active
            print(f"  📈 New trades created: {current_active}")
        
        if current_completed > trades_completed:
            new_completed = current_completed - trades_completed
            trades_completed = current_completed
            print(f"  ✅ Trades completed: +{new_completed} (total: {trades_completed})")
        
        # Simulate some time passing and check for exits
        for trade in list(trading_system.active_trades.values()):
            # Get fresh market data
            try:
                ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=5)
                if ohlcv:
                    current_price = ohlcv[-1][4]
                    await trading_system._check_trade_exit(trade, current_price)
            except Exception as e:
                logger.error(f"Error checking exit for {trade.id}: {e}")
        
        # Small delay between cycles
        await asyncio.sleep(0.1)
    
    print(f"\n📊 MULTIPLE CYCLES SUMMARY")
    print("-" * 40)
    print(f"Cycles completed: {cycles}")
    print(f"Max active trades: {trades_created}")
    print(f"Total completed trades: {len(trading_system.trade_history)}")
    print(f"Current active trades: {len(trading_system.active_trades)}")
    print(f"Total PnL: ${trading_system.total_pnl:.2f}")
    
    return len(trading_system.trade_history) > 0

async def main():
    """Main test function."""
    
    print("🚀 COMPLETE TRADING CYCLE TEST")
    print("=" * 80)
    
    # Test 1: Single cycle
    success1 = await test_complete_trading_cycle()
    
    # Test 2: Multiple cycles
    success2 = await test_multiple_trading_cycles()
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 40)
    print(f"Single cycle test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Multiple cycles test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    overall_success = success1 or success2
    print(f"\nOverall: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)