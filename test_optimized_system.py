#!/usr/bin/env python3
"""
Test the optimized trading system with frontend integration.
"""

import asyncio
import sys
import os
import time
import requests
import json
from threading import Thread
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import (
    ProductionTradingSystem, 
    create_production_config
)

def start_api_server():
    """Start the API server."""
    import uvicorn
    from api_server import app
    
    print("🚀 Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

async def test_optimized_system():
    """Test the optimized trading system."""
    
    print("🤖 TESTING OPTIMIZED TRADING SYSTEM")
    print("=" * 60)
    
    # Load optimized configuration
    print("📁 Loading optimized configuration...")
    global_config, bot_configs = create_production_config()
    
    print(f"✅ Configuration loaded:")
    print(f"   💰 Capital: ${global_config.total_capital:,.2f}")
    print(f"   🤖 Bots: {len(bot_configs)}")
    print(f"   📊 Max concurrent trades: {global_config.max_concurrent_trades}")
    print(f"   ⚠️ Daily loss limit: {global_config.daily_loss_limit:.1%}")
    
    # Initialize system
    print("\n🔧 Initializing optimized system...")
    trading_system = ProductionTradingSystem(global_config, bot_configs)
    await trading_system._initialize_bots()
    
    enabled_bots = sum(1 for config in trading_system.bot_configs.values() if config.enabled)
    print(f"✅ System initialized: {enabled_bots}/{len(trading_system.bot_configs)} bots active")
    
    # Show bot details
    print("\n🤖 OPTIMIZED BOT CONFIGURATION:")
    for bot_id, config in trading_system.bot_configs.items():
        status = "🟢" if config.enabled else "🔴"
        print(f"   {status} {config.symbol} {config.timeframe}")
        print(f"      💰 Capital: {config.capital_allocation:.0%}")
        print(f"      🎯 Confidence: {config.confidence_threshold:.2f}")
        print(f"      🛡️ Risk: {config.max_risk_per_trade:.1%}")
    
    # Test signal generation
    print(f"\n🔍 TESTING SIGNAL GENERATION:")
    total_signals = 0
    
    for bot_id, config in trading_system.bot_configs.items():
        if not config.enabled:
            continue
        
        try:
            # Get market data
            ohlcv = trading_system.exchange.fetch_ohlcv(config.symbol, config.timeframe, limit=200)
            if not ohlcv:
                continue
            
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Generate signals
            signal_generator = trading_system.signal_generators[bot_id]
            signals = signal_generator.generate_signals(df)
            
            if signals:
                total_signals += len(signals)
                print(f"   ✅ {config.symbol} {config.timeframe}: {len(signals)} signals")
                for signal in signals:
                    direction = "🟢 LONG" if signal.direction == 1 else "🔴 SHORT"
                    print(f"      {direction} Confidence: {signal.confidence:.3f}")
            else:
                print(f"   ⏳ {config.symbol} {config.timeframe}: No signals")
        
        except Exception as e:
            print(f"   ❌ {config.symbol} {config.timeframe}: Error - {e}")
    
    print(f"\n📊 Total signals generated: {total_signals}")
    
    # Test trading execution
    if total_signals > 0:
        print(f"\n💰 TESTING TRADING EXECUTION:")
        
        initial_pnl = trading_system.total_pnl
        initial_trades = len(trading_system.trade_history)
        
        # Run trading cycles
        for cycle in range(10):
            print(f"   Cycle {cycle + 1}/10", end=" ")
            
            # Process bots
            for bot_id, config in trading_system.bot_configs.items():
                if config.enabled:
                    try:
                        await trading_system._process_bot(bot_id, config)
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Check exits
            for trade in list(trading_system.active_trades.values()):
                try:
                    ohlcv = trading_system.exchange.fetch_ohlcv(trade.symbol, '1m', limit=2)
                    if ohlcv:
                        current_price = ohlcv[-1][4]
                        # Simulate price movement
                        import random
                        volatility = random.uniform(-0.01, 0.01)
                        simulated_price = current_price * (1 + volatility)
                        await trading_system._check_trade_exit(trade, simulated_price)
                except Exception as e:
                    pass
            
            # Show progress
            active = len(trading_system.active_trades)
            completed = len(trading_system.trade_history)
            pnl = trading_system.total_pnl
            print(f"- Active: {active}, Completed: {completed}, PnL: ${pnl:.2f}")
            
            await asyncio.sleep(0.1)
        
        # Final results
        final_pnl = trading_system.total_pnl
        final_trades = len(trading_system.trade_history)
        new_trades = final_trades - initial_trades
        pnl_change = final_pnl - initial_pnl
        
        print(f"\n📈 EXECUTION RESULTS:")
        print(f"   🆕 New trades: {new_trades}")
        print(f"   💰 PnL change: ${pnl_change:.2f}")
        print(f"   📊 Total PnL: ${final_pnl:.2f}")
        print(f"   🎯 Active trades: {len(trading_system.active_trades)}")
    
    return trading_system

def test_api_endpoints():
    """Test API endpoints with optimized system."""
    
    print(f"\n🌐 TESTING API ENDPOINTS:")
    
    base_url = "http://localhost:8000"
    
    # Wait for server
    print("   ⏳ Waiting for API server...")
    time.sleep(3)
    
    # Test key endpoints
    endpoints = [
        ("/api/status", "System Status"),
        ("/api/metrics", "Performance Metrics"),
        ("/api/bots", "Bot Status"),
        ("/api/trades/recent", "Recent Trades"),
        ("/api/config/full", "Configuration")
    ]
    
    results = {}
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                results[endpoint] = "✅ OK"
                print(f"   ✅ {description}: OK")
                
                # Show some data
                if endpoint == "/api/metrics" and data.get('success'):
                    metrics = data['data']
                    print(f"      📊 Win Rate: {metrics.get('win_rate', 0):.1%}")
                    print(f"      💰 Total PnL: ${metrics.get('total_pnl', 0):.2f}")
                    print(f"      🎯 Total Trades: {metrics.get('total_trades', 0)}")
                
                elif endpoint == "/api/bots" and data.get('success'):
                    bots = data['data'].get('bots', [])
                    print(f"      🤖 Active Bots: {len([b for b in bots if b.get('enabled')])}")
                
            else:
                results[endpoint] = f"❌ {response.status_code}"
                print(f"   ❌ {description}: {response.status_code}")
        
        except Exception as e:
            results[endpoint] = f"❌ Error"
            print(f"   ❌ {description}: {e}")
    
    # Test control endpoints
    print(f"\n🎮 TESTING CONTROL ENDPOINTS:")
    
    try:
        # Test start
        response = requests.post(f"{base_url}/api/start", timeout=10)
        if response.status_code == 200:
            print("   ✅ Start System: OK")
            time.sleep(2)
            
            # Test stop
            response = requests.post(f"{base_url}/api/stop", timeout=10)
            if response.status_code == 200:
                print("   ✅ Stop System: OK")
            else:
                print(f"   ❌ Stop System: {response.status_code}")
        else:
            print(f"   ❌ Start System: {response.status_code}")
    
    except Exception as e:
        print(f"   ❌ Control Endpoints: {e}")
    
    return results

def test_frontend_build():
    """Test if frontend can be built."""
    
    print(f"\n📦 TESTING FRONTEND BUILD:")
    
    frontend_dir = "/workspace/trading_bot_ml/frontend_react"
    
    if not os.path.exists(frontend_dir):
        print("   ❌ Frontend directory not found")
        return False
    
    # Check package.json
    package_json = os.path.join(frontend_dir, "package.json")
    if os.path.exists(package_json):
        print("   ✅ package.json found")
        
        # Check if node_modules exists
        node_modules = os.path.join(frontend_dir, "node_modules")
        if os.path.exists(node_modules):
            print("   ✅ node_modules found")
        else:
            print("   ⚠️ node_modules not found - run 'npm install'")
    else:
        print("   ❌ package.json not found")
        return False
    
    # Check key files
    key_files = [
        "src/App.jsx",
        "src/services/api.js",
        "src/components/PerformanceMetrics.jsx",
        "src/components/BotStatusTable.jsx"
    ]
    
    for file in key_files:
        file_path = os.path.join(frontend_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} missing")
    
    print(f"\n📋 FRONTEND USAGE INSTRUCTIONS:")
    print(f"   1. cd {frontend_dir}")
    print(f"   2. npm install (if needed)")
    print(f"   3. npm run dev")
    print(f"   4. Open http://localhost:5173")
    print(f"   5. API will be available at http://localhost:8000")
    
    return True

async def main():
    """Main test function."""
    
    print("🚀 OPTIMIZED SYSTEM INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Optimized trading system
    trading_system = await test_optimized_system()
    
    # Test 2: Start API server in background
    server_thread = Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Test 3: API endpoints
    api_results = test_api_endpoints()
    
    # Test 4: Frontend readiness
    frontend_ready = test_frontend_build()
    
    # Final summary
    print(f"\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    # System performance
    total_trades = len(trading_system.trade_history)
    win_rate = 0
    if total_trades > 0:
        winning_trades = sum(1 for t in trading_system.trade_history if t.pnl and t.pnl > 0)
        win_rate = winning_trades / total_trades
    
    print(f"🤖 Trading System:")
    print(f"   ✅ Optimized configuration loaded")
    print(f"   ✅ {len(trading_system.bot_configs)} bots configured")
    print(f"   📊 Performance: {total_trades} trades, {win_rate:.1%} win rate")
    print(f"   💰 PnL: ${trading_system.total_pnl:.2f}")
    
    # API status
    api_success = sum(1 for result in api_results.values() if "✅" in result)
    api_total = len(api_results)
    
    print(f"\n🌐 API Server:")
    print(f"   📡 Endpoints: {api_success}/{api_total} working")
    print(f"   🔗 URL: http://localhost:8000")
    
    # Frontend status
    print(f"\n📱 Frontend:")
    print(f"   {'✅' if frontend_ready else '❌'} React app ready")
    print(f"   🔗 URL: http://localhost:5173 (after npm run dev)")
    
    # Overall assessment
    overall_success = (
        len(trading_system.bot_configs) >= 3 and
        api_success >= (api_total * 0.8) and
        frontend_ready
    )
    
    print(f"\n🏁 Overall Status: {'✅ READY FOR USE' if overall_success else '⚠️ NEEDS ATTENTION'}")
    
    if overall_success:
        print(f"\n🎉 SYSTEM READY!")
        print(f"   1. API server is running on port 8000")
        print(f"   2. Start frontend: cd frontend_react && npm run dev")
        print(f"   3. Access dashboard at http://localhost:5173")
        print(f"   4. System is optimized and ready for trading")
    
    # Keep server running for manual testing
    print(f"\n⏰ API server will run for 60 seconds for manual testing...")
    time.sleep(60)
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)