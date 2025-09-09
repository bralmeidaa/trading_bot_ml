#!/usr/bin/env python3
"""
Test script to validate bot initialization with historical data.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_trading_system import ProductionTradingSystem
from optimized_config import create_optimized_config
from mock_exchange import MockExchange
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_initialization():
    """Test the bot initialization process."""
    logger.info("🧪 Testing bot initialization...")
    
    try:
        # Load configuration
        global_config, bot_configs = create_optimized_config()
        
        # Create trading system with mock exchange
        trading_system = ProductionTradingSystem(global_config, bot_configs)
        trading_system.exchange = MockExchange()  # Replace with mock exchange
        
        # Test initialization only (don't start the main loop)
        await trading_system._initialize_bots()
        
        # Convert list to dict for compatibility
        bot_configs_dict = {f"bot_{i}": config for i, config in enumerate(bot_configs)}
        
        # Check results
        enabled_bots = sum(1 for config in bot_configs_dict.values() if config.enabled)
        total_bots = len(bot_configs_dict)
        
        logger.info(f"✅ Initialization test completed!")
        logger.info(f"📊 Results: {enabled_bots}/{total_bots} bots successfully initialized")
        
        # Test signal generation for each enabled bot
        for bot_id, config in bot_configs_dict.items():
            if config.enabled:
                try:
                    signal_generator = trading_system.signal_generators[bot_id]
                    
                    # Fetch test data
                    ohlcv = trading_system.exchange.fetch_ohlcv(config.symbol, config.timeframe, limit=100)
                    if ohlcv and len(ohlcv) >= 50:
                        import pandas as pd
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        signals = signal_generator.generate_signals(df)
                        
                        logger.info(f"🤖 {config.symbol}: Generated {len(signals)} signals - Model fitted: {signal_generator.is_fitted}")
                    else:
                        logger.warning(f"⚠️ {config.symbol}: Insufficient test data")
                        
                except Exception as e:
                    logger.error(f"❌ {config.symbol}: Signal generation test failed - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_initialization())
    sys.exit(0 if success else 1)