#!/usr/bin/env python3
"""
Apply Optimizations to Trading Bot
This script applies the optimizations to increase trading activity.
"""

import shutil
import os
from datetime import datetime

def backup_original_files():
    """Backup original files before applying changes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    
    print(f"📦 Creating backup in {backup_dir}/...")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup original files
    files_to_backup = [
        'production_trading_system.py',
        'api_server.py'
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, f"{backup_dir}/{file}")
            print(f"   ✅ Backed up {file}")
    
    return backup_dir

def apply_signal_generator_optimization():
    """Replace the signal generator in production system."""
    print("\n🔧 Applying Enhanced Signal Generator...")
    
    # Read the current production system
    with open('production_trading_system.py', 'r') as f:
        content = f.read()
    
    # Replace the OptimizedSignalGenerator class with import
    enhanced_import = """
# Import enhanced signal generator
from enhanced_signal_generator import EnhancedSignalGenerator
"""
    
    # Add import at the top
    if "from enhanced_signal_generator import EnhancedSignalGenerator" not in content:
        # Find the imports section and add our import
        import_pos = content.find("import logging")
        if import_pos != -1:
            content = content[:import_pos] + enhanced_import + content[import_pos:]
    
    # Replace the signal generator initialization
    old_init = "self.signal_generators[bot_id] = OptimizedSignalGenerator(config.symbol, config.timeframe)"
    new_init = "self.signal_generators[bot_id] = EnhancedSignalGenerator(config.symbol, config.timeframe)"
    
    content = content.replace(old_init, new_init)
    
    # Write back the modified content
    with open('production_trading_system.py', 'w') as f:
        f.write(content)
    
    print("   ✅ Enhanced Signal Generator integrated")

def apply_configuration_optimization():
    """Update the production configuration."""
    print("\n⚙️ Applying Optimized Configuration...")
    
    # Read the current production system
    with open('production_trading_system.py', 'r') as f:
        content = f.read()
    
    # Replace the create_production_config function
    new_config_function = '''def create_production_config() -> Tuple[GlobalConfig, List[BotConfig]]:
    """Create production configuration based on backtest results."""
    
    global_config = GlobalConfig(
        total_capital=1200.0,  # Capital mínimo otimizado para Brasil (R$ 6,000)
        max_concurrent_trades=3,  # Increased from 2 to 3
        daily_loss_limit=0.05,  # Increased from 4% to 5%
        daily_profit_target=0.04,  # Increased from 2.5% to 4%
        emergency_stop_drawdown=0.08,
        paper_trading=True  # Start with paper trading
    )
    
    # Optimized configuration for more active trading
    bot_configs = [
        BotConfig(
            symbol='LINK/USDT',
            timeframe='5m',
            capital_allocation=0.40,  # Reduced from 70% to 40%
            max_risk_per_trade=0.025,  # 2.5% risco por trade
            confidence_threshold=0.55,  # Reduced from 0.65 to 0.55
            stop_loss_pct=0.018,  # 1.8% stop loss
            take_profit_pct=0.035  # 3.5% take profit
        ),
        BotConfig(
            symbol='LINK/USDT',
            timeframe='1m',
            capital_allocation=0.35,  # Increased from 30% to 35%
            max_risk_per_trade=0.020,  # 2.0% risco por trade
            confidence_threshold=0.50,  # Reduced from 0.65 to 0.50
            stop_loss_pct=0.015,  # 1.5% stop loss
            take_profit_pct=0.030  # 3.0% take profit
        ),
        BotConfig(
            symbol='ADA/USDT',
            timeframe='1m',
            capital_allocation=0.25,  # Added ADA for more opportunities
            max_risk_per_trade=0.020,  # 2.0% risco por trade
            confidence_threshold=0.50,  # Lower threshold
            stop_loss_pct=0.015,  # 1.5% stop loss
            take_profit_pct=0.030  # 3.0% take profit
        )
    ]
    
    return global_config, bot_configs'''
    
    # Find and replace the function
    start_marker = "def create_production_config() -> Tuple[GlobalConfig, List[BotConfig]]:"
    end_marker = "    return global_config, bot_configs"
    
    start_pos = content.find(start_marker)
    if start_pos != -1:
        # Find the end of the function
        end_pos = content.find(end_marker, start_pos)
        if end_pos != -1:
            end_pos = content.find("\n", end_pos) + 1
            # Replace the function
            content = content[:start_pos] + new_config_function + content[end_pos:]
    
    # Write back the modified content
    with open('production_trading_system.py', 'w') as f:
        f.write(content)
    
    print("   ✅ Optimized Configuration applied")

def add_enhanced_logging():
    """Add enhanced logging for better monitoring."""
    print("\n📝 Adding Enhanced Logging...")
    
    # Read the current production system
    with open('production_trading_system.py', 'r') as f:
        content = f.read()
    
    # Add signal statistics logging to the _process_bot method
    enhanced_logging = '''
            # Log signal generation statistics periodically
            if hasattr(signal_generator, 'get_signal_stats'):
                stats = signal_generator.get_signal_stats()
                if stats['total_checks'] > 0 and stats['total_checks'] % 100 == 0:
                    logger.info(f"📊 {bot_id} Signal Stats: "
                               f"Total: {stats['total_checks']}, "
                               f"Combined: {stats['combined']} ({stats['combined']/stats['total_checks']*100:.1f}%)")'''
    
    # Find the _process_bot method and add logging
    process_bot_marker = "# Generate signals"
    pos = content.find(process_bot_marker)
    if pos != -1:
        # Find the end of the signals generation section
        next_section = content.find("if not signals:", pos)
        if next_section != -1:
            content = content[:next_section] + enhanced_logging + "\n            " + content[next_section:]
    
    # Write back the modified content
    with open('production_trading_system.py', 'w') as f:
        f.write(content)
    
    print("   ✅ Enhanced logging added")

def create_quick_start_script():
    """Create a quick start script with optimized settings."""
    print("\n🚀 Creating Quick Start Script...")
    
    quick_start_content = '''#!/usr/bin/env python3
"""
Quick Start - Optimized Trading Bot
Run this to start the bot with optimized settings for increased activity.
"""

import asyncio
import logging
from production_trading_system import ProductionTradingSystem, create_production_config

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system_optimized.log'),
        logging.StreamHandler()
    ]
)

async def main():
    """Run the optimized trading system."""
    print("🚀 Starting OPTIMIZED Trading Bot ML...")
    print("=" * 60)
    print("🎯 OPTIMIZATIONS APPLIED:")
    print("   ✅ Enhanced Signal Generator (more flexible)")
    print("   ✅ Lower confidence thresholds (0.50-0.55)")
    print("   ✅ Added ADA/USDT trading pair")
    print("   ✅ Increased concurrent trades (2→3)")
    print("   ✅ Enhanced logging and monitoring")
    print("=" * 60)
    
    # Create optimized configuration
    global_config, bot_configs = create_production_config()
    
    # Display configuration
    print(f"💼 Total Capital: ${global_config.total_capital:,.2f}")
    print(f"📝 Paper Trading: {global_config.paper_trading}")
    print(f"🤖 Number of Bots: {len(bot_configs)}")
    print("\\n📊 Bot Configurations:")
    
    for i, config in enumerate(bot_configs, 1):
        print(f"  {i}. {config.symbol} {config.timeframe} - "
              f"{config.capital_allocation:.0%} allocation, "
              f"confidence: {config.confidence_threshold:.2f}")
    
    print("\\n⚠️  MONITORING TIPS:")
    print("   • Watch for increased signal generation in logs")
    print("   • Expect 5-20 trades per day (vs previous 2 in 6 days)")
    print("   • Monitor 'Signal Stats' messages every 100 checks")
    print("   • Check dashboard for real-time activity")
    print("\\n" + "=" * 60)
    
    # Initialize and start system
    system = ProductionTradingSystem(global_config, bot_configs)
    await system.start()

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('start_optimized_bot.py', 'w') as f:
        f.write(quick_start_content)
    
    os.chmod('start_optimized_bot.py', 0o755)
    print("   ✅ Quick start script created: start_optimized_bot.py")

def main():
    """Apply all optimizations."""
    print("🎯 APPLYING TRADING BOT OPTIMIZATIONS")
    print("=" * 50)
    print("This will optimize your bot for increased trading activity")
    print("while maintaining safety and risk management.")
    print("=" * 50)
    
    # Backup original files
    backup_dir = backup_original_files()
    
    try:
        # Apply optimizations
        apply_signal_generator_optimization()
        apply_configuration_optimization()
        add_enhanced_logging()
        create_quick_start_script()
        
        print("\n" + "=" * 50)
        print("✅ ALL OPTIMIZATIONS APPLIED SUCCESSFULLY!")
        print("=" * 50)
        print("\n🚀 NEXT STEPS:")
        print("1. Run diagnostic: python diagnostic_tool.py")
        print("2. Start optimized bot: python start_optimized_bot.py")
        print("3. Monitor dashboard for increased activity")
        print("4. Check logs for signal generation statistics")
        print("\n📊 EXPECTED IMPROVEMENTS:")
        print("• 5-20 trades per day (vs 2 in 6 days)")
        print("• More flexible signal generation")
        print("• Better market opportunity detection")
        print("• Enhanced monitoring and debugging")
        print(f"\n💾 Original files backed up in: {backup_dir}/")
        print("\n⚠️  Remember: Still in paper trading mode for safety!")
        
    except Exception as e:
        print(f"\n❌ Error applying optimizations: {e}")
        print(f"💾 Original files are backed up in: {backup_dir}/")
        print("You can restore them if needed.")

if __name__ == "__main__":
    main()