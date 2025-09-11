#!/usr/bin/env python3
"""
Dynamic Configuration Manager
Handles real-time configuration updates without system restarts.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from production_trading_system import GlobalConfig, BotConfig

logger = logging.getLogger(__name__)

@dataclass
class ConfigUpdate:
    """Configuration update record."""
    timestamp: str
    user: str
    changes: Dict[str, Any]
    reason: str

class ConfigManager:
    """Manages dynamic configuration updates."""
    
    def __init__(self, config_file: str = "trading_config.json"):
        self.config_file = config_file
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        # Configuration history
        self.update_history: List[ConfigUpdate] = []
        self.load_update_history()
        
        # Current configuration
        self.global_config: Optional[GlobalConfig] = None
        self.bot_configs: List[BotConfig] = []
        
        # Load existing configuration
        self.load_configuration()
    
    def load_configuration(self) -> tuple[GlobalConfig, List[BotConfig]]:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Load global config
                global_data = config_data.get('global_config', {})
                self.global_config = GlobalConfig(**global_data)
                
                # Load bot configs
                self.bot_configs = []
                for bot_data in config_data.get('bot_configs', []):
                    self.bot_configs.append(BotConfig(**bot_data))
                
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                # Create default configuration
                self.create_default_configuration()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.create_default_configuration()
        
        return self.global_config, self.bot_configs
    
    def create_default_configuration(self):
        """Create default optimized configuration."""
        from production_trading_system import create_production_config
        
        self.global_config, self.bot_configs = create_production_config()
        self.save_configuration("System", "Default configuration created")
        logger.info("Default configuration created")
    
    def save_configuration(self, user: str = "System", reason: str = "Configuration update"):
        """Save current configuration to file."""
        try:
            # Create backup first
            self.create_backup()
            
            # Prepare configuration data
            config_data = {
                'global_config': asdict(self.global_config),
                'bot_configs': [asdict(config) for config in self.bot_configs],
                'last_updated': datetime.now().isoformat(),
                'updated_by': user,
                'update_reason': reason
            }
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Record update
            self.record_update(user, config_data, reason)
            
            logger.info(f"Configuration saved by {user}: {reason}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def create_backup(self):
        """Create configuration backup."""
        if os.path.exists(self.config_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"config_backup_{timestamp}.json"
            
            with open(self.config_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            
            # Keep only last 30 backups
            self.cleanup_old_backups()
    
    def cleanup_old_backups(self, keep_count: int = 30):
        """Remove old backup files."""
        backup_files = sorted(self.backup_dir.glob("config_backup_*.json"))
        if len(backup_files) > keep_count:
            for old_file in backup_files[:-keep_count]:
                old_file.unlink()
    
    def update_global_config(self, updates: Dict[str, Any], user: str = "User") -> bool:
        """Update global configuration parameters."""
        try:
            # Validate updates
            valid_updates = self.validate_global_config_updates(updates)
            
            # Apply updates
            for key, value in valid_updates.items():
                if hasattr(self.global_config, key):
                    setattr(self.global_config, key, value)
            
            # Save configuration
            self.save_configuration(user, f"Global config updated: {list(valid_updates.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating global config: {e}")
            return False
    
    def update_bot_config(self, bot_id: str, updates: Dict[str, Any], user: str = "User") -> bool:
        """Update specific bot configuration."""
        try:
            # Find bot config
            bot_config = None
            for config in self.bot_configs:
                if f"{config.symbol}_{config.timeframe}" == bot_id:
                    bot_config = config
                    break
            
            if not bot_config:
                raise ValueError(f"Bot config not found: {bot_id}")
            
            # Validate updates
            valid_updates = self.validate_bot_config_updates(updates)
            
            # Apply updates
            for key, value in valid_updates.items():
                if hasattr(bot_config, key):
                    setattr(bot_config, key, value)
            
            # Save configuration
            self.save_configuration(user, f"Bot {bot_id} updated: {list(valid_updates.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating bot config {bot_id}: {e}")
            return False
    
    def add_bot_config(self, bot_config: BotConfig, user: str = "User") -> bool:
        """Add new bot configuration."""
        try:
            # Validate bot config
            self.validate_bot_config(bot_config)
            
            # Check for duplicates
            bot_id = f"{bot_config.symbol}_{bot_config.timeframe}"
            existing = any(f"{c.symbol}_{c.timeframe}" == bot_id for c in self.bot_configs)
            
            if existing:
                raise ValueError(f"Bot configuration already exists: {bot_id}")
            
            # Add bot config
            self.bot_configs.append(bot_config)
            
            # Save configuration
            self.save_configuration(user, f"Added bot: {bot_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding bot config: {e}")
            return False
    
    def remove_bot_config(self, bot_id: str, user: str = "User") -> bool:
        """Remove bot configuration."""
        try:
            # Find and remove bot config
            original_count = len(self.bot_configs)
            self.bot_configs = [
                config for config in self.bot_configs 
                if f"{config.symbol}_{config.timeframe}" != bot_id
            ]
            
            if len(self.bot_configs) == original_count:
                raise ValueError(f"Bot config not found: {bot_id}")
            
            # Save configuration
            self.save_configuration(user, f"Removed bot: {bot_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing bot config {bot_id}: {e}")
            return False
    
    def validate_global_config_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate global configuration updates."""
        valid_updates = {}
        
        # Define validation rules
        validation_rules = {
            'total_capital': (100.0, 100000.0),  # Min/Max capital
            'max_concurrent_trades': (1, 20),     # Min/Max concurrent trades
            'daily_loss_limit': (0.01, 0.20),     # 1% to 20%
            'daily_profit_target': (0.005, 0.10), # 0.5% to 10%
            'emergency_stop_drawdown': (0.05, 0.30), # 5% to 30%
            'paper_trading': (True, False)         # Boolean values
        }
        
        for key, value in updates.items():
            if key in validation_rules:
                if key == 'paper_trading':
                    if isinstance(value, bool):
                        valid_updates[key] = value
                else:
                    min_val, max_val = validation_rules[key]
                    if isinstance(value, (int, float)) and min_val <= value <= max_val:
                        valid_updates[key] = float(value)
                    else:
                        logger.warning(f"Invalid value for {key}: {value} (range: {min_val}-{max_val})")
            else:
                logger.warning(f"Unknown global config parameter: {key}")
        
        return valid_updates
    
    def validate_bot_config_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bot configuration updates."""
        valid_updates = {}
        
        # Define validation rules
        validation_rules = {
            'capital_allocation': (0.01, 1.0),      # 1% to 100%
            'max_risk_per_trade': (0.005, 0.10),    # 0.5% to 10%
            'confidence_threshold': (0.40, 0.90),   # 40% to 90%
            'stop_loss_pct': (0.005, 0.05),         # 0.5% to 5%
            'take_profit_pct': (0.01, 0.15),        # 1% to 15%
            'enabled': (True, False)                 # Boolean values
        }
        
        for key, value in updates.items():
            if key in validation_rules:
                if key == 'enabled':
                    if isinstance(value, bool):
                        valid_updates[key] = value
                else:
                    min_val, max_val = validation_rules[key]
                    if isinstance(value, (int, float)) and min_val <= value <= max_val:
                        valid_updates[key] = float(value)
                    else:
                        logger.warning(f"Invalid value for {key}: {value} (range: {min_val}-{max_val})")
            else:
                logger.warning(f"Unknown bot config parameter: {key}")
        
        return valid_updates
    
    def validate_bot_config(self, bot_config: BotConfig):
        """Validate complete bot configuration."""
        # Validate symbol format
        if '/' not in bot_config.symbol:
            raise ValueError(f"Invalid symbol format: {bot_config.symbol}")
        
        # Validate timeframe
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        if bot_config.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {bot_config.timeframe}")
        
        # Validate ranges using existing validation
        bot_dict = asdict(bot_config)
        self.validate_bot_config_updates(bot_dict)
    
    def record_update(self, user: str, changes: Dict[str, Any], reason: str):
        """Record configuration update in history."""
        update = ConfigUpdate(
            timestamp=datetime.now().isoformat(),
            user=user,
            changes=changes,
            reason=reason
        )
        
        self.update_history.append(update)
        
        # Keep only last 100 updates
        if len(self.update_history) > 100:
            self.update_history = self.update_history[-100:]
        
        # Save update history
        self.save_update_history()
    
    def save_update_history(self):
        """Save update history to file."""
        try:
            history_file = "config_update_history.json"
            history_data = [asdict(update) for update in self.update_history]
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving update history: {e}")
    
    def load_update_history(self):
        """Load update history from file."""
        try:
            history_file = "config_update_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                self.update_history = [
                    ConfigUpdate(**update) for update in history_data
                ]
                
        except Exception as e:
            logger.error(f"Error loading update history: {e}")
            self.update_history = []
    
    def get_configuration_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        # Always reload from file to ensure we have the latest data
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                return file_config
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
        
        # Fallback to in-memory config if file read fails
        return {
            'global_config': asdict(self.global_config),
            'bot_configs': [asdict(config) for config in self.bot_configs],
            'update_history': [asdict(update) for update in self.update_history[-10:]]  # Last 10 updates
        }
    
    def restore_backup(self, backup_file: str, user: str = "User") -> bool:
        """Restore configuration from backup."""
        try:
            backup_path = self.backup_dir / backup_file
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            # Load backup
            with open(backup_path, 'r') as f:
                config_data = json.load(f)
            
            # Restore configuration
            self.global_config = GlobalConfig(**config_data['global_config'])
            self.bot_configs = [BotConfig(**bot) for bot in config_data['bot_configs']]
            
            # Save restored configuration
            self.save_configuration(user, f"Restored from backup: {backup_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup {backup_file}: {e}")
            return False
    
    def get_available_backups(self) -> List[Dict[str, Any]]:
        """Get list of available backup files."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("config_backup_*.json"), reverse=True):
            try:
                stat = backup_file.stat()
                backups.append({
                    'filename': backup_file.name,
                    'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'size': stat.st_size
                })
            except Exception as e:
                logger.error(f"Error reading backup file {backup_file}: {e}")
        
        return backups

# Global configuration manager instance
config_manager = ConfigManager()