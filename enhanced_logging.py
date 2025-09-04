#!/usr/bin/env python3
"""
Enhanced Logging System
Comprehensive logging with retention, filtering, and export capabilities.
"""

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import gzip
import shutil
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    SIGNAL = "SIGNAL"
    RISK = "RISK"
    CONFIG = "CONFIG"
    API = "API"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    category: str
    message: str
    data: Optional[Dict[str, Any]] = None
    bot_id: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None

class EnhancedLogger:
    """Enhanced logging system with structured logging and retention."""
    
    def __init__(self, log_dir: str = "logs", retention_days: int = 15):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.retention_days = retention_days
        
        # Create subdirectories for different log types
        (self.log_dir / "daily").mkdir(exist_ok=True)
        (self.log_dir / "trading").mkdir(exist_ok=True)
        (self.log_dir / "system").mkdir(exist_ok=True)
        (self.log_dir / "archived").mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # In-memory log buffer for real-time viewing
        self.log_buffer: List[LogEntry] = []
        self.max_buffer_size = 1000
        
        # Start cleanup scheduler
        self.cleanup_old_logs()
    
    def setup_logging(self):
        """Setup enhanced logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # Daily log file handler
        daily_log_file = self.log_dir / "daily" / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(daily_log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # JSON log file handler for structured logs
        json_log_file = self.log_dir / "daily" / f"trading_{datetime.now().strftime('%Y%m%d')}.json"
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(json_formatter)
        json_handler.setLevel(logging.INFO)
        root_logger.addHandler(json_handler)
        
        # Custom handler for structured logging
        self.custom_handler = StructuredLogHandler(self)
        self.custom_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(self.custom_handler)
    
    def log_structured(self, level: LogLevel, category: LogCategory, message: str, 
                      data: Optional[Dict[str, Any]] = None, bot_id: Optional[str] = None,
                      symbol: Optional[str] = None, trade_id: Optional[str] = None):
        """Log structured entry."""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            message=message,
            data=data,
            bot_id=bot_id,
            symbol=symbol,
            trade_id=trade_id
        )
        
        # Add to buffer
        self.log_buffer.append(entry)
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]
        
        # Write to category-specific log file
        self.write_to_category_log(entry)
        
        # Also log to standard logger
        logger = logging.getLogger(f"trading.{category.value.lower()}")
        log_method = getattr(logger, level.value.lower())
        
        log_message = message
        if data:
            log_message += f" | Data: {json.dumps(data, default=str)}"
        if bot_id:
            log_message += f" | Bot: {bot_id}"
        if symbol:
            log_message += f" | Symbol: {symbol}"
        if trade_id:
            log_message += f" | Trade: {trade_id}"
        
        log_method(log_message)
    
    def write_to_category_log(self, entry: LogEntry):
        """Write log entry to category-specific file."""
        try:
            category_dir = self.log_dir / entry.category.lower()
            category_dir.mkdir(exist_ok=True)
            
            log_file = category_dir / f"{entry.category.lower()}_{datetime.now().strftime('%Y%m%d')}.json"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
                
        except Exception as e:
            logging.error(f"Error writing to category log: {e}")
    
    def log_trade_signal(self, bot_id: str, symbol: str, signal_data: Dict[str, Any]):
        """Log trading signal generation."""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.SIGNAL,
            f"Signal generated for {symbol}",
            data=signal_data,
            bot_id=bot_id,
            symbol=symbol
        )
    
    def log_trade_entry(self, trade_id: str, bot_id: str, symbol: str, trade_data: Dict[str, Any]):
        """Log trade entry."""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.TRADING,
            f"Trade entered: {symbol}",
            data=trade_data,
            bot_id=bot_id,
            symbol=symbol,
            trade_id=trade_id
        )
    
    def log_trade_exit(self, trade_id: str, symbol: str, exit_data: Dict[str, Any]):
        """Log trade exit."""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.TRADING,
            f"Trade exited: {symbol}",
            data=exit_data,
            symbol=symbol,
            trade_id=trade_id
        )
    
    def log_risk_event(self, event_type: str, data: Dict[str, Any], bot_id: Optional[str] = None):
        """Log risk management event."""
        self.log_structured(
            LogLevel.WARNING,
            LogCategory.RISK,
            f"Risk event: {event_type}",
            data=data,
            bot_id=bot_id
        )
    
    def log_config_change(self, change_type: str, data: Dict[str, Any], user: str = "System"):
        """Log configuration change."""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.CONFIG,
            f"Configuration {change_type} by {user}",
            data=data
        )
    
    def log_system_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Log system event."""
        self.log_structured(
            LogLevel.INFO,
            LogCategory.SYSTEM,
            event,
            data=data
        )
    
    def log_api_request(self, endpoint: str, method: str, data: Optional[Dict[str, Any]] = None):
        """Log API request."""
        self.log_structured(
            LogLevel.DEBUG,
            LogCategory.API,
            f"{method} {endpoint}",
            data=data
        )
    
    def get_logs(self, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 level: Optional[LogLevel] = None,
                 category: Optional[LogCategory] = None,
                 bot_id: Optional[str] = None,
                 symbol: Optional[str] = None,
                 limit: int = 1000) -> List[LogEntry]:
        """Get filtered logs."""
        
        # Start with buffer for recent logs
        filtered_logs = []
        
        for entry in reversed(self.log_buffer):
            if len(filtered_logs) >= limit:
                break
                
            # Apply filters
            if start_date and datetime.fromisoformat(entry.timestamp) < start_date:
                continue
            if end_date and datetime.fromisoformat(entry.timestamp) > end_date:
                continue
            if level and entry.level != level.value:
                continue
            if category and entry.category != category.value:
                continue
            if bot_id and entry.bot_id != bot_id:
                continue
            if symbol and entry.symbol != symbol:
                continue
            
            filtered_logs.append(entry)
        
        # If we need more logs, read from files
        if len(filtered_logs) < limit and start_date:
            file_logs = self.read_logs_from_files(start_date, end_date, category)
            
            for entry in file_logs:
                if len(filtered_logs) >= limit:
                    break
                    
                # Apply remaining filters
                if level and entry.level != level.value:
                    continue
                if bot_id and entry.bot_id != bot_id:
                    continue
                if symbol and entry.symbol != symbol:
                    continue
                
                # Avoid duplicates
                if not any(e.timestamp == entry.timestamp and e.message == entry.message 
                          for e in filtered_logs):
                    filtered_logs.append(entry)
        
        return sorted(filtered_logs, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def read_logs_from_files(self, start_date: datetime, end_date: Optional[datetime] = None,
                           category: Optional[LogCategory] = None) -> List[LogEntry]:
        """Read logs from files within date range."""
        logs = []
        
        if not end_date:
            end_date = datetime.now()
        
        # Determine which files to read
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime('%Y%m%d')
            
            # Read from category-specific files if specified
            if category:
                log_file = self.log_dir / category.value.lower() / f"{category.value.lower()}_{date_str}.json"
            else:
                # Read from daily JSON logs
                log_file = self.log_dir / "daily" / f"trading_{date_str}.json"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    if category:
                                        # Structured log entry
                                        entry_data = json.loads(line)
                                        logs.append(LogEntry(**entry_data))
                                    else:
                                        # Standard JSON log entry - convert to LogEntry
                                        entry_data = json.loads(line)
                                        logs.append(LogEntry(
                                            timestamp=entry_data.get('timestamp', ''),
                                            level=entry_data.get('level', 'INFO'),
                                            category='SYSTEM',
                                            message=entry_data.get('message', ''),
                                            data=None
                                        ))
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logging.error(f"Error reading log file {log_file}: {e}")
            
            current_date += timedelta(days=1)
        
        return logs
    
    def export_logs(self, 
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   format: str = "json") -> str:
        """Export logs to file."""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)  # Last 7 days
        if not end_date:
            end_date = datetime.now()
        
        logs = self.get_logs(start_date=start_date, end_date=end_date, limit=10000)
        
        # Create export filename
        export_filename = f"trading_logs_export_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.{format}"
        export_path = self.log_dir / export_filename
        
        try:
            if format.lower() == "json":
                with open(export_path, 'w') as f:
                    json.dump([asdict(log) for log in logs], f, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                with open(export_path, 'w', newline='') as f:
                    if logs:
                        writer = csv.DictWriter(f, fieldnames=asdict(logs[0]).keys())
                        writer.writeheader()
                        for log in logs:
                            row = asdict(log)
                            # Convert data dict to string for CSV
                            if row['data']:
                                row['data'] = json.dumps(row['data'], default=str)
                            writer.writerow(row)
            
            elif format.lower() == "txt":
                with open(export_path, 'w') as f:
                    for log in logs:
                        f.write(f"[{log.timestamp}] {log.level} - {log.category} - {log.message}")
                        if log.bot_id:
                            f.write(f" | Bot: {log.bot_id}")
                        if log.symbol:
                            f.write(f" | Symbol: {log.symbol}")
                        if log.trade_id:
                            f.write(f" | Trade: {log.trade_id}")
                        if log.data:
                            f.write(f" | Data: {json.dumps(log.data, default=str)}")
                        f.write("\n")
            
            return str(export_path)
            
        except Exception as e:
            logging.error(f"Error exporting logs: {e}")
            raise
    
    def cleanup_old_logs(self):
        """Clean up logs older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up daily logs
        for log_file in self.log_dir.glob("**/*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    # Archive before deleting
                    self.archive_log_file(log_file)
                    log_file.unlink()
            except Exception as e:
                logging.error(f"Error cleaning up log file {log_file}: {e}")
        
        # Clean up JSON logs
        for log_file in self.log_dir.glob("**/*.json"):
            try:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    # Archive before deleting
                    self.archive_log_file(log_file)
                    log_file.unlink()
            except Exception as e:
                logging.error(f"Error cleaning up JSON log file {log_file}: {e}")
    
    def archive_log_file(self, log_file: Path):
        """Archive log file by compressing it."""
        try:
            archive_path = self.log_dir / "archived" / f"{log_file.name}.gz"
            
            with open(log_file, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
        except Exception as e:
            logging.error(f"Error archiving log file {log_file}: {e}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'total_logs_in_buffer': len(self.log_buffer),
            'log_levels': {},
            'log_categories': {},
            'recent_activity': {},
            'disk_usage': {}
        }
        
        # Analyze buffer
        for entry in self.log_buffer:
            # Count by level
            stats['log_levels'][entry.level] = stats['log_levels'].get(entry.level, 0) + 1
            
            # Count by category
            stats['log_categories'][entry.category] = stats['log_categories'].get(entry.category, 0) + 1
            
            # Recent activity (last hour)
            entry_time = datetime.fromisoformat(entry.timestamp)
            if entry_time > datetime.now() - timedelta(hours=1):
                hour_key = entry_time.strftime('%H:00')
                stats['recent_activity'][hour_key] = stats['recent_activity'].get(hour_key, 0) + 1
        
        # Calculate disk usage
        try:
            total_size = sum(f.stat().st_size for f in self.log_dir.rglob('*') if f.is_file())
            stats['disk_usage'] = {
                'total_bytes': total_size,
                'total_mb': round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            stats['disk_usage'] = {'error': str(e)}
        
        return stats

class StructuredLogHandler(logging.Handler):
    """Custom log handler for structured logging."""
    
    def __init__(self, enhanced_logger: EnhancedLogger):
        super().__init__()
        self.enhanced_logger = enhanced_logger
    
    def emit(self, record):
        """Emit log record to structured logger."""
        # Skip if this is already from structured logging to avoid recursion
        if hasattr(record, 'structured_logged'):
            return
        
        # Mark to avoid recursion
        record.structured_logged = True
        
        # Convert standard log to structured format
        level = LogLevel(record.levelname)
        category = LogCategory.SYSTEM  # Default category
        
        # Try to determine category from logger name
        if 'trading' in record.name.lower():
            category = LogCategory.TRADING
        elif 'signal' in record.name.lower():
            category = LogCategory.SIGNAL
        elif 'risk' in record.name.lower():
            category = LogCategory.RISK
        elif 'config' in record.name.lower():
            category = LogCategory.CONFIG
        elif 'api' in record.name.lower():
            category = LogCategory.API
        
        # Create structured entry (but don't log to avoid recursion)
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=level.value,
            category=category.value,
            message=record.getMessage(),
            data=getattr(record, 'data', None)
        )
        
        # Add to buffer only
        self.enhanced_logger.log_buffer.append(entry)
        if len(self.enhanced_logger.log_buffer) > self.enhanced_logger.max_buffer_size:
            self.enhanced_logger.log_buffer = self.enhanced_logger.log_buffer[-self.enhanced_logger.max_buffer_size:]

# Global enhanced logger instance
enhanced_logger = EnhancedLogger()