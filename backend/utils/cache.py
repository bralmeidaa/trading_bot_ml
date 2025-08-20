"""
Caching system for performance optimization.
"""
import pickle
import hashlib
import os
import time
from typing import Any, Optional, Callable
from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MemoryCache:
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.access_times = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.default_ttl
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self.cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.remove(oldest_key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            return self.cache[key]
        elif key in self.cache:
            # Remove expired item
            self.remove(key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        self._evict_lru()
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.access_times[key] = time.time()
    
    def remove(self, key: str) -> None:
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.timestamps.clear()
        self.access_times.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        current_time = time.time()
        expired_count = sum(1 for ts in self.timestamps.values() 
                           if current_time - ts > self.default_ttl)
        
        return {
            'total_items': len(self.cache),
            'expired_items': expired_count,
            'active_items': len(self.cache) - expired_count,
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size
        }


class DiskCache:
    """Persistent disk cache."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 1000):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        os.makedirs(cache_dir, exist_ok=True)
        
        # Metadata file
        self.metadata_file = os.path.join(cache_dir, "metadata.pkl")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _get_cache_path(self, key: str) -> str:
        """Get file path for cache key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.pkl")
    
    def _cleanup_old_files(self) -> None:
        """Remove old cache files to stay under size limit."""
        # Get all cache files with their sizes and modification times
        files_info = []
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl') and filename != 'metadata.pkl':
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.exists(filepath):
                    stat = os.stat(filepath)
                    files_info.append({
                        'path': filepath,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime
                    })
                    total_size += stat.st_size
        
        # If over limit, remove oldest files
        max_size_bytes = self.max_size_mb * 1024 * 1024
        if total_size > max_size_bytes:
            # Sort by modification time (oldest first)
            files_info.sort(key=lambda x: x['mtime'])
            
            for file_info in files_info:
                if total_size <= max_size_bytes:
                    break
                
                os.remove(file_info['path'])
                total_size -= file_info['size']
                
                # Remove from metadata
                key_to_remove = None
                for key, meta in self.metadata.items():
                    if meta.get('path') == file_info['path']:
                        key_to_remove = key
                        break
                
                if key_to_remove:
                    del self.metadata[key_to_remove]
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        if key not in self.metadata:
            return None
        
        meta = self.metadata[key]
        
        # Check if expired
        if time.time() - meta['timestamp'] > meta.get('ttl', 3600):
            self.remove(key)
            return None
        
        cache_path = self._get_cache_path(key)
        if not os.path.exists(cache_path):
            # File was deleted, remove from metadata
            del self.metadata[key]
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update access time
            self.metadata[key]['last_access'] = time.time()
            return data
        except:
            # Corrupted file, remove it
            self.remove(key)
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set item in disk cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            self.metadata[key] = {
                'path': cache_path,
                'timestamp': time.time(),
                'last_access': time.time(),
                'ttl': ttl,
                'size': os.path.getsize(cache_path)
            }
            
            self._save_metadata()
            self._cleanup_old_files()
            
        except Exception as e:
            print(f"Error saving to disk cache: {e}")
    
    def remove(self, key: str) -> None:
        """Remove item from disk cache."""
        if key in self.metadata:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            del self.metadata[key]
            self._save_metadata()
    
    def clear(self) -> None:
        """Clear all disk cache."""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
        self.metadata.clear()
        self._save_metadata()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total_size = sum(meta.get('size', 0) for meta in self.metadata.values())
        current_time = time.time()
        expired_count = sum(1 for meta in self.metadata.values() 
                           if current_time - meta['timestamp'] > meta.get('ttl', 3600))
        
        return {
            'total_items': len(self.metadata),
            'expired_items': expired_count,
            'active_items': len(self.metadata) - expired_count,
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_size_mb,
            'utilization': (total_size / (1024 * 1024)) / self.max_size_mb
        }


class CacheManager:
    """Unified cache manager with memory and disk tiers."""
    
    def __init__(self, 
                 memory_cache_size: int = 500,
                 disk_cache_size_mb: int = 1000,
                 cache_dir: str = "cache"):
        self.memory_cache = MemoryCache(max_size=memory_cache_size)
        self.disk_cache = DiskCache(cache_dir=cache_dir, max_size_mb=disk_cache_size_mb)
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.hits += 1
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            self.hits += 1
            return value
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, disk_only: bool = False) -> None:
        """Set item in cache."""
        if not disk_only:
            self.memory_cache.set(key, value, ttl)
        self.disk_cache.set(key, value, ttl)
    
    def remove(self, key: str) -> None:
        """Remove item from both caches."""
        self.memory_cache.remove(key)
        self.disk_cache.remove(key)
    
    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self) -> dict:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.stats()
        disk_stats = self.disk_cache.stats()
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.hits,
            'total_misses': self.misses,
            'memory_cache': memory_stats,
            'disk_cache': disk_stats
        }


# Global cache instance
cache_manager = CacheManager()


def cached(ttl: int = 3600, disk_only: bool = False, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                
                # Add args
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    elif isinstance(arg, (pd.DataFrame, pd.Series)):
                        # Use hash of data for pandas objects
                        key_parts.append(str(hash(str(arg.values.tobytes()))))
                    elif isinstance(arg, np.ndarray):
                        key_parts.append(str(hash(arg.tobytes())))
                    else:
                        key_parts.append(str(hash(str(arg))))
                
                # Add kwargs
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl, disk_only=disk_only)
            
            return result
        
        return wrapper
    return decorator


def cache_dataframe_key(*args, **kwargs) -> str:
    """Generate cache key for DataFrame-based functions."""
    key_parts = []
    
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            # Use shape, columns, and sample of data for key
            shape_str = f"shape_{arg.shape[0]}x{arg.shape[1]}"
            cols_str = f"cols_{hash(tuple(arg.columns))}"
            
            # Sample a few values for uniqueness
            if not arg.empty:
                sample_data = arg.iloc[::max(1, len(arg)//10)].values.flatten()[:20]
                data_str = f"data_{hash(sample_data.tobytes())}"
            else:
                data_str = "empty"
            
            key_parts.append(f"{shape_str}_{cols_str}_{data_str}")
        else:
            key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    return "|".join(key_parts)


# Specialized caching decorators
def cache_features(ttl: int = 1800):  # 30 minutes
    """Cache feature engineering results."""
    return cached(ttl=ttl, key_func=cache_dataframe_key)


def cache_model_predictions(ttl: int = 300):  # 5 minutes
    """Cache model predictions."""
    return cached(ttl=ttl, key_func=cache_dataframe_key)


def cache_indicators(ttl: int = 3600):  # 1 hour
    """Cache technical indicators."""
    return cached(ttl=ttl, key_func=cache_dataframe_key)


def cache_backtest_results(ttl: int = 7200):  # 2 hours
    """Cache backtest results."""
    return cached(ttl=ttl, disk_only=True)  # Store on disk due to size