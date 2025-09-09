#!/usr/bin/env python3
"""
Mock exchange for testing purposes.
"""

import random
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class MockExchange:
    """Mock exchange that simulates market data."""
    
    def __init__(self):
        self.symbols = ['LINK/USDT', 'ADA/USDT', 'DOT/USDT']
        self.base_prices = {
            'LINK/USDT': 7.5,
            'ADA/USDT': 0.35,
            'DOT/USDT': 5.2
        }
        
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> List[List[float]]:
        """Generate mock OHLCV data."""
        if symbol not in self.symbols:
            return []
            
        base_price = self.base_prices[symbol]
        current_time = int(time.time() * 1000)
        
        # Generate realistic OHLCV data
        ohlcv_data = []
        price = base_price
        
        for i in range(limit):
            # Simulate price movement with some volatility
            change = random.uniform(-0.02, 0.02)  # ±2% change
            price = price * (1 + change)
            
            # Generate OHLC based on the price
            open_price = price
            high_price = price * (1 + random.uniform(0, 0.01))  # Up to 1% higher
            low_price = price * (1 - random.uniform(0, 0.01))   # Up to 1% lower
            close_price = price * (1 + random.uniform(-0.005, 0.005))  # ±0.5% from price
            volume = random.uniform(1000, 10000)
            
            timestamp = current_time - (limit - i) * 300000  # 5-minute intervals
            
            ohlcv_data.append([timestamp, open_price, high_price, low_price, close_price, volume])
            price = close_price
            
        return ohlcv_data
    
    def create_market_buy_order(self, symbol: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Mock market buy order."""
        return {
            'id': f'buy_{int(time.time())}_{random.randint(1000, 9999)}',
            'symbol': symbol,
            'side': 'buy',
            'amount': amount,
            'price': price or self.base_prices.get(symbol, 1.0),
            'status': 'closed',
            'filled': amount,
            'timestamp': int(time.time() * 1000)
        }
    
    def create_market_sell_order(self, symbol: str, amount: float, price: float = None) -> Dict[str, Any]:
        """Mock market sell order."""
        return {
            'id': f'sell_{int(time.time())}_{random.randint(1000, 9999)}',
            'symbol': symbol,
            'side': 'sell',
            'amount': amount,
            'price': price or self.base_prices.get(symbol, 1.0),
            'status': 'closed',
            'filled': amount,
            'timestamp': int(time.time() * 1000)
        }
    
    def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """Mock balance."""
        return {
            'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
            'LINK': {'free': 0.0, 'used': 0.0, 'total': 0.0},
            'ADA': {'free': 0.0, 'used': 0.0, 'total': 0.0},
            'DOT': {'free': 0.0, 'used': 0.0, 'total': 0.0}
        }