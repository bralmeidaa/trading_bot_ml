from typing import List, Tuple
import ccxt


class BinancePublicClient:
    """Lightweight public data client using ccxt (no API key required for candles)."""

    def __init__(self):
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Returns list of OHLCV candles: [timestamp, open, high, low, close, volume]
        timestamp is in milliseconds.
        """
        return self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
