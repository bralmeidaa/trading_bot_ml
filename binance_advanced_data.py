#!/usr/bin/env python3
"""
Binance Advanced Data Integration
Integra dados avançados da Binance API para melhorar precisão dos sinais
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class BinanceAdvancedData:
    """Integração com dados avançados da Binance."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://fapi.binance.com"
        self.session = None
        
        # Cache para evitar muitas chamadas à API
        self.cache = {}
        self.cache_duration = 300  # 5 minutos
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Obtém funding rate atual do símbolo."""
        cache_key = f"funding_rate_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/fapi/v1/fundingRate"
            params = {"symbol": symbol.replace('/', ''), "limit": 1}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        funding_rate = float(data[0]['fundingRate'])
                        self._cache_data(cache_key, funding_rate)
                        return funding_rate
        except Exception as e:
            print(f"Erro ao obter funding rate para {symbol}: {e}")
        
        return None
    
    async def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """Obtém Open Interest e sua tendência."""
        cache_key = f"open_interest_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Obter OI atual
            url = f"{self.base_url}/fapi/v1/openInterest"
            params = {"symbol": symbol.replace('/', '')}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    current_data = await response.json()
                    current_oi = float(current_data['openInterest'])
                    
                    # Obter histórico para calcular tendência
                    hist_url = f"{self.base_url}/futures/data/openInterestHist"
                    hist_params = {
                        "symbol": symbol.replace('/', ''),
                        "period": "5m",
                        "limit": 12  # Última hora
                    }
                    
                    async with self.session.get(hist_url, params=hist_params) as hist_response:
                        if hist_response.status == 200:
                            hist_data = await hist_response.json()
                            
                            if len(hist_data) >= 2:
                                # Calcular tendência
                                recent_oi = [float(item['sumOpenInterest']) for item in hist_data[-6:]]
                                older_oi = [float(item['sumOpenInterest']) for item in hist_data[-12:-6]]
                                
                                recent_avg = np.mean(recent_oi)
                                older_avg = np.mean(older_oi)
                                
                                trend = 'increasing' if recent_avg > older_avg * 1.02 else \
                                       'decreasing' if recent_avg < older_avg * 0.98 else 'stable'
                                
                                result = {
                                    'current_oi': current_oi,
                                    'trend': trend,
                                    'change_pct': (recent_avg - older_avg) / older_avg * 100
                                }
                                
                                self._cache_data(cache_key, result)
                                return result
        except Exception as e:
            print(f"Erro ao obter Open Interest para {symbol}: {e}")
        
        return None
    
    async def get_long_short_ratio(self, symbol: str) -> Optional[Dict]:
        """Obtém ratio de posições long/short."""
        cache_key = f"long_short_ratio_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/futures/data/globalLongShortAccountRatio"
            params = {
                "symbol": symbol.replace('/', ''),
                "period": "5m",
                "limit": 12
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        # Pegar dados mais recentes
                        latest = data[-1]
                        current_ratio = float(latest['longShortRatio'])
                        
                        # Calcular média e tendência
                        ratios = [float(item['longShortRatio']) for item in data]
                        avg_ratio = np.mean(ratios)
                        trend = 'increasing' if current_ratio > avg_ratio * 1.05 else \
                               'decreasing' if current_ratio < avg_ratio * 0.95 else 'stable'
                        
                        result = {
                            'current_ratio': current_ratio,
                            'avg_ratio': avg_ratio,
                            'trend': trend,
                            'sentiment': 'bullish' if current_ratio > 1.2 else \
                                       'bearish' if current_ratio < 0.8 else 'neutral'
                        }
                        
                        self._cache_data(cache_key, result)
                        return result
        except Exception as e:
            print(f"Erro ao obter Long/Short ratio para {symbol}: {e}")
        
        return None
    
    async def get_taker_buy_sell_volume(self, symbol: str) -> Optional[Dict]:
        """Obtém volume de compra/venda dos takers."""
        cache_key = f"taker_volume_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/futures/data/takerlongshortRatio"
            params = {
                "symbol": symbol.replace('/', ''),
                "period": "5m",
                "limit": 12
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        latest = data[-1]
                        buy_vol = float(latest['buySellRatio'])
                        
                        # Calcular tendência
                        ratios = [float(item['buySellRatio']) for item in data]
                        avg_ratio = np.mean(ratios)
                        
                        result = {
                            'buy_sell_ratio': buy_vol,
                            'avg_ratio': avg_ratio,
                            'trend': 'buying_pressure' if buy_vol > avg_ratio * 1.1 else \
                                   'selling_pressure' if buy_vol < avg_ratio * 0.9 else 'balanced',
                            'strength': abs(buy_vol - 1.0)  # Distância de 1.0 (neutro)
                        }
                        
                        self._cache_data(cache_key, result)
                        return result
        except Exception as e:
            print(f"Erro ao obter Taker volume para {symbol}: {e}")
        
        return None
    
    async def get_order_book_analysis(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Analisa order book para micro-timing."""
        cache_key = f"order_book_{symbol}_{limit}"
        
        if self._is_cached(cache_key, duration=30):  # Cache mais curto para order book
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/fapi/v1/depth"
            params = {"symbol": symbol.replace('/', ''), "limit": limit}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    bids = [[float(price), float(qty)] for price, qty in data['bids']]
                    asks = [[float(price), float(qty)] for price, qty in data['asks']]
                    
                    if bids and asks:
                        # Análise básica
                        best_bid = bids[0][0]
                        best_ask = asks[0][0]
                        spread = best_ask - best_bid
                        spread_pct = spread / best_bid * 100
                        
                        # Volume nos primeiros níveis
                        bid_volume_top5 = sum([qty for _, qty in bids[:5]])
                        ask_volume_top5 = sum([qty for _, qty in asks[:5]])
                        
                        # Imbalance
                        total_bid_vol = sum([qty for _, qty in bids])
                        total_ask_vol = sum([qty for _, qty in asks])
                        imbalance = total_bid_vol / (total_bid_vol + total_ask_vol)
                        
                        # Detectar walls (grandes ordens)
                        avg_bid_size = np.mean([qty for _, qty in bids])
                        avg_ask_size = np.mean([qty for _, qty in asks])
                        
                        bid_walls = [price for price, qty in bids if qty > avg_bid_size * 3]
                        ask_walls = [price for price, qty in asks if qty > avg_ask_size * 3]
                        
                        result = {
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'imbalance': imbalance,  # > 0.5 = mais bids, < 0.5 = mais asks
                            'liquidity': 'high' if spread_pct < 0.05 else 'medium' if spread_pct < 0.1 else 'low',
                            'bid_walls': len(bid_walls),
                            'ask_walls': len(ask_walls),
                            'sentiment': 'bullish' if imbalance > 0.6 else 'bearish' if imbalance < 0.4 else 'neutral'
                        }
                        
                        self._cache_data(cache_key, result, duration=30)
                        return result
        except Exception as e:
            print(f"Erro ao analisar order book para {symbol}: {e}")
        
        return None
    
    async def get_24hr_stats(self, symbol: str) -> Optional[Dict]:
        """Obtém estatísticas 24h para filtrar volatilidade."""
        cache_key = f"24hr_stats_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            params = {"symbol": symbol.replace('/', '')}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    result = {
                        'price_change_pct': float(data['priceChangePercent']),
                        'volume': float(data['volume']),
                        'quote_volume': float(data['quoteVolume']),
                        'high_low_pct': (float(data['highPrice']) - float(data['lowPrice'])) / float(data['lowPrice']) * 100,
                        'volatility': 'high' if abs(float(data['priceChangePercent'])) > 5 else \
                                    'medium' if abs(float(data['priceChangePercent'])) > 2 else 'low'
                    }
                    
                    self._cache_data(cache_key, result)
                    return result
        except Exception as e:
            print(f"Erro ao obter stats 24h para {symbol}: {e}")
        
        return None
    
    def _is_cached(self, key: str, duration: int = None) -> bool:
        """Verifica se dados estão em cache e ainda válidos."""
        if key not in self.cache:
            return False
        
        cache_duration = duration or self.cache_duration
        return time.time() - self.cache[key]['timestamp'] < cache_duration
    
    def _cache_data(self, key: str, data, duration: int = None):
        """Armazena dados no cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }

class MarketSentimentAnalyzer:
    """Analisador de sentiment baseado em dados da Binance."""
    
    def __init__(self, binance_data: BinanceAdvancedData):
        self.binance_data = binance_data
    
    async def analyze_market_sentiment(self, symbol: str) -> Dict:
        """Analisa sentiment geral do mercado para o símbolo."""
        
        # Obter todos os dados necessários
        funding_rate = await self.binance_data.get_funding_rate(symbol)
        oi_data = await self.binance_data.get_open_interest(symbol)
        ls_ratio = await self.binance_data.get_long_short_ratio(symbol)
        taker_data = await self.binance_data.get_taker_buy_sell_volume(symbol)
        order_book = await self.binance_data.get_order_book_analysis(symbol)
        stats_24h = await self.binance_data.get_24hr_stats(symbol)
        
        sentiment_score = 0.0
        confidence = 0.0
        factors = []
        
        # Análise do Funding Rate
        if funding_rate is not None:
            if abs(funding_rate) < 0.0005:  # Neutro
                sentiment_score += 0.5
                factors.append("Funding rate neutro")
            elif funding_rate > 0.001:  # Muito bullish (cuidado)
                sentiment_score += 0.2
                factors.append("Funding rate muito alto (possível reversão)")
            elif funding_rate < -0.001:  # Muito bearish (oportunidade)
                sentiment_score += 0.8
                factors.append("Funding rate negativo (oportunidade long)")
            confidence += 0.2
        
        # Análise do Open Interest
        if oi_data:
            if oi_data['trend'] == 'increasing':
                sentiment_score += 0.7
                factors.append("Open Interest crescente")
            elif oi_data['trend'] == 'stable':
                sentiment_score += 0.5
                factors.append("Open Interest estável")
            else:
                sentiment_score += 0.3
                factors.append("Open Interest decrescente")
            confidence += 0.2
        
        # Análise Long/Short Ratio
        if ls_ratio:
            ratio = ls_ratio['current_ratio']
            if 0.8 <= ratio <= 1.2:  # Balanceado
                sentiment_score += 0.6
                factors.append("Long/Short ratio balanceado")
            elif ratio > 2.0:  # Muitos longs (contrarian)
                sentiment_score += 0.3
                factors.append("Excesso de longs (risco de squeeze)")
            elif ratio < 0.5:  # Muitos shorts (contrarian)
                sentiment_score += 0.8
                factors.append("Excesso de shorts (oportunidade)")
            confidence += 0.2
        
        # Análise Taker Volume
        if taker_data:
            if taker_data['trend'] == 'buying_pressure':
                sentiment_score += 0.7
                factors.append("Pressão compradora forte")
            elif taker_data['trend'] == 'selling_pressure':
                sentiment_score += 0.3
                factors.append("Pressão vendedora forte")
            else:
                sentiment_score += 0.5
                factors.append("Pressão balanceada")
            confidence += 0.2
        
        # Análise Order Book
        if order_book:
            if order_book['sentiment'] == 'bullish':
                sentiment_score += 0.7
                factors.append("Order book bullish")
            elif order_book['sentiment'] == 'bearish':
                sentiment_score += 0.3
                factors.append("Order book bearish")
            else:
                sentiment_score += 0.5
                factors.append("Order book neutro")
            confidence += 0.2
        
        # Normalizar score
        if confidence > 0:
            sentiment_score = sentiment_score / (confidence * 5)  # Normalizar para 0-1
        
        return {
            'sentiment_score': min(max(sentiment_score, 0), 1),
            'confidence': confidence,
            'sentiment_label': 'bullish' if sentiment_score > 0.6 else \
                             'bearish' if sentiment_score < 0.4 else 'neutral',
            'factors': factors,
            'raw_data': {
                'funding_rate': funding_rate,
                'open_interest': oi_data,
                'long_short_ratio': ls_ratio,
                'taker_data': taker_data,
                'order_book': order_book,
                'stats_24h': stats_24h
            }
        }

# Exemplo de uso
async def test_binance_advanced_data():
    """Testa integração com dados avançados da Binance."""
    
    async with BinanceAdvancedData() as binance_data:
        analyzer = MarketSentimentAnalyzer(binance_data)
        
        symbol = "BTC/USDT"
        print(f"🔍 ANÁLISE AVANÇADA PARA {symbol}")
        print("=" * 50)
        
        # Testar cada função
        print("📊 Funding Rate...")
        funding = await binance_data.get_funding_rate(symbol)
        print(f"Funding Rate: {funding}")
        
        print("\n📈 Open Interest...")
        oi = await binance_data.get_open_interest(symbol)
        print(f"Open Interest: {oi}")
        
        print("\n⚖️ Long/Short Ratio...")
        ls = await binance_data.get_long_short_ratio(symbol)
        print(f"Long/Short: {ls}")
        
        print("\n💰 Taker Volume...")
        taker = await binance_data.get_taker_buy_sell_volume(symbol)
        print(f"Taker Volume: {taker}")
        
        print("\n📋 Order Book...")
        ob = await binance_data.get_order_book_analysis(symbol)
        print(f"Order Book: {ob}")
        
        print("\n🎯 Sentiment Geral...")
        sentiment = await analyzer.analyze_market_sentiment(symbol)
        print(f"Sentiment Score: {sentiment['sentiment_score']:.3f}")
        print(f"Label: {sentiment['sentiment_label']}")
        print(f"Confidence: {sentiment['confidence']:.3f}")
        print(f"Fatores: {sentiment['factors']}")

if __name__ == "__main__":
    asyncio.run(test_binance_advanced_data())