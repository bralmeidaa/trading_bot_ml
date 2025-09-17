import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  DollarSign, 
  Users, 
  BarChart3,
  Activity
} from 'lucide-react';

const MarketSentimentPanel = () => {
  const [sentimentData, setSentimentData] = useState({
    BTC: null,
    ETH: null,
    SOL: null,
    LINK: null
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchSentimentData();
    const interval = setInterval(fetchSentimentData, 10000); // Atualizar a cada 10 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchSentimentData = async () => {
    try {
      const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT'];
      const promises = symbols.map(async (symbol) => {
        const response = await fetch(`/api/market-sentiment/${symbol}`);
        if (response.ok) {
          const data = await response.json();
          return { symbol: symbol.split('/')[0], data };
        }
        return null;
      });

      const results = await Promise.all(promises);
      const newData = {};
      
      results.forEach(result => {
        if (result) {
          newData[result.symbol] = result.data;
        }
      });

      setSentimentData(newData);
    } catch (error) {
      console.error('Erro ao buscar dados de sentiment:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'bullish':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'bearish':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      default:
        return <Minus className="h-4 w-4 text-gray-600" />;
    }
  };

  const getSentimentBadge = (sentiment, score) => {
    const intensity = score > 0.7 ? 'forte' : score > 0.3 ? 'moderado' : 'fraco';
    
    if (sentiment === 'bullish') {
      return <Badge className="bg-green-100 text-green-800">Bullish {intensity}</Badge>;
    } else if (sentiment === 'bearish') {
      return <Badge className="bg-red-100 text-red-800">Bearish {intensity}</Badge>;
    } else {
      return <Badge className="bg-gray-100 text-gray-800">Neutro</Badge>;
    }
  };

  const formatFundingRate = (rate) => {
    if (!rate) return 'N/A';
    return `${(rate * 100).toFixed(4)}%`;
  };

  const formatRatio = (ratio) => {
    if (!ratio) return 'N/A';
    return ratio.toFixed(2);
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Sentiment de Mercado
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">Carregando dados de sentiment...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Overview Geral */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Sentiment de Mercado - Binance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(sentimentData).map(([symbol, data]) => (
              <div key={symbol} className="text-center p-3 border rounded-lg">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <span className="font-bold text-lg">{symbol}</span>
                  {data && getSentimentIcon(data.sentiment_label)}
                </div>
                
                {data ? (
                  <>
                    <div className="text-2xl font-bold mb-1">
                      {(data.sentiment_score * 100).toFixed(0)}
                    </div>
                    <Progress 
                      value={data.sentiment_score * 100} 
                      className="h-2 mb-2"
                    />
                    {getSentimentBadge(data.sentiment_label, data.sentiment_score)}
                  </>
                ) : (
                  <div className="text-gray-500">Sem dados</div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Detalhes por Símbolo */}
      {Object.entries(sentimentData).map(([symbol, data]) => (
        data && (
          <Card key={symbol}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>{symbol}/USDT - Análise Detalhada</span>
                {getSentimentBadge(data.sentiment_label, data.sentiment_score)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                
                {/* Funding Rate */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">Funding Rate</span>
                  </div>
                  <div className="text-lg font-bold">
                    {formatFundingRate(data.raw_data?.funding_rate)}
                  </div>
                  <div className="text-xs text-gray-600">
                    {Math.abs(data.raw_data?.funding_rate || 0) < 0.0005 ? 'Neutro' : 
                     (data.raw_data?.funding_rate || 0) > 0 ? 'Longs pagam' : 'Shorts pagam'}
                  </div>
                </div>

                {/* Long/Short Ratio */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Users className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium">Long/Short</span>
                  </div>
                  <div className="text-lg font-bold">
                    {formatRatio(data.raw_data?.long_short_ratio?.current_ratio)}
                  </div>
                  <div className="text-xs text-gray-600">
                    {data.raw_data?.long_short_ratio?.sentiment || 'N/A'}
                  </div>
                </div>

                {/* Open Interest */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium">Open Interest</span>
                  </div>
                  <div className="text-lg font-bold">
                    {data.raw_data?.open_interest?.trend || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">
                    {data.raw_data?.open_interest?.change_pct ? 
                     `${data.raw_data.open_interest.change_pct.toFixed(2)}%` : 'N/A'}
                  </div>
                </div>

                {/* Order Book */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-green-600" />
                    <span className="text-sm font-medium">Order Book</span>
                  </div>
                  <div className="text-lg font-bold">
                    {data.raw_data?.order_book?.sentiment || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">
                    Spread: {data.raw_data?.order_book?.spread_pct ? 
                            `${(data.raw_data.order_book.spread_pct * 100).toFixed(3)}%` : 'N/A'}
                  </div>
                </div>

              </div>

              {/* Fatores de Sentiment */}
              {data.factors && data.factors.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium mb-2">Fatores Identificados:</h4>
                  <div className="flex flex-wrap gap-2">
                    {data.factors.map((factor, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {factor}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Confidence Score */}
              <div className="mt-4">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium">Confiança da Análise</span>
                  <span className="text-sm font-bold">
                    {(data.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={data.confidence * 100} className="h-2" />
              </div>

            </CardContent>
          </Card>
        )
      ))}
    </div>
  );
};

export default MarketSentimentPanel;