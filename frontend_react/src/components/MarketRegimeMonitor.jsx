import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Zap,
  RotateCcw,
  Target,
  Shield,
  AlertTriangle,
  Clock,
  BarChart3
} from 'lucide-react';

const MarketRegimeMonitor = () => {
  const [regimeData, setRegimeData] = useState({
    BTC: null,
    ETH: null,
    SOL: null,
    LINK: null
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchRegimeData();
    const interval = setInterval(fetchRegimeData, 15000); // Atualizar a cada 15 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchRegimeData = async () => {
    try {
      const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT'];
      const promises = symbols.map(async (symbol) => {
        const response = await fetch(`/api/market-regime/${symbol}`);
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

      setRegimeData(newData);
    } catch (error) {
      console.error('Erro ao buscar dados de regime:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getRegimeIcon = (regime) => {
    switch (regime) {
      case 'trending_bull':
        return <TrendingUp className="h-5 w-5 text-green-600" />;
      case 'trending_bear':
        return <TrendingDown className="h-5 w-5 text-red-600" />;
      case 'ranging':
        return <Minus className="h-5 w-5 text-blue-600" />;
      case 'high_volatility':
        return <Zap className="h-5 w-5 text-yellow-600" />;
      case 'transitional':
        return <RotateCcw className="h-5 w-5 text-gray-600" />;
      default:
        return <BarChart3 className="h-5 w-5 text-gray-400" />;
    }
  };

  const getRegimeBadge = (regime, confidence) => {
    const regimeLabels = {
      'trending_bull': 'Trending Bull',
      'trending_bear': 'Trending Bear', 
      'ranging': 'Ranging',
      'high_volatility': 'High Volatility',
      'transitional': 'Transitional'
    };

    const regimeColors = {
      'trending_bull': 'bg-green-100 text-green-800',
      'trending_bear': 'bg-red-100 text-red-800',
      'ranging': 'bg-blue-100 text-blue-800',
      'high_volatility': 'bg-yellow-100 text-yellow-800',
      'transitional': 'bg-gray-100 text-gray-800'
    };

    return (
      <Badge className={regimeColors[regime] || 'bg-gray-100 text-gray-800'}>
        {regimeLabels[regime] || 'Unknown'}
      </Badge>
    );
  };

  const getStrategyIcon = (strategyType) => {
    switch (strategyType) {
      case 'trend_following':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'trend_following_short':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'mean_reversion':
        return <Target className="h-4 w-4 text-blue-600" />;
      case 'volatility_breakout':
        return <Zap className="h-4 w-4 text-yellow-600" />;
      case 'conservative':
        return <Shield className="h-4 w-4 text-gray-600" />;
      default:
        return <BarChart3 className="h-4 w-4 text-gray-400" />;
    }
  };

  const getRegimeDescription = (regime) => {
    const descriptions = {
      'trending_bull': 'Mercado em tendência de alta consistente. Estratégia: seguir a tendência, comprar pullbacks.',
      'trending_bear': 'Mercado em tendência de baixa consistente. Estratégia: shorts em rallies, seguir a queda.',
      'ranging': 'Mercado lateral entre suporte e resistência. Estratégia: mean reversion, comprar suporte/vender resistência.',
      'high_volatility': 'Mercado com alta volatilidade e movimentos erráticos. Estratégia: aguardar estabilização, trades muito seletivos.',
      'transitional': 'Mercado em transição entre regimes. Estratégia: aguardar definição, máxima cautela.'
    };
    return descriptions[regime] || 'Regime não identificado.';
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Monitor de Regime de Mercado
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">Carregando dados de regime...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overview Geral */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Monitor de Regime de Mercado
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(regimeData).map(([symbol, data]) => (
              <div key={symbol} className="text-center p-4 border rounded-lg bg-gray-50">
                <div className="flex items-center justify-center gap-2 mb-3">
                  <span className="font-bold text-lg">{symbol}</span>
                  {data && getRegimeIcon(data.regime)}
                </div>
                
                {data ? (
                  <>
                    <div className="mb-2">
                      {getRegimeBadge(data.regime, data.confidence)}
                    </div>
                    
                    <div className="text-sm text-gray-600 mb-2">
                      Confiança: {(data.confidence * 100).toFixed(0)}%
                    </div>
                    
                    <Progress 
                      value={data.confidence * 100} 
                      className="h-2"
                    />
                    
                    <div className="mt-2 text-xs text-gray-500">
                      Duração: {data.regime_duration || 1} períodos
                    </div>
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
      {Object.entries(regimeData).map(([symbol, data]) => (
        data && (
          <Card key={symbol}>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span>{symbol}/USDT</span>
                  {getRegimeIcon(data.regime)}
                  {getRegimeBadge(data.regime, data.confidence)}
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-600">
                  <Clock className="h-4 w-4" />
                  Duração: {data.regime_duration || 1} períodos
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                
                {/* Descrição do Regime */}
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    {getRegimeDescription(data.regime)}
                  </p>
                </div>

                {/* Métricas do Regime */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  
                  {/* Força da Tendência */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-600" />
                      <span className="text-sm font-medium">Força da Tendência</span>
                    </div>
                    <div className="text-lg font-bold">
                      {(data.trend_strength * 100).toFixed(0)}%
                    </div>
                    <Progress value={data.trend_strength * 100} className="h-2" />
                  </div>

                  {/* Nível de Volatilidade */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-600" />
                      <span className="text-sm font-medium">Volatilidade</span>
                    </div>
                    <div className="text-lg font-bold">
                      {(data.volatility_level * 100).toFixed(0)}%
                    </div>
                    <Progress value={data.volatility_level * 100} className="h-2" />
                  </div>

                  {/* Perfil de Volume */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-green-600" />
                      <span className="text-sm font-medium">Volume</span>
                    </div>
                    <div className="text-lg font-bold capitalize">
                      {data.volume_profile}
                    </div>
                    <div className="text-xs text-gray-600">
                      Breakouts: {(data.breakout_frequency * 100).toFixed(0)}%
                    </div>
                  </div>

                </div>

                {/* Configuração de Estratégia */}
                <div className="border-t pt-4">
                  <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
                    {getStrategyIcon(data.strategy_config?.strategy_type)}
                    Configuração de Estratégia Adaptada
                  </h4>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Tipo:</span>
                      <div className="font-medium capitalize">
                        {data.strategy_config?.strategy_type?.replace('_', ' ') || 'N/A'}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-gray-600">Max Trades/Dia:</span>
                      <div className="font-medium">
                        {data.strategy_config?.max_trades_per_day || 'N/A'}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-gray-600">Quality Threshold:</span>
                      <div className="font-medium">
                        {data.strategy_config?.quality_threshold ? 
                         `${(data.strategy_config.quality_threshold * 100).toFixed(0)}%` : 'N/A'}
                      </div>
                    </div>
                    
                    <div>
                      <span className="text-gray-600">Risco/Trade:</span>
                      <div className="font-medium">
                        {data.strategy_config?.risk_per_trade ? 
                         `${(data.strategy_config.risk_per_trade * 100).toFixed(1)}%` : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Fatores Identificados */}
                {data.factors && data.factors.length > 0 && (
                  <div className="border-t pt-4">
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

                {/* Alertas */}
                {data.regime === 'high_volatility' && (
                  <div className="flex items-center gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    <span className="text-sm text-yellow-800">
                      <strong>Atenção:</strong> Alta volatilidade detectada. Trades muito seletivos recomendados.
                    </span>
                  </div>
                )}

                {data.regime === 'transitional' && (
                  <div className="flex items-center gap-2 p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <RotateCcw className="h-4 w-4 text-gray-600" />
                    <span className="text-sm text-gray-700">
                      <strong>Info:</strong> Mercado em transição. Aguardando definição de direção.
                    </span>
                  </div>
                )}

              </div>
            </CardContent>
          </Card>
        )
      ))}

      {/* Resumo de Trading */}
      <Card>
        <CardHeader>
          <CardTitle>Resumo de Trading por Regime</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <span className="font-medium">Trending Bull</span>
              </div>
              <div className="text-sm text-gray-600">
                • 3 trades/dia máximo<br/>
                • Seguir tendência<br/>
                • Risco: 0.8%/trade
              </div>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Minus className="h-4 w-4 text-blue-600" />
                <span className="font-medium">Ranging</span>
              </div>
              <div className="text-sm text-gray-600">
                • 4 trades/dia máximo<br/>
                • Mean reversion<br/>
                • Risco: 0.5%/trade
              </div>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-4 w-4 text-yellow-600" />
                <span className="font-medium">High Volatility</span>
              </div>
              <div className="text-sm text-gray-600">
                • 1 trade/dia máximo<br/>
                • Ultra seletivo<br/>
                • Risco: 0.3%/trade
              </div>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="h-4 w-4 text-red-600" />
                <span className="font-medium">Trending Bear</span>
              </div>
              <div className="text-sm text-gray-600">
                • 2 trades/dia máximo<br/>
                • Shorts em rallies<br/>
                • Risco: 0.6%/trade
              </div>
            </div>

          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default MarketRegimeMonitor;