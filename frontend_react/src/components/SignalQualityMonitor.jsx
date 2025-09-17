import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { AlertTriangle, CheckCircle, XCircle, TrendingUp } from 'lucide-react';

const SignalQualityMonitor = () => {
  const [qualityData, setQualityData] = useState({
    current_quality_score: 0,
    signals_evaluated: 0,
    signals_passed: 0,
    signals_rejected: 0,
    avg_quality_score: 0,
    pass_rate: 0,
    layer_scores: {
      technical: 0,
      market_structure: 0,
      binance_sentiment: 0,
      ml_confidence: 0
    },
    recent_rejections: []
  });

  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchQualityData();
    const interval = setInterval(fetchQualityData, 5000); // Atualizar a cada 5 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchQualityData = async () => {
    try {
      const response = await fetch('/api/signal-quality');
      if (response.ok) {
        const data = await response.json();
        setQualityData(data);
      }
    } catch (error) {
      console.error('Erro ao buscar dados de qualidade:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getQualityColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityBadge = (score) => {
    if (score >= 0.8) return <Badge className="bg-green-100 text-green-800">Excelente</Badge>;
    if (score >= 0.6) return <Badge className="bg-yellow-100 text-yellow-800">Boa</Badge>;
    return <Badge className="bg-red-100 text-red-800">Baixa</Badge>;
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Monitor de Qualidade dos Sinais
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse">Carregando dados de qualidade...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Card Principal de Qualidade */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Monitor de Qualidade dos Sinais
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Score Atual */}
            <div className="text-center">
              <div className={`text-3xl font-bold ${getQualityColor(qualityData.current_quality_score)}`}>
                {(qualityData.current_quality_score * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Score Atual</div>
              {getQualityBadge(qualityData.current_quality_score)}
            </div>

            {/* Taxa de Aprovação */}
            <div className="text-center">
              <div className={`text-3xl font-bold ${getQualityColor(qualityData.pass_rate)}`}>
                {(qualityData.pass_rate * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Taxa de Aprovação</div>
              <div className="flex items-center justify-center gap-2 mt-1">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <span className="text-sm">{qualityData.signals_passed}</span>
                <XCircle className="h-4 w-4 text-red-600" />
                <span className="text-sm">{qualityData.signals_rejected}</span>
              </div>
            </div>

            {/* Média Geral */}
            <div className="text-center">
              <div className={`text-3xl font-bold ${getQualityColor(qualityData.avg_quality_score)}`}>
                {(qualityData.avg_quality_score * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Média Geral</div>
              <div className="text-xs text-gray-500">
                {qualityData.signals_evaluated} sinais avaliados
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Scores por Camada */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Scores por Camada de Filtro</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {Object.entries(qualityData.layer_scores).map(([layer, score]) => (
              <div key={layer} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium capitalize">
                    {layer.replace('_', ' ')}
                  </span>
                  <span className={`text-sm font-bold ${getQualityColor(score)}`}>
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress 
                  value={score * 100} 
                  className="h-2"
                />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Rejeições Recentes */}
      {qualityData.recent_rejections && qualityData.recent_rejections.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <AlertTriangle className="h-5 w-5 text-yellow-600" />
              Rejeições Recentes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {qualityData.recent_rejections.slice(0, 5).map((rejection, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div>
                    <span className="font-medium">{rejection.symbol}</span>
                    <span className="text-sm text-gray-600 ml-2">
                      Score: {(rejection.quality_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(rejection.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default SignalQualityMonitor;