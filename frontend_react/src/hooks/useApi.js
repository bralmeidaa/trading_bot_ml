import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

export function useApi(apiCall, dependencies = [], interval = null) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await apiCall();
      
      if (result.success) {
        setData(result.data);
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    fetchData();

    if (interval) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
}

export function useSystemStatus() {
  const { data, ...rest } = useApi(() => apiService.getSystemStatus(), [], 5000);
  
  // "Desembrulha" os dados do status de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}

export function useMetrics() {
  const { data, ...rest } = useApi(() => apiService.getMetrics(), [], 5000);
  
  // "Desembrulha" os dados das métricas de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}

//export function useEquity() {
//  return useApi(() => apiService.getEquity(), [], 10000);
//}
export function useEquity() {
  const { data, ...rest } = useApi(() => apiService.getEquity(), [], 10000);
  
  // Validate and sanitize equity curve data
  const sanitizedEquityData = (data?.equity_curve || []).filter(point => {
    // Filter out invalid points
    if (!point || typeof point !== 'object') return false;
    if (!point.timestamp || !point.equity) return false;
    if (typeof point.equity !== 'number' || isNaN(point.equity)) return false;
    return true;
  }).map(point => ({
    timestamp: point.timestamp,
    equity: parseFloat(point.equity)
  }));
  
  return { data: sanitizedEquityData, ...rest };
}

//export function useBots() {
//  return useApi(() => apiService.getBots(), [], 10000);
//}
export function useBots() {
  const { data, ...rest } = useApi(() => apiService.getBots(), [], 10000);
  
  // "Desembrulha" o array 'bots' de dentro do objeto 'data'
  return { data: data?.bots || [], ...rest };
}

//export function useRecentTrades() {
//  return useApi(() => apiService.getRecentTrades(), [], 10000);
//}
export function useRecentTrades() {
  const { data, ...rest } = useApi(() => apiService.getRecentTrades(), [], 10000);
  
  // Validate and sanitize trade data
  const sanitizedTrades = (data?.trades || []).map(trade => ({
    ...trade,
    time: trade.time || 'N/A',
    entry_time: trade.entry_time || null,
    pnl: typeof trade.pnl === 'number' ? trade.pnl : 0,
    entry_price: typeof trade.entry_price === 'number' ? trade.entry_price : 0,
    exit_price: typeof trade.exit_price === 'number' ? trade.exit_price : null
  }));
  
  return { data: sanitizedTrades, ...rest };
}

export function useLogs() {
  const { data, ...rest } = useApi(() => apiService.getLogs(), [], 5000);
  
  // "Desembrulha" o array 'logs' de dentro do objeto 'data'
  return { data: data?.logs || [], ...rest };
}

export function useConfig() {
  const { data, ...rest } = useApi(() => apiService.getConfig(), [], null);
  
  // "Desembrulha" os dados da configuração de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}
