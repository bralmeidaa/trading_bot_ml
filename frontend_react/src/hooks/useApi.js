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
  // 1. Pega o resultado completo do hook genérico
  const { data, ...rest } = useApi(() => apiService.getEquity(), [], 10000);
  
  // 2. Retorna um novo objeto, substituindo 'data' pelo array desembrulhado
  // O '...rest' mantém as outras propriedades (loading, error, etc.)
  return { data: data?.equity_curve || [], ...rest };
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
  
  // "Desembrulha" o array 'trades' de dentro do objeto 'data'
  return { data: data?.trades || [], ...rest };
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
