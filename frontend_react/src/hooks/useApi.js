import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/api';

export function useApi(apiCall, dependencies = [], interval = null) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);
  const [isCircuitOpen, setIsCircuitOpen] = useState(false);

  const fetchData = useCallback(async () => {
    // Circuit breaker: skip if too many consecutive errors
    if (isCircuitOpen) {
      console.warn('Circuit breaker is open, skipping API call');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await apiCall();
      
      if (result.success) {
        setData(result.data);
        setConsecutiveErrors(0); // Reset error count on success
        setIsCircuitOpen(false); // Close circuit on success
      } else {
        setError(result.error);
        setConsecutiveErrors(prev => prev + 1);
      }
    } catch (err) {
      setError(err.message);
      setConsecutiveErrors(prev => prev + 1);
    } finally {
      setLoading(false);
    }
  }, dependencies);

  // Circuit breaker logic
  useEffect(() => {
    if (consecutiveErrors >= 5) {
      setIsCircuitOpen(true);
      console.warn('Circuit breaker opened due to consecutive errors');
      
      // Auto-reset circuit breaker after 2 minutes
      const resetTimer = setTimeout(() => {
        setIsCircuitOpen(false);
        setConsecutiveErrors(0);
        console.info('Circuit breaker reset');
      }, 120000); // 2 minutes
      
      return () => clearTimeout(resetTimer);
    }
  }, [consecutiveErrors]);

  useEffect(() => {
    fetchData();

    if (interval && !isCircuitOpen) {
      const intervalId = setInterval(fetchData, interval);
      return () => clearInterval(intervalId);
    }
  }, [fetchData, interval, isCircuitOpen]);

  return { data, loading, error, refetch: fetchData, isCircuitOpen };
}

export function useSystemStatus() {
  const { data, ...rest } = useApi(() => apiService.getSystemStatus(), [], 15000); // 5s → 15s
  
  // "Desembrulha" os dados do status de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}

export function useMetrics() {
  const { data, ...rest } = useApi(() => apiService.getMetrics(), [], 15000); // 5s → 15s
  
  // "Desembrulha" os dados das métricas de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}

//export function useEquity() {
//  return useApi(() => apiService.getEquity(), [], 10000);
//}
export function useEquity() {
  // 1. Pega o resultado completo do hook genérico
  const { data, ...rest } = useApi(() => apiService.getEquity(), [], 30000); // 10s → 30s
  
  // 2. Retorna um novo objeto, substituindo 'data' pelo array desembrulhado
  // O '...rest' mantém as outras propriedades (loading, error, etc.)
  return { data: data?.equity_curve || [], ...rest };
}

//export function useBots() {
//  return useApi(() => apiService.getBots(), [], 10000);
//}
export function useBots() {
  const { data, ...rest } = useApi(() => apiService.getBots(), [], 30000); // 10s → 30s
  
  // "Desembrulha" o array 'bots' de dentro do objeto 'data'
  return { data: data?.bots || [], ...rest };
}

//export function useRecentTrades() {
//  return useApi(() => apiService.getRecentTrades(), [], 10000);
//}
export function useRecentTrades() {
  const { data, ...rest } = useApi(() => apiService.getRecentTrades(), [], 30000); // 10s → 30s
  
  // "Desembrulha" o array 'trades' de dentro do objeto 'data'
  return { data: data?.trades || [], ...rest };
}

export function useLogs() {
  const { data, ...rest } = useApi(() => apiService.getLogs(), [], 20000); // 5s → 20s
  
  // "Desembrulha" o array 'logs' de dentro do objeto 'data'
  return { data: data?.logs || [], ...rest };
}

export function useConfig() {
  const { data, ...rest } = useApi(() => apiService.getConfig(), [], null);
  
  // "Desembrulha" os dados da configuração de dentro do objeto 'data'
  return { data: data || {}, ...rest };
}
