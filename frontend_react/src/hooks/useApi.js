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
  return useApi(() => apiService.getSystemStatus(), [], 5000);
}

export function useMetrics() {
  return useApi(() => apiService.getMetrics(), [], 5000);
}

export function useEquity() {
  return useApi(() => apiService.getEquity(), [], 10000);
}

export function useBots() {
  return useApi(() => apiService.getBots(), [], 10000);
}

export function useRecentTrades() {
  return useApi(() => apiService.getRecentTrades(), [], 10000);
}

export function useLogs() {
  return useApi(() => apiService.getLogs(), [], 5000);
}

export function useConfig() {
  return useApi(() => apiService.getConfig(), [], null);
}