/**
 * API Client with React Query integration
 * Centralized API calls with caching, error handling, and real-time updates
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

// API Base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Query Keys
export const QUERY_KEYS = {
  DASHBOARD_STATS: 'dashboard-stats',
  TRADING_STATS: 'trading-stats',
  BOT_STATS: 'bot-stats',
  ACTIVE_TRADES: 'active-trades',
  TRADES_HISTORY: 'trades-history',
  SYSTEM_LOGS: 'system-logs',
  EQUITY_CURVE: 'equity-curve',
  PERFORMANCE_METRICS: 'performance-metrics',
  DATABASE_HEALTH: 'database-health'
};

// HTTP Client
class APIClient {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return await response.text();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // GET request
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return this.request(url, { method: 'GET' });
  }

  // POST request
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // PUT request
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  // DELETE request
  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
}

const apiClient = new APIClient();

// =====================================================
// API FUNCTIONS
// =====================================================

// Dashboard & Stats
export const fetchDashboardStats = () => apiClient.get('/api/dashboard/stats');
export const fetchTradingStats = () => apiClient.get('/api/db/stats/general');
export const fetchBotStats = () => apiClient.get('/api/db/stats/bots');
export const fetchPerformanceMetrics = (params = {}) => apiClient.get('/api/db/stats/performance', params);

// Trades
export const fetchActiveTrades = () => apiClient.get('/api/db/trades/active');
export const fetchTradesHistory = (params = {}) => apiClient.get('/api/db/trades', params);
export const createTrade = (tradeData) => apiClient.post('/api/db/trades', tradeData);
export const updateTrade = (tradeId, updateData) => apiClient.put(`/api/db/trades/${tradeId}`, updateData);

// Logs
export const fetchSystemLogs = (params = {}) => apiClient.get('/api/db/logs', params);
export const createLog = (logData) => apiClient.post('/api/db/logs', logData);

// Equity Curve
export const fetchEquityCurve = (params = {}) => apiClient.get('/api/db/equity-curve', params);
export const addEquityPoint = (pointData) => apiClient.post('/api/db/equity-curve', pointData);

// Admin Functions
export const clearTradingHistory = () => apiClient.post('/api/db/admin/clear-history?confirm=true');
export const checkDatabaseHealth = () => apiClient.get('/api/db/admin/health');
export const cleanupOldData = (daysToKeep = 30) => apiClient.post(`/api/db/admin/cleanup?days_to_keep=${daysToKeep}`);

// Bots
export const fetchTradingBots = () => apiClient.get('/api/db/bots');
export const updateBotStatus = (botId, status) => apiClient.put(`/api/db/bots/${botId}/status`, { status });

// =====================================================
// REACT QUERY HOOKS
// =====================================================

// Dashboard Stats Hook
export const useDashboardStats = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.DASHBOARD_STATS],
    queryFn: fetchDashboardStats,
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
};

// Trading Stats Hook
export const useTradingStats = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.TRADING_STATS],
    queryFn: fetchTradingStats,
    refetchInterval: 30000,
    staleTime: 15000,
    retry: 3,
  });
};

// Bot Stats Hook
export const useBotStats = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.BOT_STATS],
    queryFn: fetchBotStats,
    refetchInterval: 30000,
    staleTime: 15000,
    retry: 3,
  });
};

// Active Trades Hook
export const useActiveTrades = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.ACTIVE_TRADES],
    queryFn: fetchActiveTrades,
    refetchInterval: 10000, // More frequent for active trades
    staleTime: 5000,
    retry: 3,
  });
};

// Trades History Hook
export const useTradesHistory = (filters = {}) => {
  return useQuery({
    queryKey: [QUERY_KEYS.TRADES_HISTORY, filters],
    queryFn: () => fetchTradesHistory(filters),
    refetchInterval: 60000, // Less frequent for history
    staleTime: 30000,
    retry: 3,
    enabled: true, // Always enabled, but can be controlled by filters
  });
};

// System Logs Hook
export const useSystemLogs = (filters = {}) => {
  return useQuery({
    queryKey: [QUERY_KEYS.SYSTEM_LOGS, filters],
    queryFn: () => fetchSystemLogs(filters),
    refetchInterval: 5000, // Frequent for real-time logs
    staleTime: 2000,
    retry: 3,
  });
};

// Equity Curve Hook
export const useEquityCurve = (params = {}) => {
  return useQuery({
    queryKey: [QUERY_KEYS.EQUITY_CURVE, params],
    queryFn: () => fetchEquityCurve(params),
    refetchInterval: 60000,
    staleTime: 30000,
    retry: 3,
  });
};

// Performance Metrics Hook
export const usePerformanceMetrics = (params = {}) => {
  return useQuery({
    queryKey: [QUERY_KEYS.PERFORMANCE_METRICS, params],
    queryFn: () => fetchPerformanceMetrics(params),
    refetchInterval: 300000, // 5 minutes
    staleTime: 120000, // 2 minutes
    retry: 3,
  });
};

// Database Health Hook
export const useDatabaseHealth = () => {
  return useQuery({
    queryKey: [QUERY_KEYS.DATABASE_HEALTH],
    queryFn: checkDatabaseHealth,
    refetchInterval: 60000,
    staleTime: 30000,
    retry: 2,
  });
};

// Trading Bots Hook
export const useTradingBots = () => {
  return useQuery({
    queryKey: ['trading-bots'],
    queryFn: fetchTradingBots,
    refetchInterval: 30000,
    staleTime: 15000,
    retry: 3,
  });
};

// =====================================================
// MUTATION HOOKS
// =====================================================

// Clear Trading History Mutation
export const useClearTradingHistory = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: clearTradingHistory,
    onSuccess: () => {
      // Invalidate all related queries
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADING_STATS] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.BOT_STATS] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.ACTIVE_TRADES] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADES_HISTORY] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.SYSTEM_LOGS] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.EQUITY_CURVE] });
      
      toast.success('🗑️ Trading history cleared successfully!');
    },
    onError: (error) => {
      toast.error(`❌ Failed to clear history: ${error.message}`);
    },
  });
};

// Update Bot Status Mutation
export const useUpdateBotStatus = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ botId, status }) => updateBotStatus(botId, status),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.BOT_STATS] });
      queryClient.invalidateQueries({ queryKey: ['trading-bots'] });
      
      toast.success(`✅ Bot ${variables.botId} status updated to ${variables.status}`);
    },
    onError: (error) => {
      toast.error(`❌ Failed to update bot status: ${error.message}`);
    },
  });
};

// Create Trade Mutation
export const useCreateTrade = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createTrade,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.ACTIVE_TRADES] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADES_HISTORY] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADING_STATS] });
      
      toast.success('✅ Trade created successfully!');
    },
    onError: (error) => {
      toast.error(`❌ Failed to create trade: ${error.message}`);
    },
  });
};

// Update Trade Mutation
export const useUpdateTrade = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ tradeId, updateData }) => updateTrade(tradeId, updateData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.ACTIVE_TRADES] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADES_HISTORY] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.TRADING_STATS] });
      
      toast.success('✅ Trade updated successfully!');
    },
    onError: (error) => {
      toast.error(`❌ Failed to update trade: ${error.message}`);
    },
  });
};

// Cleanup Old Data Mutation
export const useCleanupOldData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: cleanupOldData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.SYSTEM_LOGS] });
      queryClient.invalidateQueries({ queryKey: [QUERY_KEYS.EQUITY_CURVE] });
      
      toast.success('🧹 Old data cleaned up successfully!');
    },
    onError: (error) => {
      toast.error(`❌ Failed to cleanup data: ${error.message}`);
    },
  });
};

// =====================================================
// UTILITY FUNCTIONS
// =====================================================

// Invalidate all queries (useful for manual refresh)
export const useInvalidateAllQueries = () => {
  const queryClient = useQueryClient();
  
  return () => {
    queryClient.invalidateQueries();
    toast.success('🔄 Data refreshed!');
  };
};

// Prefetch data (useful for preloading)
export const usePrefetchData = () => {
  const queryClient = useQueryClient();
  
  return {
    prefetchDashboardStats: () => queryClient.prefetchQuery({
      queryKey: [QUERY_KEYS.DASHBOARD_STATS],
      queryFn: fetchDashboardStats,
      staleTime: 15000,
    }),
    prefetchTradingStats: () => queryClient.prefetchQuery({
      queryKey: [QUERY_KEYS.TRADING_STATS],
      queryFn: fetchTradingStats,
      staleTime: 15000,
    }),
  };
};

// =====================================================
// TRADING RESTRICTIONS HOOKS
// =====================================================

// Get all trading restrictions
export const useTradingRestrictions = (activeOnly = false) => {
  return useQuery({
    queryKey: ['trading-restrictions', activeOnly],
    queryFn: async () => {
      const params = activeOnly ? '?active_only=true' : '';
      const response = await fetch(`${API_BASE_URL}/api/db/restrictions${params}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch restrictions: ${response.statusText}`);
      }
      return response.json();
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Create new trading restriction
export const useCreateRestriction = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (restrictionData) => {
      const response = await fetch(`${API_BASE_URL}/api/db/restrictions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(restrictionData),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create restriction');
      }
      
      return response.json();
    },
    onSuccess: () => {
      // Invalidate and refetch restrictions
      queryClient.invalidateQueries(['trading-restrictions']);
    },
  });
};

// Update trading restriction
export const useUpdateRestriction = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async ({ id, data }) => {
      const response = await fetch(`${API_BASE_URL}/api/db/restrictions/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update restriction');
      }
      
      return response.json();
    },
    onSuccess: () => {
      // Invalidate and refetch restrictions
      queryClient.invalidateQueries(['trading-restrictions']);
    },
  });
};

// Delete trading restriction
export const useDeleteRestriction = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (restrictionId) => {
      const response = await fetch(`${API_BASE_URL}/api/db/restrictions/${restrictionId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete restriction');
      }
      
      return response.json();
    },
    onSuccess: () => {
      // Invalidate and refetch restrictions
      queryClient.invalidateQueries(['trading-restrictions']);
    },
  });
};

// Check if trading is allowed at specific time
export const useCheckTradingAllowed = (checkTime = null) => {
  return useQuery({
    queryKey: ['trading-allowed', checkTime],
    queryFn: async () => {
      const params = checkTime ? `?check_time=${encodeURIComponent(checkTime)}` : '';
      const response = await fetch(`${API_BASE_URL}/api/db/restrictions/check${params}`);
      if (!response.ok) {
        throw new Error(`Failed to check trading status: ${response.statusText}`);
      }
      return response.json();
    },
    staleTime: 1 * 60 * 1000, // 1 minute
    cacheTime: 2 * 60 * 1000, // 2 minutes
    refetchInterval: 60 * 1000, // Refetch every minute
  });
};

export default apiClient;