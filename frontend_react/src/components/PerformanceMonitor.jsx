import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';

const PerformanceMonitor = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await apiService.getPerformanceMetrics();
        if (response.success) {
          setMetrics(response.data);
          setError(null);
        } else {
          setError(response.error);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Monitor</h3>
        <div className="animate-pulse">Loading metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Performance Monitor</h3>
        <div className="text-red-600">Error: {error}</div>
      </div>
    );
  }

  const getStatusColor = (percentage, thresholds = { warning: 70, danger: 85 }) => {
    if (percentage >= thresholds.danger) return 'text-red-600';
    if (percentage >= thresholds.warning) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getProgressBarColor = (percentage, thresholds = { warning: 70, danger: 85 }) => {
    if (percentage >= thresholds.danger) return 'bg-red-500';
    if (percentage >= thresholds.warning) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Performance Monitor</h3>
      
      {metrics && (
        <div className="space-y-4">
          {/* System Metrics */}
          <div>
            <h4 className="font-medium text-gray-700 mb-2">System Resources</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Memory Usage */}
              <div className="bg-gray-50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium">Memory</span>
                  <span className={`text-sm font-bold ${getStatusColor(metrics.system_metrics.memory_percent)}`}>
                    {metrics.system_metrics.memory_percent.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${getProgressBarColor(metrics.system_metrics.memory_percent)}`}
                    style={{ width: `${metrics.system_metrics.memory_percent}%` }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {metrics.system_metrics.memory_available_mb} MB available
                </div>
              </div>

              {/* CPU Usage */}
              <div className="bg-gray-50 p-3 rounded">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium">CPU</span>
                  <span className={`text-sm font-bold ${getStatusColor(metrics.system_metrics.cpu_percent)}`}>
                    {metrics.system_metrics.cpu_percent.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${getProgressBarColor(metrics.system_metrics.cpu_percent)}`}
                    style={{ width: `${metrics.system_metrics.cpu_percent}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Application Metrics */}
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Application Status</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-3 rounded">
                <div className="text-sm font-medium">Logs Buffer</div>
                <div className={`text-lg font-bold ${getStatusColor(metrics.application_metrics.logs_utilization)}`}>
                  {metrics.application_metrics.active_logs}/{metrics.application_metrics.max_logs}
                </div>
                <div className="text-xs text-gray-500">
                  {metrics.application_metrics.logs_utilization.toFixed(1)}% full
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded">
                <div className="text-sm font-medium">Trading System</div>
                <div className={`text-lg font-bold ${metrics.trading_metrics.system_running ? 'text-green-600' : 'text-red-600'}`}>
                  {metrics.trading_metrics.system_running ? 'Running' : 'Stopped'}
                </div>
                <div className="text-xs text-gray-500">
                  Status: {metrics.trading_metrics.task_status}
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded">
                <div className="text-sm font-medium">GC Collections</div>
                <div className="text-lg font-bold text-blue-600">
                  {metrics.system_metrics.gc_collections}
                </div>
                <div className="text-xs text-gray-500">
                  {metrics.system_metrics.gc_collected} objects collected
                </div>
              </div>
            </div>
          </div>

          {/* Last Updated */}
          <div className="text-xs text-gray-500 text-center">
            Last updated: {(() => {
              try {
                if (!metrics.timestamp) return 'N/A';
                const date = new Date(metrics.timestamp);
                return isNaN(date.getTime()) ? 'Invalid Date' : date.toLocaleTimeString();
              } catch (e) {
                return 'Invalid Date';
              }
            })()}
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitor;