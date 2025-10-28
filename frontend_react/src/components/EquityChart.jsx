import React, { useState, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp, Calendar, RefreshCw } from 'lucide-react';
import { useEquityCurve } from '../services/apiClient';
import { formatTableDate, getDateRangeForPeriod, TIME_PERIODS } from '../utils/dateUtils';
import { ChartSkeleton, ErrorState, EmptyState } from './LoadingSkeletons';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function EquityChart() {
  const [selectedPeriod, setSelectedPeriod] = useState('7D');
  const [selectedBot, setSelectedBot] = useState(null); // null = all bots
  
  // Calculate date range based on selected period
  const dateRange = useMemo(() => {
    const { startDate, endDate } = getDateRangeForPeriod(selectedPeriod);
    const days = Math.ceil((new Date(endDate) - new Date(startDate)) / (1000 * 60 * 60 * 24));
    return { days, startDate, endDate };
  }, [selectedPeriod]);

  // Fetch equity curve data
  const { 
    data: equityData, 
    isLoading, 
    error, 
    refetch 
  } = useEquityCurve({
    days: dateRange.days,
    bot_id: selectedBot
  });

  // Process chart data
  const chartData = useMemo(() => {
    if (!equityData || !Array.isArray(equityData) || equityData.length === 0) {
      return {
        labels: [],
        datasets: [],
      };
    }

    // Sort data by timestamp
    const sortedData = [...equityData].sort((a, b) => 
      new Date(a.timestamp) - new Date(b.timestamp)
    );

    // Process labels with safe date formatting
    const labels = sortedData.map(point => 
      formatTableDate(point.timestamp)
    );

    // Calculate equity values
    const equityValues = sortedData.map(point => parseFloat(point.equity || 0));
    const drawdownValues = sortedData.map(point => parseFloat(point.drawdown_pct || 0));

    // Calculate some basic stats
    const initialEquity = equityValues[0] || 0;
    const currentEquity = equityValues[equityValues.length - 1] || 0;
    const totalReturn = initialEquity > 0 ? ((currentEquity - initialEquity) / initialEquity) * 100 : 0;
    const maxDrawdown = Math.max(...drawdownValues);

    return {
      labels,
      datasets: [
        {
          label: 'Equity',
          data: equityValues,
          borderColor: totalReturn >= 0 ? '#10B981' : '#EF4444',
          backgroundColor: totalReturn >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: equityValues.length > 50 ? 0 : 3,
          pointHoverRadius: 5,
        },
        {
          label: 'Drawdown %',
          data: drawdownValues.map(val => -Math.abs(val)), // Make drawdown negative for display
          borderColor: '#F59E0B',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 1,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 3,
          yAxisID: 'y1',
        }
      ],
      stats: {
        initialEquity,
        currentEquity,
        totalReturn,
        maxDrawdown,
        dataPoints: equityValues.length
      }
    };
  }, [equityData]);

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        },
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            
            if (label === 'Equity') {
              return `${label}: $${value.toFixed(2)}`;
            } else if (label === 'Drawdown %') {
              return `${label}: ${Math.abs(value).toFixed(2)}%`;
            }
            return `${label}: ${value}`;
          },
          title: function(context) {
            return `Time: ${context[0].label}`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        },
        ticks: {
          maxTicksLimit: 10,
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Equity ($)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toFixed(0);
          }
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Drawdown (%)'
        },
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          callback: function(value) {
            return Math.abs(value).toFixed(1) + '%';
          }
        }
      },
    },
  };

  // Loading state
  if (isLoading) {
    return <ChartSkeleton height="400px" />;
  }

  // Error state
  if (error) {
    return (
      <ErrorState 
        message={`Failed to load equity curve: ${error.message}`}
        onRetry={refetch}
        className="h-96"
      />
    );
  }

  // Empty state
  if (!equityData || equityData.length === 0) {
    return (
      <EmptyState
        title="No Equity Data"
        description="No equity curve data available for the selected period."
        icon={<TrendingUp className="w-8 h-8 text-gray-400" />}
        className="h-96"
      />
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div className="flex items-center gap-3 mb-4 sm:mb-0">
          <div className="w-10 h-10 bg-green-100 dark:bg-green-900/20 rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Equity Curve
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Portfolio performance over time
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Time Period Selector */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-gray-500" />
            <select
              value={selectedPeriod}
              onChange={(e) => setSelectedPeriod(e.target.value)}
              className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {TIME_PERIODS.map(period => (
                <option key={period.value} value={period.value}>
                  {period.label}
                </option>
              ))}
            </select>
          </div>

          {/* Refresh Button */}
          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Stats Summary */}
      {chartData.stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600 dark:text-gray-400">Initial</div>
            <div className="text-lg font-semibold text-gray-900 dark:text-white">
              ${chartData.stats.initialEquity.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600 dark:text-gray-400">Current</div>
            <div className="text-lg font-semibold text-gray-900 dark:text-white">
              ${chartData.stats.currentEquity.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Return</div>
            <div className={`text-lg font-semibold ${chartData.stats.totalReturn >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {chartData.stats.totalReturn >= 0 ? '+' : ''}{chartData.stats.totalReturn.toFixed(2)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</div>
            <div className="text-lg font-semibold text-red-600 dark:text-red-400">
              -{chartData.stats.maxDrawdown.toFixed(2)}%
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="h-80">
        <Line data={chartData} options={chartOptions} />
      </div>

      {/* Data Info */}
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center">
        Showing {chartData.stats?.dataPoints || 0} data points for {selectedPeriod.toLowerCase()}
        {selectedBot && ` • Bot: ${selectedBot}`}
      </div>
    </div>
  );
}