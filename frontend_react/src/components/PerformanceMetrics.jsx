import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target, Award, BarChart3, AlertCircle } from 'lucide-react';
import { useMetrics } from '../hooks/useApi';
import { formatCurrency, formatPercentage, formatNumber, getMetricColor } from '../utils/formatters';

const MetricCard = ({ title, value, icon: Icon, formatter, suffix, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-20 mb-2"></div>
            <div className="h-8 bg-gray-200 rounded w-16"></div>
          </div>
          <div className="animate-pulse">
            <div className="h-8 w-8 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  const formattedValue = formatter ? formatter(value) : value;
  const colorClass = getMetricColor(value, formatter === formatPercentage);

  return (
    <div className="card hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className={`text-2xl font-bold ${colorClass}`}>
            {formattedValue}{suffix && <span className="text-sm ml-1">{suffix}</span>}
          </p>
        </div>
        <div className={`p-3 rounded-full ${
          value > 0 ? 'bg-green-100' : 
          value < 0 ? 'bg-red-100' : 
          'bg-gray-100'
        }`}>
          <Icon className={`h-6 w-6 ${
            value > 0 ? 'text-green-600' : 
            value < 0 ? 'text-red-600' : 
            'text-gray-600'
          }`} />
        </div>
      </div>
    </div>
  );
};

export default function PerformanceMetrics() {
  const { data: metrics, loading, error } = useMetrics();

  if (error) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="col-span-full">
          <div className="card bg-red-50 border-red-200">
            <div className="flex items-center space-x-2 text-red-700">
              <AlertCircle className="h-5 w-5" />
              <span>Failed to load performance metrics: {error}</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const metricsData = [
    {
      title: 'Total PnL',
      value: metrics?.total_pnl || 0,
      icon: DollarSign,
      formatter: formatCurrency,
    },
    {
      title: 'Total ROI',
      value: metrics?.total_roi || 0,
      icon: TrendingUp,
      formatter: formatPercentage,
    },
    {
      title: 'Daily PnL',
      value: metrics?.daily_pnl || 0,
      icon: metrics?.daily_pnl >= 0 ? TrendingUp : TrendingDown,
      formatter: formatCurrency,
    },
    {
      title: 'Active Trades',
      value: metrics?.active_trades || 0,
      icon: Target,
      formatter: formatNumber,
      suffix: 'trades',
    },
    {
      title: 'Win Rate',
      value: metrics?.win_rate || 0,
      icon: Award,
      formatter: formatPercentage,
    },
    {
      title: 'Total Trades',
      value: metrics?.total_trades || 0,
      icon: BarChart3,
      formatter: formatNumber,
      suffix: 'trades',
    },
    {
      title: 'Max Drawdown',
      value: metrics?.max_drawdown || 0,
      icon: TrendingDown,
      formatter: formatPercentage,
    },
    {
      title: 'Sharpe Ratio',
      value: metrics?.sharpe_ratio || 0,
      icon: BarChart3,
      formatter: (value) => formatNumber(value, 2),
    },
  ];

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Performance Metrics</h2>
        <div className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metricsData.map((metric, index) => (
          <MetricCard
            key={index}
            title={metric.title}
            value={metric.value}
            icon={metric.icon}
            formatter={metric.formatter}
            suffix={metric.suffix}
            loading={loading}
          />
        ))}
      </div>

      {/* Additional Summary */}
      {metrics && !loading && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="card bg-gradient-to-r from-primary-50 to-primary-100 border-primary-200">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-primary-900 mb-2">Today's Performance</h3>
              <div className="flex items-center justify-center space-x-4">
                <div>
                  <p className="text-sm text-primary-600">PnL</p>
                  <p className={`text-xl font-bold ${getMetricColor(metrics.daily_pnl)}`}>
                    {formatCurrency(metrics.daily_pnl)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-primary-600">Trades</p>
                  <p className="text-xl font-bold text-primary-900">
                    {metrics.daily_trades || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card bg-gradient-to-r from-green-50 to-green-100 border-green-200">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-green-900 mb-2">Success Rate</h3>
              <div className="flex items-center justify-center space-x-4">
                <div>
                  <p className="text-sm text-green-600">Win Rate</p>
                  <p className="text-xl font-bold text-green-900">
                    {formatPercentage(metrics.win_rate)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-green-600">Wins</p>
                  <p className="text-xl font-bold text-green-900">
                    {Math.round((metrics.total_trades * metrics.win_rate) / 100) || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="card bg-gradient-to-r from-warning-50 to-warning-100 border-warning-200">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-warning-900 mb-2">Risk Metrics</h3>
              <div className="flex items-center justify-center space-x-4">
                <div>
                  <p className="text-sm text-warning-600">Drawdown</p>
                  <p className="text-xl font-bold text-warning-900">
                    {formatPercentage(metrics.max_drawdown)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-warning-600">Sharpe</p>
                  <p className="text-xl font-bold text-warning-900">
                    {formatNumber(metrics.sharpe_ratio, 1)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}