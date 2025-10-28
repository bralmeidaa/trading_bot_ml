import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Target, Award, BarChart3, AlertCircle, Activity } from 'lucide-react';
import { useTradingStats, usePerformanceMetrics } from '../services/apiClient';
import { MetricsSkeleton, ErrorState } from './LoadingSkeletons';

// Utility functions for formatting
const formatCurrency = (value) => {
  if (value === null || value === undefined) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
};

const formatPercentage = (value, decimals = 2) => {
  if (value === null || value === undefined) return '0.00%';
  return `${Number(value).toFixed(decimals)}%`;
};

const formatNumber = (value, decimals = 0) => {
  if (value === null || value === undefined) return '0';
  return Number(value).toFixed(decimals);
};

const getMetricColor = (value, isPercentage = false) => {
  if (value === null || value === undefined) return 'text-gray-500 dark:text-gray-400';
  
  if (isPercentage || typeof value === 'number') {
    if (value > 0) return 'text-green-600 dark:text-green-400';
    if (value < 0) return 'text-red-600 dark:text-red-400';
  }
  
  return 'text-gray-900 dark:text-white';
};

const MetricCard = ({ 
  title, 
  value, 
  icon: Icon, 
  formatter, 
  suffix, 
  loading = false,
  trend = null,
  description = null 
}) => {
  const formattedValue = formatter ? formatter(value) : value;
  const colorClass = getMetricColor(value, formatter === formatPercentage);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow duration-200">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</p>
            {trend !== null && (
              <div className={`flex items-center text-xs ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {trend >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                <span className="ml-1">{Math.abs(trend).toFixed(1)}%</span>
              </div>
            )}
          </div>
          <p className={`text-2xl font-bold ${colorClass}`}>
            {formattedValue}
            {suffix && <span className="text-sm ml-1 font-normal">{suffix}</span>}
          </p>
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{description}</p>
          )}
        </div>
        <div className={`p-3 rounded-full ${
          value > 0 ? 'bg-green-100 dark:bg-green-900/20' : 
          value < 0 ? 'bg-red-100 dark:bg-red-900/20' : 
          'bg-gray-100 dark:bg-gray-700'
        }`}>
          <Icon className={`h-6 w-6 ${
            value > 0 ? 'text-green-600 dark:text-green-400' : 
            value < 0 ? 'text-red-600 dark:text-red-400' : 
            'text-gray-600 dark:text-gray-400'
          }`} />
        </div>
      </div>
    </div>
  );
};

export default function PerformanceMetrics() {
  // Fetch trading stats and performance metrics
  const { 
    data: tradingStats, 
    isLoading: statsLoading, 
    error: statsError 
  } = useTradingStats();

  const { 
    data: performanceMetrics, 
    isLoading: metricsLoading, 
    error: metricsError 
  } = usePerformanceMetrics();

  const isLoading = statsLoading || metricsLoading;
  const error = statsError || metricsError;

  // Loading state
  if (isLoading) {
    return <MetricsSkeleton />;
  }

  // Error state
  if (error) {
    return (
      <ErrorState 
        message={`Failed to load performance metrics: ${error.message}`}
        className="mb-8"
      />
    );
  }

  // Calculate derived metrics
  const stats = tradingStats || {};
  const metrics = performanceMetrics || {};

  // Safe number parsing with fallbacks
  const totalPnl = parseFloat(stats.total_pnl || 0);
  const totalTrades = parseInt(stats.total_trades || 0);
  const winningTrades = parseInt(stats.winning_trades || 0);
  const losingTrades = parseInt(stats.losing_trades || 0);
  const activeTrades = parseInt(stats.active_trades || 0);
  const dailyPnl = parseFloat(stats.daily_pnl || 0);
  
  // Calculate win rate properly (as percentage)
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  
  // Calculate ROI if we have initial capital
  const initialCapital = parseFloat(stats.initial_capital || 1200);
  const totalRoi = initialCapital > 0 ? (totalPnl / initialCapital) * 100 : 0;

  // Advanced metrics from performance endpoint
  const sharpeRatio = parseFloat(metrics.sharpe_ratio || 0);
  const profitFactor = parseFloat(metrics.profit_factor || 0);
  const maxDrawdown = parseFloat(metrics.max_drawdown_pct || 0);
  const avgWin = parseFloat(metrics.avg_win || 0);
  const avgLoss = parseFloat(metrics.avg_loss || 0);

  const metricsData = [
    {
      title: 'Total P&L',
      value: totalPnl,
      icon: DollarSign,
      formatter: formatCurrency,
      description: `From ${totalTrades} trades`,
    },
    {
      title: 'Total ROI',
      value: totalRoi,
      icon: TrendingUp,
      formatter: formatPercentage,
      description: `On $${initialCapital.toFixed(0)} capital`,
    },
    {
      title: 'Daily P&L',
      value: dailyPnl,
      icon: dailyPnl >= 0 ? TrendingUp : TrendingDown,
      formatter: formatCurrency,
      description: 'Today\'s performance',
    },
    {
      title: 'Active Trades',
      value: activeTrades,
      icon: Target,
      formatter: formatNumber,
      suffix: 'trades',
      description: 'Currently running',
    },
    {
      title: 'Win Rate',
      value: winRate,
      icon: Award,
      formatter: formatPercentage,
      description: `${winningTrades}W / ${losingTrades}L`,
    },
    {
      title: 'Sharpe Ratio',
      value: sharpeRatio,
      icon: BarChart3,
      formatter: (val) => formatNumber(val, 2),
      description: 'Risk-adjusted return',
    },
    {
      title: 'Profit Factor',
      value: profitFactor,
      icon: Activity,
      formatter: (val) => formatNumber(val, 2),
      description: 'Gross profit / Gross loss',
    },
    {
      title: 'Max Drawdown',
      value: -Math.abs(maxDrawdown), // Make it negative for display
      icon: TrendingDown,
      formatter: (val) => formatPercentage(Math.abs(val)),
      description: 'Largest peak-to-trough decline',
    },
  ];

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Performance Metrics</h2>
        <div className="text-sm text-gray-500 dark:text-gray-400">
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
            description={metric.description}
          />
        ))}
      </div>

      {/* Performance Summary */}
      {(tradingStats || performanceMetrics) && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border border-blue-200 dark:border-blue-700 rounded-lg p-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">Today's Performance</h3>
              <p className={`text-2xl font-bold ${dailyPnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                {formatCurrency(dailyPnl)}
              </p>
              <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                {dailyPnl >= 0 ? 'Profitable day' : 'Loss day'}
              </p>
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border border-green-200 dark:border-green-700 rounded-lg p-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-green-900 dark:text-green-100 mb-2">Risk Metrics</h3>
              <div className="space-y-2">
                <div>
                  <span className="text-sm text-green-700 dark:text-green-300">Sharpe Ratio: </span>
                  <span className="font-semibold text-green-900 dark:text-green-100">{formatNumber(sharpeRatio, 2)}</span>
                </div>
                <div>
                  <span className="text-sm text-green-700 dark:text-green-300">Profit Factor: </span>
                  <span className="font-semibold text-green-900 dark:text-green-100">{formatNumber(profitFactor, 2)}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border border-purple-200 dark:border-purple-700 rounded-lg p-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-100 mb-2">Trade Statistics</h3>
              <div className="space-y-2">
                <div>
                  <span className="text-sm text-purple-700 dark:text-purple-300">Total Trades: </span>
                  <span className="font-semibold text-purple-900 dark:text-purple-100">{totalTrades}</span>
                </div>
                <div>
                  <span className="text-sm text-purple-700 dark:text-purple-300">Win Rate: </span>
                  <span className="font-semibold text-purple-900 dark:text-purple-100">{formatPercentage(winRate)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
