import React, { useState } from 'react';
import { Activity, FileText, TrendingUp, TrendingDown, AlertCircle, RefreshCw } from 'lucide-react';
import { useRecentTrades, useLogs } from '../hooks/useApi';
import { formatCurrency, formatDateTime, getStatusColor } from '../utils/formatters';

const TabButton = ({ active, onClick, children, icon: Icon }) => (
  <button
    onClick={onClick}
    className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-200 ${
      active
        ? 'bg-primary-100 text-primary-700 border border-primary-200'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
    }`}
  >
    <Icon className="h-4 w-4" />
    <span>{children}</span>
  </button>
);

const TradeRow = ({ trade }) => {
  const isLong = trade.direction === 'long' || trade.direction === 1;
  const isProfit = trade.pnl > 0;
  
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <div className={`p-1 rounded-full mr-3 ${
            isLong ? 'bg-success-100' : 'bg-danger-100'
          }`}>
            {isLong ? (
              <TrendingUp className="h-4 w-4 text-success-600" />
            ) : (
              <TrendingDown className="h-4 w-4 text-danger-600" />
            )}
          </div>
          <div>
            <div className="text-sm font-medium text-gray-900">{trade.symbol}</div>
            <div className="text-sm text-gray-500">
              {isLong ? 'Long' : 'Short'}
            </div>
          </div>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={`status-indicator ${getStatusColor(trade.status)}`}>
          {trade.status}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className={`text-sm font-medium ${
          isProfit ? 'text-success-600' : 'text-danger-600'
        }`}>
          {formatCurrency(trade.pnl)}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
        {formatCurrency(trade.entry_price)}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
        {trade.exit_price ? formatCurrency(trade.exit_price) : '--'}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {formatDateTime(trade.time)}
      </td>
    </tr>
  );
};

const RecentTrades = () => {
  const { data: trades, loading, error, refetch } = useRecentTrades();

  if (error) {
    return (
      <div className="text-center py-8">
        <AlertCircle className="h-12 w-12 text-danger-500 mx-auto mb-4" />
        <p className="text-danger-600">Failed to load trades: {error}</p>
        <button
          onClick={refetch}
          className="mt-2 btn-secondary flex items-center space-x-2 mx-auto"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Retry</span>
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="h-4 bg-gray-200 rounded w-20"></div>
              <div className="h-4 bg-gray-200 rounded w-16"></div>
              <div className="h-4 bg-gray-200 rounded w-24"></div>
              <div className="h-4 bg-gray-200 rounded w-20"></div>
              <div className="h-4 bg-gray-200 rounded w-20"></div>
              <div className="h-4 bg-gray-200 rounded w-32"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Trade
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Status
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              PnL
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Entry Price
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Exit Price
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Time
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {trades && trades.length > 0 ? (
            trades.map((trade, index) => (
              <TradeRow key={trade.id || index} trade={trade} />
            ))
          ) : (
            <tr>
              <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Waiting for trade data...</p>
                <p className="text-sm">Recent trades will appear here once the system starts trading</p>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

const LogRow = ({ log }) => {
  const getLevelColor = (level) => {
    switch (level.toUpperCase()) {
      case 'ERROR':
        return 'text-danger-600 bg-danger-50';
      case 'SUCCESS':
        return 'text-success-600 bg-success-50';
      case 'WARNING':
        return 'text-warning-600 bg-warning-50';
      case 'INFO':
      default:
        return 'text-blue-600 bg-blue-50';
    }
  };

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-500">
        {log.timestamp}
      </td>
      <td className="px-6 py-3 whitespace-nowrap">
        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getLevelColor(log.level)}`}>
          {log.level}
        </span>
      </td>
      <td className="px-6 py-3 text-sm text-gray-900">
        {log.message}
      </td>
      <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-500">
        {log.source}
      </td>
    </tr>
  );
};

const SystemLogs = () => {
  const { data: logs, loading, error, refetch } = useLogs();

  if (error) {
    return (
      <div className="text-center py-8">
        <AlertCircle className="h-12 w-12 text-danger-500 mx-auto mb-4" />
        <p className="text-danger-600">Failed to load logs: {error}</p>
        <button
          onClick={refetch}
          className="mt-2 btn-secondary flex items-center space-x-2 mx-auto"
        >
          <RefreshCw className="h-4 w-4" />
          <span>Retry</span>
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="h-4 bg-gray-200 rounded w-16"></div>
              <div className="h-4 bg-gray-200 rounded w-20"></div>
              <div className="h-4 bg-gray-200 rounded flex-1"></div>
              <div className="h-4 bg-gray-200 rounded w-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Time
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Level
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Message
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Source
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {logs && logs.length > 0 ? (
            logs.slice().reverse().map((log, index) => (
              <LogRow key={index} log={log} />
            ))
          ) : (
            <tr>
              <td colSpan="4" className="px-6 py-8 text-center text-gray-500">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No system logs available</p>
                <p className="text-sm">Logs will appear here when the system is active</p>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default function TradesAndLogs() {
  const [activeTab, setActiveTab] = useState('trades');

  return (
    <div className="card mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Activity Monitor</h2>
          <p className="text-sm text-gray-600 mt-1">
            Recent trades and system logs
          </p>
        </div>
        
        {/* Tab Navigation */}
        <div className="flex space-x-2">
          <TabButton
            active={activeTab === 'trades'}
            onClick={() => setActiveTab('trades')}
            icon={Activity}
          >
            Recent Trades
          </TabButton>
          <TabButton
            active={activeTab === 'logs'}
            onClick={() => setActiveTab('logs')}
            icon={FileText}
          >
            System Logs
          </TabButton>
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {activeTab === 'trades' ? <RecentTrades /> : <SystemLogs />}
      </div>
    </div>
  );
}