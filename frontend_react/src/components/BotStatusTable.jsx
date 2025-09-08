import React, { useState } from 'react';
import { Settings, Power, PowerOff, AlertCircle, Bot } from 'lucide-react';
import { useBots } from '../hooks/useApi';
import { apiService } from '../services/api';
import { notificationService } from '../utils/notifications';
import { formatCurrency, formatNumber, getStatusColor } from '../utils/formatters';

const StatusBadge = ({ status }) => {
  const colorClass = getStatusColor(status);
  
  return (
    <span className={`status-indicator ${colorClass}`}>
      {status}
    </span>
  );
};

const BotRow = ({ bot, onToggle, onConfigure }) => {
  const [loading, setLoading] = useState(false);

  const handleToggle = async () => {
    setLoading(true);
    try {
      const result = await apiService.toggleBot(bot.id, !bot.enabled);
      if (result.success) {
        notificationService.success(
          `Bot ${bot.symbol} ${bot.enabled ? 'disabled' : 'enabled'} successfully`
        );
        onToggle();
      } else {
        notificationService.error(`Failed to toggle bot: ${result.error}`);
      }
    } catch (error) {
      notificationService.error(`Error toggling bot: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <Bot className="h-5 w-5 text-gray-400 mr-2" />
          <div>
            <div className="text-sm font-medium text-gray-900">{bot.symbol}</div>
            <div className="text-sm text-gray-500">{bot.timeframe}</div>
          </div>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <StatusBadge status={bot.status} />
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className={`text-sm font-medium ${
          bot.pnl >= 0 ? 'text-success-600' : 'text-danger-600'
        }`}>
          {formatCurrency(bot.pnl)}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
        {formatNumber(bot.trades, 0)}
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <button
            onClick={handleToggle}
            disabled={loading}
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
              bot.enabled ? 'bg-success-600' : 'bg-gray-200'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                bot.enabled ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
          <span className="ml-2 text-sm text-gray-600">
            {bot.enabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
        <button
          onClick={() => onConfigure(bot)}
          className="text-primary-600 hover:text-primary-900 transition-colors duration-200"
        >
          <Settings className="h-4 w-4" />
        </button>
      </td>
    </tr>
  );
};

export default function BotStatusTable() {
  const { data: bots, loading, error, refetch } = useBots();
  const [selectedBot, setSelectedBot] = useState(null);

  const handleConfigure = (bot) => {
    setSelectedBot(bot);
    // TODO: Open configuration modal
    notificationService.info(`Configuration for ${bot.symbol} - Coming soon!`);
  };

  if (error) {
    return (
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Bot Status</h2>
        </div>
        <div className="text-center py-8">
          <AlertCircle className="h-12 w-12 text-danger-500 mx-auto mb-4" />
          <p className="text-danger-600">Failed to load bot status: {error}</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="card mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Bot Status</h2>
        </div>
        <div className="animate-pulse">
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="flex items-center space-x-4">
                <div className="h-4 bg-gray-200 rounded w-24"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
                <div className="h-4 bg-gray-200 rounded w-20"></div>
                <div className="h-4 bg-gray-200 rounded w-12"></div>
                <div className="h-4 bg-gray-200 rounded w-16"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const activeBots = (Array.isArray(bots) ? bots : []).filter(bot => bot.enabled);
  const totalPnL = (Array.isArray(bots) ? bots : []).reduce((sum, bot) => sum + (bot.pnl || 0), 0);
  const totalTrades = (Array.isArray(bots) ? bots : []).reduce((sum, bot) => sum + (bot.trades || 0), 0);

  return (
    <div className="card mb-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Bot Status</h2>
          <p className="text-sm text-gray-600 mt-1">
            Individual bot performance and controls
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-600">
            {activeBots.length} of {bots?.length || 0} bots active
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-primary-600">Active Bots</p>
              <p className="text-2xl font-bold text-primary-900">{activeBots.length}</p>
            </div>
            <Power className="h-8 w-8 text-primary-600" />
          </div>
        </div>
        
        <div className="bg-success-50 border border-success-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-success-600">Total Bot PnL</p>
              <p className={`text-2xl font-bold ${
                totalPnL >= 0 ? 'text-success-900' : 'text-danger-900'
              }`}>
                {formatCurrency(totalPnL)}
              </p>
            </div>
            <div className={`h-8 w-8 rounded-full flex items-center justify-center ${
              totalPnL >= 0 ? 'bg-success-200' : 'bg-danger-200'
            }`}>
              <span className={`text-sm font-bold ${
                totalPnL >= 0 ? 'text-success-800' : 'text-danger-800'
              }`}>
                {totalPnL >= 0 ? '+' : '-'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-warning-600">Total Trades</p>
              <p className="text-2xl font-bold text-warning-900">{totalTrades}</p>
            </div>
            <Bot className="h-8 w-8 text-warning-600" />
          </div>
        </div>
      </div>

      {/* Bot Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Bot
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                PnL
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Trades
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Enabled
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {bots && bots.length > 0 ? (
              bots.map((bot, index) => (
                <BotRow
                  key={bot.id || index}
                  bot={bot}
                  onToggle={refetch}
                  onConfigure={handleConfigure}
                />
              ))
            ) : (
              <tr>
                <td colSpan="6" className="px-6 py-8 text-center text-gray-500">
                  <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Waiting for bot data...</p>
                  <p className="text-sm">Start the trading system to see bot status</p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}