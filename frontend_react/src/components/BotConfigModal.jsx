import React, { useState, useEffect } from 'react';
import { X, Save, AlertCircle } from 'lucide-react';
import { apiService } from '../services/api';
import { notificationService } from '../utils/notifications';

const TIMEFRAMES = [
  { value: '1m', label: '1 Minute' },
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '30m', label: '30 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: '1 Day' }
];

const POPULAR_SYMBOLS = [
  'BTC/USDT',
  'ETH/USDT',
  'BNB/USDT',
  'ADA/USDT',
  'XRP/USDT',
  'SOL/USDT',
  'DOT/USDT',
  'LINK/USDT',
  'MATIC/USDT',
  'AVAX/USDT',
  'UNI/USDT',
  'LTC/USDT'
];

export default function BotConfigModal({ bot, isOpen, onClose, onSave }) {
  const [config, setConfig] = useState({
    symbol: '',
    timeframe: '5m',
    enabled: true,
    risk_per_trade: 0.02,
    max_positions: 3,
    stop_loss: 0.02,
    take_profit: 0.04,
    confidence_threshold: 0.6
  });
  const [loading, setSaving] = useState(false);
  const [customSymbol, setCustomSymbol] = useState('');
  const [showCustomSymbol, setShowCustomSymbol] = useState(false);

  useEffect(() => {
    if (bot && isOpen) {
      setConfig({
        symbol: bot.symbol || '',
        timeframe: bot.timeframe || '5m',
        enabled: bot.enabled || true,
        risk_per_trade: bot.risk_per_trade || 0.02,
        max_positions: bot.max_positions || 3,
        stop_loss: bot.stop_loss || 0.02,
        take_profit: bot.take_profit || 0.04,
        confidence_threshold: bot.confidence_threshold || 0.6
      });
      setShowCustomSymbol(!POPULAR_SYMBOLS.includes(bot.symbol));
      setCustomSymbol(POPULAR_SYMBOLS.includes(bot.symbol) ? '' : bot.symbol);
    }
  }, [bot, isOpen]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const finalSymbol = showCustomSymbol ? customSymbol : config.symbol;
      
      if (!finalSymbol) {
        notificationService.error('Please select or enter a trading symbol');
        setSaving(false);
        return;
      }

      const updatedConfig = {
        ...config,
        symbol: finalSymbol
      };

      const result = await apiService.updateBotConfig(bot.id, updatedConfig);
      
      if (result.success) {
        notificationService.success(`Bot configuration updated successfully`);
        onSave(updatedConfig);
        onClose();
      } else {
        notificationService.error(`Failed to update bot: ${result.error}`);
      }
    } catch (error) {
      notificationService.error(`Error updating bot: ${error.message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Configure Bot: {bot?.symbol}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Trading Symbol */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Trading Symbol
            </label>
            <div className="space-y-3">
              {!showCustomSymbol && (
                <select
                  value={config.symbol}
                  onChange={(e) => setConfig({ ...config, symbol: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                >
                  <option value="">Select a symbol</option>
                  {POPULAR_SYMBOLS.map(symbol => (
                    <option key={symbol} value={symbol}>{symbol}</option>
                  ))}
                </select>
              )}
              
              {showCustomSymbol && (
                <input
                  type="text"
                  value={customSymbol}
                  onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
                  placeholder="e.g., BTC/USDT"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              )}
              
              <button
                type="button"
                onClick={() => setShowCustomSymbol(!showCustomSymbol)}
                className="text-sm text-primary-600 hover:text-primary-800"
              >
                {showCustomSymbol ? 'Choose from popular symbols' : 'Enter custom symbol'}
              </button>
            </div>
          </div>

          {/* Timeframe */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Timeframe
            </label>
            <select
              value={config.timeframe}
              onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {TIMEFRAMES.map(tf => (
                <option key={tf.value} value={tf.value}>{tf.label}</option>
              ))}
            </select>
          </div>

          {/* Risk Management */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-900">Risk Management</h4>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk per Trade (%)
              </label>
              <input
                type="number"
                value={config.risk_per_trade * 100}
                onChange={(e) => setConfig({ ...config, risk_per_trade: parseFloat(e.target.value) / 100 })}
                min="0.1"
                max="10"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Positions
              </label>
              <input
                type="number"
                value={config.max_positions}
                onChange={(e) => setConfig({ ...config, max_positions: parseInt(e.target.value) })}
                min="1"
                max="10"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Stop Loss (%)
              </label>
              <input
                type="number"
                value={config.stop_loss * 100}
                onChange={(e) => setConfig({ ...config, stop_loss: parseFloat(e.target.value) / 100 })}
                min="0.5"
                max="10"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Take Profit (%)
              </label>
              <input
                type="number"
                value={config.take_profit * 100}
                onChange={(e) => setConfig({ ...config, take_profit: parseFloat(e.target.value) / 100 })}
                min="1"
                max="20"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                ML Confidence Threshold (%)
                <span className="text-xs text-gray-500 ml-1">(Higher = more conservative)</span>
              </label>
              <input
                type="number"
                value={config.confidence_threshold * 100}
                onChange={(e) => setConfig({ ...config, confidence_threshold: parseFloat(e.target.value) / 100 })}
                min="30"
                max="90"
                step="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Controls how confident the ML model must be before making trades. Higher values = fewer but more confident trades.
              </p>
            </div>
          </div>

          {/* Bot Status */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="enabled"
              checked={config.enabled}
              onChange={(e) => setConfig({ ...config, enabled: e.target.checked })}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="enabled" className="ml-2 block text-sm text-gray-900">
              Enable this bot
            </label>
          </div>

          {/* Warning */}
          <div className="bg-warning-50 border border-warning-200 rounded-md p-4">
            <div className="flex">
              <AlertCircle className="h-5 w-5 text-warning-400" />
              <div className="ml-3">
                <p className="text-sm text-warning-800">
                  Changes will take effect after restarting the trading system.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end space-x-3 p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={loading}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
            <Save className="h-4 w-4" />
            <span>Save Changes</span>
          </button>
        </div>
      </div>
    </div>
  );
}