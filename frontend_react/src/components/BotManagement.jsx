import React, { useState, useEffect } from 'react';
import { Plus, Edit, Trash2, Settings, AlertCircle, CheckCircle, X } from 'lucide-react';
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

const BotCard = ({ bot, index, onEdit, onRemove, onToggle }) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${bot.enabled ? 'bg-green-500' : 'bg-gray-400'}`}></div>
          <h3 className="font-medium text-gray-900">{bot.symbol}</h3>
          <span className="text-sm text-gray-500">{bot.timeframe}</span>
        </div>
        <div className="flex items-center space-x-1">
          <button
            onClick={() => onToggle(index, !bot.enabled)}
            className={`p-1 rounded ${bot.enabled ? 'text-green-600 hover:bg-green-50' : 'text-gray-400 hover:bg-gray-50'}`}
            title={bot.enabled ? 'Disable bot' : 'Enable bot'}
          >
            <CheckCircle className="h-4 w-4" />
          </button>
          <button
            onClick={() => onEdit(index)}
            className="p-1 text-blue-600 hover:bg-blue-50 rounded"
            title="Edit bot"
          >
            <Edit className="h-4 w-4" />
          </button>
          <button
            onClick={() => onRemove(index)}
            className="p-1 text-red-600 hover:bg-red-50 rounded"
            title="Remove bot"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-500">Capital:</span>
          <span className="ml-1 font-medium">{(bot.capital_allocation * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Risk:</span>
          <span className="ml-1 font-medium">{(bot.max_risk_per_trade * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Stop Loss:</span>
          <span className="ml-1 font-medium">{(bot.stop_loss_pct * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Take Profit:</span>
          <span className="ml-1 font-medium">{(bot.take_profit_pct * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
};

const BotForm = ({ bot, isOpen, onClose, onSave, availableSymbols, isEditing = false }) => {
  const [formData, setFormData] = useState({
    symbol: '',
    timeframe: '5m',
    capital_allocation: 0.2,
    max_risk_per_trade: 0.025,
    confidence_threshold: 0.6,
    stop_loss_pct: 0.018,
    take_profit_pct: 0.035,
    enabled: true
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (bot && isOpen) {
      setFormData({
        symbol: bot.symbol || '',
        timeframe: bot.timeframe || '5m',
        capital_allocation: bot.capital_allocation || 0.2,
        max_risk_per_trade: bot.max_risk_per_trade || 0.025,
        confidence_threshold: bot.confidence_threshold || 0.6,
        stop_loss_pct: bot.stop_loss_pct || 0.018,
        take_profit_pct: bot.take_profit_pct || 0.035,
        enabled: bot.enabled !== undefined ? bot.enabled : true
      });
    } else if (!isEditing && isOpen) {
      // Reset form for new bot
      setFormData({
        symbol: '',
        timeframe: '5m',
        capital_allocation: 0.2,
        max_risk_per_trade: 0.025,
        confidence_threshold: 0.6,
        stop_loss_pct: 0.018,
        take_profit_pct: 0.035,
        enabled: true
      });
    }
  }, [bot, isOpen, isEditing]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await onSave(formData);
      onClose();
    } catch (error) {
      console.error('Error saving bot:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            {isEditing ? 'Edit Bot' : 'Add New Bot'}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Trading Symbol *
            </label>
            <select
              value={formData.symbol}
              onChange={(e) => handleChange('symbol', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              required
            >
              <option value="">Select a symbol</option>
              {availableSymbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Timeframe *
            </label>
            <select
              value={formData.timeframe}
              onChange={(e) => handleChange('timeframe', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              required
            >
              {TIMEFRAMES.map(tf => (
                <option key={tf.value} value={tf.value}>{tf.label}</option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Capital Allocation (%)
              </label>
              <input
                type="number"
                value={formData.capital_allocation * 100}
                onChange={(e) => handleChange('capital_allocation', parseFloat(e.target.value) / 100)}
                min="1"
                max="100"
                step="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Risk per Trade (%)
              </label>
              <input
                type="number"
                value={formData.max_risk_per_trade * 100}
                onChange={(e) => handleChange('max_risk_per_trade', parseFloat(e.target.value) / 100)}
                min="0.1"
                max="10"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Stop Loss (%)
              </label>
              <input
                type="number"
                value={formData.stop_loss_pct * 100}
                onChange={(e) => handleChange('stop_loss_pct', parseFloat(e.target.value) / 100)}
                min="0.1"
                max="10"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Take Profit (%)
              </label>
              <input
                type="number"
                value={formData.take_profit_pct * 100}
                onChange={(e) => handleChange('take_profit_pct', parseFloat(e.target.value) / 100)}
                min="0.1"
                max="20"
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold
            </label>
            <input
              type="number"
              value={formData.confidence_threshold}
              onChange={(e) => handleChange('confidence_threshold', parseFloat(e.target.value))}
              min="0.1"
              max="1"
              step="0.01"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              required
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="enabled"
              checked={formData.enabled}
              onChange={(e) => handleChange('enabled', e.target.checked)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="enabled" className="ml-2 block text-sm text-gray-900">
              Enable this bot
            </label>
          </div>

          <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {loading && <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>}
              <span>{isEditing ? 'Update Bot' : 'Add Bot'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default function BotManagement({ bots, onBotsChange }) {
  const [showForm, setShowForm] = useState(false);
  const [editingBot, setEditingBot] = useState(null);
  const [editingIndex, setEditingIndex] = useState(-1);
  const [availableSymbols, setAvailableSymbols] = useState([]);
  const [botCount, setBotCount] = useState({ current_count: 0, maximum_allowed: 5, can_add_more: true });

  useEffect(() => {
    loadAvailableSymbols();
    loadBotCount();
  }, []);

  useEffect(() => {
    loadBotCount();
  }, [bots]);

  const loadAvailableSymbols = async () => {
    try {
      const response = await apiService.getAvailableSymbols();
      if (response.symbols) {
        setAvailableSymbols(response.symbols);
      }
    } catch (error) {
      console.error('Error loading available symbols:', error);
      notificationService.error('Failed to load available symbols');
    }
  };

  const loadBotCount = async () => {
    try {
      const response = await apiService.getBotCount();
      setBotCount(response);
    } catch (error) {
      console.error('Error loading bot count:', error);
    }
  };

  const handleAddBot = () => {
    if (!botCount.can_add_more) {
      notificationService.error('Maximum of 5 bots allowed');
      return;
    }
    setEditingBot(null);
    setEditingIndex(-1);
    setShowForm(true);
  };

  const handleEditBot = (index) => {
    setEditingBot(bots[index]);
    setEditingIndex(index);
    setShowForm(true);
  };

  const handleSaveBot = async (botData) => {
    try {
      if (editingIndex >= 0) {
        // Update existing bot
        const response = await apiService.updateBot(editingIndex, botData);
        if (response.success) {
          notificationService.success('Bot updated successfully');
          onBotsChange();
        }
      } else {
        // Add new bot
        const response = await apiService.addBot(botData);
        if (response.success) {
          notificationService.success('Bot added successfully');
          onBotsChange();
        }
      }
    } catch (error) {
      console.error('Error saving bot:', error);
      notificationService.error(error.message || 'Failed to save bot');
      throw error;
    }
  };

  const handleRemoveBot = async (index) => {
    if (!window.confirm('Are you sure you want to remove this bot?')) {
      return;
    }

    try {
      const response = await apiService.removeBot(index);
      if (response.success) {
        notificationService.success('Bot removed successfully');
        onBotsChange();
      }
    } catch (error) {
      console.error('Error removing bot:', error);
      notificationService.error('Failed to remove bot');
    }
  };

  const handleToggleBot = async (index, enabled) => {
    try {
      const response = await apiService.updateBot(index, { enabled });
      if (response.success) {
        notificationService.success(`Bot ${enabled ? 'enabled' : 'disabled'} successfully`);
        onBotsChange();
      }
    } catch (error) {
      console.error('Error toggling bot:', error);
      notificationService.error('Failed to toggle bot');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-gray-900">Bot Management</h3>
          <p className="text-sm text-gray-600 mt-1">
            Manage up to 5 trading bots with custom configurations
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-600">
            {botCount.current_count} / {botCount.maximum_allowed} bots
          </div>
          <button
            onClick={handleAddBot}
            disabled={!botCount.can_add_more}
            className="btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="h-4 w-4" />
            <span>Add Bot</span>
          </button>
        </div>
      </div>

      {bots && bots.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {bots.map((bot, index) => (
            <BotCard
              key={`${bot.symbol}_${bot.timeframe}_${index}`}
              bot={bot}
              index={index}
              onEdit={handleEditBot}
              onRemove={handleRemoveBot}
              onToggle={handleToggleBot}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No bots configured</h3>
          <p className="text-gray-600 mb-4">
            Add your first trading bot to get started
          </p>
          <button
            onClick={handleAddBot}
            className="btn-primary flex items-center space-x-2 mx-auto"
          >
            <Plus className="h-4 w-4" />
            <span>Add Your First Bot</span>
          </button>
        </div>
      )}

      {!botCount.can_add_more && (
        <div className="bg-warning-50 border border-warning-200 rounded-md p-4">
          <div className="flex">
            <AlertCircle className="h-5 w-5 text-warning-400" />
            <div className="ml-3">
              <p className="text-sm text-warning-800">
                You have reached the maximum limit of 5 bots. Remove a bot to add a new one.
              </p>
            </div>
          </div>
        </div>
      )}

      <BotForm
        bot={editingBot}
        isOpen={showForm}
        onClose={() => setShowForm(false)}
        onSave={handleSaveBot}
        availableSymbols={availableSymbols}
        isEditing={editingIndex >= 0}
      />
    </div>
  );
}